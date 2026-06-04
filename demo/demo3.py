import ast
import operator
import os
from datetime import datetime
from typing import Any, Iterator, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from langchain.agents.middleware import wrap_tool_call
from langchain_core.messages import ToolMessage

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


DEFAULT_MODEL = "glm-5.1"
DEFAULT_SYSTEM_PROMPT = (
    "你是一个通用 AI Agent。请先判断用户问题是否需要使用工具，"
    "需要时调用工具，不需要时直接回答。回答应简洁、准确，并使用中文。"
    "如果工具提示搜索失败或信息不足，请基于已有信息说明限制，不要反复调用同一个搜索工具。"
)
MAX_WEB_SEARCH_RESULTS = 3
MAX_WEB_SEARCH_QUERY_LENGTH = 200
MAX_WEB_SEARCH_RESULT_CHARS = 700
MAX_WEB_SEARCH_TOTAL_CHARS = 2600
WEB_SEARCH_TIMEOUT_SECONDS = 10
AGENT_RECURSION_LIMIT = 8


class DashScopeAgentError(RuntimeError):
    """Raised when the DashScope agent fails at runtime."""


def _patch_tongyi_bind_tools(chat_tongyi_cls: Any) -> None:
    if getattr(chat_tongyi_cls, "_easy_agent_bind_tools_patched", False):
        return

    original_bind_tools = chat_tongyi_cls.bind_tools

    def bind_tools_without_none_kwargs(self: Any, tools: Any, **kwargs: Any) -> Any:
        clean_kwargs = {key: value for key, value in kwargs.items() if value is not None}
        return original_bind_tools(self, tools, **clean_kwargs)

    chat_tongyi_cls.bind_tools = bind_tools_without_none_kwargs
    chat_tongyi_cls._easy_agent_bind_tools_patched = True


def _format_dashscope_error(resp: Any) -> str:
    status_code = resp.get("status_code")
    code = resp.get("code")
    message = resp.get("message")
    request_id = resp.get("request_id")
    return (
        "DashScope 调用失败："
        f"status_code={status_code}, code={code}, "
        f"message={message}, request_id={request_id}"
    )


def _load_env() -> None:
    """Load .env when python-dotenv is installed."""
    if load_dotenv is not None:
        load_dotenv()


def _get_required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"请先在环境变量或 .env 中设置 {name}")
    return value


def _safe_calculate(expression: str) -> str:
    if not expression or not expression.strip():
        return "计算表达式不能为空"

    operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
    }

    def eval_node(node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = eval_node(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp):
            op = operators.get(type(node.op))
            if op is None:
                raise ValueError("不支持的运算符")
            return op(eval_node(node.left), eval_node(node.right))
        raise ValueError("只支持数字和 + - * / % ** 运算")

    try:
        tree = ast.parse(expression, mode="eval")
        return str(eval_node(tree.body))
    except ZeroDivisionError:
        return "计算失败：不能除以 0"
    except (SyntaxError, ValueError, TypeError) as exc:
        return f"计算失败：{exc}"


def _truncate_text(value: Any, max_chars: int) -> str:
    text = str(value or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _safe_web_search(query: str) -> str:
    query = (query or "").strip()
    if not query:
        return "搜索失败：query 不能为空。请停止调用 web_search，并要求用户补充搜索关键词。"
    if len(query) > MAX_WEB_SEARCH_QUERY_LENGTH:
        return f"搜索失败：query 过长，最多 {MAX_WEB_SEARCH_QUERY_LENGTH} 个字符。请停止调用 web_search，并压缩关键词。"

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "搜索失败：未配置 TAVILY_API_KEY。请停止调用 web_search，并说明当前无法联网搜索。"

    try:
        from tavily import TavilyClient
    except ImportError:
        return "搜索失败：未安装 tavily 包。请停止调用 web_search，并说明当前无法联网搜索。"

    try:
        client = TavilyClient(api_key=api_key)
        response = client.search(
            query=query,
            max_results=MAX_WEB_SEARCH_RESULTS,
            include_answer=True,
            timeout=WEB_SEARCH_TIMEOUT_SECONDS,
        )
    except Exception as exc:
        return f"搜索失败：{type(exc).__name__}: {_truncate_text(exc, 160)}。请停止调用 web_search，并基于已有信息回答。"
    if not isinstance(response, dict):
        return "搜索失败：Tavily 返回了无法解析的结果。请停止调用 web_search，并基于已有信息回答。"

    lines = [f"搜索查询：{query}"]
    answer = _truncate_text(response.get("answer"), 600)
    if answer:
        lines.append(f"直接答案：{answer}")

    results = response.get("results", [])
    if not isinstance(results, list) or not results:
        lines.append("未找到可用搜索结果。请停止调用 web_search，并说明当前搜索没有返回有效结果。")
        return "\n".join(lines)

    lines.append("相关结果：")
    for index, item in enumerate(results[:MAX_WEB_SEARCH_RESULTS], 1):
        if not isinstance(item, dict):
            continue
        title = _truncate_text(item.get("title"), 120) or "无标题"
        url = _truncate_text(item.get("url"), 240)
        content = _truncate_text(item.get("content"), MAX_WEB_SEARCH_RESULT_CHARS)
        lines.append(f"{index}. {title}")
        if url:
            lines.append(f"   URL: {url}")
        if content:
            lines.append(f"   摘要: {content}")

    return _truncate_text("\n".join(lines), MAX_WEB_SEARCH_TOTAL_CHARS)


def _build_tools() -> list[Any]:
    try:
        from langchain_core.tools import tool
    except ImportError as exc:
        raise ImportError(
            "缺少 langchain-core，请先安装 LangChain 相关依赖。"
        ) from exc

    @tool
    def calculator(expression: str) -> str:
        """计算数学表达式。支持数字以及 + - * / % ** 运算。"""
        return _safe_calculate(expression)

    @tool
    def current_time(timezone_name: str = "Asia/Shanghai") -> str:
        """查询指定 IANA 时区的当前时间，例如 Asia/Shanghai。"""
        try:
            timezone = ZoneInfo(timezone_name)
        except ZoneInfoNotFoundError:
            return f"未知时区：{timezone_name}"
        return datetime.now(timezone).strftime("%Y-%m-%d %H:%M:%S %Z")

    @tool("web_search")  # Custom name
    def search(query: str) -> str:
        """Search the web for information."""
        return _safe_web_search(query)

    return [calculator, current_time, search]



def build_agent(
    model: Optional[str] = None,
    temperature: float = 0.2,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> Any:
    _load_env()
    _get_required_env("DASHSCOPE_API_KEY")

    try:
        from langchain.agents import create_agent
        from langchain_community.chat_models import ChatTongyi
    except ImportError as exc:
        raise ImportError(
            "缺少 LangChain DashScope 依赖，请安装 langchain、langchain-community 和 dashscope。"
        ) from exc

    _patch_tongyi_bind_tools(ChatTongyi)

    class SafeChatTongyi(ChatTongyi):
        def completion_with_retry(self, **kwargs: Any) -> Any:
            resp = self.client.call(**kwargs)
            if resp.get("status_code") == 200:
                return resp
            raise DashScopeAgentError(_format_dashscope_error(resp))

    llm = SafeChatTongyi(
        model=model or os.getenv("LLM_MODEL_ID") or DEFAULT_MODEL,
        temperature=temperature,
    )

    return create_agent(
        model=llm,
        tools=_build_tools(),
        system_prompt=system_prompt
        # middleware=handle_tool_errors
    )

@wrap_tool_call
def handle_tool_errors(request, handler):
    """使用自定义消息处理工具执行错误。"""
    try:
        return handler(request)
    except Exception as e:
        # 向模型返回自定义错误消息
        return ToolMessage(
            content=f"工具错误：请检查您的输入并重试。({str(e)})",
            tool_call_id=request.tool_call["id"]
        )

def _extract_last_text(result: Any) -> str:
    messages = result.get("messages", []) if isinstance(result, dict) else []
    if not messages:
        return str(result)

    last_message = messages[-1]
    content = getattr(last_message, "content", None)
    if isinstance(content, str):
        return content
    if content is not None:
        return str(content)
    return str(last_message)


def _extract_stream_text(chunk: Any) -> str:
    payload = chunk
    if isinstance(chunk, dict):
        if chunk.get("type") != "messages":
            return ""
        payload = chunk.get("data")

    message_chunk = payload[0] if isinstance(payload, tuple) and payload else payload
    metadata = payload[1] if isinstance(payload, tuple) and len(payload) > 1 else {}
    if isinstance(metadata, dict) and metadata.get("langgraph_node") not in (None, "model"):
        return ""

    content_blocks = getattr(message_chunk, "content_blocks", None)
    if isinstance(content_blocks, list):
        text_parts = [
            block.get("text", "")
            for block in content_blocks
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        if text_parts:
            return "".join(text_parts)

    content = getattr(message_chunk, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = [
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        if text_parts:
            return "".join(text_parts)

    return ""


def _stream_agent_chunks(runnable_agent: Any, input_data: dict[str, Any]) -> Iterator[Any]:
    try:
        yield from runnable_agent.stream(
            input_data,
            config={"recursion_limit": AGENT_RECURSION_LIMIT},
            stream_mode="messages",
            version="v2",
        )
    except TypeError as exc:
        if "version" not in str(exc):
            raise
        yield from runnable_agent.stream(
            input_data,
            config={"recursion_limit": AGENT_RECURSION_LIMIT},
            stream_mode="messages",
        )


def stream_agent(question: str, agent: Optional[Any] = None) -> Iterator[str]:
    if not question or not question.strip():
        raise ValueError("question 不能为空")

    runnable_agent = agent or build_agent()
    input_data = {"messages": [{"role": "user", "content": question.strip()}]}
    try:
        for chunk in _stream_agent_chunks(runnable_agent, input_data):
            text = _extract_stream_text(chunk)
            if text:
                yield text
    except KeyError as exc:
        if str(exc) == "'request'":
            raise DashScopeAgentError(
                "DashScope 返回了 HTTP 错误，但当前 langchain-community/dashscope "
                "组合在包装错误时隐藏了真实 message。请先检查 DASHSCOPE_API_KEY、"
                "模型名称和工具调用权限；如仍失败，建议升级 dashscope 与 "
                "langchain-community。"
            ) from exc
        raise


def run_agent(question: str, agent: Optional[Any] = None) -> str:
    if not question or not question.strip():
        raise ValueError("question 不能为空")

    runnable_agent = agent or build_agent()
    message = [{"role": "user", "content": question.strip()}]
    try:
        result = runnable_agent.invoke(
            {"messages": message},
            config={"recursion_limit": AGENT_RECURSION_LIMIT},
        )
    except KeyError as exc:
        if str(exc) == "'request'":
            raise DashScopeAgentError(
                "DashScope 返回了 HTTP 错误，但当前 langchain-community/dashscope "
                "组合在包装错误时隐藏了真实 message。请先检查 DASHSCOPE_API_KEY、"
                "模型名称和工具调用权限；如仍失败，建议升级 dashscope 与 "
                "langchain-community。"
            ) from exc
        raise
    return _extract_last_text(result)


if __name__ == "__main__":
    demo_question = "今年的世界杯去哪里举行，决赛在哪个城市"
    # for response_chunk in stream_agent(demo_question):
    #     print(response_chunk, end="", flush=True)
    print(run_agent(demo_question))
    print()
