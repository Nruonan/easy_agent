# 配置好.env中的大模型API和数据库连接
import sys
import os
import io
from pathlib import Path

# 修复Windows终端GBK编码无法输出emoji的问题
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from dotenv import load_dotenv

# 显式加载项目根目录的.env（必须在import hello_agents之前）
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent.parent / ".env")

from hello_agents import SimpleAgent, HelloAgentsLLM, ToolRegistry
from hello_agents.tools import MemoryTool, RAGTool
# 创建LLM实例
llm = HelloAgentsLLM(provider="dashscope")

# 创建Agent
agent = SimpleAgent(
    name="智能助手",
    llm=llm,
    system_prompt="你是一个有记忆和知识检索能力的AI助手"
)

# 创建工具注册表
tool_registry = ToolRegistry()

# 添加记忆工具
memory_tool = MemoryTool(user_id="user123")
tool_registry.register_tool(memory_tool)

# 添加RAG工具
rag_tool = RAGTool(knowledge_base_path="./knowledge_base")
tool_registry.register_tool(rag_tool)

# 为Agent配置工具
agent.tool_registry = tool_registry

# 开始对话
response = agent.run("你好！请记住我叫xzn，我是一名Python开发者")
print(response)