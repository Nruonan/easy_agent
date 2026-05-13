# my_plan_solve_agent.py
import json
from typing import Optional, Dict, List
from hello_agents import HelloAgentsLLM, Config, Message
from demo.demo2.agent import Agent


# 默认提示词模板
DEFAULT_PROMPTS = {
    "plan": (
        "你是一个顶级的AI规划专家。你的任务是将用户提出的复杂问题分解成一个由多个简单步骤组成的行动计划。\n"
        "请确保计划中的每个步骤都是一个独立的、可执行的子任务，并且严格按照逻辑顺序排列。\n"
        "你的输出必须是一个Python列表，其中每个元素都是一个描述子任务的字符串。\n\n"
        "问题: {question}\n\n"
        "请严格按照以下格式输出你的计划:\n"
        '```python\n["步骤1", "步骤2", "步骤3", ...]\n```'
    ),
    "solve": (
        "你是一位顶级的AI执行专家。你的任务是严格按照给定的计划，一步步地解决问题。\n"
        "你将收到原始问题、完整的计划、以及到目前为止已经完成的步骤和结果。\n"
        "请你专注于解决·当前步骤·，并仅输出该步骤的最终答案，不要输出任何额外的解释或对话。\n\n"
        "# 原始问题:\n{question}\n\n"
        "# 完整计划:\n{plan}\n\n"
        "# 历史步骤与结果:\n{history}\n\n"
        "# 当前步骤:\n{current_step}\n\n"
        '请仅输出针对"当前步骤"的回答:'
    ),
}


class MyPlanAndSolveAgent(Agent):
    """
    规划执行Agent - 通过"规划→逐步执行"流程解决复杂问题
    """

    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        custom_prompts: Optional[Dict[str, str]] = None,
    ):
        super().__init__(name, llm, system_prompt, config)
        self.prompts = custom_prompts if custom_prompts else DEFAULT_PROMPTS
        print(f"✅ {name} 初始化完成")

    def run(self, input_text: str, **kwargs) -> str:
        """运行规划执行Agent：规划 → 逐步执行"""
        print(f"\n🤖 {self.name} 开始处理任务: {input_text}")

        # 第一步：制定计划
        plan_text = self._plan(input_text, **kwargs)
        steps = self._parse_steps(plan_text)
        print(f"\n📋 计划制定完成，共 {len(steps)} 个步骤")
        for i, step in enumerate(steps, 1):
            print(f"  步骤{i}: {step}")

        # 第二步：逐步执行
        history_lines: List[str] = []
        last_result = ""
        for i, step in enumerate(steps, 1):
            print(f"\n--- 执行步骤 {i}/{len(steps)} ---")
            history = "\n".join(history_lines) if history_lines else "暂无"
            result = self._solve_step(input_text, plan_text, step, history, **kwargs)
            last_result = result
            history_lines.append(f"步骤{i}: {step}\n结果: {result}")
            print(f"  结果: {result}")

        print(f"\n✅ 执行完成")

        # 保存到历史记录
        self.add_message(Message(input_text, "user"))
        self.add_message(Message(last_result, "assistant"))

        return last_result

    def _plan(self, question: str, **kwargs) -> str:
        """制定执行计划"""
        prompt = self.prompts["plan"].format(question=question)
        messages = self._build_messages(prompt)
        return self.llm.invoke(messages, **kwargs)

    def _solve_step(
        self,
        question: str,
        plan_text: str,
        current_step: str,
        history: str,
        **kwargs,
    ) -> str:
        """执行单个步骤"""
        prompt = self.prompts["solve"].format(
            question=question,
            plan=plan_text,
            history=history,
            current_step=current_step,
        )
        messages = self._build_messages(prompt)
        return self.llm.invoke(messages, **kwargs)

    def _parse_steps(self, plan_text: str) -> List[str]:
        """解析计划文本中的步骤列表，优先解析 Python 列表格式"""
        # 尝试提取 ```python [...] ``` 代码块
        import re
        code_match = re.search(r"```(?:python)?\s*(\[.*?\])\s*```", plan_text, re.DOTALL)
        raw = code_match.group(1) if code_match else plan_text.strip()

        # 尝试直接用 json 解析（Python 列表兼容 JSON）
        try:
            steps = json.loads(raw)
            if isinstance(steps, list) and all(isinstance(s, str) for s in steps):
                return steps
        except (json.JSONDecodeError, ValueError):
            pass

        # 回退：逐行解析
        steps = []
        for line in plan_text.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            for prefix in ["步骤", "Step", "step"]:
                if line.startswith(prefix):
                    _, _, content = line.partition(":")
                    if content.strip():
                        steps.append(content.strip())
                    break
            else:
                for sep in [".", "）", ")"]:
                    if sep in line:
                        idx = line.index(sep)
                        prefix = line[:idx].strip()
                        if prefix.isdigit():
                            steps.append(line[idx + 1:].strip())
                            break

        if not steps:
            steps = [plan_text.strip()]
        return steps

    def _build_messages(self, prompt: str) -> list:
        """构建消息列表"""
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages
