# my_llm.py
import os
from typing import Optional
from openai import OpenAI
from hello_agents import HelloAgentsLLM


class MyLLM(HelloAgentsLLM):
    """
    一个自定义的LLM客户端，通过继承增加了对阿里云百炼大模型的支持。
    """

    def __init__(
            self,
            model: Optional[str] = None,
            api_key: Optional[str] = None,
            base_url: Optional[str] = None,
            provider: Optional[str] = "auto",
            **kwargs
    ):
        if provider == "dashscope":
            print("正在使用阿里云百炼大模型")
            self.provider = "dashscope"

            # 百炼凭证
            self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
            self.base_url = base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"

            if not self.api_key:
                raise ValueError("百炼 API key 未找到，请设置 DASHSCOPE_API_KEY 环境变量。")

            # 默认模型
            self.model = model or os.getenv("LLM_MODEL_ID") or "qwen-plus"
            self.temperature = kwargs.get('temperature', 0.7)
            self.max_tokens = kwargs.get('max_tokens')
            self.timeout = kwargs.get('timeout', 60)

            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)

        else:
            # 非 dashscope provider，走父类逻辑
            super().__init__(model=model, api_key=api_key, base_url=base_url, provider=provider, **kwargs)