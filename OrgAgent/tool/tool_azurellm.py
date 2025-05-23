import os
from crewai import BaseLLM
from openai import AzureOpenAI
from typing import Union, List, Dict

class AzureOpenAILLM(BaseLLM):
    def __init__(self):
        self.api_key = os.getenv("AZURE_API_KEY")
        self.azure_endpoint = os.getenv("AZURE_API_BASE")
        self.api_version = os.getenv("AZURE_API_VERSION")
        self.deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        if not all([self.api_key, self.azure_endpoint, self.api_version, self.deployment_name]):
            raise ValueError("Missing Azure OpenAI credentials.")
        super().__init__(model=f"azure/{self.deployment_name}")
        self.client = AzureOpenAI(
            api_key=self.api_key,
            azure_endpoint=self.azure_endpoint,
            api_version=self.api_version
        )

    def call(self, messages: Union[str, List[Dict[str, str]]], **kwargs):
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        response = self.client.chat.completions.create(
            model=self.deployment_name,
            messages=messages,
            temperature=0.7,
            max_tokens=4000
        )
        return response.choices[0].message.content.strip()