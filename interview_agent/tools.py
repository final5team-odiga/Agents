# tool.py
# LLM 연동, 파일 저장 등

import os
from dotenv import load_dotenv
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from openai import AzureOpenAI

# .env 로드 (최초 1회만 실행)
dotenv_path = Path(r'C:\Users\kanga\Desktop\MS\interview_agent\.env')
load_dotenv(dotenv_path=dotenv_path, override=True)

class AzureOpenAILLM:
    def __init__(self):
        self.api_key = os.getenv("AZURE_API_KEY")
        self.azure_endpoint = os.getenv("AZURE_API_BASE")
        self.api_version = os.getenv("AZURE_API_VERSION")
        self.deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        if not all([self.api_key, self.azure_endpoint, self.api_version, self.deployment_name]):
            raise ValueError("Azure OpenAI 환경변수가 설정되지 않았습니다.")
        self.client = AzureOpenAI(
            api_key=self.api_key,
            azure_endpoint=self.azure_endpoint,
            api_version=self.api_version
        )

    def call(self, messages: Union[str, List[Dict[str, str]]], **kwargs) -> str:
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        try:
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=messages,
                temperature=0.7,
                max_tokens=4000
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"LLM 호출 오류: {str(e)}")
            raise RuntimeError(f"LLM 요청 실패: {str(e)}")

def save_results_to_file(results, agents, save_path):
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("=== 여행 인터뷰 결과 ===\n\n")
        for agent in agents:
            question = agent.instruction['page_instruction']['source']
            refined = results.get(agent.name, "(결과 없음)")
            f.write(f"[질문] {question}\n")
            f.write(f"[답변]\n{refined}\n")
            f.write("\n" + "-" * 40 + "\n\n")

