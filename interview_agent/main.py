import os
import json
import datetime
from dotenv import load_dotenv
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from openai import AzureOpenAI

import interview_agents  # 인터뷰 에이전트 정의 파일

# .env 파일 로드
dotenv_path = Path(r'C:\Users\kanga\Desktop\MS\0a\.env')
load_dotenv(dotenv_path=dotenv_path, override=True)

# LLM Base 클래스
class BaseLLM:
    def __init__(self, model: Optional[str] = None):
        self.model = model

    def call(self, messages: Union[str, List[Dict[str, str]]]) -> Union[str, Any]:
        raise NotImplementedError("call() 메서드는 서브클래스에서 구현되어야 합니다.")

    def supports_function_calling(self) -> bool:
        return False

    def supports_stop_words(self) -> bool:
        return False

    def get_context_window_size(self) -> int:
        return 2048

# Azure OpenAI 연결 클래스
class AzureOpenAILLM(BaseLLM):
    def __init__(self):
        self.api_key = os.getenv("AZURE_API_KEY")
        self.azure_endpoint = os.getenv("AZURE_API_BASE")
        self.api_version = os.getenv("AZURE_API_VERSION")
        self.deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

        if not self.api_key or not self.azure_endpoint or not self.api_version or not self.deployment_name:
            raise ValueError("Azure OpenAI 환경변수가 설정되지 않았습니다.")

        super().__init__(model=f"azure/{self.deployment_name}")
        self.client = AzureOpenAI(
            api_key=self.api_key,
            azure_endpoint=self.azure_endpoint,
            api_version=self.api_version
        )

    def call(
        self,
        messages: Union[str, List[Dict[str, str]]],
        tools: Optional[List[dict]] = None,
        callbacks: Optional[List[Any]] = None,
        available_functions: Optional[Dict[str, Any]] = None,
    ) -> Union[str, Any]:
        try:
            if isinstance(messages, str):
                messages = [{"role": "user", "content": messages}]
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

    def supports_function_calling(self) -> bool:
        return True

    def supports_stop_words(self) -> bool:
        return True

    def get_context_window_size(self) -> int:
        return 8192

# Azure LLM 인스턴스 반환
def get_azure_llm():
    return AzureOpenAILLM()

# 메인 실행
if __name__ == "__main__":
    llm = get_azure_llm()
    agents = interview_agents.get_interview_agents()
    manager = interview_agents.InterviewAgentManager(agents)

    print("여행 인터뷰 질문에 대한 답변을 입력하세요.\n")

    user_inputs = {}
    for agent in agents:
        question = agent.instruction['page_instruction']['source']
        print(f"{question}")
        answer = input("답변 입력: ").strip()
        user_inputs[agent.name] = answer

    results = manager.run_all(user_inputs, llm=llm)

    # 날짜 기반 저장 파일명
    today_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"interview_results_{today_str}.txt"

    # 텍스트 파일 저장
with open(save_path, "w", encoding="utf-8") as f:
    f.write("=== 여행 인터뷰 정제 결과 ===\n\n")
    for agent in agents:
        question = agent.instruction['page_instruction']['source']
        refined = results.get(agent.name, "(정제된 결과 없음)")
        f.write(f"[질문] {question}\n")
        f.write(f"[답변]\n{refined}\n")
        f.write("\n" + "-" * 40 + "\n\n")

    print(f"\n✅ 모든 결과가 '{save_path}' 파일로 저장되었습니다.")
