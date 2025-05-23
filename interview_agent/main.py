import os
import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dotenv import load_dotenv
from openai import AzureOpenAI
import tasks  
import tools  


# 1) .env 파일 위치 지정 및 환경변수 로드
dotenv_path = Path(r'C:\Users\kanga\Desktop\MS\final\.env')  # 실제 경로로 변경하세요
load_dotenv(dotenv_path=dotenv_path, override=True)

# 2) 환경변수 로드 확인용 (필요 시 주석 처리)
print("AZURE_API_KEY:", os.getenv("AZURE_API_KEY"))
print("AZURE_API_BASE:", os.getenv("AZURE_API_BASE"))
print("AZURE_API_VERSION:", os.getenv("AZURE_API_VERSION"))


# 3) Base LLM 클래스
class BaseLLM:
    def __init__(self, model: Optional[str] = None):
        self.model = model

    def call(self, messages: Union[str, List[Dict[str, str]]]) -> Union[str, Any]:
        raise NotImplementedError("call() 메서드는 서브클래스에서 구현되어야 합니다.")


# 4) Azure OpenAI LLM 연결 클래스
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


def get_azure_llm():
    return AzureOpenAILLM()


# 5) 메인 실행부
def main():
    llm = get_azure_llm()

    # tasks.py에 정의한 에이전트들 가져오기
    agents = tasks.get_interview_agents()

    # tools.py에 정의된 매니저로 감싸기
    manager = tools.InterviewAgentManager(agents)

    print("여행 인터뷰 질문에 대한 답변을 입력하세요.\n")

    user_inputs = {}
    for agent in agents:
        question = agent.instruction['page_instruction']['source']
        print(f"[질문] {question}")
        answer = input("답변 입력: ").strip()
        user_inputs[agent.name] = answer

    # 모든 에이전트에 사용자 답변 전달 후 결과 받기
    results = manager.run_all(user_inputs, llm=llm)

    # 날짜 기반 저장 파일명
    today_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"interview_results_{today_str}.txt"

    # 결과를 텍스트 파일로 저장
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("=== 여행 인터뷰 결과 ===\n\n")
        for agent in agents:
            question = agent.instruction['page_instruction']['source']
            refined = results.get(agent.name, "(정제된 결과 없음)")
            f.write(f"[질문] {question}\n")
            f.write(f"[답변]\n{refined}\n")
            f.write("\n" + "-" * 40 + "\n\n")

    print(f"\n✅ 모든 결과가 '{save_path}' 파일로 저장되었습니다.")
