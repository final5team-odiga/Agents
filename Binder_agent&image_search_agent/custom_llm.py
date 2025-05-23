import os
import json
from dotenv import load_dotenv
from crewai import BaseLLM
from openai import AzureOpenAI
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

dotenv_path = Path(r'C:\Users\EL0021\Desktop\crewai\.env')

# 환경 변수 로드
load_dotenv(dotenv_path=dotenv_path, override=True)


class AzureOpenAILLM(BaseLLM):
    """Azure OpenAI API를 직접 사용하는 사용자 정의 LLM 클래스"""
    
    def __init__(self):
        # 환경 변수 확인 및 가져오기
        self.api_key = os.getenv("AZURE_API_KEY")
        self.azure_endpoint = os.getenv("AZURE_API_BASE")
        self.api_version = os.getenv("AZURE_API_VERSION")
        self.deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        
        if not self.api_key:
            raise ValueError("AZURE_API_KEY 환경 변수가 설정되지 않았습니다.")
        if not self.azure_endpoint:
            raise ValueError("AZURE_API_BASE 환경 변수가 설정되지 않았습니다.")
        if not self.api_version:
            raise ValueError("AZURE_API_VERSION 환경 변수가 설정되지 않았습니다.")
        if not self.deployment_name:
            raise ValueError("AZURE_OPENAI_DEPLOYMENT_NAME 환경 변수가 설정되지 않았습니다.")
        
        # 부모 클래스 초기화 - model 매개변수 추가
        super().__init__(model=f"azure/{self.deployment_name}")
        
        # Azure OpenAI 클라이언트 초기화
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
        """LLM에 메시지를 전송하고 응답을 받습니다."""
        try:
            # 문자열 메시지를 적절한 형식으로 변환
            if isinstance(messages, str):
                messages = [{"role": "user", "content": messages}]
            
            # 도구(함수 호출) 지원 여부에 따라 요청 구성
            if tools and self.supports_function_calling():
                response = self.client.chat.completions.create(
                    model=self.deployment_name,
                    messages=messages,
                    tools=tools,
                    temperature=0.7,
                    max_tokens=4000
                )
            else:
                response = self.client.chat.completions.create(
                    model=self.deployment_name,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=4000
                )
            
            # 함수 호출 응답 처리
            if (
                tools 
                and self.supports_function_calling() 
                and response.choices[0].message.tool_calls
                and available_functions
            ):
                tool_call = response.choices[0].message.tool_calls[0]
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                if function_name in available_functions:
                    function_to_call = available_functions[function_name]
                    function_response = function_to_call(**function_args)
                    
                    # 함수 응답을 포함하여 후속 요청
                    messages.append(response.choices[0].message)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": function_name,
                        "content": str(function_response)
                    })
                    
                    second_response = self.client.chat.completions.create(
                        model=self.deployment_name,
                        messages=messages,
                        temperature=0.7,
                        max_tokens=4000
                    )
                    
                    return second_response.choices[0].message.content
            
            # 일반 텍스트 응답 반환
            return response.choices[0].message.content
            
        except Exception as e:
            # 오류 로깅 및 예외 발생
            print(f"LLM 호출 오류: {str(e)}")
            raise RuntimeError(f"LLM 요청 실패: {str(e)}")
    
    def supports_function_calling(self) -> bool:
        """함수 호출 지원 여부 확인"""
        # GPT-4, GPT-4 Turbo, GPT-4o 모델은 함수 호출을 지원합니다
        return True
    
    def supports_stop_words(self) -> bool:
        """중지 단어 지원 여부 확인"""
        return True
    
    def get_context_window_size(self) -> int:
        """LLM의 컨텍스트 윈도우 크기 반환"""
        # 모델에 따라 적절한 값 반환
        return 8192  # 기본값, 필요에 따라 조정

def get_azure_llm():
    """Azure OpenAI LLM 인스턴스 생성"""
    return AzureOpenAILLM()
