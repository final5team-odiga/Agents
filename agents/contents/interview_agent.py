from typing import Dict, List
from crewai import Agent, Task
from custom_llm import get_azure_llm

class InterviewAgentBase:
    def __init__(self, name: str, instruction: Dict):
        self.name = name
        self.instruction = instruction
        self.llm = get_azure_llm()

    def get_question(self) -> str:
        return self.instruction.get("page_instruction", {}).get("source", "질문이 없습니다.")

    def create_agent(self):
        return Agent(
            role=f"여행 인터뷰 전문가 - {self.name}",
            goal=self.instruction.get("purpose", "여행 경험에 대한 인터뷰 답변 정제"),
            backstory=f"""당신은 여행 인터뷰 전문가입니다. {self.instruction.get('purpose', '')}
            자연스럽고 진솔한 대화체로 여행자의 경험을 깊이 있게 탐구하며,
            감정과 경험의 디테일을 생생하게 끌어내는 전문성을 가지고 있습니다.""",
            verbose=True,
            llm=self.llm
        )

    def process_interview(self, user_response: str) -> str:
        """사용자 응답을 바탕으로 인터뷰 형식으로 정제"""
        agent = self.create_agent()
        
        question = self.get_question()
        style = self.instruction.get("style", {})
        
        interview_task = Task(
            description=f"""
            다음 질문에 대한 사용자의 응답을 자연스럽고 진솔한 인터뷰 형식으로 정제하세요.
            
            **질문**: {question}
            
            **사용자 응답**: 
            {user_response}
            
            **정제 지침**:
            - 형식: {style.get('format', 'Q&A')}
            - 톤: {style.get('tone', '자연스럽고 진솔한 대화체')}
            - 언어: {style.get('language', '구어체 기반, 존댓말 사용')}
            
            **편집 원칙**:
            {chr(10).join(['- ' + principle for principle in style.get('editing_principle', [])])}
            
            **목표**: {self.instruction.get('page_instruction', {}).get('goal', '진솔한 경험 표현')}
            
            **출력 형식**:
            Q: {question}
            A: [정제된 답변]
            """,
            expected_output="Q&A 형식의 인터뷰 텍스트"
        )
        
        result = agent.execute_task(interview_task)
        return str(result)

class InterviewAgentManager:
    def __init__(self):
        self.agents = self._get_interview_agents()
        
    def _get_interview_agents(self) -> List[InterviewAgentBase]:
        style_common = {
            "format": "Q&A",
            "tone": "자연스럽고 진솔한 대화체",
            "language": "구어체 기반, 존댓말 사용",
            "editing_principle": [
                "말투는 부드럽고 담백하게 정제",
                "너무 긴 문장은 나누되 감정의 흐름은 유지",
                "반복되거나 모호한 표현은 자연스럽게 정리",
                "질문에 맞는 답변이 되도록 포커스를 유지"
            ]
        }

        agents_data = [
            {
                "name": "InterviewAgent1",
                "purpose": "여행 중 인상 깊었던 인물이나 장면에 대한 인터뷰 답변을 정제",
                "page_instruction": {
                    "page": "1page",
                    "source": "여행 중 인상 깊었던 인물이나 장면이 있었나요?",
                    "goal": "인물 혹은 장면에 대한 구체적이고 진솔한 묘사"
                }
            },
            {
                "name": "InterviewAgent2", 
                "purpose": "날씨, 도시 느낌과 여행과 함께한 음악에 대한 인터뷰 답변을 정제",
                "page_instruction": {
                    "page": "2page",
                    "source": "날씨와 도시는 어떤 느낌이었나요? 여행과 함께한 음악이 있나요?",
                    "goal": "날씨와 도시의 분위기, 음악과의 연관성 표현"
                }
            },
            {
                "name": "InterviewAgent3",
                "purpose": "가장 만족스러웠던 음식에 대한 인터뷰 답변을 정제", 
                "page_instruction": {
                    "page": "3page",
                    "source": "그 도시에서 가장 만족스러웠던 음식은 무엇이었나요?",
                    "goal": "음식에 대한 생생한 묘사와 만족감 표현"
                }
            },
            {
                "name": "InterviewAgent4",
                "purpose": "여행 중 꼭 해보자고 생각한 것에 대한 인터뷰 답변을 정제",
                "page_instruction": {
                    "page": "4page", 
                    "source": "여행 중 \"이건 꼭 해보자\"라고 생각한 것이 있었다면?",
                    "goal": "계획이나 다짐에 대한 구체적이고 솔직한 표현"
                }
            },
            {
                "name": "InterviewAgent5",
                "purpose": "가장 좋았던 공간에 대한 인터뷰 답변을 정제",
                "page_instruction": {
                    "page": "5page",
                    "source": "여행을 돌아보았을 때 가장 좋았던 공간은?", 
                    "goal": "공간에 대한 감정과 기억을 생생히 표현"
                }
            }
        ]

        return [InterviewAgentBase(data["name"], {
            "purpose": data["purpose"],
            "style": style_common,
            "page_instruction": data["page_instruction"],
            "output_format": {
                "type": "interview_text",
                "output": "인터뷰 Q&A 형식의 문자열"
            }
        }) for data in agents_data]

    def process_all_interviews(self, text_responses: List[str]) -> Dict[str, str]:
        """모든 텍스트 응답을 인터뷰 형식으로 처리"""
        results = {}
        
        for i, agent in enumerate(self.agents):
            if i < len(text_responses):
                user_response = text_responses[i]
                interview_result = agent.process_interview(user_response)
                results[agent.name] = interview_result
                print(f"✓ {agent.name} 인터뷰 처리 완료")
        
        return results
