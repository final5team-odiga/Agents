# agents.py
# 에이전트 클래스와 각 에이전트의 설명/목적/질문 정의

from typing import Dict, List, Optional

class InterviewAgentBase:
    def __init__(self, name: str, instruction: Dict):
        self.name = name
        self.instruction = instruction

    def get_question(self) -> str:
        return self.instruction.get("page_instruction", {}).get("source", "질문이 없습니다.")

def get_interview_agents() -> List[InterviewAgentBase]:
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
                "source": "여행 중 “이건 꼭 해보자”라고 생각한 것이 있었다면?",
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

