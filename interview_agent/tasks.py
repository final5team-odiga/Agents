# tasks.py
from typing import Dict, List, Optional

class InterviewAgentBase:
    def __init__(self, name: str, instruction: Dict):
        self.name = name
        self.instruction = instruction

    def format(self, user_input: str) -> str:
        style = self.instruction.get("style", {})
        hints = self.instruction.get("page_instruction", {})
        return (
            f"[{self.name}]\n"
            f"다음 내용을 기반으로 인터뷰 답변을 작성하되, 반드시 존댓말로 작성하라. 제목은 넣지 않음.\n"
            f"인터뷰 목적: {self.instruction.get('purpose', '')}\n"
            f"형식: {style.get('format', '')}, 톤: {style.get('tone', '')}\n"
            f"페이지: {hints.get('page', '')}\n"
            f"내용:\n{user_input}"
        )

    def rewrite_text(self, user_input: str, llm=None) -> str:
        if llm:
            prompt = self.format(user_input)
            response = llm.call(prompt)
            return response
        else:
            return f"'{user_input}' 반드시 내용을 기반으로 정제된 인터뷰 답변입니다. 존댓말을 사용한다."

class InterviewAgentManager:
    def __init__(self, agents: Optional[List[InterviewAgentBase]] = None):
        self.agents = agents or []

    def add_agent(self, agent: InterviewAgentBase):
        self.agents.append(agent)

    def run_all(self, user_input: Dict[str, str], llm=None) -> Dict[str, str]:
        results = {}
        for agent in self.agents:
            input_text = user_input.get(agent.name, "")
            result = agent.rewrite_text(input_text, llm=llm)
            results[agent.name] = result
        return results

def get_interview_agents() -> List[InterviewAgentBase]:
    agents = []

    base_style = {
        "format": "인터뷰 답변",
        "tone": "담백하고 진솔한 문어체, 존댓말을 사용한다.",
        "language": "문어체 기반",
        "paragraphing": "3~5줄 단위 단락 구분",
        "editing_principle": [
            "진실된 느낌 유지",
            "불필요한 반복 제거",
            "감정 표현은 절제"
        ]
    }

    questions = [
        ("InterviewAgent1", "여행 중 인상 깊었던 인물이나 장면이 있었나요?", "1page", "여행 중 인상 깊었던 인물이나 장면에 대한 인터뷰 답변을 정제", "인물 혹은 장면에 대한 구체적이고 진솔한 묘사"),
        ("InterviewAgent2", "날씨와 도시는 어떤 느낌이었나요? 여행과 함께한 음악이 있나요?", "2page", "날씨, 도시 느낌과 여행과 함께한 음악에 대한 인터뷰 답변을 정제", "날씨와 도시의 분위기, 음악과의 연관성 표현"),
        ("InterviewAgent3", "그 도시에서 가장 만족스러웠던 음식은 무엇이었나요?", "3page", "가장 만족스러웠던 음식에 대한 인터뷰 답변을 정제", "음식에 대한 생생한 묘사와 만족감 표현"),
        ("InterviewAgent4", "여행 중 “이건 꼭 해보자”라고 생각한 것이 있었다면?", "4page", "여행 중 꼭 해보자고 생각한 것에 대한 인터뷰 답변을 정제", "계획이나 다짐에 대한 구체적이고 솔직한 표현"),
        ("InterviewAgent5", "여행을 돌아보았을 때 가장 좋았던 공간은?", "5page", "가장 좋았던 공간에 대한 인터뷰 답변을 정제", "공간에 대한 감정과 기억을 생생히 표현")
    ]

    for name, source, page, purpose, goal in questions:
        instruction = {
            "purpose": purpose,
            "style": base_style,
            "page_instruction": {
                "page": page,
                "source": f"질문: {source}",
                "goal": goal
            },
            "output_format": {
                "type": "interview_text",
                "output": "인터뷰 답변 형식의 문자열"
            }
        }
        agents.append(InterviewAgentBase(name, instruction))

    return agents
