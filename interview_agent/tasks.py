# tasks.py
# 업무(비즈니스 로직), 행동강령, 매니저 클래스

from typing import Dict, List, Optional
from agents import InterviewAgentBase

class InterviewAgentManager:
    def __init__(self, agents: Optional[List[InterviewAgentBase]] = None):
        self.agents = agents or []

    def add_agent(self, agent: InterviewAgentBase):
        self.agents.append(agent)

    def interview_step_by_step(self, llm=None, user_inputs: Optional[Dict[str, str]] = None):
        results = {}
        for agent in self.agents:
            if user_inputs is None or agent.name not in user_inputs:
                print(f"\n🟦 [{agent.name}] 질문: {agent.get_question()}")
                user_input = input("✏️ 사용자 응답: ")
            else:
                user_input = user_inputs[agent.name]
            result = self.rewrite_text(agent, user_input, llm=llm)
            print(f"\n✅ [인터뷰 응답 - {agent.name}]\n{result}\n")
            results[agent.name] = result
        return results

    def rewrite_text(self, agent: InterviewAgentBase, user_input: str, llm=None) -> str:
        prompt = self.format(agent, user_input)
        if llm:
            return llm.call(prompt)
        else:
            return f"(예시) {user_input} → 위 내용을 바탕으로 구어체 존댓말로 정제된 인터뷰 답변입니다."

    def format(self, agent: InterviewAgentBase, user_input: str) -> str:
        style = agent.instruction["style"]
        hints = agent.instruction.get("page_instruction", {})
        editing_rules = '\n- '.join(style.get('editing_principle', []))
        return (
            f"[{agent.name}]\n"
            f"다음 내용을 기반으로 여행 인터뷰 Q&A 형식의 답변을 작성하라.\n"
            f"- 반드시 구어체 기반의 **존댓말**을 사용한다.\n"
            f"- 자연스럽고 진솔한 **대화체** 말투로 표현한다.\n"
            f"- 질문은 다시 쓰지 말고, 답변만 작성하라.\n"
            f"\n"
            f"[인터뷰 작성 목적]\n{agent.instruction['purpose']}\n\n"
            f"[스타일]\n"
            f"형식: {style.get('format', '')}\n"
            f"톤: {style.get('tone', '')}\n"
            f"언어: {style.get('language', '')}\n\n"
            f"[편집 원칙]\n- {editing_rules}\n\n"
            f"[질문 출처]\n{hints.get('source', '')}\n\n"
            f"[사용자 응답]\n{user_input}"
        )
