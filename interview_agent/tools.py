from typing import Dict, List, Optional


class InterviewAgentBase:
    def __init__(self, name: str, instruction: Dict):
        self.name = name
        self.instruction = instruction

    def format(self, user_input: str) -> str:
        style = self.instruction.get("style", {})
        hints = self.instruction.get("question_instruction", {})
        return (
            f"[{self.name}]\n"
            f"다음 내용을 기반으로 인터뷰 답변을 작성하되, 반드시 존댓말로 작성하라.\n"
            f"인터뷰 목적: {self.instruction.get('purpose', '')}\n"
            f"형식: {style.get('format', '')}, 톤: {style.get('tone', '')}\n"
            f"질문: {hints.get('question', '')}\n"
            f"출력:\n{self.rewrite_text(user_input)}"
        )

    def rewrite_text(self, user_input: str, llm=None) -> str:
        
        if llm:
            prompt = self.format(user_input)
            response = llm.call(prompt)
            return response
        else:
            return f"'{user_input}' 반드시 내용을 기반으로 정제된 인터뷰 응답입니다. 존댓말을 사용합니다"


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
