from typing import Dict, List, Optional, Union


class EssayAgentBase:
    def __init__(self, name: str, instruction: Dict):
        self.name = name
        self.instruction = instruction

    def format(self, user_input: str) -> str:
        style = self.instruction["style"]
        hints = self.instruction.get("page_instruction", {})
        return (
            f"[{self.name}]\n"
            f"다음 내용을 기반으로 에세이를 작성하되, 반드시 '-하다' 말투의 문어체로 작성하라.에세이 제목은 넣지 않음.\n"
            f"에세이 목적: {self.instruction['purpose']}\n"
            f"형식: {style['format']}, 톤: {style['tone']}\n"
            f"페이지: {hints.get('page', '')}\n"
            f"출력:\n{self.rewrite_text(user_input)}"
        )

    def rewrite_text(self, user_input: str, llm=None) -> str:
        """
        실제 LLM 호출 함수
        반드시 '-하다' 말투의 문어체로 출력되도록 강제
        """
        if llm:
            prompt = f"{self.format(user_input)}"
            response = llm.call(prompt)
            return response
        else:
            return f"'{user_input}' 내용을 기반으로 정제된 에세이 문장입니다.'-하다' 말투의 문어체로 작성됨."


class EssayAgentManager:
    def __init__(self, agents: Optional[List[EssayAgentBase]] = None):
        self.agents = agents or []

    def add_agent(self, agent: EssayAgentBase):
        self.agents.append(agent)

    def run_all(self, user_input: str, llm=None) -> Dict[str, str]:
        results = {}
        for agent in self.agents:
            input_text = user_input.get(agent.name, "")
            result = agent.rewrite_text(input_text, llm=llm)
            results[agent.name] = result
        return results
