from tools import EssayAgentBase

# 각 에이전트 인스턴스 정의


def get_essay_agents():
    agents = []

    agent1_instruction = {
        "purpose": "사용자가 작성한 여행 에세이 응답 중, '여행을 떠난 이유, 나의 감정은?'에 대한 내용을 에세이 형식으로 정제",
        "style": {
            "format": "에세이",
            "tone": "담백하고 조용한 감정선을 가진 일기체",
            "language": "문어체 기반, 존댓말 아닌 '-하다' 말투",
            "paragraphing": "3~5줄 단위 짧은 단락 구분",
            "editing_principle": [
                "인터뷰이의 의도와 감정 흐름을 절대 왜곡하지 말 것",
                "의미를 바꾸지 않는 선에서 문장 표현을 다듬을 것",
                "불필요한 반복, 추임새, 군더더기는 제거",
                "단어 선택은 감정을 드러내되 절대 과하지 않게 표현"
            ]
        },
        "page_instruction": {
            "page": "1page",
            "source": "질문: '여행을 떠난 이유, 나의 감정은?'",
            "goal": "출발 동기와 감정 상태를 자연스럽게 서술",
            "structure_hint": [
                "하루의 시작이나 공항/이동 장면으로 시작",
                "이유를 명확하게 설명하지 않아도 분위기로 감정 전달",
                "비워두고 싶거나 명확하지 않은 감정도 솔직하게 표현 가능"
            ]
        },
        "output_format": {
            "type": "essay_text",
            "output": "에세이 형식의 문자열"
        }

    }

    agent2_instruction = {
        "purpose": "사용자의 여행 에세이 응답 중, '{계절}에 {도시}를 찾은 이유가 있다면?'에 대한 내용을 에세이 형식으로 정제",
        "style": {
            "format": "에세이",
            "tone": "담백하고 조용한 감정선을 가진 일기체",
            "language": "문어체 기반, 존댓말 아닌 '-하다' 말투",
            "paragraphing": "3~5줄 단위 짧은 단락 구분",
            "editing_principle": [
                "인터뷰이의 의도와 감정 흐름을 절대 왜곡하지 말 것",
                "의미를 바꾸지 않는 선에서 문장 표현을 다듬을 것",
                "불필요한 반복, 추임새, 군더더기는 제거",
                "단어 선택은 감정을 드러내되 절대 과하지 않게 표현"
            ]
        },
        "page_instruction": {
            "page": "2page",
            "source": "질문: '{계절}에 {도시}를 찾은 이유가 있다면?'",
            "goal": "도시와 계절이 연결된 개인적인 이유 또는 인상 묘사",
            "structure_hint": [
                "도시의 풍경이나 계절적 느낌으로 서두를 시작",
                "기억, 기대, 이미지 같은 주관적 인상이 드러나도록 유도"
            ]
        },
        "output_format": agent1_instruction["output_format"]

    }

    agent3_instruction = {
        "purpose": "사용자의 여행 에세이 응답 중, '{이름}님이 그곳에서의 오늘 하루는 어떻게 흘러갔는지 궁금해진다. 어땠는지'에 대한 내용을 에세이 형식으로 정제",
        "style": {
            "format": "에세이",
            "tone": "담백하고 조용한 감정선을 가진 일기체",
            "language": "문어체 기반 , 존댓말 아닌 '-하다' 말투/ 일부 구어체 허용",
            "paragraphing": "3~5줄 단위 짧은 단락 구분",
            "editing_principle": [
                "인터뷰이의 의도와 감정 흐름을 절대 왜곡하지 말 것",
                "의미를 바꾸지 않는 선에서 문장 표현을 다듬을 것",
                "불필요한 반복, 추임새, 군더더기는 제거",
                "단어 선택은 감정을 드러내되 절대 과하지 않게 표현"
            ]
        },
        "page_instruction": {
            "page": "3page",
            "source": "질문: '{이름}님이 그곳에서의 오늘 하루는 어떻게 흘러갔는지 궁금해진다. 어땠는지'",
            "goal": "하루의 흐름과 감정을 따라가는 묘사",
            "structure_hint": [
                "산책, 카페, 만난 사람, 작은 사건 등 구체적 묘사 포함",
                "감정 변화가 있으면 드러내되 과장 없이 표현"
            ]
        },
        "output_format": agent1_instruction["output_format"]

    }

    agent4_instruction = {
        "purpose": "사용자의 여행 에세이 응답 중, 3페이지 내용의 연장선인 하루 마무리를 에세이 형식으로 정제",
        "style": {
            "format": "에세이",
            "tone": "담백하고 조용한 감정선을 가진 일기체",
            "language": "문어체 기반 , 존댓말 아닌 '-하다'말투/ 일부 구어체 허용",
            "paragraphing": "3~5줄 단위 짧은 단락 구분",
            "editing_principle": [
                "인터뷰이의 의도와 감정 흐름을 절대 왜곡하지 말 것",
                "의미를 바꾸지 않는 선에서 문장 표현을 다듬을 것",
                "불필요한 반복, 추임새, 군더더기는 제거",
                "단어 선택은 감정을 드러내되 절대 과하지 않게 표현"
            ]
        },
        "page_instruction": {
            "page": "4page",
            "continuation": "3page 내용의 연장선",
            "goal": "하루 마무리나 인상 깊은 순간으로 연결",
            "structure_hint": [
                "해질 무렵의 장면, 노을, 글을 쓰는 장면 등으로 마무리",
                "짧지만 의미 있는 인용이나 내면 독백으로 끝맺기"
            ]
        },
        "output_format": agent1_instruction["output_format"]

    }

    agents.append(EssayAgentBase("EssayFormatAgent1", agent1_instruction))
    agents.append(EssayAgentBase("EssayFormatAgent2", agent2_instruction))
    agents.append(EssayAgentBase("EssayFormatAgent3", agent3_instruction))
    agents.append(EssayAgentBase("EssayFormatAgent4", agent4_instruction))

    return agents
