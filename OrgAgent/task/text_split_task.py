from crewai import Task

def text_layout_task(section_content: str, agent):
    return Task(
        name="text_layout",
        description=f"""
아래는 하나의 매거진 섹션에 사용될 텍스트입니다. 이 텍스트를 매거진 페이지에 맞게 다음 구조로 나누어 배치해줘.

형식 예시:
- title: 섹션을 대표할 한 문장 (짧고 인상 깊게)
- subtitle: 에세이에서 강조할 인용구 스타일 문장 (주관적 느낌, 문장 그대로 써도 됨)
- tagline: 페이지 하단에 들어갈 짧은 태그라인 (영감을 주는 짧은 문장)
- body: 텍스트의 가장 메인이 되는 내용을 자세하게 서술 (조금 길고, 자세하게)

### 섹션 원문:
{section_content}
""",
        expected_output="""
title: [여기에 텍스트]
subtitle: [여기에 텍스트]
tagline: [여기에 텍스트]
body: [여기에 텍스트]
""",
        agent=agent
    )