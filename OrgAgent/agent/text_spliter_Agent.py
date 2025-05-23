from crewai import Agent

def split_text(llm):
    return Agent(
        role="Text Organizer",
        goal="정제된 텍스트를 기반으로 매거진 섹션 템플릿의 title, subtitle, tagline, body를 구성",
        backstory="여행 매거진 에디터로서 전체 맥락에 맞게 레이아웃을 조정하고 콘텐츠를 배치하는 데 능함",
        llm=llm,
        verbose=True
    )