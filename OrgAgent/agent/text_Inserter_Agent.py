from crewai import Agent

def get_code_inserter(llm):
    return Agent(
        role="React Code Inserter",
        goal="분해된 콘텐츠(title, subtitle, tagline, imagePrompt)를 JSX 컴포넌트 코드에 자동 삽입",
        backstory="매거진을 구성하는 콘텐츠를 기반으로 React 템플릿을 완성하는 프론트엔드 자동화 전문가",
        llm=llm,
        verbose=True
    )