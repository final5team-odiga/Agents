from crewai import Agent, Task
from custom_llm import get_azure_llm

class ContentCreatorAgent:
    def __init__(self):
        self.llm = get_azure_llm()
        
    def create_agent(self):
        return Agent(
            role="여행 콘텐츠 작가",
            goal="사용자의 여행 경험을 매거진 형식의 매력적인 콘텐츠로 변환",
            backstory="당신은 여행 경험을 매력적인 스토리로 변환하는 전문 여행 작가입니다. 사용자의 여행 응답을 분석하여 감성적이고 시각적인 매거진 콘텐츠를 만들어냅니다.",
            verbose=True,
            llm=self.llm
        )
    
    def create_magazine_content(self, texts, image_analysis_results):
        """텍스트 응답과 이미지 분석 결과를 바탕으로 매거진 콘텐츠 생성"""
        # 텍스트 내용 결합
        combined_text = "\n\n".join(texts)
        
        # 이미지 분석 결과 정리
        image_locations = "\n".join([f"이미지 {i+1}: {result['location']}" for i, result in enumerate(image_analysis_results)])
        
        # 콘텐츠 생성 작업 정의
        content_task = Task(
            description=f"""
            사용자의 여행 응답과 이미지 분석 결과를 바탕으로 매거진 형식의 여행 콘텐츠를 작성하세요.
            
            사용자 응답:
            {combined_text}
            
            이미지 분석 결과:
            {image_locations}
            
            다음 요소를 포함하는 매거진 콘텐츠를 작성하세요:
            1. 매력적인 제목과 부제목
            2. 여행지 소개 (위치, 특징, 분위기)
            3. 여행 경험 이야기 (감성적, 묘사적 문체)
            4. 여행지 추천 장소 및 활동
            5. 여행 팁 또는 인사이트
            
            매거진 스타일로 작성하되, 각 섹션을 명확히 구분하고 시각적으로 상상할 수 있는 묘사를 포함하세요.
            """,
            expected_output="매거진 형식의 여행 콘텐츠"
        )
        
        # 에이전트 생성 및 작업 실행
        agent = self.create_agent()
        result = agent.execute_task(content_task)
        
        return str(result)
