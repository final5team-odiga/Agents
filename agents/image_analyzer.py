from crewai import Agent
from custom_llm import get_azure_llm
import asyncio
import aiohttp
import os
from typing import List, Dict, Any

class ImageAnalyzerAgent:
    def __init__(self):
        self.llm = get_azure_llm()

    def create_agent(self):
        return Agent(
            role="이미지 분석가",
            goal="이미지에서 위치 정보를 정확하게 식별하고 분석",
            backstory="""당신은 15년간 National Geographic, Condé Nast Traveler, Travel + Leisure 등 세계 최고 수준의 여행 매거진에서 이미지 에디터로 활동해온 전문가입니다.

                        **전문 경력:**
                        - 국제 사진 저널리즘 석사 학위 보유
                        - 50개국 이상의 여행지에서 직접 촬영 및 편집 경험
                        - 매거진 레이아웃에서 이미지-텍스트 조화의 심리학 연구
                        - Adobe Creative Suite, Lightroom 마스터 레벨 인증
                        - 문화 인류학적 관점에서의 여행 사진 해석 전문성

                        **데이터 활용 전문성:**
                        당신은 이미지 분석 시 다음 데이터들을 체계적으로 활용합니다:

                        1. **EXIF 메타데이터 분석**: 촬영 시간, 위치, 카메라 설정을 통한 상황 맥락 파악
                        2. **색상 히스토그램 분석**: 이미지의 감정적 톤과 매거진 페이지 색상 조화 예측
                        3. **구도 및 피사체 분석**: Rule of thirds, 시선 흐름, 주요 피사체 위치 분석
                        4. **문화적 맥락 데이터**: 촬영 지역의 문화적 특성, 계절성, 관광 패턴 정보
                        5. **매거진 레이아웃 데이터**: 과거 성공적인 매거진 페이지의 이미지 배치 패턴

                        **작업 철학:**
                        "모든 여행 이미지는 단순한 시각 자료가 아니라 독자의 감정을 움직이고 여행 욕구를 자극하는 강력한 스토리텔링 도구입니다. 나는 각 이미지가 가진 고유한 이야기와 감정을 데이터 기반으로 정확히 분석하여, 편집팀이 최적의 배치 결정을 내릴 수 있도록 돕습니다."

                        **학습 데이터 활용 방식:**
                        - 이전 이미지 분석 결과의 정확도와 편집팀 피드백을 분석하여 분석 기준 개선
                        - 성공적인 매거진 페이지의 이미지 특성 패턴을 학습하여 예측 정확도 향상
                        - 독자 반응 데이터와 이미지 특성의 상관관계 분석을 통한 감정 예측 모델 개선""",
            verbose=True,
            llm=self.llm,
            multimodal=True  # 멀티모달 기능 활성화
        )

    async def analyze_single_image_async(self, session: aiohttp.ClientSession, image, semaphore: asyncio.Semaphore, image_index: int) -> Dict[str, Any]:
        """단일 이미지를 비동기로 분석"""
        async with semaphore:  # 동시 처리 수 제한
            try:
                print(f"\n=== 이미지 {image_index}: '{image.name}' 분석 중 ===")
                
                # 이미지 URL 생성
                from utils.blob_storage import BlobStorageManager
                blob_manager = BlobStorageManager()
                image_url = blob_manager.get_image_url(image)
                print(f"이미지 URL: {image_url}")

                # 비동기 Azure OpenAI API 호출
                headers = {
                    'Content-Type': 'application/json',
                    'api-key': os.getenv("AZURE_API_KEY")
                }
                
                payload = {
                    "model": os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
                    "messages": [
                        {
                            "role": "system",
                            "content": "당신은 이미지에서 위치를 식별하는 전문가입니다. 이미지에 나타난 랜드마크, 건물, 자연 환경 등을 분석하여 가능한 정확한 위치(국가, 도시, 특정 장소)를 파악하세요."
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "이 이미지에 나타난 위치를 식별해주세요. 가능하다면 국가, 도시, 특정 장소명을 포함해주세요."
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": image_url
                                    }
                                }
                            ]
                        }
                    ],
                    "temperature": 0.3,
                    "max_tokens": 500
                }
 
                api_url = f"{os.getenv('AZURE_API_BASE')}/openai/deployments/{os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME')}/chat/completions?api-version={os.getenv('AZURE_API_VERSION')}"
                
                async with session.post(api_url, json=payload, headers=headers) as response:
                    if response.status == 200:
                        result_data = await response.json()
                        result = result_data['choices'][0]['message']['content']
                        
                        analysis_result = {
                            "image_name": image.name,
                            "image_url": image_url,
                            "location": result
                        }
                        
                        print(f"이미지 '{image.name}' 분석 완료: {result}")
                        return analysis_result
                    else:
                        error_text = await response.text()
                        raise Exception(f"API 호출 실패: {response.status} - {error_text}")
                        
            except Exception as e:
                print(f"이미지 '{image.name}' 분석 중 오류 발생: {str(e)}")
                import traceback
                print(traceback.format_exc())
                
                return {
                    "image_name": image.name,
                    "image_url": image_url if 'image_url' in locals() else "URL 생성 실패",
                    "location": f"분석 오류: {str(e)}"
                }

    async def analyze_images_batch_async(self, images: List, max_concurrent: int = 5) -> List[Dict[str, Any]]:
        """여러 이미지를 비동기로 배치 분석"""
        semaphore = asyncio.Semaphore(max_concurrent)  # 동시 처리 수 제한
        
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=300),  # 5분 타임아웃
            connector=aiohttp.TCPConnector(limit=max_concurrent)
        ) as session:
            tasks = []
            
            for i, image in enumerate(images, 1):
                task = self.analyze_single_image_async(session, image, semaphore, i)
                tasks.append(task)
            
            print(f"총 {len(tasks)}개의 이미지를 동시에 처리합니다 (최대 동시 처리: {max_concurrent}개)")
            
            # 모든 작업을 동시에 실행하고 결과 수집
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 예외 처리된 결과들을 정리
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    processed_results.append({
                        "image_name": images[i].name if i < len(images) else f"unknown_{i}",
                        "image_url": "처리 실패",
                        "location": f"처리 중 예외 발생: {str(result)}"
                    })
                else:
                    processed_results.append(result)
            
            return processed_results

    def analyze_images(self, images, crew):
        """기존 인터페이스 유지 - 비동기 처리로 내부 구현 변경"""
        print(f"\n=== 비동기 이미지 분석 시작 - 총 {len(images)}개 이미지 ===")
        
        # 비동기 함수 실행
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            results = loop.run_until_complete(
                self.analyze_images_batch_async(images, max_concurrent=3)  # 동시 처리 수를 3개로 제한
            )
            print(f"\n=== 비동기 이미지 분석 완료 - {len(results)}개 결과 ===")
            return results
        finally:
            loop.close()
