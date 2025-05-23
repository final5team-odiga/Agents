from crewai import Agent, Task
from custom_llm import get_azure_llm

class ImageAnalyzerAgent:
    def __init__(self):
        self.llm = get_azure_llm()
        
    def create_agent(self):
        return Agent(
            role="이미지 분석가",
            goal="이미지에서 위치 정보를 정확하게 식별하고 분석",
            backstory="당신은 이미지에서 랜드마크, 지형, 건축물 등을 식별하여 위치를 파악하는 전문가입니다.",
            verbose=True,
            llm=self.llm,
            multimodal=True  # 멀티모달 기능 활성화
        )
    
    def analyze_images(self, images, crew):
        """여러 이미지 분석"""
        results = []
        
        for i, image in enumerate(images, 1):
            try:
                print(f"\n=== 이미지 {i}: '{image.name}' 분석 중 ===")
                
                # 이미지 URL 생성
                from utils.blob_storage import BlobStorageManager
                blob_manager = BlobStorageManager()
                image_url = blob_manager.get_image_url(image)
                print(f"이미지 URL: {image_url}")
                
                # 직접 LLM에 이미지 분석 요청
                from openai import AzureOpenAI
                import os
                
                client = AzureOpenAI(
                    api_key=os.getenv("AZURE_API_KEY"),
                    azure_endpoint=os.getenv("AZURE_API_BASE"),
                    api_version=os.getenv("AZURE_API_VERSION")
                )
                
                response = client.chat.completions.create(
                    model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
                    messages=[
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
                    temperature=0.3,
                    max_tokens=500
                )
                
                result = response.choices[0].message.content
                
                results.append({
                    "image_name": image.name,
                    "image_url": image_url,
                    "location": result
                })
                
                print(f"이미지 '{image.name}' 분석 결과: {result}")
                
            except Exception as e:
                print(f"이미지 '{image.name}' 분석 중 오류 발생: {str(e)}")
                import traceback
                print(traceback.format_exc())
                
                # 오류가 발생해도 계속 진행
                results.append({
                    "image_name": image.name,
                    "image_url": image_url if 'image_url' in locals() else "URL 생성 실패",
                    "location": f"분석 오류: {str(e)}"
                })
        
        return results
