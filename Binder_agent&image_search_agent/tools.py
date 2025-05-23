import os
import json
import requests
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient
from crewai.tools import tool
from pathlib import Path

dotenv_path = Path(r'C:\Users\EL0021\Desktop\crewai\.env')

# 환경 변수 로드
load_dotenv(dotenv_path=dotenv_path, override=True)

def get_blob_list(connection_string, container_name):
    """Azure Blob Storage 컨테이너에서 모든 Blob 목록을 가져옵니다."""
    try:
        # BlobServiceClient 생성
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        
        # 컨테이너 클라이언트 생성
        container_client = blob_service_client.get_container_client(container_name)
        
        # 컨테이너의 모든 Blob 목록 가져오기
        blob_list = container_client.list_blobs()
        
        # 결과를 리스트로 변환하고 이름 기준으로 정렬
        sorted_blobs = sorted([blob for blob in blob_list], key=lambda x: x.name)
        
        return sorted_blobs
    except Exception as e:
        print(f"Blob 목록을 가져오는 중 오류 발생: {e}")
        return []

class ImageTools:
    @staticmethod
    def get_blob_list():
        """Azure Blob Storage 컨테이너에서 모든 Blob 목록을 가져옵니다."""
        connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        container_name = "image"
        return get_blob_list(connection_string, container_name)
    
    @staticmethod
    def get_blob_url(blob):
        """Azure Blob Storage에서 이미지 URL 가져오기"""
        container_name = "image"
        # blob이 문자열인 경우 (이름만 전달된 경우)
        if isinstance(blob, str):
            blob_name = blob
        else:
            # blob이 객체인 경우 (get_blob_list에서 반환된 객체)
            blob_name = blob.name
            
        return f"https://5teamsfoundrystorage.blob.core.windows.net/{container_name}/{blob_name}"
    
    @staticmethod
    def list_images():
        """이미지 목록 가져오기"""
        blobs = ImageTools.get_blob_list()
        return [blob.name for blob in blobs] if blobs else [f"image{i}" for i in range(1, 11)]

@tool("이미지 위치 분석")
def analyze_image_with_vision(image_url: str):
    """이미지를 분석하여 위치를 식별합니다."""
    try:
        # Azure OpenAI API 직접 호출
        headers = {
            "Content-Type": "application/json",
            "api-key": os.getenv("AZURE_API_KEY")
        }
        
        payload = {
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
        
        # 엔드포인트 URL에서 마지막 슬래시 제거 (있는 경우)
        azure_endpoint = os.getenv("AZURE_API_BASE").rstrip('/')
        deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        api_version = os.getenv("AZURE_API_VERSION")
        
        # API 요청 URL 구성
        request_url = f"{azure_endpoint}/openai/deployments/{deployment_name}/chat/completions?api-version={api_version}"
        
        print(f"요청 URL: {request_url}")  # 디버깅용
        print(f"이미지 URL: {image_url}")  # 디버깅용
        
        response = requests.post(
            request_url,
            headers=headers,
            json=payload
        )
        
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            return f"API 호출 오류: {response.status_code}, {response.text}"
            
    except Exception as e:
        return f"이미지 분석 오류: {str(e)}"

