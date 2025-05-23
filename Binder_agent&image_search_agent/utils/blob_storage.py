import os
import json
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient, ContentSettings
from pathlib import Path

dotenv_path = Path(r'C:\Users\EL0021\Desktop\crewai\.env')

# 환경 변수 로드
load_dotenv(dotenv_path=dotenv_path, override=True)

class BlobStorageManager:
    def __init__(self):
        connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        self.container_name = os.getenv("AZURE_STORAGE_CONTAINER")
        self.blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        self.container_client = self.blob_service_client.get_container_client(self.container_name)
    
    def get_images(self):
        """images 폴더에서 모든 이미지 가져오기"""
        blob_list = self.container_client.list_blobs(name_starts_with="images/")
        return sorted([blob for blob in blob_list], key=lambda x: x.name)
    
    def get_texts(self):
        """texts 폴더에서 모든 텍스트 파일 가져오기"""
        blob_list = self.container_client.list_blobs(name_starts_with="texts/")
        return sorted([blob for blob in blob_list], key=lambda x: x.name)
    
    def get_image_url(self, blob):
        """이미지 URL 생성"""
        if isinstance(blob, str):
            blob_name = blob
        else:
            blob_name = blob.name
        return f"https://{self.blob_service_client.account_name}.blob.core.windows.net/{self.container_name}/{blob_name}"
    
    def read_text_file(self, blob):
        """텍스트 파일 내용 읽기"""
        if isinstance(blob, str):
            blob_name = blob
        else:
            blob_name = blob.name
        
        blob_client = self.container_client.get_blob_client(blob_name)
        download_stream = blob_client.download_blob()
        return download_stream.readall().decode('utf-8')
    
    def save_to_blob(self, content, blob_name, content_type="text/plain"):
        """콘텐츠를 Blob Storage에 저장"""
        blob_client = self.container_client.get_blob_client(blob_name)
        blob_client.upload_blob(
            content, 
            overwrite=True,
            content_settings=ContentSettings(content_type=content_type)
        )
        return blob_client.url
    
