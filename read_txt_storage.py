# read_all_txt_from_folder.py

from azure.storage.blob import BlobServiceClient

# 🔑 Azure Storage 연결 문자열
connect_str = ""

# 설정
container_name = "userdata"


def read_all_txt_files_from_userdata(connect_str, container_name):
    # try:
    #     # BlobServiceClient 및 ContainerClient 생성
    #     blob_service_client = BlobServiceClient.from_connection_string(
    #         connect_str)
    #     container_client = blob_service_client.get_container_client(
    #         container_name)

    #     # 📦 컨테이너 내의 모든 blob 나열
    #     blob_list = container_client.list_blobs()

    #     for blob in blob_list:
    #         if blob.name.endswith(".txt"):
    #             print(f"\n📂 파일명: {blob.name}")
    #             blob_client = container_client.get_blob_client(blob.name)
    #             content = blob_client.download_blob().readall().decode('utf-8')
    #             print("-" * 50)
    #             print(content)
    #             print("=" * 50)

    # except Exception as e:
    #     print(f"[오류] {e}")
    contents = {}  # {filename: content, ...}
    try:
        blob_service_client = BlobServiceClient.from_connection_string(
            connect_str)
        container_client = blob_service_client.get_container_client(
            container_name)
        blob_list = container_client.list_blobs()

        for blob in blob_list:
            if blob.name.endswith(".txt"):
                blob_client = container_client.get_blob_client(blob.name)
                content = blob_client.download_blob().readall().decode('utf-8')
                contents[blob.name] = content
        return contents

    except Exception as e:
        print(f"[오류] {e}")
        return {}


if __name__ == "__main__":
    txt_files = read_all_txt_files_from_userdata(connect_str, container_name)
    for fname, content in txt_files.items():
        print(f"{fname} <txt 파일 내용>:\n{content}\n{'='*40}")
