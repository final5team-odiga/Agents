from azure.storage.blob import BlobServiceClient
import os

def get_user_inputs():
    connection_string = os.getenv("AZURE_CONNECTION_STRING")
    blob_service = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service.get_container_client("userdata")
    txt_blobs = sorted([b.name for b in container_client.list_blobs() if b.name.endswith(".txt")])

    user_inputs = {}
    for i, blob_name in enumerate(txt_blobs):
        content = container_client.download_blob(blob_name).readall().decode("utf-8")
        user_inputs[f"{i+1}page"] = content
    return user_inputs

def get_answer_map_from_blob():
    connection_string = os.getenv("AZURE_CONNECTION_STRING")
    blob_service = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service.get_container_client("userdata")

    # a1.txt ~ a8.txt 형태만 필터링
    txt_blobs = [b.name for b in container_client.list_blobs() if b.name.startswith("a") and b.name.endswith(".txt")]

    answer_map = {}
    for blob_name in txt_blobs:
        answer_id = os.path.splitext(blob_name)[0]  # "a1.txt" → "a1"
        content = container_client.download_blob(blob_name).readall().decode("utf-8")
        answer_map[answer_id] = content.strip()
    return answer_map