import os
import json
from dotenv import load_dotenv
from crewai import BaseLLM, Agent, Task, Crew
from openai import AzureOpenAI
from pathlib import Path
from typing import Union, List, Dict
from azure.storage.blob import BlobServiceClient
from datetime import datetime

# 🔹 1. 환경 변수 로드
dotenv_path = Path(r'C:\Users\wlsth\OneDrive\Desktop\MS_AI\.Project3\travel-magazine\.env')
load_dotenv(dotenv_path=dotenv_path, override=True)

# 🔹 2. Azure OpenAI LLM 클래스 정의
def get_azure_llm():
    class AzureOpenAILLM(BaseLLM):
        def __init__(self):
            self.api_key = os.getenv("AZURE_API_KEY")
            self.azure_endpoint = os.getenv("AZURE_API_BASE")
            self.api_version = os.getenv("AZURE_API_VERSION")
            self.deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
            if not all([self.api_key, self.azure_endpoint, self.api_version, self.deployment_name]):
                raise ValueError("Missing Azure OpenAI credentials.")
            super().__init__(model=f"azure/{self.deployment_name}")
            self.client = AzureOpenAI(
                api_key=self.api_key,
                azure_endpoint=self.azure_endpoint,
                api_version=self.api_version
            )
        def call(self, messages: Union[str, List[Dict[str, str]]], **kwargs):
            if isinstance(messages, str):
                messages = [{"role": "user", "content": messages}]
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=messages,
                temperature=0.7,
                max_tokens=4000
            )
            return response.choices[0].message.content.strip()
    return AzureOpenAILLM()

# 🔹 3. LLM 및 Azure Blob 연결
llm = get_azure_llm()
AZURE_CONNECTION_STRING = os.getenv("AZURE_CONNECTION_STRING")
blob_service = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
container_client = blob_service.get_container_client("userdata")

# ✅ [1] Azure Blob에서 .txt 파일 읽기
txt_blobs = sorted([b.name for b in container_client.list_blobs() if b.name.endswith(".txt")])
user_inputs = {}
for i, blob_name in enumerate(txt_blobs):
    content = container_client.download_blob(blob_name).readall().decode("utf-8")
    user_inputs[f"{i+1}page"] = content

# 🔹 4. 에이전트 정의
scheduler = Agent(
    role="Travel Scheduler",
    goal="사용자의 여행 데이터를 요약하여 여정과 지역을 정리, 소제목 텍스트 작성",
    backstory="여행 전문가로서 장소, 일정, 동선 요약에 능숙함",
    llm=llm,
    verbose=True
)
orchestrator = Agent(
    role="Magazine Orchestrator",
    goal="중복 내용을 축약하고 문맥을 정리해 매거진 스타일로 재구성",
    backstory="여행 매거진 전문가로서 전체 흐름을 부드럽게 이어주는 데 능함",
    llm=llm,
    verbose=True
)

# ✅ [2] CrewAI 기반 요약 + 정제 작업 수행
# ✅ [3] 각 결과를 edited_sections에 저장
edited_sections = {}
for page, user_input in user_inputs.items():
    summarize = Task(
        name="summarize",
        description=f"[{page}] 사용자 응답을 요약해서 소제목 구문으로 정리해줘:\n{user_input}",
        agent=scheduler,
        expected_output="소제목 중심 요약 텍스트"
    )

    edit = Task(
        name="edit",
        description="위 요약을 바탕으로 매거진 스타일로 정제:\n{summarize}",
        agent=orchestrator,
        expected_output="정제된 문장 흐름"
    )

    crew = Crew(
        agents=[scheduler, orchestrator],
        tasks=[summarize, edit],
        verbose=True
    )

    result = crew.kickoff()
    edited_sections[page] = result.raw.strip()  # CrewOutput → 문자열 저장

# ✅ [4] React에서 사용할 magazine_pages.json 파일 생성
magazine_data = []
for page in sorted(edited_sections.keys(), key=lambda x: int(x.replace("page", ""))):
    magazine_data.append({
        "section": int(page.replace("page", "")),
        "content": edited_sections[page].strip()
    })

os.makedirs("magazine_txts", exist_ok=True)
json_path = os.path.join("magazine_txts", "magazine_pages.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(magazine_data, f, ensure_ascii=False, indent=2)

print(f"✅ React용 JSON 저장 완료: {json_path}")

# ✅ [5] 최종 전체 에세이 텍스트도 .txt로 저장
# (Orchestrator가 전체 텍스트 연결 및 흐름 조정)
compiled_text = "\n\n".join(f"[{d['section']}]\n{d['content']}" for d in magazine_data)

orchestration_task = Task(
    name="finalize",
    description=f"""
다음은 정제된 여행 페이지입니다. 이 전체 내용을 감정선과 흐름에 맞게 하나의 매거진 내러티브로 엮어줘.
- 문단 연결을 부드럽게
- 반복 줄이고 감정선 유지
- 가능한 수정은 최소한으로 할 것

{compiled_text}
""",
    agent=orchestrator,
    expected_output="하나로 연결된 여행 에세이"
)

final_crew = Crew(
    agents=[orchestrator],
    tasks=[orchestration_task],
    verbose=True
)
final_magazine = final_crew.kickoff()

# 결과 파일 저장
timestamp = datetime.now().strftime('%Y%m%d_%H%M')
txt_path = f"final_magazine_{timestamp}.txt"
with open(txt_path, "w", encoding="utf-8") as f:
    f.write(final_magazine.raw.strip())

print(f"✅ 최종 매거진 텍스트 저장 완료: {txt_path}")