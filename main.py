import os
import json
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

from OrgAgent.tool.tool_azurellm import AzureOpenAILLM
from OrgAgent.agent.text_spliter_Agent import split_text
from OrgAgent.agent.text_Inserter_Agent import get_code_inserter
from OrgAgent.task.text_split_task import text_layout_task
from OrgAgent.task.text_insert_task import create_text_insert_task
from blob_storage import get_answer_map_from_blob

from crewai import Crew

# 1. 환경 변수 로드
dotenv_path = Path('.env')
load_dotenv(dotenv_path=dotenv_path, override=True)

# 2. 섹션 매핑 JSON 로드
with open('config/section_layout_map.json', encoding='utf-8') as f:
    layout_map = json.load(f)

# 3. Blob에서 정제된 텍스트 불러오기 (answerId 기준)
answer_map = get_answer_map_from_blob()

# 4. LLM 및 에이전트 초기화
llm = AzureOpenAILLM()
spilter = split_text(llm)
code_inserter = get_code_inserter(llm)

# 5. 섹션별 JSX 생성
jsx_results = {}

for section in layout_map:
    section_id = section["sectionId"]
    answer_id = section["answerId"]
    fields = section["fields"]

    content = answer_map.get(answer_id, "")
    if not content:
        print(f"⚠️ {answer_id} 응답 없음. 스킵합니다.")
        continue

    # 5-1 텍스트 레이아웃 분해
    layout_task = text_layout_task(content, spilter)
    layout_crew = Crew(agents=[spilter], tasks=[layout_task], verbose=True)
    layout_result = layout_crew.kickoff()
    try:
        content_dict = json.loads(layout_result.raw)
    except Exception:
        print(f"❌ {section_id} 레이아웃 분해 실패")
        continue

    # 5-2 JSX 코드 생성
    jsx_task = create_text_insert_task(section_id, fields, content_dict, code_inserter)
    jsx_crew = Crew(agents=[code_inserter], tasks=[jsx_task], verbose=True)
    jsx_result = jsx_crew.kickoff()
    jsx_results[section_id] = jsx_result.raw.strip()

# 6. JSX 코드 파일 저장
os.makedirs("generated_jsx", exist_ok=True)
for sec_id, jsx_code in jsx_results.items():
    with open(f"generated_jsx/{sec_id}.jsx", "w", encoding="utf-8") as f:
        f.write(jsx_code)

print("✅ 모든 JSX 컴포넌트 생성 완료!")