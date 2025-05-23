from crewai import Task

def create_text_insert_task(section_id: str, layout_fields: list, content_dict: dict, agent):
    """
    section_id: JSX 템플릿 컴포넌트 이름 (예: Section02)
    layout_fields: 해당 섹션에서 사용할 필드 목록
    content_dict: 텍스트 레이아웃 결과 (예: {"title": "...", "body": "...", ...})
    """

    # 필드별 텍스트 추출
    props_text = []
    for field in layout_fields:
        value = content_dict.get(field, "")
        if isinstance(value, str):
            escaped = value.replace('"', '\\"')  # 따옴표 이스케이프 처리
            props_text.append(f'{field}="{escaped}"')
        elif isinstance(value, list):  # imageUrls 등
            list_str = "[" + ", ".join(f'"{v}"' for v in value) + "]"
            props_text.append(f'{field}={list_str}')

    jsx_string = f"<{section_id} " + " ".join(props_text) + " />"

    return Task(
        name=f"text_insert_for_{section_id.lower()}",
        description=f"""
다음과 같은 텍스트 데이터를 바탕으로 React JSX 컴포넌트 `{section_id}`를 위한 JSX 코드를 생성해줘.

필드와 값은 다음과 같아:
{props_text}

원하는 형식:
<{section_id} 
  {chr(10).join(props_text)}
/>
""",
        expected_output=f"<{section_id} ... /> 형태의 JSX 코드",
        agent=agent
    )
