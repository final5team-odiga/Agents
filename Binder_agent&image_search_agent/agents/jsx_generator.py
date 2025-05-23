import os
import re
import json
from typing import Dict, List
from crewai import Agent, Task
from custom_llm import get_azure_llm

class JSXCreatorAgent:
    """template_data.json의 모든 데이터를 완전히 적용하는 JSX 생성 에이전트"""
    
    def __init__(self):
        self.llm = get_azure_llm()
        
    def create_agent(self):
        return Agent(
            role="React JSX 전문 개발자 및 데이터 통합 전문가",
            goal="template_data.json의 모든 데이터를 빠짐없이 JSX 템플릿에 적용하여 완벽한 매거진 생성",
            backstory="""당신은 React JSX 개발과 데이터 통합에 특화된 전문가입니다. 
            주어진 JSON 데이터의 모든 정보를 분석하고, 템플릿의 구조를 유지하면서도 
            필요시 템플릿을 확장하여 모든 데이터가 포함되도록 할 수 있습니다.
            특히 이미지 개수, 텍스트 길이, 복잡한 데이터 구조에 관계없이 
            모든 정보를 적절히 배치하는 전문성을 가지고 있습니다.""",
            verbose=True,
            llm=self.llm
        )
    
    def generate_jsx_components(self, template_data_path: str, templates_dir: str = "templates") -> List[Dict]:
        """template_data.json을 바탕으로 모든 JSX 컴포넌트 생성"""
        
        # template_data.json 읽기
        try:
            with open(template_data_path, 'r', encoding='utf-8') as f:
                template_data = json.load(f)
        except Exception as e:
            print(f"template_data.json 읽기 오류: {str(e)}")
            return []
        
        generated_components = []
        
        # 각 content_section에 대해 JSX 생성
        for i, content_section in enumerate(template_data.get("content_sections", [])):
            template_name = content_section.get("template")
            
            # 원본 템플릿 파일 읽기
            template_path = os.path.join(templates_dir, template_name)
            if not os.path.exists(template_path):
                print(f"템플릿 파일을 찾을 수 없습니다: {template_path}")
                continue
            
            try:
                with open(template_path, 'r', encoding='utf-8') as f:
                    original_template = f.read()
            except Exception as e:
                print(f"템플릿 파일 읽기 오류 ({template_name}): {str(e)}")
                continue
            
            # 새로운 컴포넌트 이름 생성
            base_name = template_name.replace('.jsx', '')
            new_component_name = f"{base_name}Generated{i+1}"
            new_file_name = f"{new_component_name}.jsx"
            
            print(f"\n=== {template_name} JSX 생성 시작 ===")
            print(f"컴포넌트명: {new_component_name}")
            print(f"제목: {content_section.get('title', '')[:50]}...")
            print(f"본문 길이: {len(content_section.get('body', ''))}자")
            print(f"이미지 수: {len(content_section.get('images', []))}")
            
            # JSX 생성 에이전트로 JSX 생성
            jsx_code = self._generate_complete_jsx(original_template, content_section, new_component_name)
            
            generated_components.append({
                'name': new_component_name,
                'file': new_file_name,
                'jsx_code': jsx_code,
                'template_name': template_name
            })
            
            print(f"JSX 생성 완료: {new_file_name}")
        
        return generated_components
    
    def _generate_complete_jsx(self, original_template: str, content_section: Dict, component_name: str) -> str:
        """모든 데이터를 완전히 적용한 JSX 컴포넌트 생성"""
        
        jsx_generation_task = Task(
            description=f"""
            당신은 React JSX 전문가입니다. 주어진 원본 템플릿과 콘텐츠 데이터를 바탕으로 
            **모든 데이터를 빠짐없이 포함한** 완벽한 JSX 컴포넌트를 생성해야 합니다.
            
            **원본 템플릿:**
            ```
            {original_template}
            ```
            
            **적용할 콘텐츠 데이터 (모든 데이터를 반드시 사용):**
            ```
            {json.dumps(content_section, ensure_ascii=False, indent=2)}
            ```
            
            **생성할 컴포넌트 이름:** {component_name}
            
            **핵심 요구사항:**
            1. **모든 데이터 사용**: JSON의 모든 텍스트와 이미지를 빠짐없이 포함
            2. **템플릿 구조 보존**: 원본 템플릿의 styled-components와 기본 구조 유지
            3. **데이터 적응**: 템플릿 변수보다 데이터가 많으면 템플릿을 확장하여 모든 데이터 포함
            4. **이미지 완전 활용**: 모든 이미지 URL을 적절한 위치에 배치
            5. **텍스트 완전 활용**: 제목, 부제목, 본문의 모든 내용을 포함 (잘리지 않도록)
            6. **Props 제거**: Props 없는 독립적인 컴포넌트로 생성
            7. **완벽한 구문**: 모든 JSX 태그가 올바르게 열리고 닫히도록 보장
            8. **Fragment 사용**: 여러 요소를 반환할 때는 반드시 Fragment(<></>)로 감싸기
            
            **데이터 적용 규칙:**
            - `{{title}}` → "{content_section.get('title', '')}" (전체 제목, 잘리지 않게)
            - `{{subtitle}}` → "{content_section.get('subtitle', '')}" (전체 부제목)
            - `{{body}}` → "{content_section.get('body', '')}" (전체 본문, 모든 내용 포함)
            - `{{tagline}}` → "{content_section.get('tagline', 'TRAVEL & CULTURE')}"
            - `src={{imageUrl}}` → `src="{"첫번째_이미지_URL"}"`
            - 추가 이미지들은 템플릿을 확장하여 모두 포함
            
            **템플릿 확장 규칙:**
            - 이미지가 템플릿 요구량보다 많으면: 추가 이미지 컨테이너 생성
            - 텍스트가 길면: 적절한 단락으로 분할하여 모두 표시
            - 데이터 구조가 복잡하면: 템플릿을 확장하여 모든 정보 포함
            
            **JSX 표현식 처리:**
            ```
            // 원본
            {{title.split("\\n").map((line, i) => (
              <span key={{i}}>{{line}}</span>
            ))}}
            
            // 변환 후 (실제 데이터 적용)
            <span key="0">실제_제목_첫줄</span><br />
            <span key="1">실제_제목_둘째줄</span>
            ```
            
            **출력 형식:**
            완전한 JSX 파일을 생성하세요. 여러 요소를 반환할 때는 Fragment(<></>)로 감싸세요:
            ```
            import React from "react";
            import styled from "styled-components";
            
            // 모든 styled components (원본 유지)
            
            export const {component_name} = () => {{
              return (
                <>
                  {"/* 모든 데이터가 포함된 완전한 JSX 구조 */"}
                </>
              );
            }};
            ```
            
            **중요 주의사항:**
            1. 모든 이미지 URL은 반드시 따옴표로 감싸기: `src="URL"`
            2. 텍스트 데이터는 절대 잘리지 않게 전체 포함
            3. 템플릿 변수보다 데이터가 많으면 템플릿을 확장
            4. 마지막에 반드시 `}};`로 끝나야 함
            5. 모든 JSX 태그가 올바르게 닫혀야 함
            6. 여러 요소를 반환할 때는 반드시 Fragment(<></>)로 감싸기
            """,
            expected_output="모든 데이터가 포함된 완전한 JSX 파일 코드"
        )
        
        try:
            agent = self.create_agent()
            result = agent.execute_task(jsx_generation_task)
            
            # 결과에서 JSX 코드 추출 및 검증
            jsx_code = self._extract_and_validate_jsx(str(result), content_section, component_name)
            
            # 데이터 완전성 검증
            jsx_code = self._ensure_all_data_included(jsx_code, content_section, component_name)
            
            return jsx_code
            
        except Exception as e:
            print(f"JSX 생성 중 오류: {str(e)}")
            return self._create_comprehensive_fallback_jsx(original_template, content_section, component_name)
    
    def _extract_and_validate_jsx(self, result: str, content_section: Dict, component_name: str) -> str:
        """결과에서 JSX 코드 추출 및 기본 검증"""
        
        # 1. 코드 블록에서 JSX 추출
        jsx_code = self._extract_jsx_from_result(result)
        
        # 2. 기본 구조 검증
        jsx_code = self._validate_basic_structure(jsx_code, component_name)
        
        # 3. 이미지 URL 따옴표 수정
        jsx_code = self._fix_image_urls(jsx_code)
        
        return jsx_code
    
    def _extract_jsx_from_result(self, result: str) -> str:
        """결과에서 JSX 코드 추출"""
        # 코드 블록 패턴들
        code_patterns = [
            r'``````',
            r'``````',
            r'``````'
        ]
        
        for pattern in code_patterns:
            match = re.search(pattern, result, re.DOTALL)
            if match:
                return match.group(1).strip()
        
        # import로 시작하는 부분 찾기
        import_pattern = r'(import React.*?export const.*?};)'
        import_match = re.search(import_pattern, result, re.DOTALL)
        if import_match:
            return import_match.group(1)
        
        return result.strip()
    
    def _validate_basic_structure(self, jsx_code: str, component_name: str) -> str:
        """기본 JSX 구조 검증"""
        
        # import 문 확인
        if not jsx_code.startswith('import React'):
            jsx_code = 'import React from "react";\nimport styled from "styled-components";\n\n' + jsx_code
        
        # 컴포넌트 이름 확인
        if f"export const {component_name}" not in jsx_code:
            jsx_code = re.sub(r'export const \w+', f'export const {component_name}', jsx_code)
        
        # Props 제거
        jsx_code = re.sub(r'= \(\{[^}]*\}\) =>', '= () =>', jsx_code)
        
        # 마지막 }; 확인
        if not jsx_code.rstrip().endswith('};'):
            jsx_code = jsx_code.rstrip() + '\n};'
        
        return jsx_code
    
    def _fix_image_urls(self, jsx_code: str) -> str:
        """이미지 URL 따옴표 수정"""
        # src= 뒤에 따옴표 없이 URL이 오는 패턴 수정
        jsx_code = re.sub(r'src=(https?://[^\s>"\']+)', r'src="\1"', jsx_code)
        jsx_code = re.sub(r'src=([^"\s][^\s>]*)', r'src="\1"', jsx_code)
        
        return jsx_code
    
    def _ensure_all_data_included(self, jsx_code: str, content_section: Dict, component_name: str) -> str:
        """모든 데이터가 포함되었는지 검증하고 누락된 데이터 추가"""
        
        title = content_section.get('title', '')
        subtitle = content_section.get('subtitle', '')
        body = content_section.get('body', '')
        images = content_section.get('images', [])
        
        # 데이터 포함 여부 확인
        missing_data = []
        
        if title and title not in jsx_code:
            missing_data.append(f"제목: {title[:50]}...")
        
        if subtitle and subtitle not in jsx_code:
            missing_data.append(f"부제목: {subtitle[:50]}...")
        
        if body and body not in jsx_code:
            missing_data.append(f"본문: {len(body)}자")
        
        for i, img in enumerate(images):
            if img and img not in jsx_code:
                missing_data.append(f"이미지 {i+1}: {img}")
        
        if missing_data:
            print(f"⚠️  누락된 데이터 감지: {missing_data}")
            jsx_code = self._add_missing_data(jsx_code, content_section)
        
        return jsx_code
    
    def _add_missing_data(self, jsx_code: str, content_section: Dict) -> str:
        """누락된 데이터를 JSX에 추가"""
        
        # return 문 찾기
        return_pattern = r'return\s*\(\s*(.*?)\s*\)\s*;'
        return_match = re.search(return_pattern, jsx_code, re.DOTALL)
        
        if return_match:
            jsx_content = return_match.group(1)
            
            # 누락된 데이터를 추가
            additional_content = self._create_additional_content(content_section)
            
            if additional_content:
                # Fragment로 전체 감싸기
                enhanced_jsx = f'''
      <>
{jsx_content}
{additional_content}
      </>'''
                
                # 전체 JSX 코드 업데이트
                jsx_code = jsx_code.replace(
                    return_match.group(0),
                    f'return ({enhanced_jsx}\n  );'
                )
        
        return jsx_code
    
    def _create_additional_content(self, content_section: Dict) -> str:
        """누락된 데이터를 위한 추가 콘텐츠 생성"""
        additional_jsx = []
        
        # Fragment로 감싸기 시작
        additional_jsx.append('        <>')
        
        title = content_section.get('title', '')
        subtitle = content_section.get('subtitle', '')
        body = content_section.get('body', '')
        images = content_section.get('images', [])
        
        # 추가 텍스트 콘텐츠
        if title:
            additional_jsx.append(f'          <div style={{{{marginBottom: "20px"}}}}>')
            additional_jsx.append(f'            <h1>{title}</h1>')
            additional_jsx.append(f'          </div>')
        
        if subtitle:
            additional_jsx.append(f'          <div style={{{{marginBottom: "15px"}}}}>')
            additional_jsx.append(f'            <h2>{subtitle}</h2>')
            additional_jsx.append(f'          </div>')
        
        if body:
            # 본문을 문단으로 분할
            body_paragraphs = [p.strip() for p in body.split('\n') if p.strip()]
            additional_jsx.append(f'          <div style={{{{marginBottom: "20px"}}}}>')
            for para in body_paragraphs:
                additional_jsx.append(f'            <p>{para}</p>')
            additional_jsx.append(f'          </div>')
        
        # 추가 이미지들
        if images:
            additional_jsx.append(f'          <div style={{{{display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))", gap: "20px", margin: "20px 0"}}}}>')
            for i, img_url in enumerate(images):
                additional_jsx.append(f'            <img src="{img_url}" alt="추가 이미지 {i+1}" style={{{{maxWidth: "100%", height: "auto", borderRadius: "8px"}}}} />')
            additional_jsx.append(f'          </div>')
        
        # Fragment로 감싸기 끝
        additional_jsx.append('        </>')
        
        return '\n'.join(additional_jsx)
    
    def _create_comprehensive_fallback_jsx(self, original_template: str, content_section: Dict, component_name: str) -> str:
        """포괄적인 폴백 JSX 생성 - 모든 데이터 포함"""
        
        title = content_section.get('title', '여행 이야기')
        subtitle = content_section.get('subtitle', '특별한 순간들')
        body = content_section.get('body', '여행의 아름다운 기억들')
        tagline = content_section.get('tagline', 'TRAVEL & CULTURE')
        images = content_section.get('images', [])
        
        # 본문을 문단으로 분할
        body_paragraphs = [p.strip() for p in body.split('\n') if p.strip()]
        
        # 이미지 JSX 생성
        images_jsx = []
        if images:
            for i, img_url in enumerate(images):
                images_jsx.append(f'          <img src="{img_url}" alt="여행 이미지 {i+1}" style={{{{maxWidth: "100%", height: "auto", borderRadius: "8px", margin: "10px 0"}}}} />')
        
        return f'''import React from "react";
import styled from "styled-components";

const StyledContainer = styled.div`
  background: white;
  padding: 30px;
  margin-bottom: 40px;
  border-radius: 10px;
  box-shadow: 0 4px 6px rgba(0,0,0,0.1);
  max-width: 1200px;
  margin: 0 auto 40px auto;
`;

const StyledTitle = styled.h1`
  font-size: 2.5em;
  margin-bottom: 20px;
  color: #2c3e50;
  line-height: 1.3;
`;

const StyledSubtitle = styled.h2`
  font-size: 1.5em;
  margin-bottom: 15px;
  color: #7f8c8d;
  font-style: italic;
`;

const StyledContent = styled.div`
  font-size: 1.1em;
  line-height: 1.8;
  margin-bottom: 20px;
  
  p {{
    margin-bottom: 15px;
  }}
`;

const StyledTagline = styled.p`
  text-align: center;
  font-size: 0.9em;
  color: #95a5a6;
  font-weight: bold;
  margin-top: 30px;
`;

const StyledImageGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 20px;
  margin: 30px 0;
`;

export const {component_name} = () => {{
  return (
    <>
      <StyledContainer>
        <StyledTitle>{title}</StyledTitle>
        <StyledSubtitle>{subtitle}</StyledSubtitle>
        <StyledContent>
{chr(10).join([f"          <p>{para}</p>" for para in body_paragraphs])}
        </StyledContent>
        <StyledImageGrid>
{chr(10).join(images_jsx)}
        </StyledImageGrid>
        <StyledTagline>{tagline}</StyledTagline>
      </StyledContainer>
    </>
  );
}};
'''
    
    def save_jsx_components(self, generated_components: List[Dict], components_folder: str) -> List[Dict]:
        """생성된 JSX 컴포넌트들을 파일로 저장"""
        os.makedirs(components_folder, exist_ok=True)
        saved_components = []
        
        for component in generated_components:
            file_path = os.path.join(components_folder, component['file'])
            
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(component['jsx_code'])
                
                saved_components.append({
                    'name': component['name'],
                    'file': component['file']
                })
                
                print(f"✓ {component['file']} 저장 완료")
                
            except Exception as e:
                print(f"✗ {component['file']} 저장 실패: {str(e)}")
        
        return saved_components
