import os
import re
import json
from typing import Dict, List
from crewai import Agent, Task
from custom_llm import get_azure_llm

class JSXCreatorAgent:
    """template_data.json을 바탕으로 JSX 파일을 생성하는 CrewAI 에이전트"""
    
    def __init__(self):
        self.llm = get_azure_llm()
        
    def create_agent(self):
        return Agent(
            role="React JSX 전문 개발자",
            goal="template_data.json과 원본 템플릿을 바탕으로 완벽한 JSX 컴포넌트 생성",
            backstory="""당신은 React와 JSX 개발에 특화된 전문 개발자입니다. 
            JSON 데이터 구조를 분석하고, 기존 JSX 템플릿의 구조를 완벽히 유지하면서 
            실제 콘텐츠를 주입한 완전한 JSX 컴포넌트를 생성할 수 있습니다.
            특히 styled-components, 복잡한 JSX 표현식, 이미지 URL 처리에 전문성을 가지고 있습니다.
            생성하는 모든 JSX는 문법적으로 완벽하고 실행 가능해야 합니다.""",
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
            print(f"이미지 수: {len(content_section.get('images', []))}")
            
            # JSX 생성 에이전트로 JSX 생성
            jsx_code = self._generate_single_jsx(original_template, content_section, new_component_name)
            
            generated_components.append({
                'name': new_component_name,
                'file': new_file_name,
                'jsx_code': jsx_code,
                'template_name': template_name
            })
            
            print(f"JSX 생성 완료: {new_file_name}")
        
        return generated_components
    
    def _generate_single_jsx(self, original_template: str, content_section: Dict, component_name: str) -> str:
        """단일 JSX 컴포넌트 생성 - 완전성 보장"""
        
        jsx_generation_task = Task(
            description=f"""
            당신은 React JSX 전문가입니다. 주어진 원본 템플릿과 콘텐츠 데이터를 바탕으로 
            완벽하게 작동하는 JSX 컴포넌트를 생성해야 합니다.
            
            **원본 템플릿:**
            ```
            {original_template}
            ```
            
            **적용할 콘텐츠 데이터:**
            ```
            {json.dumps(content_section, ensure_ascii=False, indent=2)}
            ```
            
            **생성할 컴포넌트 이름:** {component_name}
            
            **핵심 요구사항:**
            1. **완전한 구조**: 반드시 마지막에 }};로 끝나야 함
            2. **구조 완벽 보존**: 원본 템플릿의 모든 styled-components와 JSX 구조를 그대로 유지
            3. **Props 제거**: Props를 받지 않는 독립적인 컴포넌트로 변경
            4. **콘텐츠 주입**: JSON 데이터의 실제 값으로 모든 props 교체
            5. **이미지 URL 처리**: 모든 이미지 URL을 올바른 JSX 형식으로 처리
            6. **JSX 표현식 처리**: 복잡한 JSX 표현식을 정적 콘텐츠로 변환
            7. **문법 완벽성**: 모든 JSX 태그가 올바르게 열리고 닫히도록 보장
            
            **출력 형식:**
            반드시 다음 구조로 완전한 JSX 파일을 생성하세요:
            ```
            import React from "react";
            import styled from "styled-components";
            
            // styled components...
            
            export const {component_name} = () => {{
            return (
                // JSX content...
            );
            }};
            ```
            
            **중요**: 마지막에 반드시 }};로 끝나야 하며, 모든 중괄호가 올바르게 닫혀야 합니다.
            """,
            expected_output="완전한 JSX 파일 코드 (마지막 }}; 포함)"
        )
        
        try:
            agent = self.create_agent()
            result = agent.execute_task(jsx_generation_task)
            
            # 결과에서 JSX 코드 추출 및 검증
            jsx_code = self._extract_and_validate_jsx(str(result), content_section, component_name)
            
            # 최종 완전성 검증
            jsx_code = self._final_structure_check(jsx_code, component_name)
            
            return jsx_code
            
        except Exception as e:
            print(f"JSX 생성 중 오류: {str(e)}")
            return self._create_emergency_jsx(original_template, content_section, component_name)

    def _final_structure_check(self, jsx_code: str, component_name: str) -> str:
        """최종 구조 검증"""
        # 마지막 }; 확인
        if not jsx_code.rstrip().endswith('};'):
            print(f"경고: {component_name}의 마지막 }};가 누락되어 추가합니다.")
            jsx_code = jsx_code.rstrip() + '\n};'
        
        # 중괄호 균형 재확인
        open_count = jsx_code.count('{')
        close_count = jsx_code.count('}')
        
        if open_count != close_count:
            print(f"경고: {component_name}의 중괄호 불균형 감지. 수정합니다.")
            jsx_code = self._balance_braces(jsx_code)
            
            # 다시 }; 확인
            if not jsx_code.rstrip().endswith('};'):
                jsx_code = jsx_code.rstrip() + '\n};'
        
        return jsx_code

    
    def _extract_and_validate_jsx(self, result: str, content_section: Dict, component_name: str) -> str:
        """결과에서 JSX 코드 추출 및 검증"""
        
        # 1. 코드 블록에서 JSX 추출
        jsx_code = self._extract_jsx_from_result(result)
        
        # 2. 기본 검증
        jsx_code = self._validate_jsx_structure(jsx_code, component_name)
        
        # 3. 이미지 URL 따옴표 수정
        jsx_code = self._fix_image_url_quotes(jsx_code)
        
        # 4. 남은 JSX 표현식 정리
        jsx_code = self._clean_remaining_expressions(jsx_code)
        
        return jsx_code
    
    def _extract_jsx_from_result(self, result: str) -> str:
        """결과에서 JSX 코드 추출 - 완전한 구조 보장"""
        # 코드 블록 패턴들
        code_patterns = [
            r'``````',
            r'``````',
            r'``````'
        ]
        
        for pattern in code_patterns:
            match = re.search(pattern, result, re.DOTALL)
            if match:
                jsx_code = match.group(1).strip()
                # 완전한 구조인지 확인
                if self._is_complete_jsx_structure(jsx_code):
                    return jsx_code
        
        # import로 시작하는 완전한 컴포넌트 찾기
        complete_component_pattern = r'(import React.*?export const \w+ = \(\) => \{.*?\}\s*;\s*)'
        complete_match = re.search(complete_component_pattern, result, re.DOTALL)
        if complete_match:
            return complete_match.group(1)
        
        # 부분적인 컴포넌트라도 찾기
        partial_component_pattern = r'(import React.*?export const \w+ = \(\) => \{.*)'
        partial_match = re.search(partial_component_pattern, result, re.DOTALL)
        if partial_match:
            jsx_code = partial_match.group(1)
            # 마지막 }; 확인 및 추가
            return self._ensure_complete_structure(jsx_code)
        
        # 전체 결과에서 추출 시도
        return self._ensure_complete_structure(result.strip())

    def _is_complete_jsx_structure(self, jsx_code: str) -> bool:
        """JSX 구조가 완전한지 확인"""
        # 기본 구조 확인
        has_import = 'import React' in jsx_code
        has_export = 'export const' in jsx_code
        has_return = 'return (' in jsx_code
        
        # 중괄호 균형 확인
        open_braces = jsx_code.count('{')
        close_braces = jsx_code.count('}')
        
        # 마지막 }; 확인
        ends_properly = jsx_code.rstrip().endswith('};')
        
        return has_import and has_export and has_return and open_braces == close_braces and ends_properly

    def _ensure_complete_structure(self, jsx_code: str) -> str:
        """JSX 구조를 완전하게 만들기"""
        jsx_code = jsx_code.strip()
        
        # import 문이 없으면 추가
        if not jsx_code.startswith('import React'):
            jsx_code = 'import React from "react";\nimport styled from "styled-components";\n\n' + jsx_code
        
        # export const가 있는지 확인
        if 'export const' not in jsx_code:
            return jsx_code  # 기본 구조가 없으면 그대로 반환
        
        # 중괄호 균형 맞추기
        jsx_code = self._balance_braces(jsx_code)
        
        # 마지막 }; 확인 및 추가
        if not jsx_code.rstrip().endswith('};'):
            # return 문이 닫히지 않았는지 확인
            if jsx_code.rstrip().endswith(')'):
                jsx_code = jsx_code.rstrip() + ';\n};'
            elif jsx_code.rstrip().endswith('}'):
                jsx_code = jsx_code.rstrip() + ';'
            else:
                jsx_code = jsx_code.rstrip() + '\n};'
        
        return jsx_code

    def _balance_braces(self, jsx_code: str) -> str:
        """중괄호 균형 맞추기"""
        lines = jsx_code.split('\n')
        brace_count = 0
        result_lines = []
        
        for line in lines:
            result_lines.append(line)
            
            # 중괄호 카운트
            for char in line:
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
        
        # 부족한 닫는 중괄호 추가
        while brace_count > 0:
            result_lines.append('}')
            brace_count -= 1
        
        return '\n'.join(result_lines)

    
    def _validate_jsx_structure(self, jsx_code: str, component_name: str) -> str:
        """JSX 구조 검증 및 수정"""
        
        # 컴포넌트 이름 확인
        if f"export const {component_name}" not in jsx_code:
            jsx_code = re.sub(r'export const \w+', f'export const {component_name}', jsx_code)
        
        # Props 제거 확인
        jsx_code = re.sub(r'= \(\{[^}]*\}\) =>', '= () =>', jsx_code)
        
        # import 문 확인
        if not jsx_code.startswith('import React'):
            jsx_code = 'import React from "react";\nimport styled from "styled-components";\n\n' + jsx_code
        
        return jsx_code
    
    def _fix_image_url_quotes(self, jsx_code: str) -> str:
        """이미지 URL 따옴표 수정"""
        # src= 뒤에 따옴표 없이 URL이 오는 패턴 수정
        jsx_code = re.sub(r'src=(https?://[^\s>"\']+)', r'src="\1"', jsx_code)
        jsx_code = re.sub(r'src=([^"\s][^\s>]*)', r'src="\1"', jsx_code)
        
        # 이중 따옴표 제거
        jsx_code = re.sub(r'src=""([^"]+)""', r'src="\1"', jsx_code)
        
        return jsx_code
    
    def _clean_remaining_expressions(self, jsx_code: str) -> str:
        """남은 JSX 표현식 정리"""
        # {변수명} 형태의 표현식 제거
        jsx_code = re.sub(r'\{[a-zA-Z_][a-zA-Z0-9_]*\}', '""', jsx_code)
        
        # {변수명.메서드()} 형태의 복잡한 표현식 제거
        jsx_code = re.sub(r'\{[a-zA-Z_][a-zA-Z0-9_]*\.[^}]*\}', '""', jsx_code)
        
        return jsx_code
    
    def _create_emergency_jsx(self, original_template: str, content_section: Dict, component_name: str) -> str:
        """응급 상황용 JSX 생성"""
        images = content_section.get('images', [])
        image_jsx = f'<img src="{images[0]}" alt="여행 이미지" style={{{{maxWidth: "100%", height: "auto"}}}} />' if images else ''
        
        return f'''import React from "react";
import styled from "styled-components";

const StyledContainer = styled.div`
  background: white;
  padding: 30px;
  margin-bottom: 40px;
  border-radius: 10px;
  box-shadow: 0 4px 6px rgba(0,0,0,0.1);
`;

const StyledTitle = styled.h1`
  font-size: 2em;
  margin-bottom: 20px;
  color: #2c3e50;
`;

const StyledSubtitle = styled.h2`
  font-size: 1.2em;
  margin-bottom: 15px;
  color: #7f8c8d;
`;

const StyledContent = styled.p`
  font-size: 1.1em;
  line-height: 1.8;
  margin-bottom: 15px;
`;

export const {component_name} = () => {{
  return (
    <StyledContainer>
      <StyledTitle>{content_section.get('title', '여행 이야기')}</StyledTitle>
      <StyledSubtitle>{content_section.get('subtitle', '특별한 순간들')}</StyledSubtitle>
      <StyledContent>{content_section.get('body', '여행의 아름다운 기억들')}</StyledContent>
      {image_jsx}
    </StyledContainer>
  );
}};
'''
    
    def save_jsx_components(self, generated_components: List[Dict], components_folder: str) -> List[Dict]:
        """생성된 JSX 컴포넌트들을 파일로 저장"""
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
                
                print(f"JSX 파일 저장 완료: {component['file']}")
                
            except Exception as e:
                print(f"JSX 파일 저장 오류 ({component['file']}): {str(e)}")
        
        return saved_components
