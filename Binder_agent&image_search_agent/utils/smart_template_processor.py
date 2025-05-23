import re
import json
import ast
from typing import Dict, List, Any

class SmartTemplateProcessor:
    """지능형 템플릿 처리기 - LLM을 활용한 정확한 JSX 생성"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def process_template_with_llm(self, template_content: str, content_data: Dict, images: List[str]) -> str:
        """LLM을 활용하여 템플릿에 콘텐츠를 정확히 주입"""
        
        # 1. 템플릿 구조 분석
        template_analysis = self._analyze_template_structure(template_content)
        
        # 2. LLM에게 정확한 JSX 생성 요청
        processed_jsx = self._generate_jsx_with_llm(
            template_content, 
            template_analysis, 
            content_data, 
            images
        )
        
        return processed_jsx
    
    def _analyze_template_structure(self, template_content: str) -> Dict:
        """템플릿 구조 분석"""
        # JSX 구조 추출
        jsx_structure = self._extract_jsx_structure(template_content)
        
        # Props 추출
        props = self._extract_props(template_content)
        
        # Styled Components 추출
        styled_components = self._extract_styled_components(template_content)
        
        return {
            "jsx_structure": jsx_structure,
            "props": props,
            "styled_components": styled_components,
            "component_name": self._extract_component_name(template_content)
        }
    
    def _extract_jsx_structure(self, template_content: str) -> str:
        """JSX return 구조 추출"""
        return_pattern = r'return\s*\(\s*(.*?)\s*\)\s*;'
        match = re.search(return_pattern, template_content, re.DOTALL)
        return match.group(1).strip() if match else ""
    
    def _extract_props(self, template_content: str) -> List[str]:
        """Props 추출"""
        props_pattern = r'export const \w+ = \(\{ ([^}]*) \}\)'
        match = re.search(props_pattern, template_content)
        if match:
            return [prop.strip() for prop in match.group(1).split(',')]
        return []
    
    def _extract_styled_components(self, template_content: str) -> List[str]:
        """Styled Components 추출"""
        styled_pattern = r'const\s+(Styled\w+)\s+=\s+styled\.\w+'
        return re.findall(styled_pattern, template_content)
    
    def _extract_component_name(self, template_content: str) -> str:
        """컴포넌트 이름 추출"""
        name_pattern = r'export const (\w+) ='
        match = re.search(name_pattern, template_content)
        return match.group(1) if match else "UnknownComponent"
    
    def _generate_jsx_with_llm(self, template_content: str, analysis: Dict, content_data: Dict, images: List[str]) -> str:
        """LLM을 활용한 정확한 JSX 생성"""
        
        prompt = f"""
        당신은 React JSX 전문가입니다. 주어진 템플릿의 구조를 완벽히 유지하면서 실제 콘텐츠를 주입하여 완전한 JSX 파일을 생성해야 합니다.
        
        **원본 템플릿:**
        ```
        {template_content}
        ```
        
        **주입할 콘텐츠:**
        - 제목: {content_data.get('title', '')}
        - 부제목: {content_data.get('subtitle', '')}
        - 본문: {content_data.get('body', '')}
        - 태그라인: {content_data.get('tagline', '')}
        
        **이미지 URL:**
        {json.dumps(images, indent=2)}
        
        완전한 JSX 파일을 생성하세요.
        """
        
        try:
            # 수정된 LLM 호출 방식
            response = self.llm(prompt)  # invoke() 대신 직접 호출
            
            jsx_code = self._extract_jsx_from_response(response)
            return jsx_code
            
        except Exception as e:
            print(f"LLM JSX 생성 중 오류: {str(e)}")
            return self._create_fallback_jsx(template_content, content_data, images, analysis)

    
    def _extract_jsx_from_response(self, response: str) -> str:
        """LLM 응답에서 JSX 코드 추출"""
        # 코드 블록 패턴 찾기
        code_patterns = [
            r'``````',
            r'``````',
            r'``````'
        ]
        
        for pattern in code_patterns:
            match = re.search(pattern, str(response), re.DOTALL)
            if match:
                return match.group(1).strip()
        
        # 코드 블록이 없으면 전체 응답 반환
        return str(response).strip()
    
    def _create_fallback_jsx(self, template_content: str, content_data: Dict, images: List[str], analysis: Dict) -> str:
        """LLM 실패 시 폴백 JSX 생성 - 원본 템플릿 구조 유지"""
        component_name = analysis['component_name'] + "Generated"
        
        # 원본 템플릿에서 Styled Components 추출
        styled_components_def = self._extract_styled_components_definitions(template_content)
        
        # 원본 JSX 구조 추출
        original_jsx = self._extract_jsx_structure(template_content)
        
        # JSX에서 props를 실제 값으로 치환
        processed_jsx = self._replace_props_with_values(original_jsx, content_data, images)
        
        fallback_jsx = f"""import React from "react";
    import styled from "styled-components";

    {styled_components_def}

    export const {component_name} = () => {{
    return (
    {self._indent_jsx(processed_jsx, 4)}
    );
    }};
    """
        return fallback_jsx

    def _replace_props_with_values(self, jsx_content: str, content_data: Dict, images: List[str]) -> str:
        """JSX에서 props를 실제 값으로 치환"""
        modified_jsx = jsx_content
        
        # props 치환
        for key, value in content_data.items():
            if isinstance(value, str):
                escaped_value = value.replace('"', '\\"').replace('\n', '\\n')
                # {prop} 형태 치환
                modified_jsx = re.sub(rf'\{{{key}\}}', escaped_value, modified_jsx)
        
        # 이미지 URL 치환
        if images:
            # imageUrl prop 치환
            modified_jsx = re.sub(r'\{imageUrl\}', images[0], modified_jsx)
            # src={imageUrl} 형태 치환
            modified_jsx = re.sub(r'src=\{imageUrl\}', f'src="{images[0]}"', modified_jsx)
            
            # placeholder URL 치환
            placeholder_patterns = [
                r'https://placehold\.co/[^"\']+',
                r'https://via\.placeholder\.com/[^"\']+',
            ]
            
            for pattern in placeholder_patterns:
                if images:
                    modified_jsx = re.sub(pattern, images[0], modified_jsx)
        
        return modified_jsx

    def _indent_jsx(self, jsx_content: str, spaces: int) -> str:
        """JSX 콘텐츠에 들여쓰기 적용"""
        lines = jsx_content.split('\n')
        indented_lines = []
        
        for line in lines:
            if line.strip():
                indented_lines.append(' ' * spaces + line)
            else:
                indented_lines.append('')
        
        return '\n'.join(indented_lines)

    def _extract_styled_components_definitions(self, template_content: str) -> str:
        """Styled Components 정의 추출"""
        # const StyledXXX = styled.xxx`...`; 패턴 찾기
        styled_pattern = r'(const\s+Styled\w+\s+=\s+styled\.\w+`[^`]*`;)'
        matches = re.findall(styled_pattern, template_content, re.DOTALL)
        return '\n\n'.join(matches)
