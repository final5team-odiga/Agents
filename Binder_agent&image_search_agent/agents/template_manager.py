import os
import re
import json
from typing import Dict, List
from crewai import Agent, Task
from custom_llm import get_azure_llm

class TemplateManagerAgent:
    def __init__(self):
        self.llm = get_azure_llm()
        
    def create_agent(self):
        return Agent(
            role="매거진 디자이너",
            goal="완벽한 매거진 콘텐츠를 생성하고 템플릿 데이터를 구조화",
            backstory="""당신은 매거진 디자인에 전문성을 가진 디자이너입니다. 
            사용자의 여행 경험을 바탕으로 풍부하고 상세한 매거진 콘텐츠를 생성하며, 
            이를 적절한 템플릿에 배치할 수 있는 구조화된 데이터를 만들 수 있습니다.""",
            verbose=True,
            llm=self.llm
        )
    
    def get_available_templates(self):
        """사용 가능한 템플릿 목록 가져오기"""
        templates_dir = "templates"
        if not os.path.exists(templates_dir):
            print(f"템플릿 디렉토리를 찾을 수 없습니다: {templates_dir}")
            return []
            
        template_files = [f for f in os.listdir(templates_dir) if f.endswith('.jsx')]
        
        templates = []
        for template_file in template_files:
            template_path = os.path.join(templates_dir, template_file)
            try:
                with open(template_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    templates.append({
                        "name": template_file,
                        "content": content,
                        "description": self._analyze_template_structure(content)
                    })
            except Exception as e:
                print(f"템플릿 파일 읽기 오류 ({template_file}): {str(e)}")
        
        return templates
    
    def _analyze_template_structure(self, template_content: str) -> str:
        """템플릿 구조 분석"""
        structure_info = []
        
        # Props 분석
        props_pattern = r'export const \w+ = \(\{ ([^}]*) \}\)'
        props_match = re.search(props_pattern, template_content)
        if props_match:
            props = [prop.strip() for prop in props_match.group(1).split(',') if prop.strip()]
            structure_info.append(f"Props: {', '.join(props)}")
        
        # Styled Components 분석
        styled_components = re.findall(r'const\s+(Styled\w+)', template_content)
        if styled_components:
            structure_info.append(f"Components: {len(styled_components)}개")
        
        # 레이아웃 타입 분석
        layout_types = []
        if 'display: flex' in template_content or 'display: inline-flex' in template_content:
            layout_types.append("Flex")
        if 'display: grid' in template_content:
            layout_types.append("Grid")
        if layout_types:
            structure_info.append(f"Layout: {', '.join(layout_types)}")
        
        return " | ".join(structure_info) if structure_info else "기본 레이아웃"
    
    def select_and_apply_templates(self, magazine_content, image_analysis_results):
        """템플릿 선택 및 모든 콘텐츠를 활용한 풍부한 콘텐츠 생성"""
        templates = self.get_available_templates()
        if not templates:
            print("사용 가능한 템플릿이 없습니다.")
            return self._create_emergency_template_data(magazine_content, image_analysis_results)
            
        templates_info = "\n".join([f"- {t['name']}: {t['description']}" for t in templates])
        
        # 이미지 URL 목록 생성
        image_urls = [result.get('image_url', '') for result in image_analysis_results if result.get('image_url')]
        image_locations = [result.get('location', '') for result in image_analysis_results if result.get('location')]
        
        # 콘텐츠 분석
        content_length = len(magazine_content)
        content_sections = self._analyze_content_structure(magazine_content)
        
        print(f"템플릿 선택을 위한 데이터:")
        print(f"- 사용 가능한 템플릿: {len(templates)}개")
        print(f"- 이미지 URL: {len(image_urls)}개")
        print(f"- 전체 콘텐츠 길이: {content_length}자")
        print(f"- 분석된 콘텐츠 섹션: {len(content_sections)}개")
        
        # 모든 콘텐츠를 활용하기 위한 강화된 프롬프트
        template_selection_task = Task(
            description=f"""
            당신은 전문 여행 매거진 에디터입니다. 주어진 **모든** 여행 콘텐츠와 이미지를 빠짐없이 활용하여 
            **매우 풍부하고 상세한** 매거진을 생성하고 적절한 템플릿에 배치하세요.
            
            **전체 여행 콘텐츠 (모두 활용해야 함):**
            {magazine_content}
            
            **콘텐츠 분석 결과:**
            - 전체 길이: {content_length}자
            - 주요 섹션: {len(content_sections)}개
            - 각 섹션 미리보기: {[section[:100] + '...' for section in content_sections[:3]]}
            
            **이미지 정보:**
            - 이미지 URL: {', '.join(image_urls[:10])}{'...' if len(image_urls) > 10 else ''}
            - 위치 정보: {', '.join(image_locations[:5])}{'...' if len(image_locations) > 5 else ''}
            
            **사용 가능한 템플릿:**
            {templates_info}
            
            **매거진 콘텐츠 생성 요구사항:**
            1. **모든 콘텐츠 활용**: 제공된 모든 텍스트 콘텐츠를 빠짐없이 사용해야 합니다
            2. **풍부한 스토리텔링**: 각 섹션마다 최소 400-600자의 상세한 내용
            3. **감성적 묘사**: 여행의 감정과 분위기를 생생하게 전달
            4. **구체적 디테일**: 장소, 시간, 경험의 구체적 묘사 포함
            5. **매거진 스타일**: 전문 여행 매거진 수준의 글쓰기
            6. **완전한 스토리**: 여행의 시작부터 끝까지 완전한 내러티브
            7. **충분한 템플릿 사용**: 모든 콘텐츠를 담기 위해 최소 6-8개 템플릿 선택
            
            **템플릿 선택 기준:**
            - Section01: 여행 소개 및 첫인상 (대형 제목, 감성적 소개)
            - Section03: 주요 경험 상세 (제목+부제목+긴 본문)
            - Section06: 문화/예술 체험 (상세 묘사)
            - Section08: 일상적 순간들 (여러 이미지 활용)
            - Section13: 여행 마무리 및 소감 (감성적 마무리)
            - 추가 템플릿들도 적극 활용하여 모든 콘텐츠 포함
            
            결과 형식 (반드시 JSON 형식으로):
            {{
                "selected_templates": ["Section01.jsx", "Section03.jsx", "Section06.jsx", "Section08.jsx", "Section13.jsx", "Section02.jsx", "Section04.jsx", "Section05.jsx"],
                "content_sections": [
                    {{
                        "template": "Section01.jsx",
                        "title": "매력적이고 감성적인 여행 제목",
                        "subtitle": "여행의 핵심을 담은 부제목",
                        "body": "여행 소개와 첫인상에 대한 상세하고 감성적인 묘사 (최소 500자)",
                        "tagline": "TRAVEL & CULTURE",
                        "images": ["{image_urls[0] if image_urls else ''}"]
                    }},
                    {{
                        "template": "Section03.jsx",
                        "title": "주요 여행지나 경험의 제목",
                        "subtitle": "해당 경험의 부제목이나 장소명",
                        "body": "구체적이고 상세한 여행 경험 묘사 (최소 600자, 감정과 디테일 포함)",
                        "images": ["{image_urls[1] if len(image_urls) > 1 else ''}"]
                    }},
                    {{
                        "template": "Section06.jsx",
                        "title": "문화나 예술 체험 제목",
                        "subtitle": "체험의 특별함을 나타내는 부제목",
                        "body": "문화적 경험과 그로 인한 감상, 깨달음에 대한 깊이 있는 묘사 (최소 500자)",
                        "images": ["{image_urls[2] if len(image_urls) > 2 else ''}"]
                    }},
                    {{
                        "template": "Section08.jsx",
                        "title": "일상적 순간들이나 음식 경험",
                        "subtitle": "현지의 일상을 담은 부제목",
                        "body": "현지 음식, 사람들과의 만남, 일상적 순간들에 대한 생생한 묘사 (최소 450자)",
                        "images": ["{image_urls[3] if len(image_urls) > 3 else ''}", "{image_urls[4] if len(image_urls) > 4 else ''}"]
                    }},
                    {{
                        "template": "Section13.jsx",
                        "title": "여행의 마무리와 소감",
                        "subtitle": "여행이 남긴 의미",
                        "body": "여행 전체에 대한 회고와 깊은 소감, 여행이 자신에게 미친 영향 (최소 500자)",
                        "images": ["{image_urls[5] if len(image_urls) > 5 else ''}"]
                    }},
                    {{
                        "template": "Section02.jsx",
                        "title": "추가 여행 경험 1",
                        "subtitle": "더 많은 이야기",
                        "body": "아직 다루지 않은 여행 콘텐츠의 상세한 묘사 (최소 400자)",
                        "images": ["{image_urls[6] if len(image_urls) > 6 else ''}"]
                    }},
                    {{
                        "template": "Section04.jsx",
                        "title": "추가 여행 경험 2",
                        "subtitle": "계속되는 발견",
                        "body": "남은 여행 콘텐츠의 풍부한 스토리텔링 (최소 400자)",
                        "images": ["{image_urls[7] if len(image_urls) > 7 else ''}"]
                    }},
                    {{
                        "template": "Section05.jsx",
                        "title": "추가 여행 경험 3",
                        "subtitle": "마지막 이야기들",
                        "body": "모든 남은 콘텐츠를 포함한 완성된 여행 이야기 (최소 400자)",
                        "images": ["{image_urls[8] if len(image_urls) > 8 else ''}"]
                    }}
                ]
            }}
            
            **중요:** 
            1. 제공된 모든 콘텐츠({content_length}자)를 빠짐없이 활용해야 합니다
            2. 각 섹션의 body는 반드시 풍부하고 상세해야 합니다
            3. 실제 여행 매거진에 실릴 수 있는 수준의 완성도를 가져야 합니다
            4. 충분한 수의 템플릿을 사용하여 모든 콘텐츠를 포함하세요
            """,
            expected_output="모든 콘텐츠가 포함된 풍부한 템플릿 선택 및 콘텐츠 배치 결과 (JSON 형식)"
        )
        
        try:
            agent = self.create_agent()
            result = agent.execute_task(template_selection_task)
            return self._extract_json_from_result(result, magazine_content, image_urls)
        except Exception as e:
            print(f"템플릿 선택 작업 중 오류: {str(e)}")
            return self._create_comprehensive_default_template_data(magazine_content, image_urls)
    
    def _analyze_content_structure(self, content: str) -> List[str]:
        """콘텐츠 구조 분석 및 섹션 분할"""
        if not content:
            return []
        
        sections = []
        
        # 1. 문단 기반 분할
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip() and len(p.strip()) > 50]
        if len(paragraphs) >= 3:
            sections.extend(paragraphs)
        
        # 2. 문장 기반 분할 (문단이 부족한 경우)
        if len(sections) < 5:
            sentences = [s.strip() + '.' for s in content.split('.') if s.strip() and len(s.strip()) > 30]
            sentence_groups = []
            current_group = []
            current_length = 0
            
            for sentence in sentences:
                if current_length + len(sentence) > 300 and current_group:
                    sentence_groups.append(' '.join(current_group))
                    current_group = [sentence]
                    current_length = len(sentence)
                else:
                    current_group.append(sentence)
                    current_length += len(sentence)
            
            if current_group:
                sentence_groups.append(' '.join(current_group))
            
            sections.extend(sentence_groups)
        
        # 3. 길이 기반 균등 분할 (여전히 부족한 경우)
        if len(sections) < 6:
            section_size = max(200, len(content) // 8)
            for i in range(0, len(content), section_size):
                section = content[i:i+section_size].strip()
                if section and len(section) > 100:
                    sections.append(section)
        
        return sections[:10]
    
    def generate_react_app(self, template_data, file_manager, project_name):
        """React 앱 기본 구조 생성 - template_data.json만 생성"""
        # 프로젝트 폴더 생성
        project_folder = file_manager.create_project_folder(project_name)
        src_folder, components_folder = file_manager.create_react_app(project_folder)
        
        # template_data.json 저장
        template_data_path = os.path.join(project_folder, "template_data.json")
        file_manager.save_json(template_data, template_data_path)
        
        print(f"template_data.json이 저장되었습니다: {template_data_path}")
        print(f"React 앱 기본 구조가 생성되었습니다: {project_folder}")
        
        return project_folder, template_data_path
    
    def _extract_json_from_result(self, result, magazine_content: str, image_urls: List[str]) -> Dict:
        """결과에서 JSON 추출"""
        try:
            result_str = str(result)
            
            # JSON 블록 찾기
            json_patterns = [
                r'``````',
                r'``````',
                r'\{[^{}]*"selected_templates"[^{}]*\}',
                r'\{.*?"content_sections".*?\}',
            ]
            
            for pattern in json_patterns:
                match = re.search(pattern, result_str, re.DOTALL)
                if match:
                    json_content = match.group(1) if 'json' in pattern else match.group(0)
                    try:
                        template_data = json.loads(json_content)
                        if 'selected_templates' in template_data and 'content_sections' in template_data:
                            print("JSON 파싱 성공")
                            return template_data
                    except json.JSONDecodeError:
                        continue
            
            print("유효한 JSON 형식을 찾을 수 없습니다. 기본 템플릿을 사용합니다.")
            
        except Exception as e:
            print(f"JSON 추출 중 오류: {e}. 기본 템플릿을 사용합니다.")
        
        return self._create_comprehensive_default_template_data(magazine_content, image_urls)
    
    def _create_comprehensive_default_template_data(self, magazine_content: str, image_urls: List[str]) -> Dict:
        """모든 콘텐츠를 포함하는 포괄적인 기본 템플릿 데이터 생성"""
        sections = self._analyze_content_structure(magazine_content)
        
        template_configs = [
            {"template": "Section01.jsx", "title_prefix": "여행의 시작", "subtitle_prefix": "새로운 발견의 여정"},
            {"template": "Section03.jsx", "title_prefix": "특별한 경험", "subtitle_prefix": "잊을 수 없는 순간들"},
            {"template": "Section06.jsx", "title_prefix": "문화와 만남", "subtitle_prefix": "깊이 있는 체험"},
            {"template": "Section08.jsx", "title_prefix": "일상 속 발견", "subtitle_prefix": "소소한 행복들"},
            {"template": "Section13.jsx", "title_prefix": "여행의 마무리", "subtitle_prefix": "소중한 기억들"},
            {"template": "Section02.jsx", "title_prefix": "계속되는 모험", "subtitle_prefix": "더 많은 이야기"},
            {"template": "Section04.jsx", "title_prefix": "숨겨진 보석", "subtitle_prefix": "예상치 못한 발견"},
            {"template": "Section05.jsx", "title_prefix": "마지막 여정", "subtitle_prefix": "완성된 이야기"}
        ]
        
        components = []
        total_content_used = 0
        
        for i, config in enumerate(template_configs):
            if i < len(sections):
                section_content = sections[i]
            else:
                remaining_content = magazine_content[total_content_used:]
                if remaining_content and len(remaining_content) > 100:
                    section_content = remaining_content[:len(remaining_content)//2]
                else:
                    section_content = magazine_content[total_content_used:total_content_used+500]
            
            total_content_used += len(section_content)
            
            first_sentence = section_content.split('.')[0] if section_content else config['title_prefix']
            title = f"{config['title_prefix']}: {first_sentence[:50]}..." if len(first_sentence) > 50 else f"{config['title_prefix']}: {first_sentence}"
            
            components.append({
                "template": config["template"],
                "title": title,
                "subtitle": config["subtitle_prefix"],
                "body": section_content if len(section_content) > 50 else magazine_content[i*200:(i+1)*200],
                "tagline": "TRAVEL & CULTURE",
                "images": [image_urls[i]] if i < len(image_urls) else []
            })
            
            if total_content_used >= len(magazine_content) * 0.9:
                break
        
        print(f"기본 템플릿 데이터 생성: {len(components)}개 섹션, 총 {total_content_used}자 사용")
        
        return {
            "selected_templates": [config["template"] for config in template_configs[:len(components)]],
            "content_sections": components
        }
    
    def _create_emergency_template_data(self, magazine_content: str, image_analysis_results: List[Dict]) -> Dict:
        """응급 상황용 템플릿 데이터 생성"""
        image_urls = [result.get('image_url', '') for result in image_analysis_results if result.get('image_url')]
        
        return {
            "selected_templates": ["Section01.jsx"],
            "content_sections": [
                {
                    "template": "Section01.jsx",
                    "title": "여행 이야기",
                    "subtitle": "특별한 순간들",
                    "body": magazine_content[:1000] if magazine_content else "여행의 아름다운 기억들",
                    "tagline": "TRAVEL & CULTURE",
                    "images": [image_urls[0]] if image_urls else []
                }
            ]
        }
