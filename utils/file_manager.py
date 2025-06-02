import os
import json
from typing import List, Dict

class FileManager:
    def __init__(self, output_folder="./output"):
        self.output_folder = output_folder
        os.makedirs(output_folder, exist_ok=True)

    def create_project_folder(self, project_name):
        """프로젝트 폴더 생성"""
        project_path = os.path.join(self.output_folder, project_name)
        os.makedirs(project_path, exist_ok=True)
        return project_path

    def save_content(self, content, file_path):
        """콘텐츠를 파일로 저장"""
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return file_path

    def save_magazine_content_json(self, magazine_content, file_path):
        """매거진 콘텐츠를 JSON 형식으로 저장 (구조적 데이터 포함)"""
        try:
            # magazine_content가 이미 딕셔너리인지 확인
            if isinstance(magazine_content, dict):
                content_json = magazine_content
            elif isinstance(magazine_content, str):
                # 문자열인 경우 JSON 파싱 시도
                try:
                    content_json = json.loads(magazine_content)
                except json.JSONDecodeError:
                    # JSON이 아닌 일반 텍스트인 경우 구조화
                    content_json = {
                        "magazine_title": "여행 매거진",
                        "content_type": "integrated_magazine",
                        "sections": self._parse_text_to_sections(magazine_content),
                        "raw_content": magazine_content,
                        "layout_structure": self._generate_default_layout_structure(),
                        "metadata": {
                            "content_length": len(magazine_content),
                            "creation_date": self._get_current_timestamp(),
                            "format": "json",
                            "agent_enhanced": True
                        }
                    }
            else:
                # 기타 타입인 경우 문자열로 변환 후 처리
                content_json = {
                    "content_type": "unknown",
                    "raw_content": str(magazine_content),
                    "layout_structure": self._generate_default_layout_structure(),
                    "metadata": {
                        "original_type": str(type(magazine_content)),
                        "creation_date": self._get_current_timestamp(),
                        "agent_enhanced": True
                    }
                }

            # JSON 파일로 저장
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(content_json, f, ensure_ascii=False, indent=2)
            print(f"✅ 매거진 콘텐츠 JSON 저장 성공: {file_path}")
            return file_path

        except Exception as e:
            print(f"❌ 매거진 콘텐츠 JSON 저장 실패: {e}")
            # 폴백: 텍스트로 저장
            with open(file_path.replace('.json', '.txt'), 'w', encoding='utf-8') as f:
                f.write(str(magazine_content))
            return file_path

    def _generate_default_layout_structure(self) -> Dict:
        """기본 레이아웃 구조 생성"""
        return {
            "page_grid": {
                "columns": 12,
                "gutter": "20px",
                "margin": "40px"
            },
            "text_areas": [
                {"id": "title", "position": {"x": 0, "y": 0, "width": 12, "height": 2}},
                {"id": "subtitle", "position": {"x": 0, "y": 2, "width": 12, "height": 1}},
                {"id": "body", "position": {"x": 0, "y": 3, "width": 8, "height": 6}}
            ],
            "image_areas": [
                {"id": "hero", "position": {"x": 8, "y": 3, "width": 4, "height": 6}}
            ],
            "layout_type": "grid",
            "responsive_breakpoints": {
                "mobile": "768px",
                "tablet": "1024px",
                "desktop": "1200px"
            }
        }

    def _parse_text_to_sections(self, content: str) -> List[Dict]:
        """텍스트 콘텐츠를 섹션별로 파싱 (구조적 정보 포함)"""
        import re
        sections = []

        # 헤더 기반 분할 (## 또는 ###)
        header_pattern = r'^(#{1,3})\s+(.+?)$'
        current_section = {"title": "", "content": "", "level": 1}
        current_content = []

        lines = content.split('\n')
        for line in lines:
            header_match = re.match(header_pattern, line.strip())
            if header_match:
                # 이전 섹션 저장
                if current_content or current_section["title"]:
                    current_section["content"] = '\n'.join(current_content).strip()
                    if current_section["content"] or current_section["title"]:
                        # 구조적 정보 추가
                        current_section["layout_info"] = {
                            "estimated_reading_time": len(current_section["content"]) // 200,
                            "content_density": "high" if len(current_section["content"]) > 1000 else "medium",
                            "suggested_layout": "grid" if len(current_section["content"]) > 500 else "minimal"
                        }
                        sections.append(current_section.copy())

                # 새 섹션 시작
                header_level = len(header_match.group(1))
                header_text = header_match.group(2).strip()
                current_section = {
                    "title": header_text,
                    "content": "",
                    "level": header_level,
                    "type": "header_section"
                }
                current_content = []
            else:
                # 내용 추가
                if line.strip():
                    current_content.append(line)

        # 마지막 섹션 저장
        if current_content or current_section["title"]:
            current_section["content"] = '\n'.join(current_content).strip()
            if current_section["content"] or current_section["title"]:
                current_section["layout_info"] = {
                    "estimated_reading_time": len(current_section["content"]) // 200,
                    "content_density": "high" if len(current_section["content"]) > 1000 else "medium",
                    "suggested_layout": "grid" if len(current_section["content"]) > 500 else "minimal"
                }
                sections.append(current_section)

        # 섹션이 없으면 전체를 하나의 섹션으로
        if not sections:
            sections.append({
                "title": "여행 이야기",
                "content": content,
                "level": 1,
                "type": "full_content",
                "layout_info": {
                    "estimated_reading_time": len(content) // 200,
                    "content_density": "high" if len(content) > 1000 else "medium",
                    "suggested_layout": "magazine"
                }
            })

        return sections

    def _get_current_timestamp(self) -> str:
        """현재 타임스탬프 반환"""
        from datetime import datetime
        return datetime.now().isoformat()

    def save_json(self, data, file_path):
        """데이터를 JSON 파일로 저장 - 안전한 처리 (구조적 데이터 지원)"""
        try:
            # 데이터가 문자열인 경우 파싱 시도
            if isinstance(data, str):
                try:
                    # JSON 파싱 시도
                    parsed_data = json.loads(data)
                    data = parsed_data
                except json.JSONDecodeError:
                    # Python dict 문자열 변환 시도
                    try:
                        import ast
                        cleaned_str = data.replace("'", '"').replace('True', 'true').replace('False', 'false').replace('None', 'null')
                        parsed_data = json.loads(cleaned_str)
                        data = parsed_data
                    except:
                        try:
                            parsed_data = ast.literal_eval(data)
                            data = parsed_data
                        except:
                            print(f"⚠️ JSON 저장 실패: 문자열을 파싱할 수 없습니다")
                            return file_path

            # 구조적 메타데이터 추가
            if isinstance(data, dict) and 'metadata' not in data:
                data['metadata'] = {
                    "creation_date": self._get_current_timestamp(),
                    "file_manager_version": "2.0",
                    "agent_enhanced": True
                }

            # JSON 파일로 저장
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"✅ JSON 파일 저장 성공: {file_path}")
            return file_path

        except Exception as e:
            print(f"❌ JSON 파일 저장 중 오류: {e}")
            # 폴백: 문자열로 저장
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(str(data))
            return file_path

    def create_magazine_react_app(self, project_folder, saved_components, template_data):
        """매거진 스타일 React 앱 생성 (다중 컴포넌트 캐러셀 뷰어)"""
        print(f"📱 매거진 스타일 React 앱 생성 시작: {project_folder}")

        # 기본 구조 생성
        src_folder = os.path.join(project_folder, "src")
        components_folder = os.path.join(src_folder, "components")
        public_folder = os.path.join(project_folder, "public")
        
        os.makedirs(components_folder, exist_ok=True)
        os.makedirs(public_folder, exist_ok=True)

        # 1. index.html 생성 (매거진 최적화)
        index_html = """<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <link rel="icon" href="%PUBLIC_URL%/favicon.ico" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <meta name="theme-color" content="#000000" />
  <meta name="description" content="AI Generated Magazine Components Viewer" />
  <title>AI 매거진 컴포넌트 뷰어</title>
  <style>
    body {
      margin: 0;
      padding: 0;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen',
        'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue',
        sans-serif;
      -webkit-font-smoothing: antialiased;
      -moz-osx-font-smoothing: grayscale;
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      min-height: 100vh;
    }
    
    * {
      box-sizing: border-box;
    }
  </style>
</head>
<body>
  <noscript>You need to enable JavaScript to run this app.</noscript>
  <div id="root"></div>
</body>
</html>"""

        with open(os.path.join(public_folder, "index.html"), 'w', encoding='utf-8') as f:
            f.write(index_html)

        # 2. package.json 생성
        package_json = {
            "name": "ai-magazine-viewer",
            "version": "0.1.0",
            "private": True,
            "dependencies": {
                "react": "^18.2.0",
                "react-dom": "^18.2.0",
                "react-scripts": "5.0.1",
                "styled-components": "^6.0.0",
                "react-multi-carousel": "^2.8.4"
            },
            "scripts": {
                "start": "react-scripts start",
                "build": "react-scripts build",
                "test": "react-scripts test",
                "eject": "react-scripts eject"
            },
            "eslintConfig": {
                "extends": [
                    "react-app",
                    "react-app/jest"
                ]
            },
            "browserslist": {
                "production": [
                    ">0.2%",
                    "not dead",
                    "not op_mini all"
                ],
                "development": [
                    "last 1 chrome version",
                    "last 1 firefox version",
                    "last 1 safari version"
                ]
            }
        }

        with open(os.path.join(project_folder, "package.json"), 'w', encoding='utf-8') as f:
            json.dump(package_json, f, indent=2)

        # 3. 생성된 컴포넌트들을 components 폴더에 복사
        component_imports = []
        component_list = []
        
        for i, component_data in enumerate(saved_components):
            component_name = component_data.get('name', f'Component{i+1}')
            component_file = component_data.get('file', f'{component_name}.jsx')
            jsx_code = component_data.get('jsx_code', '')
            
            if jsx_code:
                # 컴포넌트 파일 저장
                component_path = os.path.join(components_folder, component_file)
                with open(component_path, 'w', encoding='utf-8') as f:
                    f.write(jsx_code)
                
                # import 문과 컴포넌트 리스트에 추가
                component_imports.append(f"import {{ {component_name} }} from './components/{component_file.replace('.jsx', '')}';")
                component_list.append({
                    'name': component_name,
                    'component': component_name,
                    'title': f"컴포넌트 {i+1}: {component_name}",
                    'description': f"AI가 생성한 매거진 컴포넌트 #{i+1}"
                })

        # 4. 메인 App.js 생성 (다중 컴포넌트 캐러셀 뷰어)
        app_js = f"""import React, {{ useState }} from 'react';
import Carousel from 'react-multi-carousel';
import 'react-multi-carousel/lib/styles.css';
import styled from 'styled-components';
{chr(10).join(component_imports)}

const AppContainer = styled.div`
  min-height: 100vh;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  padding: 20px;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
`;

const Header = styled.div`
  text-align: center;
  margin-bottom: 30px;
  color: white;
`;

const Title = styled.h1`
  font-size: 3rem;
  margin: 0 0 10px 0;
  text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
  font-weight: 700;
`;

const Subtitle = styled.p`
  font-size: 1.2rem;
  margin: 0;
  opacity: 0.9;
`;

const CarouselContainer = styled.div`
  max-width: 1400px;
  margin: 0 auto;
  background: rgba(255, 255, 255, 0.1);
  border-radius: 20px;
  padding: 30px;
  backdrop-filter: blur(10px);
  box-shadow: 0 20px 40px rgba(0,0,0,0.1);
`;

const ComponentWrapper = styled.div`
  margin: 0 15px;
  background: white;
  border-radius: 15px;
  overflow: hidden;
  box-shadow: 0 10px 30px rgba(0,0,0,0.2);
  transition: transform 0.3s ease, box-shadow 0.3s ease;
  
  &:hover {{
    transform: translateY(-5px);
    box-shadow: 0 20px 40px rgba(0,0,0,0.3);
  }}
`;

const ComponentHeader = styled.div`
  background: linear-gradient(45deg, #667eea, #764ba2);
  color: white;
  padding: 20px;
  text-align: center;
`;

const ComponentTitle = styled.h3`
  margin: 0 0 5px 0;
  font-size: 1.4rem;
  font-weight: 600;
`;

const ComponentDescription = styled.p`
  margin: 0;
  opacity: 0.9;
  font-size: 0.9rem;
`;

const ComponentContent = styled.div`
  max-height: 80vh;
  overflow-y: auto;
  padding: 0;
  
  /* 스크롤바 스타일링 */
  &::-webkit-scrollbar {{
    width: 8px;
  }}
  
  &::-webkit-scrollbar-track {{
    background: #f1f1f1;
  }}
  
  &::-webkit-scrollbar-thumb {{
    background: #888;
    border-radius: 4px;
  }}
  
  &::-webkit-scrollbar-thumb:hover {{
    background: #555;
  }}
`;

const NavigationInfo = styled.div`
  text-align: center;
  margin-bottom: 20px;
  color: white;
  font-size: 1.1rem;
  opacity: 0.8;
`;

const ComponentCounter = styled.div`
  text-align: center;
  margin-top: 20px;
  color: white;
  font-size: 1rem;
  opacity: 0.7;
`;

const CustomDot = styled.button`
  width: 12px;
  height: 12px;
  border-radius: 50%;
  border: none;
  background: ${{props => props.active ? '#fff' : 'rgba(255,255,255,0.5)'}};
  margin: 0 5px;
  cursor: pointer;
  transition: all 0.3s ease;
  
  &:hover {{
    background: #fff;
    transform: scale(1.2);
  }}
`;

const responsive = {{
  desktop: {{
    breakpoint: {{ max: 3000, min: 1024 }},
    items: 1,
    slidesToSlide: 1
  }},
  tablet: {{
    breakpoint: {{ max: 1024, min: 464 }},
    items: 1,
    slidesToSlide: 1
  }},
  mobile: {{
    breakpoint: {{ max: 464, min: 0 }},
    items: 1,
    slidesToSlide: 1
  }}
}};

const CustomButtonGroup = ({{ next, previous, goToSlide, ...rest }}) => {{
  const {{ carouselState: {{ currentSlide, totalItems }} }} = rest;
  
  return (
    <div style={{{{ 
      textAlign: 'center', 
      marginTop: '20px',
      display: 'flex',
      justifyContent: 'center',
      alignItems: 'center',
      gap: '20px'
    }}}}>
      <button
        onClick={{() => previous()}}
        disabled={{currentSlide === 0}}
        style={{{{
          background: currentSlide === 0 ? 'rgba(255,255,255,0.3)' : 'rgba(255,255,255,0.8)',
          border: 'none',
          borderRadius: '50%',
          width: '50px',
          height: '50px',
          cursor: currentSlide === 0 ? 'not-allowed' : 'pointer',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          fontSize: '20px',
          transition: 'all 0.3s ease',
          color: currentSlide === 0 ? 'rgba(0,0,0,0.3)' : '#333'
        }}}}
      >
        ←
      </button>
      
      <span style={{{{ color: 'white', fontSize: '16px', fontWeight: '500' }}}}>
        {{currentSlide + 1}} / {{totalItems}}
      </span>
      
      <button
        onClick={{() => next()}}
        disabled={{currentSlide >= totalItems - 1}}
        style={{{{
          background: currentSlide >= totalItems - 1 ? 'rgba(255,255,255,0.3)' : 'rgba(255,255,255,0.8)',
          border: 'none',
          borderRadius: '50%',
          width: '50px',
          height: '50px',
          cursor: currentSlide >= totalItems - 1 ? 'not-allowed' : 'pointer',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          fontSize: '20px',
          transition: 'all 0.3s ease',
          color: currentSlide >= totalItems - 1 ? 'rgba(0,0,0,0.3)' : '#333'
        }}}}
      >
        →
      </button>
    </div>
  );
}};

const components = [
{chr(10).join([f"  {{ name: '{comp['name']}', component: {comp['component']}, title: '{comp['title']}', description: '{comp['description']}' }}," for comp in component_list])}
];

function App() {{
  const [currentSlide, setCurrentSlide] = useState(0);

  const handleSlideChange = (previousSlide, {{ currentSlide, onMove }}) => {{
    setCurrentSlide(currentSlide);
  }};

  return (
    <AppContainer>
      <Header>
        <Title>🎨 AI 매거진 컴포넌트 뷰어</Title>
        <Subtitle>생성된 {len(component_list)}개의 매거진 컴포넌트를 확인해보세요</Subtitle>
      </Header>
      
      <CarouselContainer>
        <NavigationInfo>
          ← → 화살표 버튼을 사용하여 컴포넌트를 탐색하세요
        </NavigationInfo>
        
        <Carousel
          responsive={{responsive}}
          infinite={{false}}
          autoPlay={{false}}
          keyBoardControl={{true}}
          customTransition="transform 300ms ease-in-out"
          transitionDuration={{300}}
          containerClass="carousel-container"
          removeArrowOnDeviceType={{[]}}
          arrows={{false}}
          renderButtonGroupOutside={{true}}
          customButtonGroup={{<CustomButtonGroup />}}
          beforeChange={{handleSlideChange}}
          showDots={{false}}
        >
          {{components.map((comp, index) => {{
            const ComponentToRender = comp.component;
            return (
              <ComponentWrapper key={{index}}>
                <ComponentHeader>
                  <ComponentTitle>{{comp.title}}</ComponentTitle>
                  <ComponentDescription>{{comp.description}}</ComponentDescription>
                </ComponentHeader>
                <ComponentContent>
                  <ComponentToRender />
                </ComponentContent>
              </ComponentWrapper>
            );
          }})}}
        </Carousel>
        
        <ComponentCounter>
          현재 보고 있는 컴포넌트: {{currentSlide + 1}} / {len(component_list)}
        </ComponentCounter>
      </CarouselContainer>
    </AppContainer>
  );
}}

export default App;"""

        with open(os.path.join(src_folder, "App.js"), 'w', encoding='utf-8') as f:
            f.write(app_js)

        # 5. index.js 생성
        index_js = """import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);"""

        with open(os.path.join(src_folder, "index.js"), 'w', encoding='utf-8') as f:
            f.write(index_js)

        # 6. App.css 생성 (추가 스타일링)
        app_css = """.carousel-container {
  position: relative;
  width: 100%;
  height: 100%;
}

.react-multi-carousel-list {
  position: relative;
  overflow: hidden;
  width: 100%;
}

.react-multi-carousel-track {
  display: flex;
  align-items: center;
  width: 100%;
}

.react-multi-carousel-item {
  width: 100% !important;
  min-height: 600px;
}

/* 반응형 조정 */
@media (max-width: 768px) {
  .react-multi-carousel-item {
    min-height: 500px;
  }
}

@media (max-width: 480px) {
  .react-multi-carousel-item {
    min-height: 400px;
  }
}

/* 컴포넌트 내부 스크롤 최적화 */
.component-content {
  max-height: 70vh;
  overflow-y: auto;
  padding: 20px;
}

/* 로딩 애니메이션 */
.loading-spinner {
  display: flex;
  justify-content: center;
  align-items: center;
  height: 200px;
}

.spinner {
  width: 40px;
  height: 40px;
  border: 4px solid #f3f3f3;
  border-top: 4px solid #3498db;
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}"""

        with open(os.path.join(src_folder, "App.css"), 'w', encoding='utf-8') as f:
            f.write(app_css)

