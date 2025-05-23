import os
from dotenv import load_dotenv
from tools import ImageTools
from agents.image_analyzer import ImageAnalyzerAgent
from agents.content_creator import ContentCreatorAgent
from agents.template_manager import TemplateManagerAgent
from agents.jsx_creator import JSXCreatorAgent 
from utils.blob_storage import BlobStorageManager
from utils.file_manager import FileManager
from crewai import Crew

# 환경 변수 로드
load_dotenv()

def main():
    print("=== CrewAI 여행 매거진 자동 생성 시스템 ===")
    
    try:
        # Blob Storage 매니저 초기화
        blob_manager = BlobStorageManager()
        
        # 파일 매니저 초기화
        file_manager = FileManager(output_folder=os.getenv("OUTPUT_FOLDER", "./output"))
        
        # 이미지 목록 가져오기
        images = blob_manager.get_images()
        if not images:
            print("이미지를 찾을 수 없습니다.")
            return
        
        print(f"총 {len(images)}개의 이미지를 찾았습니다.")
        
        # 텍스트 목록 가져오기
        text_blobs = blob_manager.get_texts()
        if not text_blobs:
            print("텍스트 파일을 찾을 수 없습니다.")
            return
        
        print(f"총 {len(text_blobs)}개의 텍스트 파일을 찾았습니다.")
        
        # 텍스트 내용 읽기
        texts = [blob_manager.read_text_file(text_blob) for text_blob in text_blobs]
        
        # 에이전트 초기화
        image_analyzer = ImageAnalyzerAgent()
        content_creator = ContentCreatorAgent()
        template_manager = TemplateManagerAgent()
        jsx_creator = JSXCreatorAgent()  # JSX 생성 에이전트 별도 초기화
        
        # Crew 생성 (template_manager까지만)
        crew = Crew(
            agents=[
                image_analyzer.create_agent(),
                content_creator.create_agent(),
                template_manager.create_agent()
            ],
            verbose=True
        )
        
        # 1. 이미지 분석
        print("\n=== 이미지 분석 시작 ===")
        image_analysis_results = image_analyzer.analyze_images(images, crew)
        
        # 분석 결과 저장
        analysis_path = os.path.join(file_manager.output_folder, "image_analysis_results.json")
        file_manager.save_json(image_analysis_results, analysis_path)
        print(f"이미지 분석 결과가 {analysis_path}에 저장되었습니다.")
        
        # 2. 매거진 콘텐츠 생성
        print("\n=== 매거진 콘텐츠 생성 시작 ===")
        magazine_content = content_creator.create_magazine_content(texts, image_analysis_results)
        
        # 콘텐츠 저장
        content_path = os.path.join(file_manager.output_folder, "magazine_content.txt")
        file_manager.save_content(str(magazine_content), content_path)
        print(f"매거진 콘텐츠가 {content_path}에 저장되었습니다.")
        
        # 3. 템플릿 선택 및 적용
        print("\n=== 템플릿 선택 및 적용 시작 ===")
        template_data = template_manager.select_and_apply_templates(str(magazine_content), image_analysis_results)
        
        # 템플릿 데이터 저장
        template_path = os.path.join(file_manager.output_folder, "template_data.json")
        file_manager.save_json(template_data, template_path)
        print(f"템플릿 데이터가 {template_path}에 저장되었습니다.")
        
        # 4. React 앱 기본 구조 생성
        print("\n=== React 앱 기본 구조 생성 시작 ===")
        project_name = "travel-magazine"
        project_folder, template_data_path = template_manager.generate_react_app(template_data, file_manager, project_name)
        print(f"React 앱 기본 구조가 {project_folder}에 생성되었습니다.")
        
        # 5. JSX 생성 (별도 에이전트)
        print("\n=== JSX 생성 에이전트 시작 ===")
        generated_components_data = jsx_creator.generate_jsx_components(template_data_path)
        
        # 6. JSX 파일 저장
        components_folder = os.path.join(project_folder, "src", "components")
        saved_components = jsx_creator.save_jsx_components(generated_components_data, components_folder)
        
        # 7. App.js 생성
        if saved_components:
            app_js = generate_app_js(saved_components)  # 별도 함수로 분리
            app_js_path = os.path.join(project_folder, "src", "App.js")
            file_manager.save_content(app_js, app_js_path)
            print(f"App.js 생성 완료 - {len(saved_components)}개 컴포넌트 포함")
        
        # 8. 실행 방법 안내
        print("\n=== 매거진 실행 방법 ===")
        print(f"1. 터미널에서 다음 명령어를 실행하세요:")
        print(f"   cd {project_folder}")
        print(f"   npm install")
        print(f"   npm start")
        print(f"2. 웹 브라우저에서 http://localhost:3000 으로 접속하세요.")
        
    except Exception as e:
        print(f"오류 발생: {str(e)}")
        import traceback
        print(traceback.format_exc())

def generate_app_js(saved_components):
    """App.js 생성 (별도 함수)"""
    imports = []
    components = []
    
    for component in saved_components:
        component_name = component['name']
        file_name = component['file']
        
        imports.append(f"import {{ {component_name} }} from './components/{file_name}';")
        components.append(f"        <{component_name} />")
    
    app_js = f'''import React from 'react';
import './App.css';
{chr(10).join(imports)}

function App() {{
  return (
    <div className="App">
      <div className="magazine-container">
        <header className="magazine-header">
          <h1>✈️ 여행 매거진</h1>
          <p>특별한 여행의 순간들을 담아낸 이야기</p>
        </header>
{chr(10).join(components)}
        <footer className="magazine-footer">
          <p>이 매거진은 CrewAI로 자동 생성되었습니다.</p>
        </footer>
      </div>
    </div>
  );
}}

export default App;
'''
    return app_js

if __name__ == "__main__":
    main()
