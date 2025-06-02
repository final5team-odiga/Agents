import os
import asyncio
from typing import Dict, List
from agents.image_analyzer import ImageAnalyzerAgent
from agents.contents.content_creator import ContentCreatorV2Crew
from agents.Editor.template_manager import MultiAgentTemplateManager
from agents.jsxcreate.jsx_generator import JSXCreatorAgent
from utils.blob_storage import BlobStorageManager
from utils.file_manager import FileManager
from utils.agent_decision_logger import get_agent_logger
from utils.logging_manager import LoggingManager
from crewai import Crew

class SystemCoordinator:
    """시스템 전체 조율자 (분할된 기능 통합 - 비동기 에이전트 지원)"""

    def __init__(self):
        self.logger = get_agent_logger()
        self.logging_manager = LoggingManager()
        self.blob_manager = BlobStorageManager()
        self.file_manager = FileManager(output_folder=os.getenv("OUTPUT_FOLDER", "./output"))

        # 에이전트 초기화
        self.image_analyzer = ImageAnalyzerAgent()
        self.content_creator = ContentCreatorV2Crew()
        self.template_manager = MultiAgentTemplateManager()
        self.jsx_creator = JSXCreatorAgent()

    async def initialize_system(self) -> Dict:
        """시스템 초기화 (비동기)"""
        print("📊 시스템 초기화 중...")

        # 출력 폴더 생성
        os.makedirs(self.file_manager.output_folder, exist_ok=True)

        # 데이터 수집
        images = self.blob_manager.get_images()
        text_blobs = self.blob_manager.get_texts()
        texts = [self.blob_manager.read_text_file(text_blob) for text_blob in text_blobs]

        print(f"✅ 데이터 수집 완료: 이미지 {len(images)}개, 텍스트 {len(texts)}개")

        # PDF 벡터 시스템 초기화 (비동기)
        should_init = await self.template_manager.should_initialize_vector_system()
        if should_init:
            print("🔄 PDF 벡터 시스템 초기화...")
            await self.template_manager.initialize_vector_system("templates")

        # 시스템 초기화 로깅 (비동기)
        await self.logging_manager.log_system_initialization(len(images), len(texts))

        return {
            'images': images,
            'texts': texts
        }

    async def process_images(self, images: List[str]) -> List[Dict]:
        """이미지 분석 처리 (비동기)"""
        print("📸 이미지 분석 중...")
        crew = Crew(agents=[self.image_analyzer.create_agent()], verbose=False)

        try:
            if hasattr(self.image_analyzer, 'analyze_images_batch_async'):
                results = await self.image_analyzer.analyze_images_batch_async(images, max_concurrent=5)
            elif hasattr(self.image_analyzer, 'analyze_images_async'):
                results = await self.image_analyzer.analyze_single_image_async(images, max_concurrent=3)
            else:
                loop = asyncio.get_event_loop()
                results = await loop.run_in_executor(None, self.image_analyzer.analyze_images, images, crew)

            print(f"✅ 이미지 분석 완료: {len(results)}개 결과")

            # 결과 저장
            analysis_path = os.path.join(self.file_manager.output_folder, "image_analysis_results.json")
            self.file_manager.save_json(results, analysis_path)

            # 이미지 분석 로깅 (비동기)
            await self.logging_manager.log_image_analysis_completion(len(images), len(results))

            return results

        except Exception as e:
            print(f"⚠️ 이미지 분석 실패: {e}")
            return []

    async def create_content(self, texts: List[str], image_results: List[Dict]) -> str:
        """콘텐츠 생성 (비동기)"""
        print("📝 콘텐츠 생성 중...")
        
        # ContentCreatorV2Crew의 비동기 메서드 호출
        if hasattr(self.content_creator, 'execute_content_creation') and asyncio.iscoroutinefunction(self.content_creator.execute_content_creation):
            magazine_content = await self.content_creator.execute_content_creation(texts, image_results)
        else:
            # 동기 버전 폴백
            magazine_content = await asyncio.get_event_loop().run_in_executor(
                None, self.content_creator.execute_content_creation_sync, texts, image_results
            )

        print(f"✅ 콘텐츠 생성 완료: {len(magazine_content)}자")

        # 콘텐츠 저장
        content_path = os.path.join(self.file_manager.output_folder, "magazine_content.json")
        self.file_manager.save_magazine_content_json(magazine_content, content_path)

        # 콘텐츠 생성 로깅 (비동기)
        await self.logging_manager.log_content_creation_completion(len(texts), len(image_results), len(magazine_content))

        return magazine_content

    async def generate_template_data(self, magazine_content: str, image_results: List[Dict]) -> Dict:
        """템플릿 데이터 생성 (비동기)"""
        print("📋 템플릿 데이터 생성 중...")
        
        # MultiAgentTemplateManager의 비동기 메서드 호출
        template_data = await self.template_manager.create_magazine_data(magazine_content, image_results)

        print(f"✅ 템플릿 데이터 생성 완료: {len(template_data.get('content_sections', []))}개 섹션")

        # 템플릿 데이터 저장
        template_path = os.path.join(self.file_manager.output_folder, "template_data.json")
        self.file_manager.save_json(template_data, template_path)

        # 템플릿 데이터 생성 로깅 (비동기)
        await self.logging_manager.log_template_data_completion(template_data)

        return template_data

    async def generate_jsx_components(self, template_data: Dict) -> List[Dict]:
        """JSX 컴포넌트 생성 (비동기)"""
        print("⚛️ JSX 컴포넌트 생성 중...")

        # 프로젝트 폴더 생성
        project_name = "travel-magazine"
        project_folder = self.file_manager.create_project_folder(project_name)

        # template_data.json을 프로젝트 폴더에 저장
        template_data_path = os.path.join(project_folder, "template_data.json")
        self.file_manager.save_json(template_data, template_data_path)

        # JSX 컴포넌트 생성 (비동기)
        generated_components = await self.jsx_creator.generate_jsx_components_async(template_data_path)

        print(f"✅ JSX 컴포넌트 생성 완료: {len(generated_components)}개")

        # 컴포넌트 저장
        components_folder = os.path.join(project_folder, "src", "components")
        os.makedirs(components_folder, exist_ok=True)

        # save_jsx_components 비동기 호출 (필요시)
        if hasattr(self.jsx_creator, 'save_jsx_components') and asyncio.iscoroutinefunction(self.jsx_creator.save_jsx_components):
            saved_components = await self.jsx_creator.save_jsx_components(generated_components, components_folder)
        else:
            saved_components = await asyncio.get_event_loop().run_in_executor(
                None, self.jsx_creator.save_jsx_components, generated_components, components_folder
            )

        print(f"✅ 컴포넌트 저장 완료: {len(saved_components)}개")

        # JSX 생성 로깅 (비동기)
        await self.logging_manager.log_jsx_generation_completion(len(generated_components), len(saved_components))

        return {
            'generated_components': generated_components,
            'saved_components': saved_components,
            'project_folder': project_folder
        }

    async def create_react_app(self, components: Dict, template_data: Dict) -> str:
        """React 앱 생성 (비동기)"""
        print("🚀 React 앱 생성 중...")

        saved_components = components['saved_components']
        project_folder = components['project_folder']

        if saved_components:
            # React 앱 생성을 비동기로 처리
            await asyncio.get_event_loop().run_in_executor(
                None, self.file_manager.create_magazine_react_app, project_folder, saved_components, template_data
            )

            print(f"✅ React 앱 생성 완료")

            # React 앱 생성 로깅 (비동기)
            await self.logging_manager.log_react_app_completion(project_folder, len(saved_components))

            return project_folder
        else:
            print("⚠️ 저장된 컴포넌트가 없어 React 앱을 생성할 수 없습니다.")
            return ""

    def display_results(self, project_path: str, components: Dict):
        """결과 출력"""
        saved_components = components['saved_components']

        print("\n=== ✅ 매거진 생성 완료 ===")
        print(f"⚛️ JSX 컴포넌트: {len(saved_components)}개 생성")
        print(f"🚀 React 앱: {project_path}")

        if project_path:
            print(f"\n🎯 실행 방법:")
            print(f"1. cd {project_path}")
            print(f"2. npm install")
            print(f"3. npm start")
            print(f"4. http://localhost:3000 접속")

        # 로깅 통계
        self._display_logging_stats()

    def _display_logging_stats(self):
        """로깅 통계 출력"""
        all_outputs = self.logger.output_manager.get_all_outputs()
        agent_stats = {}

        for output in all_outputs:
            agent_name = output.get('agent_name', 'unknown')
            if agent_name not in agent_stats:
                agent_stats[agent_name] = 0
            agent_stats[agent_name] += 1

        print(f"\n📊 로깅 통계: 총 {len(all_outputs)}개 기록")
        for agent_name, count in agent_stats.items():
            print(f" - {agent_name}: {count}개")

    async def handle_error(self, error: Exception):
        """에러 처리 (비동기)"""
        import traceback

        # 간단한 에러 로깅만 (비동기)
        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.logger.log_agent_real_output(
                agent_name="SystemError",
                agent_role="에러 처리",
                task_description="시스템 실행 중 오류 발생",
                final_answer=f"ERROR: {str(error)}",
                reasoning_process="시스템 실행 중 예외 발생",
                performance_metrics={
                    "error_occurred": True,
                    "async_processing": True
                }
            )
        )

    # 동기 버전 메서드들 (호환성 유지)
    def handle_error_sync(self, error: Exception):
        """에러 처리 (동기 버전)"""
        return asyncio.run(self.handle_error(error))
