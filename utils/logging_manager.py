import asyncio
from typing import Dict
from utils.agent_decision_logger import get_agent_logger

class LoggingManager:
    """로깅 관리자 (핵심 로깅만 - 비동기 처리 추가)"""

    def __init__(self):
        self.logger = get_agent_logger()

    async def log_system_initialization(self, images_count: int, texts_count: int):
        """시스템 초기화 로깅 (비동기)"""
        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.logger.log_agent_real_output(
                agent_name="SystemInitializer",
                agent_role="시스템 초기화",
                task_description="시스템 초기화 및 데이터 로드",
                final_answer=f"시스템 초기화 완료: 이미지 {images_count}개, 텍스트 {texts_count}개",
                reasoning_process="시스템 초기화 및 데이터 로드 완료",
                performance_metrics={
                    "images_count": images_count,
                    "texts_count": texts_count,
                    "initialization_success": True,
                    "async_processing": True
                }
            )
        )

    async def log_image_analysis_completion(self, images_count: int, results_count: int):
        """이미지 분석 완료 로깅 (비동기)"""
        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.logger.log_agent_real_output(
                agent_name="ImageAnalysisCompletion",
                agent_role="이미지 분석 완료",
                task_description=f"{images_count}개 이미지 분석 완료",
                final_answer=f"이미지 분석 완료: {results_count}개 결과 생성",
                reasoning_process="이미지 분석 에이전트 실행 완료",
                performance_metrics={
                    "images_processed": images_count,
                    "results_generated": results_count,
                    "success_rate": results_count / images_count if images_count > 0 else 0,
                    "async_processing": True
                }
            )
        )

    async def log_content_creation_completion(self, texts_count: int, images_count: int, content_length: int):
        """콘텐츠 생성 완료 로깅 (비동기)"""
        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.logger.log_agent_real_output(
                agent_name="ContentCreationCompletion",
                agent_role="콘텐츠 생성 완료",
                task_description=f"{texts_count}개 텍스트와 {images_count}개 이미지로 콘텐츠 생성",
                final_answer=f"콘텐츠 생성 완료: {content_length}자",
                reasoning_process="콘텐츠 생성 에이전트 실행 완료",
                performance_metrics={
                    "source_texts": texts_count,
                    "source_images": images_count,
                    "content_length": content_length,
                    "content_richness": content_length / texts_count if texts_count > 0 else 0,
                    "async_processing": True
                }
            )
        )

    async def log_template_data_completion(self, template_data: Dict):
        """템플릿 데이터 생성 완료 로깅 (비동기)"""
        sections_count = len(template_data.get('content_sections', []))
        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.logger.log_agent_real_output(
                agent_name="TemplateDataCompletion",
                agent_role="템플릿 데이터 생성 완료",
                task_description="템플릿 데이터 생성 완료",
                final_answer=f"템플릿 데이터 생성 완료: {sections_count}개 섹션",
                reasoning_process="에디터 에이전트들 협업으로 템플릿 데이터 생성",
                performance_metrics={
                    "sections_count": sections_count,
                    "vector_enhanced": template_data.get("vector_enhanced", False),
                    "integration_quality": template_data.get('integration_metadata', {}).get('integration_quality_score', 0),
                    "async_processing": True
                }
            )
        )

    async def log_jsx_generation_completion(self, generated_count: int, saved_count: int):
        """JSX 생성 완료 로깅 (비동기)"""
        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.logger.log_agent_real_output(
                agent_name="JSXGenerationCompletion",
                agent_role="JSX 생성 완료",
                task_description=f"{generated_count}개 JSX 컴포넌트 생성",
                final_answer=f"JSX 생성 완료: {saved_count}/{generated_count}개 저장 성공",
                reasoning_process="JSX 에이전트들이 컴포넌트 생성 완료",
                performance_metrics={
                    "components_generated": generated_count,
                    "components_saved": saved_count,
                    "save_success_rate": saved_count / generated_count if generated_count > 0 else 0,
                    "async_processing": True
                }
            )
        )

    async def log_react_app_completion(self, project_path: str, components_count: int):
        """React 앱 생성 완료 로깅 (비동기)"""
        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.logger.log_agent_real_output(
                agent_name="ReactAppCompletion",
                agent_role="React 앱 생성 완료",
                task_description="React 앱 생성 완료",
                final_answer=f"React 앱 생성 완료: {project_path}",
                reasoning_process="매거진 스타일 React 앱 생성 완료",
                performance_metrics={
                    "project_path": project_path,
                    "components_integrated": components_count,
                    "app_creation_success": True,
                    "async_processing": True
                }
            )
        )

    # 동기 버전 메서드들 (호환성 유지)
    def log_system_initialization_sync(self, images_count: int, texts_count: int):
        """시스템 초기화 로깅 (동기 버전)"""
        return asyncio.run(self.log_system_initialization(images_count, texts_count))

    def log_image_analysis_completion_sync(self, images_count: int, results_count: int):
        """이미지 분석 완료 로깅 (동기 버전)"""
        return asyncio.run(self.log_image_analysis_completion(images_count, results_count))

    def log_content_creation_completion_sync(self, texts_count: int, images_count: int, content_length: int):
        """콘텐츠 생성 완료 로깅 (동기 버전)"""
        return asyncio.run(self.log_content_creation_completion(texts_count, images_count, content_length))

    def log_template_data_completion_sync(self, template_data: Dict):
        """템플릿 데이터 생성 완료 로깅 (동기 버전)"""
        return asyncio.run(self.log_template_data_completion(template_data))

    def log_jsx_generation_completion_sync(self, generated_count: int, saved_count: int):
        """JSX 생성 완료 로깅 (동기 버전)"""
        return asyncio.run(self.log_jsx_generation_completion(generated_count, saved_count))

    def log_react_app_completion_sync(self, project_path: str, components_count: int):
        """React 앱 생성 완료 로깅 (동기 버전)"""
        return asyncio.run(self.log_react_app_completion(project_path, components_count))
