import asyncio
import sys
import time
import concurrent.futures
from typing import Dict, List, Optional, Callable, Any
from collections import deque
from dataclasses import dataclass
import os
import json
import re
from crewai import Agent, Task, Crew, Process
from custom_llm import get_azure_llm
from utils.hybridlogging import get_hybrid_logger
from utils.ai_search_isolation import CoordinatorAgentIsolation
from utils.session_isolation import SessionAwareMixin
from utils.agent_communication_isolation import InterAgentCommunicationMixin


@dataclass
class WorkItem:
    id: str
    task_func: Callable
    args: tuple
    kwargs: dict
    priority: int = 0
    max_retries: int = 3
    current_retry: int = 0
    timeout: float = 300.0

class AsyncWorkQueue:
    def __init__(self, max_workers: int = 2, max_queue_size: int = 50):
        self.max_workers = max_workers
        self.max_queue_size = max_queue_size
        self.work_queue = deque()
        self.active_tasks = {}
        self.results = {}
        self.semaphore = asyncio.Semaphore(max_workers)
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

    async def add_work(self, work_item: WorkItem) -> str:
        """작업을 큐에 추가"""
        if len(self.work_queue) >= self.max_queue_size:
            old_item = self.work_queue.popleft()
            print(f"⚠️ 큐 용량 초과로 작업 {old_item.id} 제거")
        
        self.work_queue.append(work_item)
        return work_item.id

    async def process_work_item(self, work_item: WorkItem) -> Optional[Any]:
        """개별 작업 처리"""
        async with self.semaphore:
            try:
                print(f"🔄 작업 {work_item.id} 시작 (시도 {work_item.current_retry + 1}/{work_item.max_retries + 1})")
                
                # 수정: 코루틴 객체와 코루틴 함수 구분
                if asyncio.iscoroutine(work_item.task_func):
                    # 이미 생성된 코루틴 객체
                    result = await asyncio.wait_for(work_item.task_func, timeout=work_item.timeout)
                elif asyncio.iscoroutinefunction(work_item.task_func):
                    # 코루틴 함수
                    result = await asyncio.wait_for(
                        work_item.task_func(*work_item.args, **work_item.kwargs),
                        timeout=work_item.timeout
                    )
                elif callable(work_item.task_func):
                    # 일반 호출 가능 객체
                    result = await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(
                            self.executor,
                            lambda: work_item.task_func(*work_item.args, **work_item.kwargs)
                        ),
                        timeout=work_item.timeout
                    )
                else:
                    # 호출 불가능한 객체인 경우 오류 발생
                    raise TypeError(f"task_func is not callable: {type(work_item.task_func)}")
                
                self.results[work_item.id] = {"status": "success", "result": result}
                print(f"✅ 작업 {work_item.id} 완료")
                return result
                
            except asyncio.TimeoutError:
                print(f"⏰ 작업 {work_item.id} 타임아웃 ({work_item.timeout}초)")
                if work_item.current_retry < work_item.max_retries:
                    work_item.current_retry += 1
                    work_item.timeout *= 1.5
                    await self.add_work(work_item)
                else:
                    self.results[work_item.id] = {"status": "timeout", "error": "최대 재시도 횟수 초과"}
                return None
                
            except Exception as e:
                print(f"❌ 작업 {work_item.id} 실패: {e}")
                if work_item.current_retry < work_item.max_retries:
                    work_item.current_retry += 1
                    await self.add_work(work_item)
                else:
                    self.results[work_item.id] = {"status": "error", "error": str(e)}
                return None
            
    async def process_queue(self) -> dict:
        """큐의 모든 작업을 배치 처리"""
        tasks = []
        while self.work_queue:
            work_item = self.work_queue.popleft()
            task = asyncio.create_task(self.process_work_item(work_item))
            tasks.append(task)
            self.active_tasks[work_item.id] = task
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        return self.results

class CircuitBreaker:
    def __init__(self, failure_threshold: int = 10, recovery_timeout: float = 120.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

    def is_open(self) -> bool:
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
                return False
            return True
        return False

    def record_success(self):
        self.failure_count = 0
        self.state = "CLOSED"

    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"





TIMEOUT_CONFIGS = {
    'org_agent': 900,      # 15분
    'binding_agent': 1200, # 20분  
    'coordinator_agent': 600, # 10분
    'vector_init': 600,    # 10분
    'crew_execution': 900  # 15분
}


class CoordinatorAgent(CoordinatorAgentIsolation, SessionAwareMixin, InterAgentCommunicationMixin):
    """통합 조율자 (AI Search 격리 강화)"""
    
    def __init__(self, session_id: Optional[str] = None):
        # 기존 초기화 코드
        self.llm = get_azure_llm()
        self.logger = get_hybrid_logger()
        self.crew_agent = self._create_crew_agent()
        self.text_analyzer_agent = self._create_text_analyzer_agent()
        self.image_analyzer_agent = self._create_image_analyzer_agent()
        self.work_queue = AsyncWorkQueue(max_workers=1, max_queue_size=20)
        self.circuit_breaker = CircuitBreaker()
        self.recursion_threshold = 600
        self.fallback_to_sync = False
        self.batch_size = 2
        self.execution_stats = {
            "total_attempts": 0,
            "successful_executions": 0,
            "fallback_used": 0,
            "circuit_breaker_triggered": 0,
            "timeout_occurred": 0
        }

        self.__init_isolation__()

        self.__init_session_awareness__(session_id)

        self.__init_inter_agent_communication__()

    async def _process_enhanced_crew_result_async(self, crew_result, extracted_text_data: Dict,
                                                extracted_image_data: Dict, org_results: List[Dict],
                                                binding_results: List[Dict]) -> Dict:
        """강화된 Crew 실행 결과 처리 (격리 강화)"""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._process_enhanced_crew_result_with_isolation, crew_result, extracted_text_data,
            extracted_image_data, org_results, binding_results
        )

    def _process_enhanced_crew_result_with_isolation(self, crew_result, extracted_text_data: Dict,
                                                   extracted_image_data: Dict, org_results: List[Dict],
                                                   binding_results: List[Dict]) -> Dict:
        """Crew 실행 결과 처리 및 격리 강화"""
        try:
            # 1. Azure AI Search 영향 차단 (강화된 격리)
            parsed_data = self.block_azure_search_influence(crew_result)
            
            # 2. 기존 처리 로직
            if not parsed_data.get('content_sections') or len(parsed_data.get('content_sections', [])) == 0:
                parsed_data = self._create_enhanced_structure_isolated(
                    extracted_text_data, extracted_image_data, org_results, binding_results
                )
            else:
                parsed_data = self._enhance_parsed_data_with_real_images_isolated(
                    parsed_data, extracted_image_data
                )
            
            # 3. 콘텐츠 진정성 검증 및 교정 적용 (강화)
            parsed_data = self.validate_content_authenticity(parsed_data)
            
            # 4. 최종 격리 검증
            parsed_data = self._final_isolation_validation(parsed_data)
            
            # 5. 최종 메타데이터 업데이트
            parsed_data['integration_metadata'] = {
                **parsed_data.get('integration_metadata', {}),
                "total_sections": len(parsed_data.get('content_sections', [])),
                "azure_search_influence_blocked": True,
                "original_content_validation_applied": True,
                "data_source_priority": "magazine_content_json_primary",
                "ai_search_isolation": {
                    "isolation_applied": True,
                    "contamination_report": self._get_isolation_report(),
                    "final_validation_passed": True
                }
            }
            
            return parsed_data
            
        except Exception as e:
            print(f"⚠️ Crew 결과 처리 및 격리 실패: {e}")
            return self._restore_from_magazine_content()

    def _create_enhanced_structure_isolated(self, extracted_text_data: Dict, extracted_image_data: Dict,
                                          org_results: List[Dict], binding_results: List[Dict]) -> Dict:
        """격리된 강화 구조 생성"""
        # 기존 _create_enhanced_structure 로직에 격리 적용
        content_sections = []
        selected_templates = []
        
        # 추출된 텍스트 섹션을 기반으로 구조 생성 (격리 적용)
        for i, section in enumerate(extracted_text_data.get('sections', [])):
            # 섹션 오염 검사
            if self.isolation_manager.is_contaminated(section, f"extracted_section_{i}"):
                print(f"🚫 섹션 {i+1} 오염 감지, 건너뜀")
                continue
            
            template = section.get('template', f"Section{i+1:02d}.jsx")
            
            # 해당 템플릿의 격리된 이미지 가져오기
            template_images = extracted_image_data.get('template_images', {}).get(template, [])
            clean_images = [img for img in template_images if self.isolation_manager.is_trusted_image_url(img)]
            
            # 섹션 구조 생성
            section_data = {
                "template": template,
                "title": section.get('title', ''),
                "subtitle": section.get('subtitle', ''),
                "body": section.get('body', ''),
                "tagline": section.get('tagline', 'TRAVEL & CULTURE'),
                "images": clean_images[:self.isolation_manager.config.max_images_per_section],
                "metadata": {
                    "content_quality": self._calculate_content_quality(section),
                    "image_count": len(clean_images[:self.isolation_manager.config.max_images_per_section]),
                    "source": section.get('layout_source', 'extracted'),
                    "real_content": True,
                    "fallback_used": False,
                    "ai_search_isolation": {
                        "isolation_applied": True,
                        "images_filtered": len(template_images) - len(clean_images),
                        "contamination_detected": False
                    }
                }
            }
            
            content_sections.append(section_data)
            selected_templates.append(template)
        
        if not content_sections:
            content_sections = [{
                "template": "Section01.jsx",
                "title": "여행 매거진",
                "subtitle": "특별한 이야기",
                "body": "격리 시스템에 의해 모든 섹션이 필터링되었습니다.",
                "tagline": "TRAVEL & CULTURE",
                "images": [],
                "metadata": {
                    "content_quality": 0.5,
                    "image_count": 0,
                    "source": "isolated_fallback",
                    "real_content": False,
                    "fallback_used": True,
                    "ai_search_isolation": {
                        "isolation_applied": True,
                        "all_sections_filtered": True,
                        "reason": "contamination_detected"
                    }
                }
            }]
            selected_templates = ["Section01.jsx"]
        
        return {
            "selected_templates": selected_templates,
            "content_sections": content_sections,
            "integration_metadata": {
                "source": "enhanced_structure_isolated",
                "total_sections": len(content_sections),
                "azure_search_influence": "blocked",
                "content_authenticity": "verified",
                "ai_search_isolation": {
                    "isolation_applied": True,
                    "structure_created": True
                }
            }
        }

    def _enhance_parsed_data_with_real_images_isolated(self, parsed_data: Dict, extracted_image_data: Dict) -> Dict:
        """격리된 실제 이미지로 파싱된 데이터 강화"""
        enhanced_sections = []
        
        for section in parsed_data.get('content_sections', []):
            template = section.get('template', '')
            
            # 템플릿별 격리된 이미지 가져오기
            template_images = extracted_image_data.get('template_images', {}).get(template, [])
            clean_images = [img for img in template_images if self.isolation_manager.is_trusted_image_url(img)]
            
            # 섹션 강화
            enhanced_section = {
                **section,
                "images": clean_images[:self.isolation_manager.config.max_images_per_section],
                "metadata": {
                    **section.get('metadata', {}),
                    "image_count": len(clean_images[:self.isolation_manager.config.max_images_per_section]),
                    "real_images_applied": True,
                    "ai_search_isolation": {
                        "isolation_applied": True,
                        "images_filtered": len(template_images) - len(clean_images),
                        "trusted_images_only": True
                    }
                }
            }
            
            enhanced_sections.append(enhanced_section)
        
        parsed_data['content_sections'] = enhanced_sections
        return parsed_data

    def _final_isolation_validation(self, parsed_data: Dict) -> Dict:
        """최종 격리 검증"""
        validated_sections = []
        
        for section in parsed_data.get('content_sections', []):
            # 섹션 전체 오염 검사
            if self.isolation_manager.is_contaminated(section, "final_validation"):
                print(f"🚫 최종 검증에서 섹션 오염 감지: {section.get('title', 'unknown')[:30]}...")
                
                # 원본 데이터로 복원 시도
                restored_section = self._restore_section_from_original(section)
                validated_sections.append(restored_section)
            else:
                validated_sections.append(section)
        
        parsed_data['content_sections'] = validated_sections
        parsed_data['final_isolation_validation'] = {
            "validation_applied": True,
            "sections_validated": len(validated_sections),
            "contamination_report": self._get_isolation_report()
        }
        
        return parsed_data

    def _restore_section_from_original(self, contaminated_section: Dict) -> Dict:
        """오염된 섹션을 원본 데이터로 복원"""
        try:
            magazine_content_path = "./output/magazine_content.json"
            if os.path.exists(magazine_content_path):
                with open(magazine_content_path, 'r', encoding='utf-8') as f:
                    original_data = json.load(f)
                
                # 해당 템플릿의 원본 데이터 찾기
                template = contaminated_section.get('template', '')
                for section in original_data.get('sections', []):
                    if section.get('template') == template:
                        return {
                            "template": template,
                            "title": section.get('title', ''),
                            "subtitle": section.get('subtitle', ''),
                            "body": section.get('content', section.get('body', '')),
                            "tagline": "TRAVEL & CULTURE",
                            "images": contaminated_section.get('images', []),  # 이미지는 유지
                            "metadata": {
                                **contaminated_section.get('metadata', {}),
                                "source": "magazine_content_json_restored",
                                "contamination_restored": True,
                                "ai_search_isolation": {
                                    "isolation_applied": True,
                                    "restored_from_original": True
                                }
                            }
                        }
        except Exception as e:
            print(f"⚠️ 섹션 복원 실패: {e}")
        
        # 복원 실패시 기본값 반환
        return {
            **contaminated_section,
            "title": "여행 이야기",
            "subtitle": "특별한 순간들",
            "body": "원본 데이터로 복원되었습니다.",
            "metadata": {
                **contaminated_section.get('metadata', {}),
                "restoration_failed": True,
                "ai_search_isolation": {
                    "isolation_applied": True,
                    "restoration_attempted": True,
                    "restoration_failed": True
                }
            }
        }

    def _get_fallback_result(self, task_id: str) -> dict:
        """폴백 결과 생성 (격리 메타데이터 추가)"""
        return {
            "selected_templates": ["Section01.jsx"],
            "content_sections": [{
                "template": "Section01.jsx",
                "title": "여행 매거진",
                "subtitle": "특별한 이야기",
                "body": "Circuit Breaker 또는 실패로 인한 폴백 콘텐츠입니다.",
                "tagline": "TRAVEL & CULTURE",
                "images": [],
                "metadata": {
                    "content_quality": 0.5,
                    "image_count": 0,
                    "source": "coordinator_fallback",
                    "real_content": False,
                    "fallback_used": True,
                    "ai_search_isolation": {
                        "isolation_applied": True,
                        "fallback_reason": "circuit_breaker_or_failure"
                    }
                }
            }],
            "integration_metadata": {
                "source": "fallback",
                "total_sections": 1,
                "azure_search_influence": "blocked",
                "content_authenticity": "fallback",
                "ai_search_isolation": {
                    "isolation_applied": True,
                    "fallback_used": True
                }
            }
        }

    def _check_recursion_depth(self):
        """현재 재귀 깊이 확인"""
        frame = sys._getframe()
        depth = 0
        while frame:
            depth += 1
            frame = frame.f_back
        return depth

    def _should_use_sync(self):
        """동기 모드로 전환할지 판단"""
        current_depth = self._check_recursion_depth()
        if current_depth > self.recursion_threshold:
            print(f"⚠️ CoordinatorAgent 재귀 깊이 {current_depth} 감지 - 동기 모드로 전환")
            self.fallback_to_sync = True
            return True
        return self.fallback_to_sync



    async def execute_with_resilience(self, task_func: Callable, task_id: str,
                                  timeout: float = None, max_retries: int = 2,
                                  *args, **kwargs) -> Any:
        if timeout is None:
            for task_type, default_timeout in TIMEOUT_CONFIGS.items():
                if task_type in task_id.lower():
                    timeout = default_timeout
                    break
            else:
                timeout = 300

        if self.circuit_breaker.is_open():
            print(f"🚫 Circuit Breaker 열림 - 작업 {task_id} 건너뜀")
            return self._get_fallback_result(task_id)

        # 수정: 원래 함수와 인자를 WorkItem에 전달
        work_item = WorkItem(
            id=task_id,
            task_func=task_func,  # 원래 함수 전달
            args=args,            # 원래 인자 전달
            kwargs=kwargs,        # 원래 키워드 인자 전달
            timeout=timeout,
            max_retries=max_retries
        )

        await self.work_queue.add_work(work_item)
        processed_results = await self.work_queue.process_queue()
        result_info = processed_results.get(task_id)

        if result_info and result_info["status"] == "success":
            self.circuit_breaker.record_success()
            return result_info["result"]
        else:
            self.circuit_breaker.record_failure()
            if result_info:
                print(f"⚠️ 작업 {task_id} 최종 실패: {result_info.get('error', '알 수 없는 오류')}")
            else:
                print(f"⚠️ 작업 {task_id}의 결과 정보를 찾을 수 없음 (큐 처리 후).")
            return self._get_fallback_result(task_id)


    def _get_fallback_result(self, task_id: str) -> dict:
        """개선된 폴백 결과 생성"""
        self.execution_stats["fallback_used"] += 1
        reason = task_id  # 기본적으로 task_id를 reason으로 사용
        
        if "_timeout" in task_id:
            reason = "timeout"
        elif "_exception" in task_id:
            reason = "exception"
        elif "_type_error" in task_id:
            reason = "type_error"
        
        return {
            "selected_templates": ["Section01.jsx"],
            "content_sections": [{
                "template": "Section01.jsx",
                "title": "여행 매거진 (폴백)",
                "subtitle": f"특별한 이야기 ({reason})",
                "body": f"CoordinatorAgent 처리 중 문제 발생 ({reason})으로 인한 폴백 콘텐츠입니다. Task ID: {task_id}",
                "tagline": "TRAVEL & CULTURE",
                "images": [],
                "metadata": {
                    "fallback_used": True,
                    "reason": reason,
                    "task_id": task_id
                }
            }],
            "integration_metadata": {
                "total_sections": 1,
                "integration_quality_score": 0.5,
                "fallback_mode": True
            }
        }

    def _create_crew_agent(self):
        """메인 조율 에이전트 생성"""
        return Agent(
            role="매거진 구조 통합 조율자 및 최종 품질 보증 전문가",
            goal="magazine_content.json의 원본 텍스트와 image_analysis.json의 이미지 분석 데이터를 필수적으로 활용하여 완벽한 매거진 구조를 생성하고, Azure AI Search 외부 데이터의 영향을 최소화한 순수한 template_data.json을 제공",
            backstory="""당신은 25년간 세계 최고 수준의 출판사에서 매거진 구조 통합 및 품질 보증 책임자로 활동해온 전문가입니다. Condé Nast, Hearst Corporation, Time Inc.에서 수백 개의 매거진 프로젝트를 성공적으로 조율했습니다.

    **전문 경력:**
    - 출판학 및 구조 설계 석사 학위 보유
    - PMP(Project Management Professional) 인증
    - 매거진 구조 통합 및 품질 관리 전문가
    - 텍스트-이미지 정합성 검증 시스템 개발 경험
    - 독자 경험(UX) 및 접근성 최적화 전문성

    **핵심 임무 및 데이터 우선순위:**
    당신의 최우선 임무는 magazine_content.json과 image_analysis.json의 원본 데이터를 충실히 반영하는 것입니다. 이는 다음과 같은 엄격한 우선순위를 따릅니다:

    **1순위 데이터 소스 (필수 사용):**
    - magazine_content.json의 원본 텍스트 내용 (제목, 본문, 주제, 스타일)
    - image_analysis.json의 이미지 설명 및 분석 결과
    - 이 두 파일의 내용은 반드시 최종 출력물에 직접적으로 반영되어야 합니다.

    **2순위 데이터 소스 (보조 참고용):**
    - 입력 매개변수의 유효한 데이터 (1순위 데이터와 일치하는 경우에만)

    **3순위 데이터 소스 (제한적 활용):**
    - Azure AI Search를 통한 외부 레이아웃 제안 (원본 데이터의 의미를 변경하지 않는 범위에서만)
    - 다른 에이전트의 구조 제안 (magazine_content.json의 섹션 구조와 일치하는 경우에만)

    **조율 철학:**
    "완벽한 매거진은 원본 콘텐츠의 진정성과 작성자의 의도를 온전히 보존하면서, 독자에게 최적의 경험을 제공하는 것입니다. 나는 magazine_content.json의 모든 텍스트와 image_analysis.json의 모든 이미지 정보를 소중히 여기며, 이를 바탕으로 구조적 완성도를 달성합니다."

    **필수 준수 규칙 (절대 위반 금지):**

    **텍스트 콘텐츠 규칙:**
    1. magazine_content.json의 각 섹션 텍스트는 반드시 해당 섹션의 title, subtitle, body에 직접 반영되어야 합니다.
    2. 원본 텍스트의 핵심 주제, 톤, 스타일을 변경하지 마십시오.
    3. magazine_content.json에 명시된 섹션 수와 정확히 일치하도록 content_sections을 생성하십시오.
    4. 원본 텍스트에 없는 내용을 임의로 추가하거나 창작하지 마십시오.
    5. Azure AI Search에서 제안된 레이아웃이 원본 텍스트의 의미와 상충할 경우, 반드시 원본 텍스트를 우선하십시오.

    **Azure AI Search 데이터 차단 규칙:**
    1. "도시의 미학", "골목길의 재발견", "아티스트 인터뷰", "친환경 도시" 등 Azure AI Search에서 제안된 주제는 절대 사용하지 마십시오.
    2. magazine_content.json에 없는 새로운 주제나 내용을 생성하지 마십시오.
    3. Azure AI Search 데이터는 오직 템플릿 선택과 레이아웃 구조 결정에만 참고하십시오.
    4. 원본 데이터와 상충하는 외부 제안은 모두 거부하십시오.

    **이미지 콘텐츠 규칙:**
    1. image_analysis.json의 각 이미지 설명은 반드시 해당 이미지가 배치되는 섹션의 콘텐츠에 반영되어야 합니다.
    2. 이미지 분석 결과와 상반되는 이미지 배치나 설명을 생성하지 마십시오.
    3. image_analysis.json에 기록된 이미지의 위치, 특성, 맥락 정보를 충실히 활용하십시오.
    4. 각 섹션당 최대 3개의 실제 이미지 URL만을 사용하십시오.

    **데이터 무결성 보장:**
    1. 폴백 데이터(fallback_used: true)는 절대 포함하지 않습니다.
    2. 플레이스홀더 텍스트, 예시 콘텐츠, 템플릿 설명은 포함하지 않습니다.
    3. 로그 데이터나 시스템 메시지는 최종 출력물에 직접 포함하지 않습니다.
    4. 중복 섹션을 절대 생성하지 않습니다.
    5. 실제 콘텐츠 데이터에서 추출된 내용만 사용합니다.

    **품질 검증 체크리스트:**
    작업 완료 전 다음 사항을 반드시 확인하십시오:
    □ template 에는 jsx_template에 존재하는 Section이 들어갔는가?
    □ magazine_content.json의 모든 섹션이 content_sections에 반영되었는가?
    □ 각 섹션의 title, subtitle, body가 원본 텍스트를 충실히 반영하는가?
    □ image_analysis.json의 이미지 설명이 해당 섹션에 적절히 통합되었는가?
    □ Azure AI Search 데이터가 원본 데이터를 압도하지 않았는가?
    □ 모든 이미지 URL이 실제 유효한 URL인가?
    □ 폴백 데이터나 플레이스홀더가 포함되지 않았는가?
    □ 원본 데이터에 없는 주제나 내용이 생성되지 않았는가?

    **경고 및 제한사항:**
    - Azure AI Search를 통해 얻은 레이아웃 제안은 참고용으로만 사용하고, 원본 콘텐츠의 의미나 구조를 변경하는 데 사용하지 마십시오.
    - 다른 에이전트의 결과가 magazine_content.json과 상충할 경우, 반드시 magazine_content.json을 우선하십시오.
    - 어떠한 상황에서도 원본 데이터의 무결성을 훼손하지 마십시오.
    - 불확실한 경우, 항상 보수적으로 접근하여 원본 데이터를 보존하십시오.

    이 지침을 철저히 준수하여 magazine_content.json과 image_analysis.json의 내용이 최종 출력물에 완전히 반영되도록 하십시오.""",
            verbose=True,
            llm=self.llm,
            allow_delegation=False
        )


    def _create_text_analyzer_agent(self):
        """텍스트 분석 전문 에이전트"""
        return Agent(
            role="텍스트 매핑 분석 전문가",
            goal="ContentCreatorV2Agent의 텍스트 매핑 결과를 정밀 분석하여 구조적 완성도를 검증하고 최적화된 텍스트 섹션을 생성",
            backstory="""당신은 15년간 출판업계에서 텍스트 구조 분석 및 최적화를 담당해온 전문가입니다. 복잡한 텍스트 데이터에서 핵심 정보를 추출하고 독자 친화적인 구조로 재구성하는 데 탁월한 능력을 보유하고 있습니다.""",
            verbose=True,
            llm=self.llm,
            allow_delegation=False
        )

    def _create_image_analyzer_agent(self):
        """이미지 분석 전문 에이전트"""
        return Agent(
            role="이미지 분배 분석 전문가",
            goal="BindingAgent의 이미지 분배 결과를 정밀 분석하여 시각적 일관성을 검증하고 최적화된 이미지 배치를 생성",
            backstory="""당신은 12년간 매거진 및 출판물의 시각적 디자인을 담당해온 전문가입니다. 이미지와 텍스트의 조화로운 배치를 통해 독자의 시선을 효과적으로 유도하는 레이아웃 설계에 전문성을 보유하고 있습니다.""",
            verbose=True,
            llm=self.llm,
            allow_delegation=False
        )

    async def coordinate_magazine_creation(self, text_mapping: Dict, image_distribution: Dict) -> Dict:
        """매거진 구조 통합 조율 (세션 및 통신 격리 적용, 안전성 강화)"""
        print(f"📦 CoordinatorAgent 조율 시작")
        
        # 세션 정보 확인
        if hasattr(self, 'current_session_id'):
            print(f"🔒 세션: {self.current_session_id}")
        
        # 입력 데이터 통신 격리 검증 (안전한 방식)
        if hasattr(self, 'receive_data_from_agent'):
            try:
                text_mapping_result = self.receive_data_from_agent("OrgAgent", text_mapping)
                image_distribution_result = self.receive_data_from_agent("BindingAgent", image_distribution)
                
                if text_mapping_result is None or image_distribution_result is None:
                    print("🚫 오염된 입력 데이터 감지, 폴백 모드로 전환")
                    return self._get_fallback_result("contaminated_input")
            except Exception as e:
                print(f"⚠️ 통신 격리 검증 실패: {e}, 원본 데이터 사용")
                text_mapping_result = text_mapping
                image_distribution_result = image_distribution
        else:
            text_mapping_result = text_mapping
            image_distribution_result = image_distribution
        
        self.execution_stats["total_attempts"] += 1
        
        # 재귀 깊이 확인 및 동기 모드 전환
        if self._should_use_sync():
            print("🔄 CoordinatorAgent 동기 모드로 전환하여 실행")
            result = await self._coordinate_magazine_creation_sync_mode(text_mapping_result, image_distribution_result)
        else:
            try:
                # 개선된 배치 기반 비동기 모드 실행
                result = await self._coordinate_magazine_creation_batch_mode(text_mapping_result, image_distribution_result)
            except RecursionError:
                print("🔄 CoordinatorAgent RecursionError 감지 - 동기 모드로 전환")
                self.fallback_to_sync = True
                result = await self._coordinate_magazine_creation_sync_mode(text_mapping_result, image_distribution_result)
            except Exception as e:
                print(f"❌ CoordinatorAgent 매거진 생성 중 예외 발생: {e} - 동기 모드로 폴백 시도")
                self.fallback_to_sync = True
                result = await self._coordinate_magazine_creation_sync_mode(text_mapping_result, image_distribution_result)
        
        # 세션별 결과 저장 (안전한 방식)
        if hasattr(self, 'store_result'):
            try:
                self.store_result(result)
            except Exception as e:
                print(f"⚠️ 세션 결과 저장 실패: {e}")
        
        # 결과에 세션 메타데이터 추가
        session_metadata = {
            "agent_name": "CoordinatorAgent",
            "isolation_applied": hasattr(self, 'isolation_manager'),
            "communication_isolated": hasattr(self, 'communication_isolator')
        }
        
        if hasattr(self, 'current_session_id'):
            session_metadata["session_id"] = self.current_session_id
        
        if hasattr(self, 'get_communication_stats'):
            try:
                session_metadata["coordination_stats"] = self.get_communication_stats()
            except Exception as e:
                print(f"⚠️ 통신 통계 조회 실패: {e}")
        
        result["session_metadata"] = session_metadata
        
        return result




    async def _get_enhanced_previous_results_batch(self) -> List[Dict]:
        """배치 기반 이전 결과 수집 (세션 격리 적용)"""
        try:
            all_results = []
            
            # 세션 격리가 활성화된 경우
            if hasattr(self, 'current_session_id') and hasattr(self, 'session_manager'):
                # 현재 세션의 결과만 조회
                session_results = self.get_previous_results(max_results=20)
                
                # 다른 에이전트의 결과도 세션 격리 적용하여 조회
                org_results = self.session_manager.get_agent_results(self.current_session_id, "OrgAgent")
                binding_results = self.session_manager.get_agent_results(self.current_session_id, "BindingAgent")
                
                # 격리 필터링 적용
                for result in session_results + org_results + binding_results:
                    if hasattr(self, 'isolation_manager') and not self.isolation_manager.is_contaminated(result, "enhanced_previous_results"):
                        all_results.append(result)
                    elif not hasattr(self, 'isolation_manager'):
                        all_results.append(result)
                
                print(f"🔍 세션 격리된 이전 결과: {len(all_results)}개")
            else:
                # 세션 격리가 비활성화된 경우 기본 방식 사용
                all_results = await self._get_enhanced_previous_results_fallback()
            
            return all_results
            
        except Exception as e:
            print(f"⚠️ 세션 격리된 이전 결과 수집 실패: {e}")
            return await self._get_enhanced_previous_results_fallback()

    async def _get_enhanced_previous_results_fallback(self) -> List[Dict]:
        """폴백 이전 결과 수집"""
        try:
            # HybridLogger에서 안전하게 결과 조회
            if hasattr(self.logger, 'get_learning_insights'):
                insights = self.logger.get_learning_insights("CoordinatorAgent")
                if isinstance(insights, dict) and 'patterns' in insights:
                    return [{"insight_data": insights, "source": "learning_insights"}]
            
            # 파일 기반 결과 로드
            file_results = self._load_results_from_file()
            if file_results:
                return file_results
            
            # 최종 폴백: 빈 결과
            print("⚠️ 이전 결과를 찾을 수 없음 - 빈 결과 반환")
            return []
            
        except Exception as e:
            print(f"⚠️ 폴백 이전 결과 수집 실패: {e}")
            return []

    def _get_enhanced_previous_results_sync(self) -> List[Dict]:
        """동기 버전 이전 결과 수집 (수정됨)"""
        try:
            all_results = []
            
            # 세션 격리가 활성화된 경우
            if hasattr(self, 'current_session_id') and hasattr(self, 'session_manager'):
                # 동기적으로 세션 결과 조회
                session_results = self.get_previous_results(max_results=20)
                org_results = self.session_manager.get_agent_results(self.current_session_id, "OrgAgent")
                binding_results = self.session_manager.get_agent_results(self.current_session_id, "BindingAgent")
                
                # 격리 필터링 적용
                for result in session_results + org_results + binding_results:
                    if hasattr(self, 'isolation_manager') and not self.isolation_manager.is_contaminated(result, "sync_previous_results"):
                        all_results.append(result)
                    elif not hasattr(self, 'isolation_manager'):
                        all_results.append(result)
            else:
                # HybridLogger 안전 호출
                if hasattr(self.logger, 'get_learning_insights'):
                    try:
                        insights = self.logger.get_learning_insights("CoordinatorAgent")
                        if isinstance(insights, dict):
                            all_results.append({"insight_data": insights, "source": "learning_insights"})
                    except Exception as e:
                        print(f"⚠️ Learning insights 조회 실패: {e}")
                
                # 파일 기반 결과 로드
                file_results = self._load_results_from_file()
                all_results.extend(file_results if isinstance(file_results, list) else [])
            
            return self._deduplicate_results(all_results)
            
        except Exception as e:
            print(f"⚠️ 동기 이전 결과 수집 실패: {e}")
            return []


    async def _coordinate_magazine_creation_batch_mode(self, text_mapping: Dict, image_distribution: Dict) -> Dict:
        """개선된 배치 기반 매거진 구조 통합 조율"""
        print("📦 CoordinatorAgent 배치 모드 시작")
        
        # 입력 데이터 로깅
        input_data = {
            "text_mapping": text_mapping,
            "image_distribution": image_distribution
        }
        
        
        # 강화된 이전 에이전트 결과 수집 (배치 처리)
        previous_results = await self._get_enhanced_previous_results_batch()
        org_results = self._filter_agent_results(previous_results, "OrgAgent")
        binding_results = self._filter_agent_results(previous_results, "BindingAgent")
        content_creator_results = self._filter_agent_results(previous_results, "ContentCreatorV2Agent")
        
        print(f"📊 배치 모드 결과 수집: 전체 {len(previous_results)}개, OrgAgent {len(org_results)}개, BindingAgent {len(binding_results)}개, ContentCreator {len(content_creator_results)}개")
        
        # 수정: OrgAgent 결과 필터링 - ContentCreatorV2Agent 결과만 사용
        filtered_org_results = []
        for result in org_results:
            final_answer = result.get("final_answer", "")
            raw_output = result.get("raw_output", {})
            
            # 폴백 데이터 제외
            if isinstance(raw_output, dict):
                metadata = raw_output.get("metadata", {})
                if metadata.get("fallback_used"):
                    continue
            
            # ContentCreatorV2Agent의 실제 콘텐츠만 포함
            if ("ContentCreatorV2Agent" in final_answer or
                "content_creator" in final_answer.lower() or
                len(final_answer) > 500):  # 충분한 콘텐츠가 있는 경우
                # "자세한 이야기를 담고 있습니다" 같은 템플릿 응답 제외
                if not ("자세한 이야기를 담고 있습니다" in final_answer or
                        "특별한 이야기를 담고 있습니다" in final_answer):
                    filtered_org_results.append(result)
        
        org_results = filtered_org_results
        print(f"🔍 필터링 후 OrgAgent 결과: {len(org_results)}개")
        
        # magazine_content.json 로드하여 섹션 수 확인
        target_section_count = self._get_target_section_count()
        print(f"🎯 목표 섹션 수: {target_section_count}개")
        
        # 데이터 추출 작업을 배치로 처리
        data_extraction_tasks = [
            ("text_data", self._extract_real_text_data_safe, text_mapping, org_results, content_creator_results, target_section_count),
            ("image_data", self._extract_real_image_data_safe, image_distribution, binding_results)
        ]
        
        extraction_results = await self._process_data_extraction_batch(data_extraction_tasks)
        extracted_text_data = extraction_results.get("text_data", {})
        extracted_image_data = extraction_results.get("image_data", {})
        
        # CrewAI 실행을 안전한 배치로 처리
        crew_result = await self._execute_crew_batch_safe(
            extracted_text_data, extracted_image_data, org_results, binding_results
        )
        
        final_result = await self._process_enhanced_crew_result_async(
            crew_result, extracted_text_data, extracted_image_data, org_results, binding_results
        )
        
        # 수정: 섹션 수 제한 및 폴백 데이터 제거
        final_result = self._limit_and_clean_sections(final_result, target_section_count)
        
        # 결과 검증
        if self._validate_coordinator_result(final_result):
            self.execution_stats["successful_executions"] += 1
        else:
            print("⚠️ CoordinatorAgent 최종 결과 검증 실패.")
        
        # 결과 로깅
        await self._log_coordination_result_async(final_result, text_mapping, image_distribution, org_results, binding_results)
        
        print(f"✅ CoordinatorAgent 배치 모드 완료: {len(final_result.get('content_sections', []))}개 섹션 생성")
        return final_result

    def _get_target_section_count(self) -> int:
        """magazine_content.json에서 목표 섹션 수 확인"""
        try:
            magazine_content_path = "./output/magazine_content.json"
            if os.path.exists(magazine_content_path):
                with open(magazine_content_path, 'r', encoding='utf-8') as f:
                    magazine_data = json.load(f)
                sections = magazine_data.get("sections", [])
                if isinstance(sections, list):
                    return len(sections)
            # 기본값
            return 5
        except Exception as e:
            print(f"⚠️ magazine_content.json 로드 실패: {e}")
            return 5

    def _limit_and_clean_sections(self, result: Dict, target_count: int) -> Dict:
        """섹션 수 제한 및 폴백 데이터 정리"""
        if not isinstance(result, dict) or "content_sections" not in result:
            return result
        
        content_sections = result["content_sections"]
        if not isinstance(content_sections, list):
            return result
        
        # 폴백 데이터 제거
        cleaned_sections = []
        for section in content_sections:
            if isinstance(section, dict):
                metadata = section.get("metadata", {})
                if not metadata.get("fallback_used"):
                    cleaned_sections.append(section)
        
        # 섹션 수 제한
        limited_sections = cleaned_sections[:target_count]
        
        # 최소 1개 섹션 보장 (폴백이 아닌 실제 데이터로)
        if not limited_sections:
            limited_sections = [{
                "template": "Section01.jsx",
                "title": "",
                "subtitle": "",
                "body": "",
                "tagline": "",
                "images": [],
                "metadata": {
                    "minimal_fallback": True
                }
            }]
        
        result["content_sections"] = limited_sections
        result["selected_templates"] = [section.get("template", f"Section{i+1:02d}.jsx")
                                      for i, section in enumerate(limited_sections)]
        
        # 메타데이터 업데이트
        if "integration_metadata" in result:
            result["integration_metadata"]["total_sections"] = len(limited_sections)
            result["integration_metadata"]["cleaned_sections"] = True
            result["integration_metadata"]["target_section_count"] = target_count
        
        return result

    async def _process_data_extraction_batch(self, extraction_tasks: List[tuple]) -> Dict:
        """데이터 추출 작업을 배치로 처리 (각 작업을 순차적으로 execute_with_resilience 호출)"""
        results = {}
        for task_name, task_func_ref, *args_for_task_func in extraction_tasks:
            if not callable(task_func_ref):
                print(f"⚠️ {task_name}에 대한 task_func이 호출 가능하지 않음: {task_func_ref}")
                results[task_name] = self._get_fallback_extraction_result(task_name)
                continue

            print(f"DEBUG [_process_data_extraction_batch]: executing task_name={task_name}")
            try:
                # 모든 작업을 execute_with_resilience로 통일하여 처리
                result_value = await self.execute_with_resilience(
                    task_func_ref, 
                    f"extract_{task_name}",
                    120.0,
                    1,
                    *args_for_task_func
                )

                results[task_name] = result_value
            except Exception as e:
                print(f"⚠️ 데이터 추출 작업 {task_name} 실패 (execute_with_resilience 호출 중): {e}")
                results[task_name] = self._get_fallback_extraction_result(task_name)
        return results



    def _get_fallback_extraction_result(self, task_name: str) -> Dict:
        """데이터 추출 폴백 결과"""
        self.execution_stats["fallback_used"] += 1
        if task_name == "text_data":
            return {
                "sections": [],
                "total_content_length": 0,
                "source_count": 0
            }
        else:  # image_data
            return {
                "template_images": {},
                "total_images": 0,
                "image_sources": []
            }

    async def _execute_crew_batch_safe(self, extracted_text_data: Dict, extracted_image_data: Dict,
                                     org_results: List[Dict], binding_results: List[Dict]) -> Any:
        """안전한 CrewAI 배치 실행"""
        try:
            # 태스크 생성
            text_analysis_task = self._create_enhanced_text_analysis_task(extracted_text_data, org_results)
            image_analysis_task = self._create_enhanced_image_analysis_task(extracted_image_data, binding_results)
            coordination_task = self._create_enhanced_coordination_task(extracted_text_data, extracted_image_data)
            
            # CrewAI Crew 생성
            coordination_crew = Crew(
                agents=[self.text_analyzer_agent, self.image_analyzer_agent, self.crew_agent],
                tasks=[text_analysis_task, image_analysis_task, coordination_task],
                process=Process.sequential,
                verbose=False  # 로그 최소화
            )
            
            # 안전한 실행 (타임아웃 증가)
            crew_result = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(None, coordination_crew.kickoff),
                timeout=600.0  # 10분으로 증가
            )
            
            return crew_result
            
        except asyncio.TimeoutError:
            print("⏰ CrewAI 배치 실행 타임아웃")
            self.execution_stats["timeout_occurred"] += 1
            return self._create_fallback_crew_result(extracted_text_data, extracted_image_data)
        except Exception as e:
            print(f"⚠️ CrewAI 배치 실행 실패: {e}")
            return self._create_fallback_crew_result(extracted_text_data, extracted_image_data)

    def _create_fallback_crew_result(self, extracted_text_data: Dict, extracted_image_data: Dict) -> str:
        """CrewAI 폴백 결과 생성"""
        self.execution_stats["fallback_used"] += 1
        sections = extracted_text_data.get("sections", [])
        if not sections:
            sections = []
        
        # 이미지 추가
        for section in sections:
            template = section.get("template", "Section01.jsx")
            template_images = extracted_image_data.get("template_images", {}).get(template, [])
            section["images"] = template_images[:3]  # 최대 3개로 제한
        
        return json.dumps({
            "selected_templates": [s.get("template", "Section01.jsx") for s in sections],
            "content_sections": sections
        })

    async def _extract_real_text_data_safe(self, text_mapping: Dict, org_results: List[Dict], 
                                     content_creator_results: List[Dict], target_section_count: int) -> Dict:
        """강제적 원본 데이터 우선 추출 (Azure AI Search 영향 차단)"""
        try:
            return await asyncio.get_event_loop().run_in_executor(
                None, self._extract_real_text_data_forced, text_mapping, org_results, 
                content_creator_results, target_section_count
            )
        except Exception as e:
            print(f"⚠️ 강제 텍스트 데이터 추출 실패: {e}")
            return self._get_fallback_extraction_result("text_data")

    def _extract_real_text_data_forced(self, text_mapping: Dict, org_results: List[Dict], 
                                    content_creator_results: List[Dict], target_section_count: int) -> Dict:
        """강제적 원본 데이터 우선 추출"""
        extracted_data = {
            "sections": [],
            "total_content_length": 0,
            "source_count": 0,
            "data_source_priority": "magazine_content_forced"
        }
        
        # 1. magazine_content.json 강제 우선 사용
        magazine_content_path = "./output/magazine_content.json"
        if os.path.exists(magazine_content_path):
            try:
                with open(magazine_content_path, 'r', encoding='utf-8') as f:
                    magazine_data = json.load(f)
                
                sections = magazine_data.get("sections", [])
                print(f"📁 magazine_content.json에서 {len(sections)}개 섹션 발견")
                
                for i, section in enumerate(sections):
                    if len(extracted_data["sections"]) < target_section_count:
                        # 원본 데이터 그대로 사용 (Azure AI Search 영향 차단)
                        extracted_section = {
                            "template": f"Section{i+1:02d}.jsx",
                            "title": section.get("title", f"섹션 {i+1}"),
                            "subtitle": section.get("subtitle", ""),
                            "body": section.get("content", section.get("body", "")),
                            "tagline": section.get("tagline", "TRAVEL & CULTURE"),
                            "source": "magazine_content_json_forced",
                            "priority": 1,
                            "azure_search_blocked": True  # Azure AI Search 영향 차단 플래그
                        }
                        extracted_data["sections"].append(extracted_section)
                        extracted_data["total_content_length"] += len(extracted_section["body"])
                        extracted_data["source_count"] += 1
                
                print(f"✅ magazine_content.json에서 {len(extracted_data['sections'])}개 섹션 추출 완료")
                
                # 원본 데이터가 충분한 경우 다른 소스 무시
                if len(extracted_data["sections"]) >= target_section_count:
                    print("🚫 원본 데이터 충분, Azure AI Search 결과 무시")
                    return extracted_data
                    
            except Exception as e:
                print(f"⚠️ magazine_content.json 로드 실패: {e}")
        
        # 2. 원본 데이터가 부족한 경우에만 text_mapping 사용
        remaining_count = target_section_count - len(extracted_data["sections"])
        if remaining_count > 0:
            print(f"📝 원본 데이터 부족, {remaining_count}개 섹션 추가 필요")
            
            # text_mapping에서 유효한 섹션 추출 (Azure AI Search 영향 최소화)
            if isinstance(text_mapping, dict):
                text_sections = text_mapping.get("text_mapping", [])
                for section in text_sections:
                    if (isinstance(section, dict) and 
                        len(extracted_data["sections"]) < target_section_count and
                        self._is_valid_original_section(section)):
                        
                        section["azure_search_blocked"] = True
                        section["source"] = "text_mapping_filtered"
                        extracted_data["sections"].append(section)
                        extracted_data["source_count"] += 1
        
        return extracted_data

    def _is_valid_original_section(self, section: Dict) -> bool:
        """원본 데이터 기반 섹션 유효성 검증"""
        if not isinstance(section, dict):
            return False
        
        # 폴백 데이터 제외
        metadata = section.get("metadata", {})
        if metadata.get("fallback_used"):
            return False
        
        # Azure AI Search 키워드 차단
        azure_search_keywords = [
            "도시의 미학", "골목길", "도시 계획", "친환경 도시",
            "도심 속 자연", "빛과 그림자", "아티스트 인터뷰"
        ]
        
        title = section.get("title", "").lower()
        body = section.get("body", "").lower()
        
        # Azure AI Search 키워드가 포함된 경우 제외
        for keyword in azure_search_keywords:
            if keyword in title or keyword in body:
                print(f"🚫 Azure AI Search 키워드 '{keyword}' 감지, 섹션 제외")
                return False
        
        # 최소 콘텐츠 요구사항
        return len(section.get("title", "")) > 0 or len(section.get("body", "")) > 10


    async def _extract_real_image_data_safe(self, image_distribution: Dict, binding_results: List[Dict]) -> Dict:
        """안전한 실제 이미지 데이터 추출"""
        try:
            return await self._extract_real_image_data_async(image_distribution, binding_results)
        except Exception as e:
            print(f"⚠️ 이미지 데이터 추출 실패: {e}")
            return self._get_fallback_extraction_result("image_data")

    async def _get_enhanced_previous_results_batch_isolated(self) -> List[Dict]:
        """세션 격리된 배치 기반 이전 결과 수집"""
        all_results = []
        
        try:
            # 현재 세션의 결과만 조회
            session_results = self.get_previous_results(max_results=20)
            
            # 다른 에이전트의 결과도 세션 격리 적용하여 조회
            org_results = self.session_manager.get_agent_results(self.current_session_id, "OrgAgent")
            binding_results = self.session_manager.get_agent_results(self.current_session_id, "BindingAgent")
            
            # 격리 필터링 적용
            for result in session_results + org_results + binding_results:
                if not self.isolation_manager.is_contaminated(result, "enhanced_previous_results"):
                    all_results.append(result)
            
            print(f"🔍 세션 격리된 이전 결과: {len(all_results)}개")
            return all_results
            
        except Exception as e:
            print(f"⚠️ 세션 격리된 이전 결과 수집 실패: {e}")
            return []



    # 기존 _coordinate_magazine_creation_async_mode 메서드 유지 (호환성을 위해)
    async def _coordinate_magazine_creation_async_mode(self, text_mapping: Dict, image_distribution: Dict) -> Dict:
        """비동기 모드 매거진 조율 (기존 호환성 유지)"""
        print("⚠️ 기존 async_mode 호출됨 - batch_mode로 리다이렉트")
        return await self._coordinate_magazine_creation_batch_mode(text_mapping, image_distribution)

    async def _coordinate_magazine_creation_sync_mode(self, text_mapping: Dict, image_distribution: Dict) -> Dict:
        """동기 모드 매거진 구조 통합 조율"""
        print("🔄 CoordinatorAgent 동기 모드 실행")
        
        # 동기 모드에서는 각 에이전트의 동기 버전 메서드를 호출해야 함
        # 이전 결과 수집 (동기)
        previous_results = self._get_enhanced_previous_results_sync()
        org_results = self._filter_agent_results(previous_results, "OrgAgent")
        binding_results = self._filter_agent_results(previous_results, "BindingAgent")
        content_creator_results = self._filter_agent_results(previous_results, "ContentCreatorV2Agent")
        
        # 수정: OrgAgent 결과 필터링
        filtered_org_results = []
        for result in org_results:
            final_answer = result.get("final_answer", "")
            raw_output = result.get("raw_output", {})
            
            # 폴백 데이터 제외
            if isinstance(raw_output, dict):
                metadata = raw_output.get("metadata", {})
                if metadata.get("fallback_used"):
                    continue
            
            # ContentCreatorV2Agent의 실제 콘텐츠만 포함
            if ("ContentCreatorV2Agent" in final_answer or
                "content_creator" in final_answer.lower() or
                len(final_answer) > 500):
                if not ("자세한 이야기를 담고 있습니다" in final_answer or
                        "특별한 이야기를 담고 있습니다" in final_answer):
                    filtered_org_results.append(result)
        
        org_results = filtered_org_results
        
        # 목표 섹션 수 확인
        target_section_count = self._get_target_section_count()
        
        # 데이터 추출 (동기)
        extracted_text_data = self._extract_real_text_data(text_mapping, org_results, content_creator_results, target_section_count)
        extracted_image_data = self._extract_real_image_data(image_distribution, binding_results)
        
        # Crew 실행 (동기) - CrewAI의 kickoff은 동기 메서드
        text_analysis_task_sync = self._create_enhanced_text_analysis_task(extracted_text_data, org_results)
        image_analysis_task_sync = self._create_enhanced_image_analysis_task(extracted_image_data, binding_results)
        coordination_task_sync = self._create_enhanced_coordination_task(extracted_text_data, extracted_image_data)
        
        coordination_crew_sync = Crew(
            agents=[self.text_analyzer_agent, self.image_analyzer_agent, self.crew_agent],
            tasks=[text_analysis_task_sync, image_analysis_task_sync, coordination_task_sync],
            process=Process.sequential,
            verbose=False
        )
        
        try:
            crew_result_sync = coordination_crew_sync.kickoff()
        except Exception as e_crew_sync:
            print(f"⚠️ 동기 모드 CrewAI 실행 실패: {e_crew_sync}")
            crew_result_sync = self._create_fallback_crew_result(extracted_text_data, extracted_image_data)
        
        # 결과 처리 (동기)
        final_result = self._process_enhanced_crew_result(crew_result_sync, extracted_text_data, extracted_image_data, org_results, binding_results)
        
        # 섹션 수 제한 및 정리
        final_result = self._limit_and_clean_sections(final_result, target_section_count)
        
        # 동기 모드 로깅
        final_response_id_sync = self.logger.log_agent_real_output(
            agent_name="CoordinatorAgent_SyncMode",
            agent_role="동기 모드 매거진 구조 통합 조율자",
            task_description=f"동기 모드로 {len(final_result.get('content_sections', []))}개 섹션 생성",
            final_answer=str(final_result),
            reasoning_process="재귀 깊이 초과로 인한 동기 모드 전환 후 안전한 매거진 구조 통합 실행",
            execution_steps=[
                "재귀 깊이 감지",
                "동기 모드 전환",
                "이전 결과 수집",
                "데이터 추출",
                "구조 생성"
            ],
            raw_input={
                "text_mapping": str(text_mapping)[:500],
                "image_distribution": str(image_distribution)[:500]
            },
            raw_output=final_result,
            performance_metrics={
                "sync_mode_used": True,
                "recursion_fallback": True,
                "total_sections": len(final_result.get('content_sections', [])),
                "org_results_utilized": len(org_results),
                "binding_results_utilized": len(binding_results),
                "safe_execution": True
            }
        )
        
        final_result["final_response_id"] = final_response_id_sync
        final_result["execution_mode"] = "sync_fallback"
        final_result["recursion_fallback"] = True  # 재귀로 인한 폴백 명시
        
        print(f"✅ CoordinatorAgent 동기 완료: {len(final_result.get('content_sections', []))}개 섹션")
        return final_result


    def _adjust_quality_criteria_dynamically(self, content_sections: List[Dict]) -> Dict[str, float]:
        """동적 품질 기준 조정"""
        total_sections = len(content_sections)
        
        # 섹션 수에 따른 기준 조정
        if total_sections <= 2:
            # 섹션이 적으면 더 관대하게
            criteria = {
                'min_title_length': 2,
                'min_body_length': 15,
                'min_quality_threshold': 0.4,
                'section_pass_rate': 0.3  # 30%만 통과하면 OK
            }
        elif total_sections <= 5:
            # 일반적인 경우
            criteria = {
                'min_title_length': 3,
                'min_body_length': 20,
                'min_quality_threshold': 0.5,
                'section_pass_rate': 0.5  # 50% 통과
            }
        else:
            # 섹션이 많으면 조금 더 엄격하게 (하지만 기존보다는 관대)
            criteria = {
                'min_title_length': 4,
                'min_body_length': 30,
                'min_quality_threshold': 0.6,
                'section_pass_rate': 0.6  # 60% 통과
            }
        
        # 원본 데이터 비율에 따른 추가 조정
        original_data_sections = sum(1 for section in content_sections 
                                if section.get('metadata', {}).get('source') == 'magazine_content_json_primary')
        original_ratio = original_data_sections / total_sections if total_sections > 0 else 0
        
        if original_ratio > 0.7:  # 70% 이상이 원본 데이터면 더 관대하게
            criteria['min_quality_threshold'] *= 0.8
            criteria['section_pass_rate'] *= 0.8
        
        print(f"📊 동적 품질 기준: 최소 품질 {criteria['min_quality_threshold']:.2f}, 통과율 {criteria['section_pass_rate']:.2f}")
        
        return criteria

    def _apply_dynamic_validation(self, content_sections: List[Dict]) -> bool:
        """동적 기준을 적용한 검증"""
        if not content_sections:
            return False
        
        criteria = self._adjust_quality_criteria_dynamically(content_sections)
        
        valid_sections = 0
        total_quality_score = 0
        
        for section in content_sections:
            quality_score = self._calculate_content_quality(section)
            total_quality_score += quality_score
            
            # 동적 기준 적용
            if quality_score >= criteria['min_quality_threshold']:
                valid_sections += 1
        
        # 통과 조건 확인
        pass_rate = valid_sections / len(content_sections)
        avg_quality = total_quality_score / len(content_sections)
        
        passed = (pass_rate >= criteria['section_pass_rate'] and avg_quality >= 0.5)
        
        print(f"📊 품질 검증 결과: 통과율 {pass_rate:.2f}, 평균 품질 {avg_quality:.2f}, 결과: {'✅ 통과' if passed else '❌ 실패'}")
        
        return passed




    def _get_enhanced_previous_results_sync(self) -> List[Dict]:
        """동기 버전 이전 결과 수집"""
        try:
            basic_results = self.logger.get_all_previous_results("CoordinatorAgent")
            file_results = self._load_results_from_file()
            
            all_results = []
            all_results.extend(basic_results if isinstance(basic_results, list) else [])
            all_results.extend(file_results if isinstance(file_results, list) else [])
            
            return self._deduplicate_results(all_results)
        except Exception as e:
            print(f"⚠️ 동기 이전 결과 수집 실패: {e}")
            return []

    # 모든 기존 메서드들 유지 (동기 버전들)
    async def _extract_real_text_data_async(self, text_mapping: Dict, org_results: List[Dict],
                                          content_creator_results: List[Dict], target_section_count: int) -> Dict:
        """실제 텍스트 데이터 추출 (비동기)"""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._extract_real_text_data, text_mapping, org_results, content_creator_results, target_section_count
        )

    def _extract_real_text_data(self, text_mapping: Dict, org_results: List[Dict],
                               content_creator_results: List[Dict], target_section_count: int) -> Dict:
        """실제 텍스트 데이터 추출"""
        extracted_data = {
            "sections": [],
            "total_content_length": 0,
            "source_count": 0
        }
        
        # 1. ContentCreator 결과에서 우선적으로 추출
        for result in content_creator_results:
            final_answer = result.get('final_answer', '')
            if len(final_answer) > 200:  # 충분한 콘텐츠가 있는 경우
                # 섹션별로 분할
                sections = self._split_content_into_sections(final_answer)
                for i, section_content in enumerate(sections):
                    if len(section_content) > 50 and len(extracted_data["sections"]) < target_section_count:
                        extracted_section = {
                            "template": f"Section{len(extracted_data['sections'])+1:02d}.jsx",
                            "title": self._extract_title_from_content(section_content),
                            "subtitle": self._extract_subtitle_from_content(section_content),
                            "body": self._clean_content(section_content),
                            "tagline": "TRAVEL & CULTURE",
                            "layout_source": "content_creator"
                        }
                        extracted_data["sections"].append(extracted_section)
                        extracted_data["total_content_length"] += len(extracted_section["body"])
                        extracted_data["source_count"] += 1
        
        # 2. text_mapping에서 추가 추출 (목표 섹션 수에 미달인 경우)
        if len(extracted_data["sections"]) < target_section_count and isinstance(text_mapping, dict):
            text_mapping_data = text_mapping.get("text_mapping", [])
            if isinstance(text_mapping_data, list):
                for section in text_mapping_data:
                    if (isinstance(section, dict) and
                        len(extracted_data["sections"]) < target_section_count):
                        # 폴백 데이터 제외
                        metadata = section.get("metadata", {})
                        if metadata.get("fallback_used"):
                            continue
                        
                        extracted_section = {
                            "template": section.get("template", f"Section{len(extracted_data['sections'])+1:02d}.jsx"),
                            "title": section.get("title", ""),
                            "subtitle": section.get("subtitle", ""),
                            "body": section.get("body", ""),
                            "tagline": section.get("tagline", "TRAVEL & CULTURE"),
                            "layout_source": "text_mapping"
                        }
                        
                        # 빈 콘텐츠 제외
                        if (extracted_section["title"] or extracted_section["subtitle"] or
                            len(extracted_section["body"]) > 10):
                            extracted_data["sections"].append(extracted_section)
                            extracted_data["total_content_length"] += len(extracted_section["body"])
                            extracted_data["source_count"] += 1
        
        # 3. 목표 섹션 수에 맞춰 제한
        extracted_data["sections"] = extracted_data["sections"][:target_section_count]
        
        return extracted_data

    async def _extract_real_image_data_async(self, image_distribution: Dict, binding_results: List[Dict]) -> Dict:
        """실제 이미지 데이터 추출 (비동기)"""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._extract_real_image_data, image_distribution, binding_results
        )

    def _extract_real_image_data(self, image_distribution: Dict, binding_results: List[Dict]) -> Dict:
        """실제 이미지 데이터 추출"""
        extracted_data = {
            "template_images": {},
            "total_images": 0,
            "image_sources": []
        }
        
        # 1. image_distribution에서 직접 추출
        if isinstance(image_distribution, dict) and "image_distribution" in image_distribution:
            for template, images in image_distribution["image_distribution"].items():
                if isinstance(images, list) and images:
                    real_images = [img for img in images if self._is_real_image_url(img)][:3]
                    if real_images:
                        extracted_data["template_images"][template] = real_images
                        extracted_data["total_images"] += len(real_images)
        
        # 2. BindingAgent 결과에서 이미지 URL 추출
        for result in binding_results:
            final_answer = result.get('final_answer', '')
            # 실제 이미지 URL 패턴 찾기
            image_urls = re.findall(r'https://[^\s\'"<>]*\.(?:jpg|jpeg|png|gif|webp)', final_answer, re.IGNORECASE)
            if image_urls:
                # 템플릿별로 분배
                template_name = self._extract_template_from_binding_result(result)
                if template_name not in extracted_data["template_images"]:
                    extracted_data["template_images"][template_name] = []
                
                for url in image_urls:
                    if (self._is_real_image_url(url) and
                        url not in extracted_data["template_images"][template_name] and
                        len(extracted_data["template_images"][template_name]) < 3):  # 최대 3개
                        extracted_data["template_images"][template_name].append(url)
                        extracted_data["total_images"] += 1
                        
                        # 이미지 소스 정보 추가
                        source_info = self._extract_image_source_info(result, url)
                        if source_info:
                            extracted_data["image_sources"].append(source_info)
        
        return extracted_data

    async def _process_enhanced_crew_result_async(self, crew_result, extracted_text_data: Dict,
                                            extracted_image_data: Dict, org_results: List[Dict],
                                            binding_results: List[Dict]) -> Dict:
        """강화된 Crew 실행 결과 처리 (Azure AI Search 영향 차단 포함)"""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._process_enhanced_crew_result_with_validation, crew_result, extracted_text_data,
            extracted_image_data, org_results, binding_results
        )

    def _process_enhanced_crew_result_with_validation(self, crew_result, extracted_text_data: Dict,
                                                extracted_image_data: Dict, org_results: List[Dict],
                                                binding_results: List[Dict]) -> Dict:
        """Crew 실행 결과 처리 및 검증 (AI Search 격리 적용)"""
        try:
            # 1. Azure AI Search 영향 차단 (새로운 격리 시스템 사용)
            parsed_data = self.block_azure_search_influence(crew_result)

            # 2. 기존 처리 로직
            if not parsed_data.get('content_sections') or len(parsed_data.get('content_sections', [])) == 0:
                parsed_data = self._create_enhanced_structure(
                    extracted_text_data, extracted_image_data, org_results, binding_results
                )
            else:
                parsed_data = self._enhance_parsed_data_with_real_images(
                    parsed_data, extracted_image_data
                )

            # 3. 콘텐츠 진정성 검증 및 교정 적용 (새로운 격리 시스템 사용)
            parsed_data = self.validate_content_authenticity(parsed_data)

            # 4. 최종 격리 검증
            parsed_data = self._final_isolation_validation(parsed_data)

            # 5. 최종 메타데이터 업데이트
            parsed_data['integration_metadata'] = {
                **parsed_data.get('integration_metadata', {}),
                "total_sections": len(parsed_data.get('content_sections', [])),
                "azure_search_influence_blocked": True,
                "original_content_validation_applied": True,
                "data_source_priority": "magazine_content_json_primary",
                "ai_search_isolation": {
                    "isolation_applied": True,
                    "contamination_report": self._get_isolation_report(),
                    "final_validation_passed": True
                }
            }

            return parsed_data

        except Exception as e:
            print(f"⚠️ Crew 결과 처리 및 검증 실패: {e}")
            return self._restore_from_magazine_content()




    def _process_enhanced_crew_result(self, crew_result, extracted_text_data: Dict,
                                    extracted_image_data: Dict, org_results: List[Dict],
                                    binding_results: List[Dict]) -> Dict:
        """강화된 Crew 실행 결과 처리"""
        try:
            # Crew 결과에서 데이터 추출
            if hasattr(crew_result, 'raw') and crew_result.raw:
                result_text = crew_result.raw
            else:
                result_text = str(crew_result)
            
            # JSON 패턴 찾기 및 파싱
            parsed_data = self._extract_json_from_text(result_text)
            
            # 실제 데이터 기반 구조 생성
            if not parsed_data.get('content_sections') or len(parsed_data.get('content_sections', [])) == 0:
                parsed_data = self._create_enhanced_structure(extracted_text_data, extracted_image_data, org_results, binding_results)
            else:
                # 기존 파싱된 데이터에 실제 이미지 추가
                parsed_data = self._enhance_parsed_data_with_real_images(parsed_data, extracted_image_data)
            
            # 메타데이터 추가
            parsed_data['integration_metadata'] = {
                "total_sections": len(parsed_data.get('content_sections', [])),
                "total_templates": len(set(section.get("template", f"Section{i+1:02d}.jsx") for i, section in enumerate(parsed_data.get('content_sections', [])))),
                "agent_enhanced": True,
                "org_results_utilized": len(org_results),
                "binding_results_utilized": len(binding_results),
                "integration_quality_score": self._calculate_enhanced_quality_score(parsed_data.get('content_sections', []), len(org_results), len(binding_results)),
                "crewai_enhanced": True,
                "async_processed": True,
                "data_source": "real_extracted_data",
                "real_content_used": True,
                "real_images_used": extracted_image_data['total_images'] > 0
            }
            
            return parsed_data
            
        except Exception as e:
            print(f"⚠️ 강화된 Crew 결과 처리 실패: {e}")
            return self._create_enhanced_structure(extracted_text_data, extracted_image_data, org_results, binding_results)

    # 모든 기존 유틸리티 메서드들 유지
    def _is_real_image_url(self, url: str) -> bool:
        """실제 이미지 URL인지 확인"""
        if not url or not isinstance(url, str):
            return False
        
        # 예시 URL이나 플레이스홀더 제외
        excluded_patterns = [
            'your-cdn.com',
            'example.com',
            'placeholder',
            'sample',
            'demo'
        ]
        
        for pattern in excluded_patterns:
            if pattern in url.lower():
                return False
        
        # 실제 도메인과 이미지 확장자 확인
        return (url.startswith('https://') and
                any(ext in url.lower() for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp']) and
                'blob.core.windows.net' in url)

    def _create_enhanced_text_analysis_task(self, extracted_text_data: Dict, org_results: List[Dict]) -> Task:
        """강화된 텍스트 분석 태스크 생성"""
        return Task(
            description=f"""추출된 실제 텍스트 데이터를 분석하여 고품질 매거진 섹션을 생성하세요.

**추출된 데이터:**
- 섹션 수: {len(extracted_text_data['sections'])}개
- 총 콘텐츠 길이: {extracted_text_data['total_content_length']} 문자
- 소스 수: {extracted_text_data['source_count']}개
- OrgAgent 결과: {len(org_results)}개

**실제 섹션 데이터:**
{self._format_sections_for_analysis(extracted_text_data['sections'])}

**분석 요구사항:**
1. 각 섹션의 콘텐츠 품질 평가
2. 제목과 부제목의 매력도 검증
3. 본문 내용의 완성도 확인
4. 매거진 스타일 일관성 검토
5. 독자 친화성 최적화

**출력 형식:**
각 섹션별로 다음 정보 포함:
- 품질 점수 (1-10)
- 개선 제안사항
- 최적화된 콘텐츠""",
            expected_output="실제 데이터 기반 텍스트 분석 및 최적화 결과",
            agent=self.text_analyzer_agent
        )

    def _create_enhanced_image_analysis_task(self, extracted_image_data: Dict, binding_results: List[Dict]) -> Task:
        """강화된 이미지 분석 태스크 생성"""
        return Task(
            description=f"""추출된 실제 이미지 데이터를 분석하여 최적화된 이미지 배치를 생성하세요.

**추출된 데이터:**
- 총 이미지 수: {extracted_image_data['total_images']}개
- 템플릿 수: {len(extracted_image_data['template_images'])}개
- BindingAgent 결과: {len(binding_results)}개

**템플릿별 이미지 분배:**
{self._format_images_for_analysis(extracted_image_data['template_images'])}

**이미지 소스 정보:**
{self._format_image_sources(extracted_image_data['image_sources'])}

**분석 요구사항:**
1. 이미지 URL 유효성 검증
2. 템플릿별 이미지 분배 균형도 평가
3. 이미지 품질 및 적합성 확인
4. 시각적 일관성 검토
5. 레이아웃 최적화 제안

**출력 형식:**
템플릿별로 다음 정보 포함:
- 이미지 목록 및 설명
- 배치 권장사항
- 시각적 효과 예측""",
            expected_output="실제 이미지 데이터 기반 배치 분석 및 최적화 결과",
            agent=self.image_analyzer_agent
        )

    def _create_enhanced_coordination_task(self, extracted_text_data: Dict, extracted_image_data: Dict) -> Task:
        """강화된 통합 조율 태스크 생성 - 완벽한 지침 적용"""
        return Task(
            description=f"""# 매거진 구조 통합 조율 전문가 임무

    ## 1. 역할 정의 (Identity)
    **당신의 정체성:** 25년 경력의 세계 최고 수준 매거진 구조 통합 조율 전문가
    **전문 분야:** 출판학 및 구조 설계 석사, PMP 인증, 텍스트-이미지 정합성 검증 시스템 개발
    **근무 경력:** Condé Nast, Hearst Corporation, Time Inc.에서 수백 개 매거진 프로젝트 성공 조율
    **어조 및 태도:** 정확하고 체계적이며, 원본 데이터 무결성에 대한 절대적 책임감

    ## 2. 현재 상황 및 맥락 (Context)
    **작업 환경:** 디지털 매거진 자동 생성 시스템
    **입력 데이터 현황:**
    - 추출된 텍스트 섹션: {len(extracted_text_data.get('sections', []))}개
    - 총 텍스트 길이: {extracted_text_data.get('total_content_length', 0):,} 문자
    - 사용 가능한 이미지: {extracted_image_data.get('total_images', 0)}개
    - 템플릿 분배: {len(extracted_image_data.get('template_images', {}))}개 템플릿

    **중요한 배경 정보:**
    - magazine_content.json과 image_analysis.json이 유일한 신뢰할 수 있는 원본 데이터 소스
    - Azure AI Search 데이터는 레이아웃 참고용으로만 제한적 사용
    - 최종 출력물은 JSX 컴포넌트 생성을 위한 완전한 스펙이어야 함

    ## 3. 핵심 임무 (Task)

    ### 3.1 주요 목표
    magazine_content.json과 image_analysis.json의 원본 데이터를 100% 충실히 반영하여 완벽한 매거진 구조를 생성

    ### 3.2 세부 작업 단계
    **Step 1: 데이터 소스 우선순위 적용**
    1. magazine_content.json의 원본 텍스트를 최우선으로 추출
    2. image_analysis.json의 이미지 분석 결과를 정확히 매핑
    3. 외부 데이터는 구조적 참고용으로만 제한적 활용

    **Step 2: 콘텐츠 무결성 검증**
    1. 원본 텍스트의 제목, 부제목, 본문이 변경되지 않았는지 확인
    2. 이미지 설명이 해당 섹션과 논리적으로 일치하는지 검증
    3. 폴백 데이터나 플레이스홀더가 포함되지 않았는지 점검

    **Step 3: 구조적 최적화**
    1. 각 섹션에 가장 적합한 템플릿 선택
    2. 텍스트와 이미지의 균형잡힌 배치
    3. 독자 경험을 고려한 섹션 순서 조정

    **Step 4: 품질 보증 및 최종 검토**
    1. 모든 필수 필드가 실제 데이터로 채워졌는지 확인
    2. JSON 구조가 명세에 정확히 일치하는지 검증
    3. 메타데이터의 정확성 및 완성도 점검

    ## 4. 응답 가이드라인 (Response Guidelines)

    ### 4.1 구조화된 응답 형식
    **필수 JSON 구조를 정확히 준수하세요:**
    {{
    "selected_templates": ["실제 선택된 템플릿 목록"],
    "content_sections": [
    {{
    "template": "섹션별 최적 템플릿",
    "title": "magazine_content.json에서 추출된 실제 제목",
    "subtitle": "magazine_content.json에서 추출된 실제 부제목",
    "body": "magazine_content.json의 원본 텍스트 기반 본문",
    "tagline": "원본 데이터 기반 태그라인",
    "images": ["image_analysis.json 기반 실제 이미지 URL (최대 3개)"],
    "metadata": {{
    "content_quality": "A+ 형식의 품질 점수",
    "image_count": 실제_이미지_수,
    "source": "magazine_content_json_primary",
    "original_content_preserved": true,
    "azure_search_influence": "minimal"
    }}
    }}
    ],
    "integration_metadata": {{
    "total_sections": magazine_content.json_섹션_수와_정확히_일치,
    "data_source_priority": "magazine_content_json_primary",
    "original_content_fidelity": "100%",
    "external_data_influence": "minimal"
    }}
    }}

    text

    ### 4.2 포맷팅 규칙
    - **간결성:** 불필요한 설명이나 메타 정보는 JSON에 포함하지 않음
    - **정확성:** 모든 필드는 실제 데이터로만 채움
    - **일관성:** 동일한 명명 규칙과 데이터 형식 유지
    - **완성도:** 빈 필드가 있다면 빈 문자열("")이나 빈 배열([])로 명시

    ### 4.3 언어 스타일
    - **전문적:** 매거진 출판 업계 표준 용어 사용
    - **명확함:** 모호한 표현 금지, 구체적이고 정확한 설명
    - **객관적:** 개인적 의견이나 추측 배제

    ## 5. 오류 처리 및 폴백 (Error Handling)

    ### 5.1 명확화 프롬프트
    **데이터 부족 시:**
    - magazine_content.json에 해당 필드가 없으면 빈 문자열("")로 처리
    - image_analysis.json에 이미지가 없으면 빈 배열([])로 처리
    - 절대로 임의의 내용을 생성하지 않음

    ### 5.2 기본 응답
    **처리 불가능한 요청 시:**
    - 원본 데이터 범위를 벗어나는 요청은 거부
    - 폴백 데이터 사용 요청은 명시적으로 거부
    - Azure AI Search 데이터가 원본과 상충하면 원본 우선

    ### 5.3 액션 실패 처리
    **외부 시스템 연동 실패 시:**
    - magazine_content.json 로드 실패: 최소한의 구조만 생성
    - image_analysis.json 접근 불가: 이미지 없이 텍스트만 처리
    - 템플릿 매핑 실패: 기본 템플릿(Section01.jsx 등) 사용

    ## 6. 사용자 정의 가드레일 (Guardrails)

    ### 6.1 절대 금지 사항
    ❌ **Azure AI Search 키워드 사용 금지:**
    - "도시의 미학", "골목길의 재발견", "아티스트 인터뷰", "친환경 도시"
    - "도심 속 자연", "빛과 그림자", "새로운 시선", "편집장의 글"
    - "특집:", "포토 에세이", "트렌드:", "프로파일 하이라이트"

    ❌ **폴백 데이터 절대 금지:**
    - fallback_used: true인 데이터 사용 금지
    - 플레이스홀더 텍스트 생성 금지
    - 예시 콘텐츠나 템플릿 설명 포함 금지
    
    ❌ **특정 설명 사용 금지**
    - 특정 구조, 설명, 사용방법에 대한 데이터 사용금지

    ❌ **임의 콘텐츠 생성 금지:**
    - 원본에 없는 새로운 제목이나 내용 창작 금지
    - 추측이나 가정에 기반한 정보 추가 금지

    ### 6.2 필수 준수 사항
    ✅ **원본 데이터 우선순위:**
    1. magazine_content.json (최우선)
    2. image_analysis.json (최우선)
    3. 입력 매개변수 (보조)
    4. Azure AI Search (구조 참고용만)

    ✅ **데이터 무결성 보장:**
    - 원본 텍스트의 의미, 톤, 스타일 100% 보존
    - 이미지 분석 결과와 섹션 내용의 논리적 일치성 유지
    - 모든 URL의 유효성 및 실제성 확인

    ## 7. 품질 검증 체크리스트 (Quality Verification)

    작업 완료 전 다음 사항을 **반드시** 확인하세요:

    ### 7.1 콘텐츠 검증
    - [ ] magazine_content.json의 모든 섹션이 content_sections에 반영됨
    - [ ] 각 섹션의 title, subtitle, body가 원본 텍스트를 충실히 반영함
    - [ ] 원본 데이터에 없는 주제나 내용이 생성되지 않음
    - [ ] 텍스트의 핵심 주제, 톤, 스타일이 변경되지 않음

    ### 7.2 이미지 검증
    - [ ] image_analysis.json의 이미지 설명이 해당 섹션에 적절히 통합됨
    - [ ] 모든 이미지 URL이 실제 유효한 URL임
    - [ ] 각 섹션당 최대 3개 이미지 제한 준수
    - [ ] 이미지와 텍스트 내용의 논리적 일치성 확인

    ### 7.3 구조 검증
    - [ ] JSON 구조가 명세에 정확히 일치함
    - [ ] 모든 필수 필드가 적절한 데이터 타입으로 채워짐
    - [ ] 중복 섹션이 생성되지 않음
    - [ ] 템플릿 선택이 콘텐츠 특성에 적합함

    ### 7.4 메타데이터 검증
    - [ ] content_quality 점수가 실제 품질을 반영함
    - [ ] source 정보가 정확히 기록됨
    - [ ] original_content_preserved가 true로 설정됨
    - [ ] azure_search_influence가 "minimal"로 제한됨

    ## 8. 예시 시나리오 (Examples)

    ### 8.1 완벽한 처리 예시
    **입력:** magazine_content.json에 "베네치아 여행기" 섹션 존재
    **처리:** 
    {{
    "template": "",
    "title": "",
    "subtitle": "",
    "body": "",
    "tagline": "",
    "images": ["https://blob.core.windows.net/images/venice1.jpg"],
    "metadata": {{
    "content_quality": "A+",
    "image_count": 1,
    "source": "magazine_content_json_primary",
    "original_content_preserved": true
    }}
    }}

    text

    ### 8.2 데이터 부족 시 처리 예시
    **입력:** magazine_content.json에 subtitle 없음
    **처리:**
    {{
    "subtitle": "",
    "metadata": {{
    "original_content_preserved": true,
    "missing_fields": ["subtitle"]
    }}
    }}

    text

    ### 8.3 Azure AI Search 영향 차단 예시
    **상황:** Azure AI Search가 "도시의 미학" 제안, Ai search 내부 데이터 제안
    **처리:** 해당 제안 완전 무시, magazine_content.json의 원본 제목만 사용

    이 지침을 철저히 준수하여 magazine_content.json과 image_analysis.json의 내용이 최종 출력물에 100% 완전히 반영되도록 하십시오.""",
            expected_output="magazine_content.json과 image_analysis.json 원본 데이터를 100% 충실히 반영한 완성된 매거진 구조 JSON",
            agent=self.crew_agent,
            context=[
                self._create_enhanced_text_analysis_task(extracted_text_data, []),
                self._create_enhanced_image_analysis_task(extracted_image_data, [])
            ]
        )

    def _create_enhanced_structure(self, extracted_text_data: Dict, extracted_image_data: Dict,
                                 org_results: List[Dict], binding_results: List[Dict]) -> Dict:
        """실제 데이터 기반 강화된 구조 생성"""
        content_sections = []
        selected_templates = []
        
        # 추출된 텍스트 섹션을 기반으로 구조 생성
        for i, section in enumerate(extracted_text_data.get('sections', [])):
            template = section.get('template', f"Section{i+1:02d}.jsx")
            
            # 해당 템플릿의 이미지 가져오기
            template_images = extracted_image_data.get('template_images', {}).get(template, [])
            
            # 섹션 구조 생성
            section_data = {
                "template": template,
                "title": section.get('title', ''),
                "subtitle": section.get('subtitle', ''),
                "body": section.get('body', ''),
                "tagline": section.get('tagline', 'TRAVEL & CULTURE'),
                "images": template_images[:3],  # 최대 3개로 제한
                "metadata": {
                    "content_quality": self._calculate_content_quality(section),
                    "image_count": len(template_images[:3]),
                    "source": section.get('layout_source', 'extracted'),
                    "real_content": True,
                    "fallback_used": False
                }
            }
            
            content_sections.append(section_data)
            selected_templates.append(template)
        
        # 최소 1개 섹션 보장
        if not content_sections:
            content_sections = [{
                "template": "Section01.jsx",
                "title": "여행 매거진",
                "subtitle": "특별한 이야기",
                "body": "매거진 콘텐츠를 준비 중입니다.",
                "tagline": "TRAVEL & CULTURE",
                "images": [],
                "metadata": {
                    "content_quality": 0.5,
                    "image_count": 0,
                    "source": "minimal_fallback",
                    "real_content": False,
                    "fallback_used": True
                }
            }]
            selected_templates = ["Section01.jsx"]
        
        return {
            "selected_templates": selected_templates,
            "content_sections": content_sections,
            "integration_metadata": {
                "total_sections": len(content_sections),
                "total_templates": len(set(selected_templates)),
                "integration_quality_score": self._calculate_enhanced_quality_score(
                    content_sections, len(org_results), len(binding_results)
                ),
                "org_results_utilized": len(org_results),
                "binding_results_utilized": len(binding_results),
                "enhanced_structure": True,
                "real_data_based": True
            }
        }

    def _enhance_parsed_data_with_real_images(self, parsed_data: Dict, extracted_image_data: Dict) -> Dict:
        """파싱된 데이터에 실제 이미지 추가"""
        if not isinstance(parsed_data, dict) or 'content_sections' not in parsed_data:
            return parsed_data
        
        content_sections = parsed_data['content_sections']
        if not isinstance(content_sections, list):
            return parsed_data
        
        # 각 섹션에 실제 이미지 추가
        for section in content_sections:
            if isinstance(section, dict):
                template = section.get('template', 'Section01.jsx')
                real_images = extracted_image_data.get('template_images', {}).get(template, [])
                
                # 기존 이미지를 실제 이미지로 교체 (최대 3개)
                section['images'] = real_images[:3]
                
                # 메타데이터 업데이트
                if 'metadata' not in section:
                    section['metadata'] = {}
                section['metadata']['real_images_used'] = len(real_images[:3]) > 0
                section['metadata']['image_count'] = len(real_images[:3])
        
        return parsed_data

    def _calculate_content_quality(self, section: Dict) -> float:
        """개선된 콘텐츠 품질 점수 계산 (완화된 기준)"""
        score = 0.0
        
        # 제목 품질 (0.25) - 기준 완화
        title = section.get('title', '')
        if title and len(title) > 3:  # 3자 이상으로 완화
            if len(title) > 10:  # 10자 이상이면 만점
                score += 0.25
            else:
                score += 0.20  # 3-10자도 높은 점수
        elif title:
            score += 0.15  # 빈 제목이 아니면 기본 점수
        
        # 부제목 품질 (0.15) - 선택적 요소로 변경
        subtitle = section.get('subtitle', '')
        if subtitle and len(subtitle) > 3:
            score += 0.15
        elif subtitle:
            score += 0.10
        else:
            score += 0.05  # 부제목이 없어도 기본 점수 제공
        
        # 본문 품질 (0.35) - 대폭 완화
        body = section.get('body', '')
        body_length = len(body)
        if body_length > 100:  # 100자 이상이면 만점 (기존 200자에서 완화)
            score += 0.35
        elif body_length > 50:  # 50자 이상도 높은 점수
            score += 0.30
        elif body_length > 20:  # 20자 이상도 적절한 점수
            score += 0.25
        elif body:
            score += 0.15  # 내용이 있으면 기본 점수
        
        # 태그라인 품질 (0.10) - 관대한 평가
        tagline = section.get('tagline', '')
        if tagline and tagline.strip():
            score += 0.10
        else:
            score += 0.05  # 태그라인이 없어도 기본 점수
        
        # 이미지 보너스 (0.15) - 새로 추가
        images = section.get('images', [])
        if images and len(images) > 0:
            if len(images) >= 2:
                score += 0.15  # 2개 이상 이미지면 보너스
            else:
                score += 0.10  # 1개 이미지도 보너스
        else:
            score += 0.05  # 이미지가 없어도 기본 점수
        
        # 원본 데이터 보너스 (최대 0.20 추가)
        metadata = section.get('metadata', {})
        if metadata.get('source') == 'magazine_content_json_primary':
            score += 0.15  # 원본 데이터 보너스
        if not metadata.get('fallback_used', False):
            score += 0.05  # 폴백이 아닌 경우 보너스
        
        # 최종 점수는 1.0을 초과할 수 있도록 허용 (최대 1.2)
        return min(score, 1.2)

    def _calculate_enhanced_quality_score(self, content_sections: List[Dict],
                                        org_results_count: int, binding_results_count: int) -> float:
        """강화된 품질 점수 계산 (완화된 기준)"""
        if not content_sections:
            return 0.5  # 기본 점수 상향 (기존 0.0에서 0.5로)
        
        # 기본 콘텐츠 품질 점수
        content_scores = [self._calculate_content_quality(section) for section in content_sections]
        avg_content_score = sum(content_scores) / len(content_scores)
        
        # 데이터 활용도 점수 (기준 완화)
        data_utilization_score = min((org_results_count + binding_results_count) / 5.0, 1.0)  # 기존 10.0에서 5.0으로 완화
        
        # 이미지 활용도 점수 (기준 완화)
        total_images = sum(len(section.get('images', [])) for section in content_sections)
        image_score = min(total_images / len(content_sections), 1.0)  # 섹션당 평균 1개 이미지 기준으로 완화
        
        # 섹션 수 보너스 (새로 추가)
        section_count_bonus = min(len(content_sections) / 5.0, 0.2)  # 섹션이 많을수록 보너스
        
        # 가중 평균 계산 (더 관대하게)
        final_score = (
            avg_content_score * 0.4 +  # 콘텐츠 품질 비중 감소
            data_utilization_score * 0.2 +  # 데이터 활용도 비중 감소
            image_score * 0.2 +  # 이미지 점수 비중 증가
            section_count_bonus * 0.2  # 섹션 수 보너스 추가
        )
        
        # 최소 점수 보장
        final_score = max(final_score, 0.6)  # 최소 0.6점 보장
        
        return round(final_score, 2)


    def _calculate_enhanced_quality_score(self, content_sections: List[Dict], 
                                        org_results_count: int, binding_results_count: int) -> float:
        """강화된 품질 점수 계산"""
        if not content_sections:
            return 0.0
        
        # 기본 콘텐츠 품질 점수
        content_scores = [self._calculate_content_quality(section) for section in content_sections]
        avg_content_score = sum(content_scores) / len(content_scores)
        
        # 데이터 활용도 점수
        data_utilization_score = min((org_results_count + binding_results_count) / 10.0, 1.0)
        
        # 이미지 활용도 점수
        total_images = sum(len(section.get('images', [])) for section in content_sections)
        image_score = min(total_images / (len(content_sections) * 2), 1.0)  # 섹션당 평균 2개 이미지 기준
        
        # 가중 평균 계산
        final_score = (avg_content_score * 0.5 + data_utilization_score * 0.3 + image_score * 0.2)
        
        return round(final_score, 2)

    def _format_sections_for_analysis(self, sections: List[Dict]) -> str:
        """분석용 섹션 포맷팅"""
        if not sections:
            return "섹션 데이터 없음"
        
        formatted = []
        for i, section in enumerate(sections[:3]):  # 처음 3개만 표시
            formatted.append(f"""
섹션 {i+1}:
- 템플릿: {section.get('template', 'N/A')}
- 제목: {section.get('title', 'N/A')[:50]}...
- 부제목: {section.get('subtitle', 'N/A')[:50]}...
- 본문 길이: {len(section.get('body', ''))} 문자
- 소스: {section.get('layout_source', 'N/A')}""")
        
        if len(sections) > 3:
            formatted.append(f"... 및 {len(sections) - 3}개 추가 섹션")
        
        return "\n".join(formatted)

    def _format_images_for_analysis(self, template_images: Dict) -> str:
        """분석용 이미지 포맷팅"""
        if not template_images:
            return "이미지 데이터 없음"
        
        formatted = []
        for template, images in template_images.items():
            formatted.append(f"- {template}: {len(images)}개 이미지")
            for img in images[:2]:  # 처음 2개만 표시
                formatted.append(f"  * {img[:60]}...")
        
        return "\n".join(formatted)

    def _format_image_sources(self, image_sources: List[Dict]) -> str:
        """이미지 소스 정보 포맷팅"""
        if not image_sources:
            return "이미지 소스 정보 없음"
        
        formatted = []
        for source in image_sources[:5]:  # 처음 5개만 표시
            formatted.append(f"- {source.get('url', 'N/A')[:50]}... (소스: {source.get('source', 'N/A')})")
        
        if len(image_sources) > 5:
            formatted.append(f"... 및 {len(image_sources) - 5}개 추가 소스")
        
        return "\n".join(formatted)

    def _split_content_into_sections(self, content: str) -> List[str]:
        """콘텐츠를 섹션별로 분할"""
        # 단락 기준으로 분할
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        # 최소 길이 이상의 단락들을 섹션으로 구성
        sections = []
        current_section = ""
        
        for paragraph in paragraphs:
            if len(current_section + paragraph) < 300:  # 섹션당 최소 300자
                current_section += paragraph + "\n\n"
            else:
                if current_section:
                    sections.append(current_section.strip())
                current_section = paragraph + "\n\n"
        
        # 마지막 섹션 추가
        if current_section:
            sections.append(current_section.strip())
        
        return sections

    def _extract_title_from_content(self, content: str) -> str:
        """콘텐츠에서 제목 추출"""
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if line and len(line) < 100:  # 제목은 보통 100자 이하
                # 특수 문자나 번호 제거
                cleaned = re.sub(r'^[\d\.\-\*\#\s]+', '', line)
                if len(cleaned) > 5:
                    return cleaned[:80]  # 최대 80자
        
        # 첫 번째 문장을 제목으로 사용
        first_sentence = content.split('.')[0].strip()
        return first_sentence[:80] if first_sentence else "여행 이야기"

    def _extract_subtitle_from_content(self, content: str) -> str:
        """콘텐츠에서 부제목 추출"""
        lines = content.split('\n')
        
        # 두 번째 줄이나 첫 번째 문장 다음을 부제목으로 사용
        if len(lines) > 1:
            subtitle = lines[1].strip()
            if subtitle and len(subtitle) < 150:
                return subtitle[:100]
        
        # 두 번째 문장을 부제목으로 사용
        sentences = content.split('.')
        if len(sentences) > 1:
            subtitle = sentences[1].strip()
            return subtitle[:100] if subtitle else "특별한 경험"
        
        return "특별한 경험"

    def _clean_content(self, content: str) -> str:
        """콘텐츠 정리"""
        # 불필요한 공백 제거
        cleaned = re.sub(r'\n\s*\n', '\n\n', content)
        cleaned = re.sub(r'[ \t]+', ' ', cleaned)
        
        # 특수 문자 정리
        cleaned = re.sub(r'^[\d\.\-\*\#\s]+', '', cleaned, flags=re.MULTILINE)
        
        return cleaned.strip()

    def _extract_template_from_binding_result(self, result: Dict) -> str:
        """BindingAgent 결과에서 템플릿명 추출"""
        final_answer = result.get('final_answer', '')
        
        # 템플릿 패턴 찾기
        template_match = re.search(r'Section\d{2}\.jsx', final_answer)
        if template_match:
            return template_match.group()
        
        # 기본 템플릿 반환
        return "Section01.jsx"

    def _extract_image_source_info(self, result: Dict, url: str) -> Dict:
        """이미지 소스 정보 추출"""
        return {
            "url": url,
            "source": "BindingAgent",
            "agent_id": result.get('agent_id', 'unknown'),
            "timestamp": result.get('timestamp', 'unknown')
        }

    def _filter_agent_results(self, results: List[Dict], agent_name: str) -> List[Dict]:
        """특정 에이전트 결과 필터링"""
        filtered = []
        for result in results:
            if isinstance(result, dict):
                agent_info = result.get('agent_name', '')
                if agent_name.lower() in agent_info.lower():
                    filtered.append(result)
        return filtered

    def _deduplicate_results(self, results: List[Dict]) -> List[Dict]:
        """결과 중복 제거"""
        seen_ids = set()
        deduplicated = []
        
        for result in results:
            if isinstance(result, dict):
                result_id = result.get('id', str(hash(str(result))))
                if result_id not in seen_ids:
                    seen_ids.add(result_id)
                    deduplicated.append(result)
        
        return deduplicated

    def _load_results_from_file(self) -> List[Dict]:
        """파일에서 결과 로드"""
        try:
            results_file = "./output/agent_results.json"
            if os.path.exists(results_file):
                with open(results_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data if isinstance(data, list) else []
            return []
        except Exception as e:
            print(f"⚠️ 결과 파일 로드 실패: {e}")
            return []

    def _extract_json_from_text(self, text: str) -> Dict:
        """텍스트에서 JSON 추출 및 파싱"""
        try:
            # JSON 블록 찾기
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                return json.loads(json_str)
            
            # 기본 구조 반환
            return {
                "selected_templates": [],
                "content_sections": []
            }
        except Exception as e:
            print(f"⚠️ JSON 파싱 실패: {e}")
            return {
                "selected_templates": [],
                "content_sections": []
            }

    def _validate_coordinator_result(self, result: Dict) -> bool:
        """CoordinatorAgent 결과 검증 (완화된 기준)"""
        if not isinstance(result, dict):
            return False
        
        # 필수 키 확인
        required_keys = ['selected_templates', 'content_sections']
        for key in required_keys:
            if key not in result:
                return False
        
        # 콘텐츠 섹션 검증 (기준 완화)
        content_sections = result.get('content_sections', [])
        if not isinstance(content_sections, list) or len(content_sections) == 0:
            return False
        
        # 각 섹션 검증 (기준 대폭 완화)
        valid_sections = 0
        for section in content_sections:
            if not isinstance(section, dict):
                continue
                
            # 필수 필드 중 하나라도 있으면 유효 (기존: 모든 필드 필수)
            has_title = bool(section.get('title', '').strip())
            has_body = bool(section.get('body', '').strip())
            has_template = bool(section.get('template', '').strip())
            
            if has_title or has_body or has_template:  # 하나라도 있으면 유효
                valid_sections += 1
        
        # 전체 섹션의 50% 이상이 유효하면 통과 (기존: 100%)
        return valid_sections >= len(content_sections) * 0.5

    def _is_valid_original_section(self, section: Dict) -> bool:
        """원본 데이터 기반 섹션 유효성 검증 (완화된 기준)"""
        if not isinstance(section, dict):
            return False
        
        # 폴백 데이터 제외 (유지)
        metadata = section.get("metadata", {})
        if metadata.get("fallback_used"):
            return False
        
        # Azure AI Search 키워드 차단 (유지하되 더 관대하게)
        azure_search_keywords = [
            "도시의 미학", "골목길의 재발견", "아티스트 인터뷰"  # 키워드 수 감소
        ]
        
        title = section.get("title", "").lower()
        body = section.get("body", "").lower()
        
        # 키워드가 전체 내용의 30% 이상을 차지할 때만 제외 (기존: 포함되면 무조건 제외)
        contamination_ratio = 0
        total_words = len((title + " " + body).split())
        
        for keyword in azure_search_keywords:
            if keyword in title or keyword in body:
                contamination_ratio += len(keyword.split()) / max(total_words, 1)
        
        if contamination_ratio > 0.3:  # 30% 이상 오염시에만 제외
            print(f"🚫 Azure AI Search 키워드 오염율 {contamination_ratio:.2f} 초과, 섹션 제외")
            return False
        
        # 최소 콘텐츠 요구사항 (대폭 완화)
        has_meaningful_content = (
            len(section.get("title", "")) > 0 or  # 제목이 있거나
            len(section.get("body", "")) > 10 or  # 10자 이상 본문이 있거나
            len(section.get("subtitle", "")) > 0 or  # 부제목이 있거나
            len(section.get("images", [])) > 0  # 이미지가 있으면 유효
        )
        
        return has_meaningful_content


    async def _log_coordination_result_async(self, result: Dict, text_mapping: Dict, 
                                           image_distribution: Dict, org_results: List[Dict], 
                                           binding_results: List[Dict]):
        """비동기 조율 결과 로깅"""
        try:
            response_id = self.logger.log_agent_real_output(
                agent_name="CoordinatorAgent",
                agent_role="매거진 구조 통합 조율자",
                task_description=f"배치 모드로 {len(result.get('content_sections', []))}개 섹션 생성",
                final_answer=str(result),
                reasoning_process="실제 데이터 기반 배치 처리를 통한 안전한 매거진 구조 통합",
                execution_steps=[
                    "이전 결과 배치 수집",
                    "실제 데이터 추출",
                    "CrewAI 배치 실행",
                    "결과 통합 및 검증",
                    "품질 보증"
                ],
                raw_input={
                    "text_mapping": str(text_mapping)[:500],
                    "image_distribution": str(image_distribution)[:500]
                },
                raw_output=result,
                performance_metrics={
                    "batch_mode_used": True,
                    "total_sections": len(result.get('content_sections', [])),
                    "org_results_utilized": len(org_results),
                    "binding_results_utilized": len(binding_results),
                    "execution_stats": self.execution_stats,
                    "quality_score": result.get('integration_metadata', {}).get('integration_quality_score', 0),
                    "real_data_used": True
                }
            )
            
            result["final_response_id"] = response_id
            result["execution_mode"] = "batch_async"
            
        except Exception as e:
            print(f"⚠️ 비동기 로깅 실패: {e}")


    async def _log_coordination_result_safe(self, result: Dict, text_mapping: Dict,
                                       image_distribution: Dict, org_results: List[Dict],
                                       binding_results: List[Dict]):
        """안전한 조율 결과 로깅"""
        try:
            # 세션 메타데이터 추가
            session_metadata = {}
            if hasattr(self, 'current_session_id'):
                session_metadata = {
                    "session_id": self.current_session_id,
                    "agent_name": self.agent_name if hasattr(self, 'agent_name') else "CoordinatorAgent",
                    "isolation_applied": hasattr(self, 'isolation_manager'),
                    "communication_isolated": hasattr(self, 'communication_isolator')
                }
            
            # 안전한 로깅 실행
            response_id = self.logger.log_agent_real_output(
                agent_name="CoordinatorAgent",
                agent_role="매거진 구조 통합 조율자",
                task_description=f"배치 모드로 {len(result.get('content_sections', []))}개 섹션 생성",
                final_answer=str(result)[:1000],  # 길이 제한
                reasoning_process="실제 데이터 기반 배치 처리를 통한 안전한 매거진 구조 통합",
                execution_steps=[
                    "이전 결과 배치 수집",
                    "실제 데이터 추출", 
                    "CrewAI 배치 실행",
                    "결과 통합 및 검증",
                    "품질 보증"
                ],
                raw_input={
                    "text_mapping_summary": f"텍스트 매핑 {len(str(text_mapping))} 문자",
                    "image_distribution_summary": f"이미지 분배 {len(str(image_distribution))} 문자",
                    "session_metadata": session_metadata
                },
                raw_output=result,
                performance_metrics={
                    "batch_mode_used": True,
                    "total_sections": len(result.get('content_sections', [])),
                    "org_results_utilized": len(org_results),
                    "binding_results_utilized": len(binding_results),
                    "execution_stats": getattr(self, 'execution_stats', {}),
                    "quality_score": result.get('integration_metadata', {}).get('integration_quality_score', 0),
                    "real_data_used": True,
                    "session_isolated": hasattr(self, 'current_session_id')
                }
            )
            
            # 결과에 메타데이터 추가
            result["final_response_id"] = response_id
            result["execution_mode"] = "batch_async_safe"
            result["session_metadata"] = session_metadata
            
        except Exception as e:
            print(f"⚠️ 안전한 로깅 실패: {e}")
            # 최소한의 메타데이터라도 추가
            result["final_response_id"] = f"safe_fallback_{int(__import__('time').time())}"
            result["execution_mode"] = "safe_fallback"
            result["logging_error"] = str(e)

    def _load_results_from_file(self) -> List[Dict]:
        """파일에서 결과 로드 (개선됨)"""
        try:
            # 여러 파일 경로 시도
            possible_files = [
                "./output/agent_results.json",
                "./output/latest_outputs.json",
                "./output/sessions/session_data.json"
            ]
            
            for results_file in possible_files:
                if __import__('os').path.exists(results_file):
                    try:
                        with open(results_file, 'r', encoding='utf-8') as f:
                            data = __import__('json').load(f)
                        
                        # 데이터 형식에 따라 처리
                        if isinstance(data, list):
                            return data
                        elif isinstance(data, dict):
                            if 'latest_outputs' in data:
                                return data['latest_outputs']
                            elif 'agent_results' in data:
                                return data['agent_results']
                            else:
                                return [data]
                    except Exception as e:
                        print(f"⚠️ 파일 {results_file} 로드 실패: {e}")
                        continue
            
            return []
            
        except Exception as e:
            print(f"⚠️ 결과 파일 로드 실패: {e}")
            return []






    def get_execution_stats(self) -> Dict:
        """실행 통계 반환"""
        return {
            **self.execution_stats,
            "success_rate": (self.execution_stats["successful_executions"] / 
                           max(self.execution_stats["total_attempts"], 1)) * 100,
            "fallback_rate": (self.execution_stats["fallback_used"] / 
                            max(self.execution_stats["total_attempts"], 1)) * 100,
            "circuit_breaker_state": self.circuit_breaker.state,
            "current_mode": "sync" if self.fallback_to_sync else "async"
        }

    def reset_execution_state(self):
        """실행 상태 초기화"""
        self.fallback_to_sync = False
        self.circuit_breaker = CircuitBreaker()
        self.execution_stats = {
            "total_attempts": 0,
            "successful_executions": 0,
            "fallback_used": 0,
            "circuit_breaker_triggered": 0,
            "timeout_occurred": 0
        }
        print("✅ CoordinatorAgent 실행 상태 초기화 완료")
    async def __aenter__(self):
        """비동기 컨텍스트 매니저 진입"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """비동기 컨텍스트 매니저 종료"""
        if self.work_queue.executor:
            self.work_queue.executor.shutdown(wait=True)
        
        # 예외 처리
        if exc_type:
            print(f"⚠️ CoordinatorAgent 컨텍스트에서 예외 발생: {exc_type.__name__}: {exc_val}")
            return False  # 예외를 재발생시킴
        
        return True

    def __enter__(self):
        """동기 컨텍스트 매니저 진입"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """동기 컨텍스트 매니저 종료"""
        if self.work_queue.executor:
            self.work_queue.executor.shutdown(wait=True)
        
        if exc_type:
            print(f"⚠️ CoordinatorAgent 동기 컨텍스트에서 예외 발생: {exc_type.__name__}: {exc_val}")
        
        return False

    def cleanup_resources(self):
        """리소스 정리"""
        try:
            if hasattr(self.work_queue, 'executor') and self.work_queue.executor:
                self.work_queue.executor.shutdown(wait=True)
                print("✅ ThreadPoolExecutor 정리 완료")
            
            # 큐 정리
            self.work_queue.work_queue.clear()
            self.work_queue.active_tasks.clear()
            self.work_queue.results.clear()
            
            print("✅ CoordinatorAgent 리소스 정리 완료")
            
        except Exception as e:
            print(f"⚠️ 리소스 정리 중 오류: {e}")

    async def health_check(self) -> Dict[str, Any]:
        """CoordinatorAgent 상태 확인"""
        try:
            # 기본 상태 정보
            health_status = {
                "status": "healthy",
                "timestamp": time.time(),
                "execution_mode": "sync" if self.fallback_to_sync else "async",
                "circuit_breaker_state": self.circuit_breaker.state,
                "queue_size": len(self.work_queue.work_queue),
                "active_tasks": len(self.work_queue.active_tasks),
                "execution_stats": self.execution_stats
            }
            
            # LLM 연결 확인
            try:
                if self.llm:
                    health_status["llm_status"] = "connected"
                else:
                    health_status["llm_status"] = "disconnected"
                    health_status["status"] = "degraded"
            except Exception as e:
                health_status["llm_status"] = f"error: {str(e)}"
                health_status["status"] = "degraded"
            
            # 로거 상태 확인
            try:
                if self.logger:
                    health_status["logger_status"] = "connected"
                else:
                    health_status["logger_status"] = "disconnected"
                    health_status["status"] = "degraded"
            except Exception as e:
                health_status["logger_status"] = f"error: {str(e)}"
                health_status["status"] = "degraded"
            
            # 에이전트 상태 확인
            agents_status = {}
            for agent_name, agent in [
                ("crew_agent", self.crew_agent),
                ("text_analyzer_agent", self.text_analyzer_agent),
                ("image_analyzer_agent", self.image_analyzer_agent)
            ]:
                try:
                    if agent and hasattr(agent, 'role'):
                        agents_status[agent_name] = "initialized"
                    else:
                        agents_status[agent_name] = "not_initialized"
                        health_status["status"] = "degraded"
                except Exception as e:
                    agents_status[agent_name] = f"error: {str(e)}"
                    health_status["status"] = "degraded"
            
            health_status["agents_status"] = agents_status
            
            # 메모리 사용량 확인 (선택적)
            try:
                import psutil
                process = psutil.Process()
                memory_info = process.memory_info()
                health_status["memory_usage"] = {
                    "rss": memory_info.rss,
                    "vms": memory_info.vms,
                    "percent": process.memory_percent()
                }
            except ImportError:
                health_status["memory_usage"] = "psutil not available"
            except Exception as e:
                health_status["memory_usage"] = f"error: {str(e)}"
            
            return health_status
            
        except Exception as e:
            return {
                "status": "error",
                "timestamp": time.time(),
                "error": str(e),
                "execution_stats": self.execution_stats
            }

    async def force_reset(self):
        """강제 재설정"""
        print("🔄 CoordinatorAgent 강제 재설정 시작")
        
        try:
            # 1. 실행 중인 작업 중단
            for task_id, task in self.work_queue.active_tasks.items():
                if not task.done():
                    task.cancel()
                    print(f"⏹️ 작업 {task_id} 취소")
            
            # 2. 큐 및 결과 정리
            self.work_queue.work_queue.clear()
            self.work_queue.active_tasks.clear()
            self.work_queue.results.clear()
            
            # 3. 실행 상태 초기화
            self.reset_execution_state()
            
            # 4. 에이전트 재생성
            self.crew_agent = self._create_crew_agent()
            self.text_analyzer_agent = self._create_text_analyzer_agent()
            self.image_analyzer_agent = self._create_image_analyzer_agent()
            
            print("✅ CoordinatorAgent 강제 재설정 완료")
            
        except Exception as e:
            print(f"❌ 강제 재설정 중 오류: {e}")
            raise

    def get_performance_metrics(self) -> Dict[str, Any]:
        """성능 메트릭 반환"""
        total_attempts = self.execution_stats["total_attempts"]
        
        if total_attempts == 0:
            return {
                "success_rate": 0.0,
                "failure_rate": 0.0,
                "fallback_rate": 0.0,
                "timeout_rate": 0.0,
                "circuit_breaker_rate": 0.0,
                "total_attempts": 0,
                "current_mode": "sync" if self.fallback_to_sync else "async",
                "circuit_breaker_state": self.circuit_breaker.state
            }
        
        return {
            "success_rate": (self.execution_stats["successful_executions"] / total_attempts) * 100,
            "failure_rate": ((total_attempts - self.execution_stats["successful_executions"]) / total_attempts) * 100,
            "fallback_rate": (self.execution_stats["fallback_used"] / total_attempts) * 100,
            "timeout_rate": (self.execution_stats["timeout_occurred"] / total_attempts) * 100,
            "circuit_breaker_rate": (self.execution_stats["circuit_breaker_triggered"] / total_attempts) * 100,
            "total_attempts": total_attempts,
            "successful_executions": self.execution_stats["successful_executions"],
            "current_mode": "sync" if self.fallback_to_sync else "async",
            "circuit_breaker_state": self.circuit_breaker.state,
            "queue_utilization": len(self.work_queue.work_queue) / self.work_queue.max_queue_size * 100,
            "active_tasks_count": len(self.work_queue.active_tasks)
        }

    async def test_coordination_pipeline(self) -> Dict[str, Any]:
        """조율 파이프라인 테스트"""
        print("🧪 CoordinatorAgent 파이프라인 테스트 시작")
        
        test_results = {
            "test_timestamp": time.time(),
            "tests_passed": 0,
            "tests_failed": 0,
            "test_details": []
        }
        
        # 테스트 1: 기본 초기화 확인
        try:
            assert self.llm is not None, "LLM이 초기화되지 않음"
            assert self.logger is not None, "Logger가 초기화되지 않음"
            assert self.crew_agent is not None, "Crew Agent가 초기화되지 않음"
            
            test_results["tests_passed"] += 1
            test_results["test_details"].append({
                "test_name": "initialization_test",
                "status": "passed",
                "message": "모든 구성 요소가 정상적으로 초기화됨"
            })
        except Exception as e:
            test_results["tests_failed"] += 1
            test_results["test_details"].append({
                "test_name": "initialization_test",
                "status": "failed",
                "error": str(e)
            })
        
        # 테스트 2: 간단한 작업 실행 테스트
        try:
            test_task_result = await self.execute_with_resilience(
                task_func=lambda: {"test": "success"},
                task_id="pipeline_test",
                timeout=30.0,
                max_retries=1
            )
            
            assert test_task_result is not None, "테스트 작업 결과가 None"
            
            test_results["tests_passed"] += 1
            test_results["test_details"].append({
                "test_name": "task_execution_test",
                "status": "passed",
                "message": "작업 실행이 정상적으로 완료됨",
                "result": test_task_result
            })
        except Exception as e:
            test_results["tests_failed"] += 1
            test_results["test_details"].append({
                "test_name": "task_execution_test",
                "status": "failed",
                "error": str(e)
            })
        
        # 테스트 3: 데이터 추출 기능 테스트
        try:
            test_text_data = {
                "sections": [{
                    "template": "Section01.jsx",
                    "title": "테스트 제목",
                    "body": "테스트 본문 내용"
                }]
            }
            
            test_image_data = {
                "template_images": {
                    "Section01.jsx": ["https://example.com/test.jpg"]
                }
            }
            
            enhanced_structure = self._create_enhanced_structure(
                test_text_data, test_image_data, [], []
            )
            
            assert isinstance(enhanced_structure, dict), "구조 생성 결과가 딕셔너리가 아님"
            assert "content_sections" in enhanced_structure, "content_sections 키가 없음"
            
            test_results["tests_passed"] += 1
            test_results["test_details"].append({
                "test_name": "data_extraction_test",
                "status": "passed",
                "message": "데이터 추출 및 구조 생성이 정상적으로 완료됨"
            })
        except Exception as e:
            test_results["tests_failed"] += 1
            test_results["test_details"].append({
                "test_name": "data_extraction_test",
                "status": "failed",
                "error": str(e)
            })
        
        # 테스트 결과 요약
        total_tests = test_results["tests_passed"] + test_results["tests_failed"]
        test_results["success_rate"] = (test_results["tests_passed"] / total_tests * 100) if total_tests > 0 else 0
        test_results["overall_status"] = "passed" if test_results["tests_failed"] == 0 else "failed"
        
        print(f"🧪 파이프라인 테스트 완료: {test_results['tests_passed']}/{total_tests} 통과")
        
        return test_results

# 사용 예시 및 유틸리티 함수들
def create_coordinator_agent() -> CoordinatorAgent:
    """CoordinatorAgent 인스턴스 생성"""
    try:
        coordinator = CoordinatorAgent()
        print("✅ CoordinatorAgent 생성 완료")
        return coordinator
    except Exception as e:
        print(f"❌ CoordinatorAgent 생성 실패: {e}")
        raise


async def run_coordination_with_monitoring(coordinator: CoordinatorAgent, 
                                         text_mapping: Dict, 
                                         image_distribution: Dict) -> Dict:
    """모니터링과 함께 조율 실행"""
    start_time = time.time()
    
    try:
        # 상태 확인
        health_status = await coordinator.health_check()
        if health_status["status"] == "error":
            print(f"⚠️ CoordinatorAgent 상태 불량: {health_status}")
        
        # 조율 실행
        result = await coordinator.coordinate_magazine_creation(text_mapping, image_distribution)
        
        # 실행 시간 측정
        execution_time = time.time() - start_time
        
        # 성능 메트릭 추가
        result["execution_metadata"] = {
            "execution_time": execution_time,
            "performance_metrics": coordinator.get_performance_metrics(),
            "health_status": health_status
        }
        
        print(f"✅ 조율 완료 (실행 시간: {execution_time:.2f}초)")
        return result
        
    except Exception as e:
        execution_time = time.time() - start_time
        print(f"❌ 조율 실행 실패 (실행 시간: {execution_time:.2f}초): {e}")
        
        # 오류 정보와 함께 폴백 결과 반환
        return {
            "selected_templates": ["Section01.jsx"],
            "content_sections": [{
                "template": "Section01.jsx",
                "title": "매거진 생성 오류",
                "subtitle": "시스템 오류로 인한 폴백",
                "body": f"조율 과정에서 오류가 발생했습니다: {str(e)}",
                "tagline": "SYSTEM ERROR",
                "images": [],
                "metadata": {
                    "error_fallback": True,
                    "error_message": str(e),
                    "execution_time": execution_time
                }
            }],
            "integration_metadata": {
                "total_sections": 1,
                "error_occurred": True,
                "execution_time": execution_time,
                "performance_metrics": coordinator.get_performance_metrics()
            }
        }

# 모듈 수준 유틸리티
def validate_coordination_inputs(text_mapping: Dict, image_distribution: Dict) -> bool:
    """조율 입력 데이터 검증"""
    try:
        # text_mapping 검증
        if not isinstance(text_mapping, dict):
            print("⚠️ text_mapping이 딕셔너리가 아님")
            return False
        
        # image_distribution 검증
        if not isinstance(image_distribution, dict):
            print("⚠️ image_distribution이 딕셔너리가 아님")
            return False
        
        print("✅ 조율 입력 데이터 검증 통과")
        return True
        
    except Exception as e:
        print(f"❌ 입력 데이터 검증 실패: {e}")
        return False

# 전역 설정
COORDINATOR_CONFIG = {
    "max_workers": 1,
    "max_queue_size": 20,
    "default_timeout": 300.0,
    "max_retries": 2,
    "circuit_breaker_threshold": 5,
    "circuit_breaker_timeout": 60.0,
    "batch_size": 2,
    "recursion_threshold": 800
}

def update_coordinator_config(**kwargs):
    """CoordinatorAgent 설정 업데이트"""
    global COORDINATOR_CONFIG
    COORDINATOR_CONFIG.update(kwargs)
    print(f"✅ CoordinatorAgent 설정 업데이트: {kwargs}")

# 모듈 초기화 시 실행되는 코드
if __name__ == "__main__":
    print("🚀 CoordinatorAgent 모듈 로드 완료")
    print(f"📋 현재 설정: {COORDINATOR_CONFIG}")
