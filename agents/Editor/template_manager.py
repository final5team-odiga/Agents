import sys
import asyncio
import os
import time
import concurrent.futures
from typing import Dict, List, Optional, Callable, Any
from collections import deque
from dataclasses import dataclass

from crewai import Agent, Task, Crew, Process
from custom_llm import get_azure_llm
from agents.Editor.OrgAgent import OrgAgent
from agents.Editor.BindingAgent import BindingAgent
from agents.Editor.CoordinatorAgent import CoordinatorAgent
from utils.pdf_vector_manager import PDFVectorManager
from utils.hybridlogging import get_hybrid_logger
try:
    from utils.ai_search_isolation import AISearchIsolationManager
    ISOLATION_AVAILABLE = True
except ImportError:
    print("⚠️ AI Search 격리 모듈을 찾을 수 없습니다. TemplateManager는 기본 모드로 실행됩니다.")
    ISOLATION_AVAILABLE = False


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
                
                if asyncio.iscoroutine(work_item.task_func):
                    # 코루틴 객체인 경우 인자를 전달할 수 없으므로 직접 실행
                    result = await asyncio.wait_for(work_item.task_func, timeout=work_item.timeout)
                elif asyncio.iscoroutinefunction(work_item.task_func):
                    # 코루틴 함수인 경우 인자와 함께 호출
                    result = await asyncio.wait_for(
                        work_item.task_func(*work_item.args, **work_item.kwargs),
                        timeout=work_item.timeout
                    )

                else:
                    result = await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(
                            self.executor,
                            lambda: work_item.task_func(*work_item.args, **work_item.kwargs)
                        ),
                        timeout=work_item.timeout
                    )
                
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
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0):  # 임계값 조정
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"
    
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

class MultiAgentTemplateManager:
    """PDF 벡터 데이터 기반 다중 에이전트 템플릿 관리자 (CrewAI 통합 로깅 시스템 - 비동기 처리, AI Search 격리 적용)"""
    
    def __init__(self, isolation_enabled=True):
        self.llm = get_azure_llm()
        self.org_agent = OrgAgent()
        self.binding_agent = BindingAgent()
        self.coordinator_agent = CoordinatorAgent()
        
        # PDF Vector Manager에 격리 시스템 적용
        self.vector_manager = PDFVectorManager(isolation_enabled=isolation_enabled)
        
        self.recursion_threshold = 600
        self.fallback_to_sync = False
        self.logger = get_hybrid_logger()

        # 새로운 복원력 시스템 추가
        self.work_queue = AsyncWorkQueue(max_workers=1, max_queue_size=20)
        self.circuit_breaker = CircuitBreaker()
        self.batch_size = 2

        # 실행 통계 추가
        self.execution_stats = {
            "total_attempts": 0,
            "successful_executions": 0,
            "fallback_used": 0,
            "circuit_breaker_triggered": 0,
            "timeout_occurred": 0
        }

        # AI Search 격리 시스템 초기화 (새로 추가)
        self.isolation_enabled = isolation_enabled and ISOLATION_AVAILABLE
        if self.isolation_enabled:
            self.isolation_manager = AISearchIsolationManager()
            print("🛡️ MultiAgentTemplateManager AI Search 격리 시스템 활성화")
        else:
            self.isolation_manager = None
            print("⚠️ MultiAgentTemplateManager AI Search 격리 시스템 비활성화")

        # CrewAI 에이전트들 생성
        self.vector_init_agent = self._create_vector_init_agent()
        self.template_loader_agent = self._create_template_loader_agent()
        self.requirement_analyzer_agent = self._create_requirement_analyzer_agent()
        self.data_prep_agent = self._create_data_prep_agent()
        self.coordination_agent = self._create_coordination_agent()

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
            print(f"⚠️ MultiAgentTemplateManager 재귀 깊이 {current_depth} 감지 - 동기 모드로 전환")
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
                timeout = 300 # 기본값

        print(f"DEBUG [execute_with_resilience]: task_id={task_id}, task_func type={type(task_func)}")

        if self.circuit_breaker.is_open():
            print(f"🚫 Circuit Breaker 열림 - 작업 {task_id} 건너뜀")
            return self._get_fallback_result(task_id)

        # 수정: 원래 함수와 인자를 WorkItem에 전달
        work_item = WorkItem(
            id=task_id,
            task_func=task_func, # 원래 함수 전달
            args=args, # 원래 인자 전달
            kwargs=kwargs, # 원래 키워드 인자 전달
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



    def _get_fallback_result(self, task_id: str) -> Any:
        """개선된 폴백 결과 생성"""
        self.execution_stats["fallback_used"] += 1
        
        if "vector_init" in task_id:
            return True
        elif "template" in task_id:
            return ["Section01.jsx", "Section03.jsx", "Section06.jsx"]
        elif "requirements" in task_id:
            return [{"template": "Section01.jsx", "image_requirements": {"total_estimated": 2}}]
        elif "magazine_data" in task_id or "_timeout" in task_id or "_exception" in task_id or "_type_error" in task_id or "_sync_fallback_exception" in task_id:
            reason = "unknown_error"
            if "_timeout" in task_id: reason = "timeout"
            elif "_exception" in task_id: reason = "exception"
            elif "_type_error" in task_id: reason = "type_error"
            elif "_sync_fallback_exception" in task_id: reason = "sync_fallback_exception"

            return {
                "selected_templates": ["Section01.jsx"],
                "content_sections": [{
                    "template": "Section01.jsx",
                    "title": "여행 매거진 (폴백)",
                    "subtitle": f"특별한 이야기 ({reason})",
                    "body": f"Circuit Breaker 또는 {reason}으로 인한 폴백 콘텐츠입니다. Task ID: {task_id}",
                    "tagline": "TRAVEL & CULTURE",
                    "images": [],
                    "metadata": {"fallback_used": True, "reason": reason, "task_id": task_id}
                }],
                "vector_enhanced": False,
                "fallback_mode": True,
                "error_info": {"task_id": task_id, "reason": reason}
            }
        else:
            return {"fallback": True, "task_id": task_id}

    def _create_vector_init_agent(self):
        """벡터 시스템 초기화 에이전트"""
        return Agent(
            role="PDF 벡터 시스템 초기화 전문가",
            goal="Azure Cognitive Search 기반 PDF 벡터 시스템을 안정적으로 초기화하고 템플릿 데이터를 효율적으로 처리",
            backstory="""당신은 10년간 검색 엔진 및 벡터 데이터베이스 시스템을 설계하고 운영해온 전문가입니다. Azure Cognitive Search, Elasticsearch, 그리고 다양한 벡터 데이터베이스 시스템의 최적화에 특화되어 있습니다.""",
            verbose=True,
            llm=self.llm,
            allow_delegation=False
        )

    def _create_template_loader_agent(self):
        """템플릿 로더 에이전트"""
        return Agent(
            role="JSX 템플릿 관리 및 로딩 전문가",
            goal="템플릿 폴더에서 JSX 파일들을 효율적으로 스캔하고 매거진 생성에 최적화된 템플릿 목록을 제공",
            backstory="""당신은 8년간 React 및 JSX 기반 웹 개발 프로젝트를 관리해온 전문가입니다. 템플릿 시스템 설계와 동적 컴포넌트 로딩에 특화되어 있습니다.""",
            verbose=True,
            llm=self.llm,
            allow_delegation=False
        )

    def _create_requirement_analyzer_agent(self):
        """요구사항 분석 에이전트"""
        return Agent(
            role="템플릿 요구사항 분석 전문가",
            goal="각 JSX 템플릿의 구조적 특성을 분석하여 이미지 요구사항과 레이아웃 스펙을 정확히 도출",
            backstory="""당신은 12년간 UI/UX 설계 및 템플릿 시스템 분석을 담당해온 전문가입니다. 다양한 레이아웃 패턴과 이미지 배치 최적화에 대한 깊은 이해를 보유하고 있습니다.""",
            verbose=True,
            llm=self.llm,
            allow_delegation=False
        )

    def _create_data_prep_agent(self):
        """데이터 준비 에이전트"""
        return Agent(
            role="매거진 데이터 준비 및 전처리 전문가",
            goal="매거진 생성에 필요한 모든 데이터를 수집, 정리, 검증하여 다중 에이전트 시스템이 효율적으로 작동할 수 있도록 준비",
            backstory="""당신은 15년간 출판업계에서 데이터 관리 및 전처리를 담당해온 전문가입니다. 복잡한 멀티미디어 데이터의 구조화와 품질 관리에 특화되어 있습니다.""",
            verbose=True,
            llm=self.llm,
            allow_delegation=False
        )

    def _create_coordination_agent(self):
        """조율 관리 에이전트"""
        return Agent(
            role="다중 에이전트 조율 및 프로세스 관리 전문가",
            goal="OrgAgent, BindingAgent, CoordinatorAgent의 순차적 실행을 관리하고 각 단계의 결과를 최적화하여 최고 품질의 매거진 데이터를 생성",
            backstory="""당신은 20년간 복잡한 소프트웨어 시스템의 프로젝트 관리와 다중 에이전트 조율을 담당해온 전문가입니다. 워크플로우 최적화와 품질 보증에 특화되어 있습니다.""",
            verbose=True,
            llm=self.llm,
            allow_delegation=False
        )

    async def initialize_vector_system(self, template_folder: str = "templates"):
        """벡터 시스템 초기화 - PDF 처리 및 인덱싱 (개선된 배치 기반 처리)"""
        # 재귀 깊이 확인 및 동기 모드 전환
        if self._should_use_sync():
            print("🔄 MultiAgentTemplateManager 벡터 초기화 동기 모드로 전환")
            return await self._initialize_vector_system_sync_mode(template_folder)

        try:
            # 개선된 배치 기반 비동기 모드 실행
            return await self._initialize_vector_system_batch_mode(template_folder)
        except RecursionError:
            print("🔄 MultiAgentTemplateManager 벡터 초기화 RecursionError 감지 - 동기 모드로 전환")
            self.fallback_to_sync = True
            return await self._initialize_vector_system_sync_mode(template_folder)

    async def _initialize_vector_system_batch_mode(self, template_folder: str):
        """개선된 배치 기반 벡터 시스템 초기화"""
        print("📦 벡터 시스템 초기화 배치 모드 시작")

        # 초기화 작업을 배치로 처리
        init_tasks = [
            ("crew_task", self._execute_init_crew_safe, template_folder),
            ("vector_init", self._execute_vector_init_safe, template_folder)  # ✅ 배치 처리에서 await 처리됨
        ]

        results = await self._process_init_batch(init_tasks)
        return results

    async def _process_init_batch(self, init_tasks: List[tuple]) -> Dict:
        batch_tasks = []
        for task_name, task_func, args in init_tasks:
            # args를 항상 튜플로 정규화
            if args is None:
                normalized_args = ()
            elif isinstance(args, (list, tuple)):
                normalized_args = tuple(args)
            else:
                # 단일 값인 경우 튜플로 변환
                normalized_args = (args,)
            
            task_result = await self.execute_with_resilience(
                task_func,  # task_func= 키워드 제거
                f"init_{task_name}",
                300.0,
                1,
                *normalized_args
            )
            batch_tasks.append((task_name, task_result))
        return {"tasks": batch_tasks}



    async def _execute_init_crew_safe(self, template_folder: str):
        """안전한 CrewAI 초기화 실행 (AI Search 격리 적용)"""
        try:
            # 격리된 초기화 설명 생성
            if self.isolation_enabled:
                clean_description = self.isolation_manager.clean_query_from_azure_keywords(
                    f"PDF 벡터 시스템을 초기화하고 템플릿 폴더 '{template_folder}'를 처리하세요."
                )
            else:
                clean_description = f"PDF 벡터 시스템을 초기화하고 템플릿 폴더 '{template_folder}'를 처리하세요."

            init_task = Task(
                description=f"""
    {clean_description}

    **초기화 단계:**
    1. Azure Cognitive Search 인덱스 상태 확인
    2. 기존 인덱스가 없으면 새로 생성
    3. PDF 템플릿 파일들을 스캔하고 벡터화
    4. 인덱스에 데이터 저장 및 검증
    5. AI Search 격리 시스템 적용 확인

    초기화 과정에서 발생하는 모든 단계를 상세히 보고하세요.
    AI Search 격리가 {'활성화' if self.isolation_enabled else '비활성화'}되었습니다.
    """,
                expected_output="벡터 시스템 초기화 완료 보고서 (성공/실패 상태, 상세 로그, 격리 시스템 상태 포함)",
                agent=self.vector_init_agent
            )

            init_crew = Crew(
                agents=[self.vector_init_agent],
                tasks=[init_task],
                process=Process.sequential,
                verbose=False
            )

            crew_result = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(None, init_crew.kickoff),
                timeout=300.0
            )

            # 격리 시스템 결과 검증
            if self.isolation_enabled:
                validated_result = self.isolation_manager.validate_original_preservation(
                    {"crew_result": str(crew_result)}, 
                    clean_description, 
                    "init_crew"
                )
                print(f"🛡️ 초기화 결과 격리 검증 완료: {validated_result.get('preservation_rate', 0):.2f}")

            return crew_result

        except Exception as e:
            print(f"⚠️ CrewAI 초기화 실행 실패: {e}")
            if self.isolation_enabled:
                print("🛡️ 격리된 폴백 초기화 결과 반환")
                return "격리된 초기화 폴백 완료"
            return "초기화 실패"

    async def _execute_vector_init_safe(self, template_folder: str):
        """안전한 벡터 초기화 실행"""
        try:
            await asyncio.gather(
                asyncio.get_event_loop().run_in_executor(None, self.vector_manager.initialize_search_index),
                asyncio.get_event_loop().run_in_executor(None, self.vector_manager.process_pdf_templates, template_folder)
            )
            return True
        except Exception as e:
            print(f"⚠️ 벡터 초기화 실패: {e}")
            return False

    async def _initialize_vector_system_sync_mode(self, template_folder: str):
        """동기 모드 벡터 시스템 초기화"""
        print("🔄 벡터 시스템 초기화 동기 모드 실행")
        
        try:
            await asyncio.get_event_loop().run_in_executor(
                None, self._initialize_vector_system_sync, template_folder
            )
            return True
        except Exception as e:
            print(f"❌ 벡터 시스템 초기화 동기 모드 실패: {e}")
            return False

    def _initialize_vector_system_sync(self, template_folder: str):
        """벡터 시스템 초기화 (동기 버전)"""
        try:
            self.vector_manager.initialize_search_index()
            self.vector_manager.process_pdf_templates(template_folder)
            print("✅ PDF 벡터 시스템 초기화 완료 (동기)")
        except Exception as e:
            print(f"❌ PDF 벡터 시스템 초기화 실패 (동기): {e}")
            raise e

    async def should_initialize_vector_system(self) -> bool:
        """벡터 시스템 초기화 필요 여부 확인 (개선된 배치 기반 처리)"""
        # 재귀 깊이 확인 및 동기 모드 전환
        if self._should_use_sync():
            print("🔄 벡터 시스템 상태 확인 동기 모드로 전환")
            return await self._should_initialize_vector_system_sync_mode()

        try:
            # 개선된 배치 기반 비동기 모드 실행
            return await self._should_initialize_vector_system_batch_mode()
        except RecursionError:
            print("🔄 벡터 시스템 상태 확인 RecursionError 감지 - 동기 모드로 전환")
            self.fallback_to_sync = True
            return await self._should_initialize_vector_system_sync_mode()

    async def _should_initialize_vector_system_batch_mode(self) -> bool:
        """배치 기반 벡터 시스템 상태 확인"""
        try:
            # CrewAI 태스크와 실제 확인을 병렬로 처리
            check_tasks = [
                ("crew_check", self._execute_check_crew_safe),
                ("index_check", self._check_index_exists_async),
                ("data_check", self._check_data_exists_async)
            ]

            results = await self._process_check_batch(check_tasks)
            
            # 데이터가 있으면 초기화 불필요
            if results.get("data_check"):
                await self._log_existing_system_found_async()
                print("✅ 기존 벡터 인덱스와 데이터 발견 - 초기화 생략")
                return False
            
            # 인덱스는 있지만 데이터 없음
            if results.get("index_check"):
                await self._log_index_exists_no_data_async()
                print("⚠️ 인덱스는 있지만 데이터 없음 - 초기화 필요")
                return True
            
            # 인덱스 없음
            await self._log_no_index_found_async()
            print("📄 벡터 인덱스 없음 - 초기화 필요")
            return True

        except Exception as e:
            await self._log_no_index_found_async()
            print(f"📄 벡터 시스템 확인 실패 - 초기화 필요: {e}")
            return True

    async def _process_check_batch(self, check_tasks: List[tuple]) -> Dict:
        batch_tasks = []
        for task_name, task_func, *args_list in check_tasks: # args 변수명 변경 (execute_with_resilience의 args와 구분)
            # args_list를 항상 튜플로 정규화
            normalized_args = tuple(args_list)

            task_result = await self.execute_with_resilience(
                task_func,  # task_func= 키워드 제거
                f"check_{task_name}",
                60.0,
                1,
                *normalized_args
            )

            batch_tasks.append((task_name, task_result))

        results = {}
        for task_name, result_value in batch_tasks: # 변수명 변경 (result는 예약어와 혼동 가능)
            try:
                results[task_name] = result_value
            except Exception as e:
                print(f"⚠️ 상태 확인 실패 {task_name}: {e}")
                results[task_name] = False
        return results

        


    async def _execute_check_crew_safe(self):
        """안전한 CrewAI 상태 확인 실행"""
        try:
            check_task = Task(
                description="""
벡터 시스템의 현재 상태를 확인하고 초기화가 필요한지 판단하세요.

**확인 항목:**
1. Azure Cognitive Search 인덱스 존재 여부
2. 인덱스 내 데이터 존재 여부
3. 벡터 검색 기능 정상 작동 여부

상태 확인 결과를 상세히 보고하세요.
""",
                expected_output="벡터 시스템 상태 확인 결과 및 초기화 필요성 판단",
                agent=self.vector_init_agent
            )

            check_crew = Crew(
                agents=[self.vector_init_agent],
                tasks=[check_task],
                process=Process.sequential,
                verbose=False
            )

            crew_result = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(None, check_crew.kickoff),
                timeout=60.0
            )

            return crew_result

        except Exception as e:
            print(f"⚠️ CrewAI 상태 확인 실행 실패: {e}")
            return "상태 확인 실패"

    async def _should_initialize_vector_system_sync_mode(self) -> bool:
        """동기 모드 벡터 시스템 상태 확인"""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._should_initialize_vector_system_sync
        )

    def _should_initialize_vector_system_sync(self) -> bool:
        """벡터 시스템 초기화 필요 여부 확인 (동기 버전)"""
        try:
            index_exists = self._check_index_exists_sync()
            data_exists = self._check_data_exists_sync()
            if data_exists:
                print("✅ 기존 벡터 인덱스와 데이터 발견 - 초기화 생략 (동기)")
                return False
            elif index_exists:
                print("⚠️ 인덱스는 있지만 데이터 없음 - 초기화 필요 (동기)")
                return True
            else:
                print("📄 벡터 인덱스 없음 - 초기화 필요 (동기)")
                return True
        except Exception as e:
            print(f"📄 벡터 시스템 확인 실패 - 초기화 필요 (동기): {e}")
            return True

    async def get_available_templates(self):
        """사용 가능한 템플릿 목록 (개선된 배치 기반 처리)"""
        # 재귀 깊이 확인 및 동기 모드 전환
        if self._should_use_sync():
            print("🔄 템플릿 목록 조회 동기 모드로 전환")
            return await self._get_available_templates_sync_mode()

        try:
            # 개선된 배치 기반 비동기 모드 실행
            return await self._get_available_templates_batch_mode()
        except RecursionError:
            print("🔄 템플릿 목록 조회 RecursionError 감지 - 동기 모드로 전환")
            self.fallback_to_sync = True
            return await self._get_available_templates_sync_mode()

    async def _get_available_templates_batch_mode(self):
        """배치 기반 템플릿 목록 조회"""
        # CrewAI 태스크와 실제 파일 스캔을 병렬로 처리
        template_tasks = [
            ("crew_scan", self._execute_template_crew_safe),
            ("file_scan", self._scan_template_files_async)
        ]

        results = await self._process_template_batch(template_tasks)
        
        # 실제 파일 스캔 결과 우선 사용
        final_templates = results.get("file_scan", ["Section01.jsx", "Section03.jsx", "Section06.jsx"])
        
        # 로깅
        templates_dir = "jsx_template"
        await self._log_templates_loaded_async(templates_dir, final_templates, final_templates)
        
        return final_templates

    async def _process_template_batch(self, template_tasks: List[tuple]) -> Dict:
        batch_tasks = []
        for task_name, task_func, *args_list in template_tasks: # args 변수명 변경
            # args_list를 항상 튜플로 정규화
            normalized_args = tuple(args_list)

            task_result = await self.execute_with_resilience(
                task_func,  # task_func= 키워드 제거
                f"template_{task_name}",
                60.0,
                1,
                *normalized_args
            )
            batch_tasks.append((task_name, task_result))

        results = {}
        for task_name, result_value in batch_tasks: # 변수명 변경
            try:
                results[task_name] = result_value
            except Exception as e:
                print(f"⚠️ 템플릿 작업 실패 {task_name}: {e}")
                results[task_name] = ["Section01.jsx", "Section03.jsx", "Section06.jsx"] # 폴백 값 유지
        return results



    async def _execute_template_crew_safe(self):
        """안전한 CrewAI 템플릿 스캔 실행 (AI Search 격리 적용)"""
        try:
            # 격리된 템플릿 스캔 설명 생성
            base_description = """
    템플릿 폴더에서 사용 가능한 JSX 템플릿 파일들을 스캔하고 목록을 생성하세요.

    **스캔 요구사항:**
    1. 'jsx_template' 폴더 존재 여부 확인
    2. .jsx 확장자를 가진 파일들 검색
    3. 파일명 유효성 검증
    4. 기본 템플릿 목록 준비 (폴더가 없는 경우)

    스캔 결과를 상세히 보고하세요.
    """

            if self.isolation_enabled:
                clean_description = self.isolation_manager.clean_query_from_azure_keywords(base_description)
                clean_description += f"\nAI Search 격리 시스템이 활성화되어 있습니다."
            else:
                clean_description = base_description

            template_task = Task(
                description=clean_description,
                expected_output="사용 가능한 JSX 템플릿 파일 목록 (격리 시스템 적용)",
                agent=self.template_loader_agent
            )

            template_crew = Crew(
                agents=[self.template_loader_agent],
                tasks=[template_task],
                process=Process.sequential,
                verbose=False
            )

            crew_result = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(None, template_crew.kickoff),
                timeout=60.0
            )

            # 격리 시스템 결과 검증
            if self.isolation_enabled:
                if self.isolation_manager.is_contaminated(str(crew_result), "template_crew"):
                    print("🚫 템플릿 스캔 결과에서 AI Search 오염 감지")
                    return "격리된 템플릿 스캔 폴백"

            return crew_result

        except Exception as e:
            print(f"⚠️ CrewAI 템플릿 스캔 실행 실패: {e}")
            if self.isolation_enabled:
                print("🛡️ 격리된 폴백 템플릿 스캔 결과 반환")
                return "격리된 템플릿 스캔 폴백"
            return "템플릿 스캔 실패"


    async def _scan_template_files_async(self):
        """비동기 템플릿 파일 스캔"""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._get_available_templates_sync
        )

    async def _get_available_templates_sync_mode(self):
        """동기 모드 템플릿 목록 조회"""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._get_available_templates_sync
        )

    def _get_available_templates_sync(self):
        """사용 가능한 템플릿 목록 (동기 버전)"""
        templates_dir = "jsx_template"
        if not os.path.exists(templates_dir):
            return ["Section01.jsx", "Section03.jsx", "Section06.jsx", "Section08.jsx"]
        template_files = [f for f in os.listdir(templates_dir) if f.endswith('.jsx')]
        return template_files if template_files else ["Section01.jsx", "Section03.jsx", "Section06.jsx"]

    async def analyze_template_requirements(self, template_files: List[str]) -> List[Dict]:
        """템플릿 요구사항 분석 (개선된 배치 기반 처리)"""
        # 재귀 깊이 확인 및 동기 모드 전환
        if self._should_use_sync():
            print("🔄 템플릿 요구사항 분석 동기 모드로 전환")
            return await self._analyze_template_requirements_sync_mode(template_files)

        try:
            # 개선된 배치 기반 비동기 모드 실행
            return await self._analyze_template_requirements_batch_mode(template_files)
        except RecursionError:
            print("🔄 템플릿 요구사항 분석 RecursionError 감지 - 동기 모드로 전환")
            self.fallback_to_sync = True
            return await self._analyze_template_requirements_sync_mode(template_files)

    async def _analyze_template_requirements_batch_mode(self, template_files: List[str]) -> List[Dict]:
        """배치 기반 템플릿 요구사항 분석"""
        # CrewAI 분석과 기본 분석을 병렬로 처리
        analysis_tasks = [
            ("crew_analysis", self._execute_analysis_crew_safe, template_files),
            ("basic_analysis", self._analyze_requirements_sync, template_files)
        ]

        results = await self._process_analysis_batch(analysis_tasks)
        
        # 기본 분석 결과 사용 (더 안정적)
        requirements = results.get("basic_analysis", [])
        
        # 로깅
        await self._log_requirements_analysis_async(template_files, requirements)
        
        return requirements

    async def _process_analysis_batch(self, analysis_tasks: List[tuple]) -> Dict:
        batch_tasks = []
        for task_name, task_func, *args_list in analysis_tasks: # args 변수명 변경
            # args_list를 항상 튜플로 정규화
            normalized_args = tuple(args_list)

            task_result = await self.execute_with_resilience(
                task_func,  # task_func= 키워드 제거
                f"analysis_{task_name}",
                120.0,
                1,
                *normalized_args
            )
            batch_tasks.append((task_name, task_result))

        results = {}
        for task_name, result_value in batch_tasks: # 변수명 변경
            try:
                results[task_name] = result_value
            except Exception as e:
                print(f"⚠️ 분석 작업 실패 {task_name}: {e}")
                if task_name == "basic_analysis": # 폴백 값 유지
                    results[task_name] = [{"template": "Section01.jsx", "image_requirements": {"total_estimated": 2}}]
                # 다른 task_name에 대한 폴백이 필요하면 여기에 추가
        return results




    async def _execute_analysis_crew_safe(self, template_files: List[str]):
        """안전한 CrewAI 분석 실행 (AI Search 격리 적용)"""
        try:
            # 격리된 분석 설명 생성
            base_description = f"""
    제공된 {len(template_files)}개의 JSX 템플릿 파일들을 분석하여 각각의 요구사항을 도출하세요.

    **분석 대상 템플릿:** {', '.join(template_files)}

    **분석 항목:**
    1. 각 템플릿의 이미지 요구사항 (메인 이미지, 서브 이미지)
    2. 예상 이미지 개수
    3. 레이아웃 특성 및 구조적 요구사항

    분석 결과를 상세히 보고하세요.
    """

            if self.isolation_enabled:
                clean_description = self.isolation_manager.clean_query_from_azure_keywords(base_description)
                # 템플릿 파일 목록도 격리 검증
                clean_template_files = []
                for template_file in template_files:
                    if not self.isolation_manager.is_contaminated(template_file, "template_file"):
                        clean_template_files.append(template_file)
                    else:
                        print(f"🚫 오염된 템플릿 파일 제외: {template_file}")
                
                if not clean_template_files:
                    print("⚠️ 모든 템플릿 파일이 격리됨 - 기본 템플릿 사용")
                    clean_template_files = ["Section01.jsx", "Section03.jsx", "Section06.jsx"]
                
                clean_description += f"\nAI Search 격리 시스템 적용: {len(template_files)} → {len(clean_template_files)}개 템플릿"
            else:
                clean_description = base_description
                clean_template_files = template_files

            analysis_task = Task(
                description=clean_description,
                expected_output="템플릿별 요구사항 분석 결과 목록 (격리 시스템 적용)",
                agent=self.requirement_analyzer_agent
            )

            analysis_crew = Crew(
                agents=[self.requirement_analyzer_agent],
                tasks=[analysis_task],
                process=Process.sequential,
                verbose=False
            )

            crew_result = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(None, analysis_crew.kickoff),
                timeout=120.0
            )

            # 격리 시스템 결과 검증
            if self.isolation_enabled:
                validated_result = self.isolation_manager.validate_original_preservation(
                    {"crew_result": str(crew_result)}, 
                    clean_description, 
                    "analysis_crew"
                )
                
                if validated_result.get("contamination_detected", False):
                    print("🚫 분석 결과에서 AI Search 오염 감지 - 기본 분석 사용")
                    return self._get_isolated_analysis_fallback(clean_template_files)

            return crew_result

        except Exception as e:
            print(f"⚠️ CrewAI 분석 실행 실패: {e}")
            if self.isolation_enabled:
                print("🛡️ 격리된 폴백 분석 결과 반환")
                return self._get_isolated_analysis_fallback(template_files)
            return "분석 실패"

    def _get_isolated_analysis_fallback(self, template_files: List[str]) -> str:
        """격리된 분석 폴백 결과"""
        fallback_analysis = []
        for template_file in template_files:
            if "Section01" in template_file or "Section03" in template_file:
                image_count = 2
            elif "Section06" in template_file or "Section08" in template_file:
                image_count = 3
            else:
                image_count = 1
            
            fallback_analysis.append(f"{template_file}: {image_count}개 이미지 요구")
        
        return f"격리된 분석 결과: {', '.join(fallback_analysis)}"


    async def _analyze_template_requirements_sync_mode(self, template_files: List[str]) -> List[Dict]:
        """동기 모드 템플릿 요구사항 분석"""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._analyze_requirements_sync, template_files
        )

    async def create_magazine_data(self, magazine_content, image_analysis_results: List[Dict]) -> Dict:
        """재귀 깊이 감지 후 동기/비동기 선택적 실행 (개선된 배치 기반 처리)"""
        print("=== PDF 벡터 기반 매거진 생성 시작 ===")
        
        self.execution_stats["total_attempts"] += 1
        
        # 시스템 상태 사전 점검
        await self._perform_health_check()
        
        # 재귀 깊이 확인 및 동기 모드 전환
        if self._should_use_sync():
            print("🔄 매거진 생성 동기 모드로 전환")
            return await self._create_magazine_data_sync_mode(magazine_content, image_analysis_results)

        try:
            # 개선된 배치 기반 비동기 모드 실행
            return await self._create_magazine_data_batch_mode(magazine_content, image_analysis_results)
        except RecursionError:
            print("🔄 매거진 생성 RecursionError 감지 - 동기 모드로 전환")
            self.fallback_to_sync = True
            return await self._create_magazine_data_sync_mode(magazine_content, image_analysis_results)
        except Exception as e:
            print(f"❌ 매거진 생성 중 예외 발생: {e} - 동기 모드로 폴백 시도")
            self.fallback_to_sync = True
            return await self._create_magazine_data_sync_mode(magazine_content, image_analysis_results)

    async def _create_magazine_data_batch_mode(self, magazine_content, image_analysis_results: List[Dict]) -> Dict:
        """개선된 배치 기반 매거진 데이터 생성"""
        print("📦 매거진 생성 배치 모드 시작")

        # 전체 프로세스 시작 로깅
        await self._log_process_start_async(magazine_content, image_analysis_results)

        # 1. 벡터 시스템 확인 (순차)
        try:
            should_init = await asyncio.wait_for(
                self.should_initialize_vector_system(),
                timeout=120.0
            )
            if should_init:
                print("\n=== PDF 벡터 시스템 초기화 ===")
                await asyncio.wait_for(
                    self.initialize_vector_system("templates"),
                    timeout=300.0
                )
            else:
                print("\n=== 기존 벡터 데이터 사용 ===")
        except asyncio.TimeoutError:
            print("⚠️ 벡터 시스템 확인/초기화 타임아웃 - 기본 모드로 진행")
        except Exception as e_vec_init:
            print(f"⚠️ 벡터 시스템 확인/초기화 중 예외 발생: {e_vec_init} - 기본 모드로 진행")

        # 2. 기본 데이터 준비 (배치 처리)
        print("\n=== 기본 데이터 준비 (배치) ===")
        try:
            prep_results = await self._prepare_basic_data_batch(image_analysis_results)
        except Exception as e_prep:
            print(f"⚠️ 데이터 준비 실패: {e_prep}")
            prep_results = self._get_minimal_prep_data()

        available_templates = prep_results.get("templates", ["Section01.jsx"])
        template_requirements = prep_results.get("requirements", [])
        image_urls = prep_results.get("image_urls", [])
        image_locations = prep_results.get("image_locations", [])

        # 데이터 준비 완료 로깅
        await self._log_data_prep_complete_async(available_templates, image_urls, template_requirements)

        # 3. 에이전트들을 안전한 배치로 순차 실행 (재시도 로직 포함)
        print("\n=== 안전한 배치 에이전트 처리 시작 ===")
        try:
            agent_results = await self._execute_agents_with_retry(
                magazine_content, available_templates, image_urls, image_locations, template_requirements
            )
            
            if self._validate_agent_results(agent_results):
                self.execution_stats["successful_executions"] += 1
            else:
                print("⚠️ 에이전트 결과 검증 실패 - 기본 결과 생성")
                agent_results = self._create_basic_magazine_result(
                    magazine_content, available_templates, image_urls
                )
        except Exception as e_agent_exec:
            print(f"⚠️ 에이전트 배치 실행 실패: {e_agent_exec}")
            agent_results = self._create_basic_magazine_result(
                magazine_content, available_templates, image_urls
            )

        # 4. 메타데이터 추가
        final_template_data = agent_results
        final_template_data["vector_enhanced"] = True
        final_template_data["crewai_enhanced"] = True
        final_template_data["batch_processed"] = True
        final_template_data["execution_mode"] = "batch_async"
        final_template_data["safe_execution"] = True

        # 최종 완료 로깅
        await self._log_final_complete_async(final_template_data)

        print("✅ PDF 벡터 기반 배치 매거진 데이터 생성 완료")
        return final_template_data

    async def _execute_agents_with_retry(self, magazine_content, available_templates: List[str],
                                       image_urls: List[str], image_locations: List[str],
                                       template_requirements: List[Dict]) -> Dict:
        """재시도 로직이 있는 에이전트 실행"""
        
        # OrgAgent 실행 (재시도 포함)
        text_mapping = await self._execute_org_agent_with_retry(magazine_content, available_templates)
        await self._log_org_agent_complete_async(text_mapping)

        # BindingAgent 실행 (재시도 포함)
        image_distribution = await self._execute_binding_agent_with_retry(
            image_urls, image_locations, template_requirements
        )
        await self._log_binding_agent_complete_async(image_distribution)

        # CoordinatorAgent 실행 (재시도 포함)
        final_template_data = await self._execute_coordinator_agent_with_retry(
            text_mapping, image_distribution
        )
        await self._log_coordinator_agent_complete_async(final_template_data)

        return final_template_data

    async def _execute_org_agent_with_retry(self, magazine_content, available_templates: List[str]) -> Dict:
        """재시도 로직이 있는 OrgAgent 실행"""
        max_retries = 2
        
        for attempt in range(max_retries + 1):
            try:
                print(f"🔄 OrgAgent 실행 시도 {attempt + 1}/{max_retries + 1}")
                
                result = await self._run_agent_with_depth_check_safe('org', self.org_agent.process_content, magazine_content, available_templates)
                
                if self._validate_org_result(result):
                    print("✅ OrgAgent 성공")
                    return result
                else:
                    print("⚠️ OrgAgent 결과 검증 실패")
                    if attempt < max_retries:
                        await asyncio.sleep(5)
                        continue
                    
            except RecursionError:
                print(f"⚠️ OrgAgent 실행 중 재귀 깊이 초과 (시도 {attempt + 1}) - 동기 모드 전환 필요")
                self.fallback_to_sync = True
                raise
            except Exception as e:
                print(f"❌ OrgAgent 실행 실패 (시도 {attempt + 1}): {e}")
                if attempt < max_retries:
                    await asyncio.sleep(5)
                    continue
        
        print("⚠️ OrgAgent 모든 시도 실패 - 기본 결과 생성")
        self.execution_stats["fallback_used"] += 1
        return self._create_basic_org_result(magazine_content, available_templates)

    async def _execute_binding_agent_with_retry(self, image_urls: List[str], 
                                              image_locations: List[str], 
                                              template_requirements: List[Dict]) -> Dict:
        """재시도 로직이 있는 BindingAgent 실행"""
        max_retries = 2
        
        for attempt in range(max_retries + 1):
            try:
                print(f"🔄 BindingAgent 실행 시도 {attempt + 1}/{max_retries + 1}")
                
                result = await self._run_agent_with_depth_check_safe('binding', self.binding_agent.process_images, image_urls, image_locations, template_requirements)
                
                if self._validate_binding_result(result):
                    print("✅ BindingAgent 성공")
                    return result
                else:
                    print("⚠️ BindingAgent 결과 검증 실패")
                    if attempt < max_retries:
                        await asyncio.sleep(5)
                        continue
                        
            except RecursionError:
                print(f"⚠️ BindingAgent 실행 중 재귀 깊이 초과 (시도 {attempt + 1}) - 동기 모드 전환 필요")
                self.fallback_to_sync = True
                raise
            except Exception as e:
                print(f"❌ BindingAgent 실행 실패 (시도 {attempt + 1}): {e}")
                if attempt < max_retries:
                    await asyncio.sleep(5)
                    continue
        
        print("⚠️ BindingAgent 모든 시도 실패 - 기본 결과 생성")
        self.execution_stats["fallback_used"] += 1
        return self._create_basic_binding_result(image_urls, template_requirements)

    async def _execute_coordinator_agent_with_retry(self, text_mapping: Dict, image_distribution: Dict) -> Dict:
        """재시도 로직이 있는 CoordinatorAgent 실행"""
        max_retries = 2
        
        for attempt in range(max_retries + 1):
            try:
                print(f"🔄 CoordinatorAgent 실행 시도 {attempt + 1}/{max_retries + 1}")
                
                result = await self._run_agent_with_depth_check_safe('coordinator', self.coordinator_agent.coordinate_magazine_creation, text_mapping, image_distribution)
                
                if self._validate_coordinator_result(result):
                    print("✅ CoordinatorAgent 성공")
                    return result
                else:
                    print("⚠️ CoordinatorAgent 결과 검증 실패")
                    if attempt < max_retries:
                        await asyncio.sleep(5)
                        continue
                        
            except RecursionError:
                print(f"⚠️ CoordinatorAgent 실행 중 재귀 깊이 초과 (시도 {attempt + 1}) - 동기 모드 전환 필요")
                self.fallback_to_sync = True
                raise
            except Exception as e:
                print(f"❌ CoordinatorAgent 실행 실패 (시도 {attempt + 1}): {e}")
                if attempt < max_retries:
                    await asyncio.sleep(5)
                    continue
        
        print("⚠️ CoordinatorAgent 모든 시도 실패 - 기본 결과 생성")
        self.execution_stats["fallback_used"] += 1
        return self._create_basic_coordinator_result(text_mapping, image_distribution)

    async def _run_agent_with_depth_check_safe(self, agent_name: str, agent_method: Callable, *args):
        current_depth = self._check_recursion_depth()
        print(f"DEBUG [{agent_name}]: 현재 재귀 깊이: {current_depth}, 임계값: {self.recursion_threshold}")
        print(f"DEBUG [{agent_name}]: agent_method type: {type(agent_method)}, is coroutine function: {asyncio.iscoroutinefunction(agent_method)}")

        if current_depth > self.recursion_threshold:
            print(f"⚠️ [{agent_name}] 재귀 깊이 초과 ({current_depth}) - 동기 폴백 실행은 상위에서 처리되어야 함. 예외 발생.")
            raise RecursionError(f"[{agent_name}] 재귀 깊이 초과로 동기 모드 전환 필요")

        timeout_map = {
            'org': 600, 'binding': 900, 'coordinator': 600
        }
        timeout = timeout_map.get(agent_name, 300)

        try:
            print(f"🔄 [{agent_name}] 에이전트 안전 실행 시작 (타임아웃: {timeout}초)")
            if asyncio.iscoroutinefunction(agent_method):
                coro_obj = agent_method(*args)
                print(f"DEBUG [{agent_name}]: 생성된 코루틴 객체: {type(coro_obj)}")
                result = await asyncio.wait_for(coro_obj, timeout=timeout)
            else:
                result = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(None, agent_method, *args),
                    timeout=timeout
                )
            print(f"✅ [{agent_name}] 에이전트 안전 실행 완료")
            return result
        except asyncio.TimeoutError:
            print(f"⏰ [{agent_name}] 에이전트 타임아웃 ({timeout}초) - 폴백 결과 반환")
            self.execution_stats["timeout_occurred"] += 1
            self.execution_stats["fallback_used"] += 1
            return self._get_fallback_result(f"{agent_name}_timeout")
        except TypeError as te:
            print(f"❌ [{agent_name}] 에이전트 호출 시 TypeError 발생: {te}")
            print(f"DEBUG [{agent_name}]: agent_method: {agent_method}, args: {args}")
            if asyncio.iscoroutine(agent_method):
                print(f"ℹ️ [{agent_name}] agent_method가 코루틴 객체입니다. 직접 await 시도합니다.")
                try:
                    result = await asyncio.wait_for(agent_method, timeout=timeout)
                    print(f"✅ [{agent_name}] 에이전트 (코루틴 객체 직접 실행) 완료")
                    return result
                except Exception as e_coro_direct:
                    print(f"❌ [{agent_name}] 코루틴 객체 직접 실행 실패: {e_coro_direct}")
            self.execution_stats["fallback_used"] += 1
            return self._get_fallback_result(f"{agent_name}_type_error")
        except Exception as e:
            print(f"❌ [{agent_name}] 에이전트 실행 중 일반 예외 발생: {e}")
            self.execution_stats["fallback_used"] += 1
            return self._get_fallback_result(f"{agent_name}_exception")

    async def _prepare_basic_data_batch(self, image_analysis_results: List[Dict]) -> Dict:
        """기본 데이터 준비 배치 처리"""
        prep_tasks = [
            ("templates", self.get_available_templates),
            ("image_data", self._extract_image_data_safe, image_analysis_results)
        ]

        # 템플릿과 이미지 데이터를 병렬로 준비
        results = {}
        for task_name, task_func, *args in prep_tasks:
            try:
                if args:
                    result = await task_func(*args)  # ✅ 이미 올바름
                else:
                    result = await task_func()  # ✅ 이미 올바름
                results[task_name] = result
            except Exception as e:
                print(f"⚠️ 데이터 준비 실패 {task_name}: {e}")
                if task_name == "templates":
                    results[task_name] = ["Section01.jsx"]
                else:
                    results[task_name] = {"image_urls": [], "image_locations": []}

        # 템플릿 요구사항 분석
        templates = results.get("templates", ["Section01.jsx"])
        try:
            requirements = await self.analyze_template_requirements(templates)
            results["requirements"] = requirements
        except Exception as e:
            print(f"⚠️ 요구사항 분석 실패: {e}")
            results["requirements"] = [{"template": "Section01.jsx", "image_requirements": {"total_estimated": 2}}]

        # 이미지 데이터 추출
        image_data = results.get("image_data", {})
        results["image_urls"] = image_data.get("image_urls", [])
        results["image_locations"] = image_data.get("image_locations", [])

        return results

    async def _extract_image_data_safe(self, image_analysis_results: List[Dict]) -> Dict:
        """안전한 이미지 데이터 추출"""
        try:
            image_urls = [result.get('image_url', '') for result in image_analysis_results if result.get('image_url')]
            image_locations = [result.get('location', '') for result in image_analysis_results if result.get('location')]
            
            return {
                "image_urls": image_urls,
                "image_locations": image_locations
            }
        except Exception as e:
            print(f"⚠️ 이미지 데이터 추출 실패: {e}")
            return {"image_urls": [], "image_locations": []}

    def _validate_org_result(self, result: Dict) -> bool:
        """OrgAgent 결과 검증"""
        if not isinstance(result, dict):
            return False
        
        required_fields = ["text_mapping", "total_sections"]
        for field in required_fields:
            if field not in result:
                return False
        
        text_mapping = result.get("text_mapping", {})
        if isinstance(text_mapping, dict) and "text_mapping" in text_mapping:
            sections = text_mapping["text_mapping"]
            if isinstance(sections, list) and len(sections) > 0:
                for section in sections:
                    if not isinstance(section, dict):
                        return False
                    if not all(key in section for key in ["template", "title", "body"]):
                        return False
                return True
        
        return False

    def _validate_binding_result(self, result: Dict) -> bool:
        """BindingAgent 결과 검증"""
        if not isinstance(result, dict):
            return False
        return "image_distribution" in result

    def _validate_coordinator_result(self, result: Dict) -> bool:
        """CoordinatorAgent 결과 검증"""
        if not isinstance(result, dict):
            return False
        
        required_fields = ["content_sections", "selected_templates"]
        for field in required_fields:
            if field not in result:
                return False
        
        content_sections = result.get("content_sections", [])
        if not isinstance(content_sections, list) or len(content_sections) == 0:
            return False
        
        for section in content_sections:
            if not isinstance(section, dict):
                return False
            required_section_fields = ["template", "title", "subtitle", "body", "tagline"]
            if not all(field in section for field in required_section_fields):
                return False
        
        return True

    def _validate_agent_results(self, agent_results: Dict) -> bool:
        """에이전트 결과 전체 검증"""
        if not isinstance(agent_results, dict):
            return False
        
        required_fields = ["content_sections", "selected_templates"]
        for field in required_fields:
            if field not in agent_results:
                return False
        
        content_sections = agent_results.get("content_sections", [])
        if not isinstance(content_sections, list) or len(content_sections) == 0:
            return False
        
        return True

    def _create_basic_org_result(self, magazine_content, available_templates: List[str]) -> Dict:
        """기본 OrgAgent 결과 생성"""
        content_text = str(magazine_content)
        
        sections = []
        for i, template in enumerate(available_templates[:3]):
            section = {
                "template": template,
                "title": f"여행 이야기 {i+1}",
                "subtitle": "특별한 순간들",
                "body": content_text[:500] if content_text else "여행의 아름다운 순간들을 담은 이야기입니다.",
                "tagline": "TRAVEL & CULTURE",
                "layout_source": "basic_fallback"
            }
            sections.append(section)
        
        return {
            "text_mapping": {"text_mapping": sections},
            "total_sections": len(sections),
            "vector_enhanced": False,
            "execution_mode": "basic_fallback",
            "agent_responses": []
        }

    def _create_basic_binding_result(self, image_urls: List[str], template_requirements: List[Dict]) -> Dict:
        """기본 BindingAgent 결과 생성"""
        image_distribution = {}
        
        if image_urls and template_requirements:
            images_per_template = len(image_urls) // len(template_requirements)
            remainder = len(image_urls) % len(template_requirements)
            
            start_idx = 0
            for i, template_req in enumerate(template_requirements):
                template_name = template_req.get("template", f"Section{i+1:02d}.jsx")
                
                end_idx = start_idx + images_per_template
                if i < remainder:
                    end_idx += 1
                
                assigned_images = image_urls[start_idx:end_idx]
                image_distribution[template_name] = assigned_images
                start_idx = end_idx
        
        return {
            "image_distribution": image_distribution,
            "vector_enhanced": False,
            "execution_mode": "basic_fallback",
            "template_distributions": []
        }

    def _create_basic_coordinator_result(self, text_mapping: Dict, image_distribution: Dict) -> Dict:
        """기본 CoordinatorAgent 결과 생성"""
        content_sections = []
        
        text_sections = []
        if isinstance(text_mapping, dict) and "text_mapping" in text_mapping:
            text_sections = text_mapping["text_mapping"]
            if isinstance(text_sections, dict) and "text_mapping" in text_sections:
                text_sections = text_sections["text_mapping"]
        
        images_by_template = {}
        if isinstance(image_distribution, dict) and "image_distribution" in image_distribution:
            images_by_template = image_distribution["image_distribution"]
        
        for section in text_sections:
            if isinstance(section, dict):
                template = section.get("template", "Section01.jsx")
                template_images = images_by_template.get(template, [])
                
                content_section = {
                    "template": template,
                    "title": section.get("title", "여행 이야기"),
                    "subtitle": section.get("subtitle", "특별한 순간들"),
                    "body": section.get("body", "여행의 아름다운 순간들을 담은 이야기입니다."),
                    "tagline": section.get("tagline", "TRAVEL & CULTURE"),
                    "images": template_images,
                    "metadata": {
                        "basic_fallback": True,
                        "content_source": "basic_generation"
                    }
                }
                content_sections.append(content_section)
        
        if not content_sections:
            content_sections = [{
                "template": "Section01.jsx",
                "title": "여행 매거진",
                "subtitle": "특별한 이야기",
                "body": "여행의 특별한 순간들을 담은 매거진입니다.",
                "tagline": "TRAVEL & CULTURE",
                "images": list(images_by_template.values())[0] if images_by_template else [],
                "metadata": {"basic_fallback": True}
            }]
        
        return {
            "selected_templates": [section["template"] for section in content_sections],
            "content_sections": content_sections,
            "integration_metadata": {
                "total_sections": len(content_sections),
                "basic_fallback": True,
                "integration_quality_score": 0.7
            }
        }

    def _create_basic_magazine_result(self, magazine_content, available_templates: List[str], image_urls: List[str]) -> Dict:
        """기본 매거진 결과 생성"""
        content_text = str(magazine_content)
        
        content_sections = []
        for i, template in enumerate(available_templates[:3]):
            # 이미지 분배
            start_idx = i * (len(image_urls) // len(available_templates))
            end_idx = start_idx + (len(image_urls) // len(available_templates))
            template_images = image_urls[start_idx:end_idx]
            
            section = {
                "template": template,
                "title": f"여행 이야기 {i+1}",
                "subtitle": "특별한 순간들",
                "body": content_text[:500] if content_text else "여행의 아름다운 순간들을 담은 이야기입니다.",
                "tagline": "TRAVEL & CULTURE",
                "images": template_images,
                "metadata": {"basic_fallback": True}
            }
            content_sections.append(section)
        
        return {
            "selected_templates": [section["template"] for section in content_sections],
            "content_sections": content_sections,
            "integration_metadata": {
                "total_sections": len(content_sections),
                "basic_fallback": True,
                "integration_quality_score": 0.7
            }
        }

    def _get_minimal_prep_data(self) -> Dict:
        """최소한의 준비 데이터 생성"""
        return {
            "templates": ["Section01.jsx", "Section03.jsx", "Section06.jsx"],
            "requirements": [
                {"template": "Section01.jsx", "image_requirements": {"total_estimated": 2}},
                {"template": "Section03.jsx", "image_requirements": {"total_estimated": 2}},
                {"template": "Section06.jsx", "image_requirements": {"total_estimated": 3}}
            ],
            "image_urls": [],
            "image_locations": []
        }

    async def _perform_health_check(self):
        """시스템 상태 점검"""
        if self.circuit_breaker.state == "OPEN":
            print("🔄 Circuit Breaker가 열려있음 - 리셋 시도")
            self.circuit_breaker.failure_count = max(0, self.circuit_breaker.failure_count - 2)
            if self.circuit_breaker.failure_count < self.circuit_breaker.failure_threshold:
                self.circuit_breaker.state = "CLOSED"
                print("✅ Circuit Breaker 리셋 완료")

    async def _create_magazine_data_sync_mode(self, magazine_content, image_analysis_results: List[Dict]) -> Dict:
        """동기 모드 매거진 데이터 생성"""
        print("🔄 매거진 생성 동기 모드 실행")
        
        loop = asyncio.get_event_loop()

        # 기본 데이터 준비 (동기)
        available_templates = await loop.run_in_executor(None, self._get_available_templates_sync)
        template_requirements = await loop.run_in_executor(None, self._analyze_requirements_sync, available_templates)
        image_urls = [result.get('image_url', '') for result in image_analysis_results if result.get('image_url')]
        image_locations = [result.get('location', '') for result in image_analysis_results if result.get('location')]

        # 에이전트들을 동기 모드로 실행
        text_mapping = await loop.run_in_executor(
            None, self._run_org_agent_sync, magazine_content, available_templates
        )

        image_distribution = await loop.run_in_executor(
            None, self._run_binding_agent_sync, image_urls, image_locations, template_requirements
        )

        final_template_data = await loop.run_in_executor(
            None, self._run_coordinator_agent_sync, text_mapping, image_distribution
        )

        # 메타데이터 추가
        final_template_data["vector_enhanced"] = True
        final_template_data["sync_processed"] = True
        final_template_data["execution_mode"] = "sync_fallback"
        final_template_data["recursion_fallback"] = True

        print("✅ 동기 모드 매거진 데이터 생성 완료")
        return final_template_data

    def _run_org_agent_sync(self, magazine_content, available_templates):
        """동기 버전 OrgAgent 실행"""
        try:
            if hasattr(self.org_agent, 'process_content_sync'):
                return self.org_agent.process_content_sync(magazine_content, available_templates)
            else:
                print("⚠️ OrgAgent의 동기 메서드(process_content_sync)를 찾을 수 없음. 기본 폴백 사용.")
                return self._create_basic_org_result(magazine_content, available_templates)
        except Exception as e:
            print(f"⚠️ OrgAgent 동기 실행 실패: {e}")
            return self._create_basic_org_result(magazine_content, available_templates)

    def _run_binding_agent_sync(self, image_urls, image_locations, template_requirements):
        """동기 버전 BindingAgent 실행"""
        try:
            if hasattr(self.binding_agent, 'process_images_sync'):
                return self.binding_agent.process_images_sync(image_urls, image_locations, template_requirements)
            else:
                print("⚠️ BindingAgent의 동기 메서드(process_images_sync)를 찾을 수 없음. 기본 폴백 사용.")
                return self._create_basic_binding_result(image_urls, template_requirements)
        except Exception as e:
            print(f"⚠️ BindingAgent 동기 실행 실패: {e}")
            return self._create_basic_binding_result(image_urls, template_requirements)

    def _run_coordinator_agent_sync(self, text_mapping, image_distribution):
        """동기 버전 CoordinatorAgent 실행"""
        try:
            if hasattr(self.coordinator_agent, 'coordinate_magazine_creation_sync'):
                return self.coordinator_agent.coordinate_magazine_creation_sync(text_mapping, image_distribution)
            else:
                print("⚠️ CoordinatorAgent의 동기 메서드(coordinate_magazine_creation_sync)를 찾을 수 없음. 기본 폴백 사용.")
                return self._create_basic_coordinator_result(text_mapping, image_distribution)
        except Exception as e:
            print(f"⚠️ CoordinatorAgent 동기 실행 실패: {e}")
            return self._create_basic_coordinator_result(text_mapping, image_distribution)

    def _analyze_requirements_sync(self, template_files: List[str]) -> List[Dict]:
        """템플릿 요구사항 분석 (동기 버전)"""
        requirements = []
        for template_file in template_files:
            if "Section01" in template_file or "Section03" in template_file:
                image_count = 2
            elif "Section06" in template_file or "Section08" in template_file:
                image_count = 3
            else:
                image_count = 1

            requirement = {
                "template": template_file,
                "image_requirements": {
                    "total_estimated": image_count,
                    "main_image": 1,
                    "sub_images": image_count - 1,
                    "layout_type": "standard"
                }
            }
            requirements.append(requirement)

        return requirements

    async def _check_index_exists_async(self) -> bool:
        """인덱스 존재 여부 확인 (비동기)"""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._check_index_exists_sync
        )

    async def _check_data_exists_async(self) -> bool:
        """데이터 존재 여부 확인 (비동기)"""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._check_data_exists_sync
        )

    def _check_index_exists_sync(self) -> bool:
        """인덱스 존재 여부 확인 (동기)"""
        try:
            index_client = self.vector_manager.search_index_client
            index_client.get_index(self.vector_manager.search_index_name)
            return True
        except:
            return False

    def _check_data_exists_sync(self) -> bool:
        """데이터 존재 여부 확인 (동기)"""
        try:
            search_client = self.vector_manager.search_client
            results = search_client.search("*", top=1)
            for _ in results:
                return True
            return False
        except:
            return False

    # 로깅 메서드들
    async def _log_process_start_async(self, magazine_content, image_analysis_results: List[Dict]):
        """전체 프로세스 시작 로깅 (비동기)"""
        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.logger.log_agent_real_output(
                agent_name="MultiAgentTemplateManager",
                agent_role="PDF 벡터 데이터 기반 다중 에이전트 템플릿 관리자",
                task_description="CrewAI 기반 배치 매거진 생성 프로세스 시작",
                final_answer="PDF 벡터 기반 다중 에이전트 매거진 생성 시작",
                reasoning_process="CrewAI 통합 로깅 시스템과 배치 처리를 통한 안전한 매거진 데이터 생성 프로세스 시작",
                execution_steps=[
                    "입력 데이터 검증",
                    "벡터 시스템 상태 확인",
                    "다중 에이전트 배치 처리 준비"
                ],
                raw_input={
                    "magazine_content_length": len(str(magazine_content)),
                    "image_analysis_count": len(image_analysis_results)
                },
                performance_metrics={
                    "process_started": True,
                    "input_images": len(image_analysis_results),
                    "crewai_integration": True,
                    "batch_processing": True,
                    "safe_execution": True
                }
            )
        )

    async def _log_initialization_complete_async(self, template_folder: str, crew_result):
        """초기화 완료 로깅 (비동기)"""
        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.logger.log_agent_real_output(
                agent_name="MultiAgentTemplateManager_VectorInit",
                agent_role="벡터 시스템 초기화 관리자",
                task_description="PDF 벡터 시스템 초기화 완료",
                final_answer="PDF 벡터 시스템 초기화 성공적으로 완료",
                reasoning_process="CrewAI 기반 배치 Azure Cognitive Search 인덱스 생성 및 PDF 템플릿 벡터화 완료",
                execution_steps=[
                    "CrewAI 초기화 태스크 배치 실행 완료",
                    "Azure Cognitive Search 인덱스 생성 완료",
                    "PDF 템플릿 처리 및 벡터화 완료",
                    "벡터 시스템 활성화"
                ],
                raw_output={
                    "initialization_success": True,
                    "crew_result": str(crew_result)[:500]
                },
                performance_metrics={
                    "initialization_completed": True,
                    "vector_system_active": True,
                    "template_folder_processed": template_folder,
                    "crewai_execution_success": True,
                    "batch_processing": True
                }
            )
        )




    async def _log_existing_system_found_async(self):
        """기존 시스템 발견 로깅 (비동기)"""
        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.logger.log_agent_real_output(
                agent_name="MultiAgentTemplateManager_VectorCheck",
                agent_role="벡터 시스템 상태 확인자",
                task_description="기존 벡터 시스템 발견",
                final_answer="기존 PDF 벡터 시스템 및 데이터 발견 - 초기화 생략",
                reasoning_process="CrewAI 기반 배치 Azure Cognitive Search 인덱스 및 벡터 데이터 존재 확인",
                execution_steps=[
                    "인덱스 존재 여부 확인",
                    "벡터 데이터 존재 여부 확인",
                    "시스템 상태 검증 완료"
                ],
                performance_metrics={
                    "existing_system_found": True,
                    "initialization_skipped": True,
                    "vector_data_available": True,
                    "batch_processing": True
                }
            )
        )

    async def _log_index_exists_no_data_async(self):
        """인덱스 존재하지만 데이터 없음 로깅 (비동기)"""
        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.logger.log_agent_real_output(
                agent_name="MultiAgentTemplateManager_VectorCheck",
                agent_role="벡터 시스템 상태 확인자",
                task_description="인덱스 존재하지만 데이터 없음",
                final_answer="Azure Cognitive Search 인덱스는 존재하지만 벡터 데이터 없음 - 초기화 필요",
                reasoning_process="CrewAI 기반 배치 인덱스 구조는 있으나 실제 PDF 벡터 데이터 부재 확인",
                execution_steps=[
                    "인덱스 존재 확인",
                    "벡터 데이터 부재 확인",
                    "초기화 필요성 판단"
                ],
                performance_metrics={
                    "index_exists": True,
                    "data_missing": True,
                    "initialization_required": True,
                    "batch_processing": True
                }
            )
        )

    async def _log_no_index_found_async(self):
        """인덱스 없음 로깅 (비동기)"""
        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.logger.log_agent_real_output(
                agent_name="MultiAgentTemplateManager_VectorCheck",
                agent_role="벡터 시스템 상태 확인자",
                task_description="벡터 인덱스 없음",
                final_answer="Azure Cognitive Search 인덱스 없음 - 전체 초기화 필요",
                reasoning_process="CrewAI 기반 배치 벡터 시스템 인프라 부재 확인",
                execution_steps=[
                    "인덱스 부재 확인",
                    "전체 시스템 초기화 필요성 판단"
                ],
                performance_metrics={
                    "index_missing": True,
                    "full_initialization_required": True,
                    "batch_processing": True
                }
            )
        )

    async def _log_templates_loaded_async(self, templates_dir: str, template_files: List[str], crew_result):
        """템플릿 로드 완료 로깅 (비동기, 격리 정보 포함)"""
        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.logger.log_agent_real_output(
                agent_name="MultiAgentTemplateManager_TemplateLoader",
                agent_role="JSX 템플릿 관리 및 로딩 전문가",
                task_description=f"{templates_dir} 폴더에서 JSX 템플릿 로딩",
                final_answer=f"{len(template_files)}개 JSX 템플릿 로딩 완료: {', '.join(template_files)}",
                reasoning_process="CrewAI 기반 배치 템플릿 폴더 스캔 및 JSX 파일 목록 생성 (AI Search 격리 적용)",
                execution_steps=[
                    "템플릿 폴더 존재 여부 확인",
                    "JSX 파일 스캔 실행",
                    "AI Search 격리 필터링 적용",
                    "템플릿 목록 생성 완료"
                ],
                raw_output={
                    "template_files": template_files,
                    "crew_result": str(crew_result)[:300]
                },
                performance_metrics={
                    "templates_loaded": len(template_files),
                    "template_directory": templates_dir,
                    "crewai_execution_success": True,
                    "batch_processing": True,
                    "ai_search_isolation": {
                        "isolation_enabled": self.isolation_enabled,
                        "contamination_filtered": True if self.isolation_enabled else False
                    }
                }
            )
        )

    async def _log_requirements_analysis_async(self, template_files: List[str], requirements: List[Dict]):
        """요구사항 분석 완료 로깅 (비동기, 격리 정보 포함)"""
        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.logger.log_agent_real_output(
                agent_name="MultiAgentTemplateManager_RequirementAnalyzer",
                agent_role="템플릿 요구사항 분석 전문가",
                task_description=f"{len(template_files)}개 JSX 템플릿 요구사항 분석",
                final_answer=f"템플릿 요구사항 분석 완료: {len(requirements)}개 템플릿 분석",
                reasoning_process="CrewAI 기반 배치 각 JSX 템플릿의 구조적 특성 및 이미지 요구사항 도출 (AI Search 격리 적용)",
                execution_steps=[
                    "템플릿 구조 분석",
                    "이미지 요구사항 도출",
                    "AI Search 격리 검증",
                    "레이아웃 스펙 정의"
                ],
                raw_input={"template_files": template_files},
                raw_output={"requirements": requirements},
                performance_metrics={
                    "templates_analyzed": len(template_files),
                    "requirements_generated": len(requirements),
                    "analysis_depth": "comprehensive",
                    "crewai_enhanced": True,
                    "batch_processing": True,
                    "ai_search_isolation": {
                        "isolation_enabled": self.isolation_enabled,
                        "contamination_report": self.isolation_manager.get_contamination_report() if self.isolation_enabled else {}
                    }
                }
            )
        )


    async def _log_data_prep_complete_async(self, available_templates: List[str], image_urls: List[str], template_requirements: List[Dict]):
        """데이터 준비 완료 로깅 (비동기)"""
        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.logger.log_agent_real_output(
                agent_name="MultiAgentTemplateManager_DataPrep",
                agent_role="매거진 데이터 준비 및 전처리 전문가",
                task_description="CrewAI 기반 배치 매거진 생성 데이터 준비 완료",
                final_answer=f"데이터 준비 완료: {len(available_templates)}개 템플릿, {len(image_urls)}개 이미지",
                reasoning_process="CrewAI 기반 배치 템플릿 로딩, 요구사항 분석, 이미지 데이터 추출 완료",
                execution_steps=[
                    "템플릿 목록 로딩",
                    "템플릿 요구사항 분석",
                    "이미지 데이터 추출",
                    "다중 에이전트 실행 준비"
                ],
                raw_output={
                    "available_templates": available_templates,
                    "image_count": len(image_urls),
                    "template_requirements": len(template_requirements)
                },
                performance_metrics={
                    "templates_prepared": len(available_templates),
                    "images_prepared": len(image_urls),
                    "requirements_analyzed": len(template_requirements),
                    "data_prep_completed": True,
                    "crewai_enhanced": True,
                    "batch_processing": True
                }
            )
        )

    async def _log_org_agent_complete_async(self, text_mapping: Dict):
        """OrgAgent 완료 로깅 (비동기)"""
        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.logger.log_agent_real_output(
                agent_name="MultiAgentTemplateManager_OrgAgent",
                agent_role="OrgAgent 실행 관리자",
                task_description="OrgAgent 텍스트 처리 완료",
                final_answer=f"OrgAgent 처리 완료: {text_mapping.get('total_sections', 0)}개 섹션 생성",
                reasoning_process="CrewAI 기반 배치 PDF 벡터 데이터를 활용한 텍스트 배치 및 구조화 완료",
                execution_steps=[
                    "OrgAgent 안전 실행",
                    "텍스트 구조 분석",
                    "템플릿 매핑 완료"
                ],
                raw_output={
                    "text_mapping_summary": {
                        "total_sections": text_mapping.get('total_sections', 0),
                        "execution_mode": text_mapping.get('execution_mode', 'unknown'),
                        "vector_enhanced": text_mapping.get('vector_enhanced', False)
                    }
                },
                performance_metrics={
                    "org_agent_completed": True,
                    "sections_generated": text_mapping.get('total_sections', 0),
                    "safe_execution": True,
                    "batch_processing": True
                }
            )
        )

    async def _log_binding_agent_complete_async(self, image_distribution: Dict):
        """BindingAgent 완료 로깅 (비동기)"""
        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.logger.log_agent_real_output(
                agent_name="MultiAgentTemplateManager_BindingAgent",
                agent_role="BindingAgent 실행 관리자",
                task_description="BindingAgent 이미지 처리 완료",
                final_answer=f"BindingAgent 처리 완료: 이미지 분배 완료",
                reasoning_process="CrewAI 기반 배치 PDF 벡터 데이터를 활용한 이미지 배치 및 분배 완료",
                execution_steps=[
                    "BindingAgent 안전 실행",
                    "이미지 분석 및 배치",
                    "템플릿별 이미지 분배 완료"
                ],
                raw_output={
                    "image_distribution_summary": {
                        "execution_mode": image_distribution.get('execution_mode', 'unknown'),
                        "vector_enhanced": image_distribution.get('vector_enhanced', False),
                        "templates_processed": len(image_distribution.get('template_distributions', []))
                    }
                },
                performance_metrics={
                    "binding_agent_completed": True,
                    "image_distribution_completed": True,
                    "safe_execution": True,
                    "batch_processing": True
                }
            )
        )

    async def _log_coordinator_agent_complete_async(self, final_template_data: Dict):
        """CoordinatorAgent 완료 로깅 (비동기)"""
        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.logger.log_agent_real_output(
                agent_name="MultiAgentTemplateManager_CoordinatorAgent",
                agent_role="CoordinatorAgent 실행 관리자",
                task_description="CoordinatorAgent 결과 통합 완료",
                final_answer=f"CoordinatorAgent 처리 완료: {len(final_template_data.get('content_sections', []))}개 최종 섹션 생성",
                reasoning_process="CrewAI 기반 배치 OrgAgent와 BindingAgent 결과 통합 및 최종 매거진 구조 생성 완료",
                execution_steps=[
                    "CoordinatorAgent 안전 실행",
                    "텍스트-이미지 통합",
                    "최종 매거진 구조 생성 완료"
                ],
                raw_output={
                    "final_data_summary": {
                        "content_sections": len(final_template_data.get('content_sections', [])),
                        "selected_templates": len(final_template_data.get('selected_templates', [])),
                        "integration_quality": final_template_data.get('integration_metadata', {}).get('integration_quality_score', 0)
                    }
                },
                performance_metrics={
                    "coordinator_agent_completed": True,
                    "final_sections_generated": len(final_template_data.get('content_sections', [])),
                    "integration_completed": True,
                    "safe_execution": True,
                    "batch_processing": True
                }
            )
        )

    async def _log_final_complete_async(self, final_template_data: Dict):
        """최종 완료 로깅 (비동기)"""
        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.logger.log_agent_real_output(
                agent_name="MultiAgentTemplateManager",
                agent_role="PDF 벡터 데이터 기반 다중 에이전트 템플릿 관리자",
                task_description="CrewAI 기반 배치 매거진 생성 프로세스 완료",
                final_answer=f"PDF 벡터 기반 배치 매거진 데이터 생성 성공적으로 완료: {len(final_template_data.get('content_sections', []))}개 섹션",
                reasoning_process="CrewAI 통합 로깅 시스템과 배치 처리를 통한 안전한 다중 에이전트 협업으로 고품질 매거진 데이터 생성 완료",
                execution_steps=[
                    "벡터 시스템 초기화/확인 완료",
                    "템플릿 및 요구사항 분석 완료",
                    "OrgAgent 텍스트 처리 완료",
                    "BindingAgent 이미지 처리 완료",
                    "CoordinatorAgent 통합 완료",
                    "최종 매거진 데이터 생성 완료"
                ],
                raw_output=final_template_data,
                performance_metrics={
                    "process_completed": True,
                    "total_sections": len(final_template_data.get('content_sections', [])),
                    "vector_enhanced": final_template_data.get('vector_enhanced', False),
                    "crewai_enhanced": final_template_data.get('crewai_enhanced', False),
                    "batch_processed": final_template_data.get('batch_processed', False),
                    "safe_execution": final_template_data.get('safe_execution', False),
                    "execution_mode": final_template_data.get('execution_mode', 'unknown'),
                    "all_agents_completed": True
                }
            )
        )

    # 동기 버전 메서드들 (호환성 유지)
    def create_magazine_data_sync(self, magazine_content, image_analysis_results: List[Dict]) -> Dict:
        """동기 버전 매거진 데이터 생성 (호환성 유지)"""
        return asyncio.run(self.create_magazine_data(magazine_content, image_analysis_results))

    def initialize_vector_system_sync(self, template_folder: str = "templates") -> bool:
        """동기 버전 벡터 시스템 초기화 (호환성 유지)"""
        return asyncio.run(self.initialize_vector_system(template_folder))

    def should_initialize_vector_system_sync(self) -> bool:
        """동기 버전 벡터 시스템 초기화 필요 여부 확인 (호환성 유지)"""
        return asyncio.run(self.should_initialize_vector_system())

    def get_available_templates_sync(self) -> List[str]:
        """동기 버전 템플릿 목록 조회 (호환성 유지)"""
        return asyncio.run(self.get_available_templates())

    def analyze_template_requirements_sync(self, template_files: List[str]) -> List[Dict]:
        """동기 버전 템플릿 요구사항 분석 (호환성 유지)"""
        return asyncio.run(self.analyze_template_requirements(template_files))

    # 디버깅 및 모니터링 메서드
    def get_execution_statistics(self) -> Dict:
        """실행 통계 조회"""
        return {
            **self.execution_stats,
            "success_rate": (
                self.execution_stats["successful_executions"] / 
                max(self.execution_stats["total_attempts"], 1)
            ) * 100,
            "circuit_breaker_state": self.circuit_breaker.state,
            "current_queue_size": len(self.work_queue.work_queue)
        }

    def reset_system_state(self) -> None:
        """시스템 상태 리셋"""
        print("🔄 MultiAgentTemplateManager 시스템 상태 리셋")
        
        # Circuit Breaker 리셋
        self.circuit_breaker.failure_count = 0
        self.circuit_breaker.state = "CLOSED"
        self.circuit_breaker.last_failure_time = None
        
        # 폴백 플래그 리셋
        self.fallback_to_sync = False
        
        # 작업 큐 클리어
        self.work_queue.work_queue.clear()
        self.work_queue.active_tasks.clear()
        self.work_queue.results.clear()
        
        # 통계 초기화
        self.execution_stats = {
            "total_attempts": 0,
            "successful_executions": 0,
            "fallback_used": 0,
            "circuit_breaker_triggered": 0,
            "timeout_occurred": 0
        }
        
        print("✅ 시스템 상태가 리셋되었습니다.")

    def get_performance_metrics(self) -> Dict:
        """성능 메트릭 수집 (격리 정보 포함)"""
        base_metrics = {
            "circuit_breaker": {
                "state": self.circuit_breaker.state,
                "failure_count": self.circuit_breaker.failure_count,
                "failure_threshold": self.circuit_breaker.failure_threshold
            },
            "work_queue": {
                "max_workers": self.work_queue.max_workers,
                "queue_size": len(self.work_queue.work_queue),
                "active_tasks": len(self.work_queue.active_tasks),
                "completed_results": len(self.work_queue.results)
            },
            "system": {
                "recursion_threshold": self.recursion_threshold,
                "fallback_to_sync": self.fallback_to_sync,
                "batch_size": self.batch_size,
                "current_depth": self._check_recursion_depth()
            },
            "agents": {
                "org_agent_ready": hasattr(self.org_agent, 'process_content'),
                "binding_agent_ready": hasattr(self.binding_agent, 'process_images'),
                "coordinator_agent_ready": hasattr(self.coordinator_agent, 'coordinate_magazine_creation')
            }
        }
        
        # AI Search 격리 정보 추가
        if self.isolation_enabled:
            base_metrics["ai_search_isolation"] = {
                "isolation_enabled": True,
                "contamination_report": self.isolation_manager.get_contamination_report(),
                "vector_manager_isolated": hasattr(self.vector_manager, 'isolation_manager')
            }
        else:
            base_metrics["ai_search_isolation"] = {
                "isolation_enabled": False,
                "reason": "isolation_module_unavailable" if not ISOLATION_AVAILABLE else "manually_disabled"
            }
        
        return base_metrics

    def debug_system_state(self) -> Dict:
        """시스템 상태 디버깅"""
        return {
            "circuit_breaker_state": self.circuit_breaker.state,
            "failure_count": self.circuit_breaker.failure_count,
            "work_queue_size": len(self.work_queue.work_queue),
            "active_tasks": len(self.work_queue.active_tasks),
            "recursion_fallback_active": self.fallback_to_sync,
            "recursion_threshold": self.recursion_threshold,
            "current_recursion_depth": self._check_recursion_depth(),
            "batch_size": self.batch_size
        }

    def monitor_agent_health(self) -> Dict:
        """에이전트 건강 상태 모니터링 (격리 정보 포함)"""
        health_status = {
            "org_agent_available": hasattr(self.org_agent, 'process_content'),
            "binding_agent_available": hasattr(self.binding_agent, 'process_images'),
            "coordinator_agent_available": hasattr(self.coordinator_agent, 'coordinate_magazine_creation'),
            "vector_manager_available": hasattr(self.vector_manager, 'search_similar_layouts'),
            "logger_available": hasattr(self.logger, 'log_agent_real_output'),
            "crewai_agents_created": all([
                hasattr(self, 'vector_init_agent'),
                hasattr(self, 'template_loader_agent'),
                hasattr(self, 'requirement_analyzer_agent'),
                hasattr(self, 'data_prep_agent'),
                hasattr(self, 'coordination_agent')
            ]),
            "system_status": "healthy"
        }

        # AI Search 격리 상태 확인
        if self.isolation_enabled:
            health_status["ai_search_isolation"] = {
                "status": "active",
                "manager_available": self.isolation_manager is not None,
                "vector_manager_isolated": hasattr(self.vector_manager, 'isolation_manager')
            }
        else:
            health_status["ai_search_isolation"] = {
                "status": "inactive",
                "reason": "module_unavailable" if not ISOLATION_AVAILABLE else "disabled"
            }

        # 건강 상태 평가
        if self.circuit_breaker.state == "OPEN":
            health_status["system_status"] = "degraded"
        elif self.fallback_to_sync:
            health_status["system_status"] = "fallback_mode"
        elif not all([health_status["org_agent_available"], health_status["binding_agent_available"], health_status["coordinator_agent_available"]]):
            health_status["system_status"] = "agents_unavailable"
        elif self.isolation_enabled and not health_status["ai_search_isolation"]["manager_available"]:
            health_status["system_status"] = "isolation_degraded"

        return health_status

    def validate_system_integrity(self) -> bool:
        """시스템 무결성 검증"""
        try:
            # 필수 컴포넌트 확인
            required_components = [
                self.llm,
                self.org_agent,
                self.binding_agent,
                self.coordinator_agent,
                self.vector_manager,
                self.logger
            ]
            
            for component in required_components:
                if component is None:
                    return False
            
            # CrewAI 에이전트들 확인
            crewai_agents = [
                self.vector_init_agent,
                self.template_loader_agent,
                self.requirement_analyzer_agent,
                self.data_prep_agent,
                self.coordination_agent
            ]
            
            for agent in crewai_agents:
                if agent is None:
                    return False
            
            # 복원력 시스템 확인
            if self.work_queue is None or self.circuit_breaker is None:
                return False
            
            return True
            
        except Exception as e:
            print(f"⚠️ 시스템 무결성 검증 실패: {e}")
            return False

    def get_system_info(self) -> Dict:
        """시스템 정보 조회"""
        return {
            "class_name": self.__class__.__name__,
            "version": "2.0_batch_resilient",
            "features": [
                "PDF 벡터 데이터 기반 처리",
                "CrewAI 통합 로깅 시스템",
                "비동기 배치 처리",
                "Circuit Breaker 패턴",
                "재귀 깊이 감지 및 폴백",
                "안전한 에이전트 실행",
                "복원력 있는 작업 큐"
            ],
            "agents": {
                "core_agents": ["OrgAgent", "BindingAgent", "CoordinatorAgent"],
                "crewai_agents": [
                    "VectorInitAgent",
                    "TemplateLoaderAgent", 
                    "RequirementAnalyzerAgent",
                    "DataPrepAgent",
                    "CoordinationAgent"
                ]
            },
            "execution_modes": ["batch_async", "sync_fallback"],
            "safety_features": [
                "재귀 깊이 모니터링",
                "타임아웃 처리",
                "Circuit Breaker",
                "점진적 백오프",
                "폴백 메커니즘"
            ]
        }
