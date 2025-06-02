import os
import re
import asyncio
import time
import sys
import inspect
import logging
from utils.hybridlogging import get_hybrid_logger
from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

from crewai import Agent, Task, Crew, Process
from custom_llm import get_azure_llm
from utils.pdf_vector_manager import PDFVectorManager
from utils.agent_decision_logger import get_complete_data_manager

# ==================== 표준화된 기본 인프라 클래스들 ====================

def ensure_awaitable_result(result: Any) -> Any:
    """결과가 코루틴인지 확인하고 경고 출력 (개선된 버전)"""
    if asyncio.iscoroutine(result):
        import warnings
        warnings.warn(
            f"Coroutine object detected but not awaited: {result}. "
            "This should be awaited in an async context.",
            RuntimeWarning,
            stacklevel=2
        )
        # 코루틴을 닫지 않고 그대로 반환하여 호출자가 처리하도록 함
        return result
    return result



@dataclass
class WorkItem:
    """표준화된 작업 항목 정의"""
    id: str
    task_func: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    priority: int = 0
    max_retries: int = 3
    current_retry: int = 0
    timeout: float = 300.0
    created_at: float = field(default_factory=time.time)

    def __lt__(self, other):
        return self.priority < other.priority

class CircuitBreakerState(Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"

class CircuitBreaker:
    def __init__(self, failure_threshold: int = 12, recovery_timeout: float = 90.0, half_open_attempts: int = 2):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_attempts = half_open_attempts
        self._state = CircuitBreakerState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = None
        self._internal_logger = logging.getLogger(self.__class__.__name__)
    
    @property
    def state(self):
        if self._state == CircuitBreakerState.OPEN:
            if self._last_failure_time and (time.monotonic() - self._last_failure_time) > self.recovery_timeout:
                # 수정: self.logger.info() → self._internal_logger.info()
                self._internal_logger.info("CircuitBreaker recovery timeout elapsed. Transitioning to HALF_OPEN.")
                self._state = CircuitBreakerState.HALF_OPEN
                self._success_count = 0
        return self._state
    
    def record_failure(self):
        self._failure_count += 1
        self._last_failure_time = time.monotonic()
        if self.state == CircuitBreakerState.HALF_OPEN:
            # 수정: self.logger.warning() → self._internal_logger.warning()
            self._internal_logger.warning("CircuitBreaker failed in HALF_OPEN state. Transitioning back to OPEN.")
            self._state = CircuitBreakerState.OPEN
            self._failure_count = self.failure_threshold
        elif self._failure_count >= self.failure_threshold and self.state == CircuitBreakerState.CLOSED:
            # 수정: self.logger.error() → self._internal_logger.error()
            self._internal_logger.error(f"CircuitBreaker failure threshold {self.failure_threshold} reached. Transitioning to OPEN.")
            self._state = CircuitBreakerState.OPEN
    
    def record_success(self):
        if self.state == CircuitBreakerState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self.half_open_attempts:
                # 수정: self.logger.info() → self._internal_logger.info()
                self._internal_logger.info("CircuitBreaker successful in HALF_OPEN state. Transitioning to CLOSED.")
                self._state = CircuitBreakerState.CLOSED
                self._reset_counts()
        elif self.state == CircuitBreakerState.CLOSED:
            self._reset_counts()

    def _reset_counts(self):
        self._failure_count = 0
        self._success_count = 0

    async def execute(self, task_func: Callable, *args, **kwargs) -> Any:
        """표준화된 execute 메서드 (코루틴 처리 개선)"""
        if self.state == CircuitBreakerState.OPEN:
            self._internal_logger.warning(f"CircuitBreaker is OPEN for {getattr(task_func, '__name__', 'unknown_task')}. Call rejected.")
            raise CircuitBreakerOpenError(f"CircuitBreaker is OPEN for {getattr(task_func, '__name__', 'unknown_task')}. Call rejected.")

        try:
            # Future 객체 처리
            if asyncio.isfuture(task_func):
                result = await task_func
            # 이미 생성된 코루틴 객체 처리
            elif asyncio.iscoroutine(task_func):
                result = await task_func
            # 코루틴 함수 처리 (올바른 방식)
            elif inspect.iscoroutinefunction(task_func):
                coro = task_func(*args, **kwargs)  # 코루틴 생성
                result = await coro  # 코루틴 실행
            else:
                # 일반 동기 함수 처리
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, lambda: task_func(*args, **kwargs))

            self.record_success()
            return result

        except Exception as e:
            self._internal_logger.error(f"CircuitBreaker recorded failure for {getattr(task_func, '__name__', 'unknown_task')}: {e}")
            self.record_failure()
            raise e

        
        
class CircuitBreakerOpenError(Exception):
    """Circuit Breaker가 열린 상태일 때 발생하는 예외"""
    pass

class AsyncWorkQueue:
    """표준화된 비동기 작업 큐 (결과 저장 형식 통일)"""
    def __init__(self, max_workers: int = 3, max_queue_size: int = 50):
        self._queue = asyncio.PriorityQueue(max_queue_size if max_queue_size > 0 else 0)
        self._workers: List[asyncio.Task] = []
        self._max_workers = max_workers
        self._running = False
        self.logger = logging.getLogger(self.__class__.__name__)
        self._results: Dict[str, Any] = {}  # 표준화된 결과 저장 형식

    async def _worker(self, worker_id: int):
        self._internal_logger = logging.getLogger(f"{self.__class__.__name__}_Worker{worker_id}")
        self._internal_logger.info(f"Worker {worker_id} starting.")
        
        while self._running or not self._queue.empty():
            try:
                item: WorkItem = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                self._internal_logger.info(f"Worker {worker_id} processing task {item.id} (retry {item.current_retry})")
                
                try:
                    if inspect.iscoroutinefunction(item.task_func):
                        # 수정: 코루틴 객체를 명시적으로 생성한 후 await
                        coro = item.task_func(*item.args, **item.kwargs)
                        result = await asyncio.wait_for(coro, timeout=item.timeout)
                    else:
                        loop = asyncio.get_event_loop()
                        result = await asyncio.wait_for(
                            loop.run_in_executor(None, lambda: item.task_func(*item.args, **item.kwargs)),
                            timeout=item.timeout
                        )
                    
                    # 추가: 결과가 또 다른 코루틴인지 확인하고 처리
                    if asyncio.iscoroutine(result):
                        self._internal_logger.warning(f"Task {item.id} result is a nested coroutine. Awaiting it.")
                        try:
                            result = await asyncio.wait_for(result, timeout=item.timeout)
                        except Exception as e:
                            self._results[item.id] = {"status": "error", "error": f"Nested coroutine execution failed: {e}"}
                            continue
                    
                    # 표준화된 결과 저장 형식
                    self._results[item.id] = {"status": "success", "result": result}
                    self._internal_logger.info(f"Task {item.id} completed successfully by worker {worker_id}.")
                    
                except asyncio.TimeoutError:
                    self._results[item.id] = {"status": "timeout", "error": f"Task {item.id} timed out"}
                    self._internal_logger.error(f"Task {item.id} timed out in worker {worker_id}.")
                except Exception as e:
                    self._results[item.id] = {"status": "error", "error": str(e)}
                    self._internal_logger.error(f"Task {item.id} failed in worker {worker_id}: {e}")
                finally:
                    self._queue.task_done()
                    
            except asyncio.TimeoutError:
                if not self._running and self._queue.empty():
                    break
                continue
            except Exception as e:
                self.logger.error(f"Worker {worker_id} encountered an error: {e}")
                await asyncio.sleep(1)
        
        self.logger.info(f"Worker {worker_id} stopping.")


    async def start(self):
        if not self._running:
            self._running = True
            self.logger.info(f"Starting {self._max_workers} workers.")
            self._workers = [asyncio.create_task(self._worker(i)) for i in range(self._max_workers)]

    async def stop(self, graceful=True):
        if self._running:
            self.logger.info("Stopping work queue...")
            self._running = False
            if graceful:
                await self._queue.join()
            
            if self._workers:
                for worker_task in self._workers:
                    worker_task.cancel()
                await asyncio.gather(*self._workers, return_exceptions=True)
                self._workers.clear()
            self.logger.info("Work queue stopped.")

    async def enqueue_work(self, item: WorkItem) -> bool:
        if not self._running:
            await self.start()
        try:
            await self._queue.put(item)
            self.logger.debug(f"Enqueued task {item.id}")
            return True
        except asyncio.QueueFull:
            self.logger.warning(f"Queue is full. Could not enqueue task {item.id}")
            return False

    async def get_results(self, specific_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """개선된 결과 조회 (중첩된 코루틴 객체 처리 추가)"""
        await self._queue.join()
        
        # 결과가 코루틴 객체인지 확인하고 처리
        processed_results = {}
        for key, value in self._results.items():
            if asyncio.iscoroutine(value):
                # 직접 코루틴인 경우 (방어 코드)
                try:
                    processed_results[key] = await asyncio.wait_for(value, timeout=60.0)
                except Exception as e:
                    processed_results[key] = {"status": "error", "error": f"Direct coroutine execution failed: {e}"}
            elif isinstance(value, dict) and 'result' in value:
                # 표준 형태의 결과 딕셔너리인 경우
                if asyncio.iscoroutine(value['result']):
                    # result 값이 코루틴인 경우
                    try:
                        awaited_result = await asyncio.wait_for(value['result'], timeout=60.0)
                        processed_results[key] = {
                            "status": value.get("status", "success"),
                            "result": awaited_result
                        }
                        if "error" in value:
                            processed_results[key]["error"] = value["error"]
                    except Exception as e:
                        processed_results[key] = {
                            "status": "error",
                            "error": f"Nested coroutine execution failed: {e}",
                            "result": None
                        }
                else:
                    # 정상적인 결과
                    processed_results[key] = value
            else:
                # 기타 형태의 결과
                processed_results[key] = value
        
        if specific_ids:
            return {id: processed_results.get(id) for id in specific_ids if id in processed_results}
        
        return processed_results.copy()



    async def clear_results(self):
        self._results.clear()

class BaseAsyncAgent:
    """표준화된 기본 비동기 에이전트 클래스"""
    
    def __init__(self):
        # 하이브리드 로깅 시스템 초기화
        self.logger = get_hybrid_logger(self.__class__.__name__)
        
        # 기존 초기화 코드들...
        self.work_queue = AsyncWorkQueue(max_workers=2, max_queue_size=50)
        self.circuit_breaker = CircuitBreaker(failure_threshold=8, recovery_timeout=30.0)
        self.recursion_threshold = 800
        self.fallback_to_sync = False
        self._recursion_check_buffer = 50
        
        self.timeouts = {
            'crew_kickoff': 90.0,
            'result_collection': 15.0,
            'vector_search': 10.0,
            'agent_creation': 20.0,
            'total_analysis': 180.0,
            'post_processing': 25.0
        }
        
        self.retry_config = {
            'max_attempts': 3,
            'base_delay': 1.0,
            'max_delay': 8.0,
            'exponential_base': 2
        }
        
        self.execution_stats = {
            "total_attempts": 0,
            "successful_executions": 0,
            "fallback_used": 0,
            "circuit_breaker_triggered": 0,
            "timeout_occurred": 0
        }

    def _check_recursion_depth(self):
        """현재 재귀 깊이 확인"""
        current_depth = len(inspect.stack())
        return current_depth

    def _should_use_sync(self):
        """동기 모드로 전환할지 판단"""
        current_depth = self._check_recursion_depth()
        if current_depth >= sys.getrecursionlimit() - self._recursion_check_buffer:
            self.logger.warning(f"Approaching recursion limit ({current_depth}/{sys.getrecursionlimit()}). Switching to sync mode.")
            self.fallback_to_sync = True
            return True
        return self.fallback_to_sync

    async def execute_with_resilience(
        self,
        task_id: str,
        task_func: Callable,
        args: tuple = (),
        kwargs: dict = None,
        max_retries: int = 2,
        initial_timeout: float = 180.0,
        backoff_factor: float = 1.5,
        circuit_breaker: CircuitBreaker = None
    ) -> Any:
        """표준화된 복원력 있는 작업 실행 (코루틴 처리 개선)"""
        if kwargs is None:
            kwargs = {}

        # 이미 생성된 코루틴 객체 처리
        if asyncio.iscoroutine(task_func):
            try:
                return await asyncio.wait_for(task_func, timeout=initial_timeout)
            except Exception as e:
                self.logger.error(f"Coroutine execution failed: {e}")
                return self._get_fallback_result(task_id)

        # Future 객체 처리
        if asyncio.isfuture(task_func):
            try:
                return await asyncio.wait_for(task_func, timeout=initial_timeout)
            except Exception as e:
                self.logger.error(f"Future object execution failed: {e}")
                return self._get_fallback_result(task_id)

        current_retry = 0
        current_timeout = initial_timeout
        last_exception = None
        actual_circuit_breaker = circuit_breaker if circuit_breaker else self.circuit_breaker

        while current_retry <= max_retries:
            task_full_id = f"{task_id}-attempt-{current_retry + 1}"
            self.logger.info(f"Attempt {current_retry + 1}/{max_retries + 1} for task '{task_full_id}' with timeout {current_timeout}s.")

            try:
                if self._check_recursion_depth() >= sys.getrecursionlimit() - self._recursion_check_buffer:
                    self.logger.warning(f"Preemptive recursion stop for '{task_full_id}'.")
                    raise RecursionError(f"Preemptive recursion depth stop for {task_full_id}")

                # CircuitBreaker를 통한 안전한 실행
                circuit_breaker_task = actual_circuit_breaker.execute(task_func, *args, **kwargs)
                result = await asyncio.wait_for(circuit_breaker_task, timeout=current_timeout)

                self.logger.info(f"Task '{task_full_id}' completed successfully.")
                return result

            except asyncio.TimeoutError as e:
                last_exception = e
                self.execution_stats["timeout_occurred"] += 1
                self.logger.warning(f"Task '{task_full_id}' timed out after {current_timeout}s.")

            except RecursionError as e:
                last_exception = e
                self.logger.error(f"Task '{task_full_id}' failed due to RecursionError: {e}")
                self.fallback_to_sync = True
                raise e

            except CircuitBreakerOpenError as e:
                self.execution_stats["circuit_breaker_triggered"] += 1
                self.logger.warning(f"Task '{task_full_id}' rejected by CircuitBreaker.")
                last_exception = e

            except Exception as e:
                last_exception = e
                self.logger.error(f"Task '{task_full_id}' failed: {e}")

            current_retry += 1
            if current_retry <= max_retries:
                sleep_duration = (backoff_factor ** (current_retry - 1))
                self.logger.info(f"Retrying task '{task_id}' in {sleep_duration}s...")
                await asyncio.sleep(sleep_duration)
                current_timeout *= backoff_factor
            else:
                self.logger.error(f"Task '{task_id}' failed after {max_retries + 1} attempts.")

        if last_exception:
            raise last_exception
        else:
            raise Exception(f"Task '{task_id}' failed after max retries without a specific exception.")


    def _get_fallback_result(self, task_id: str) -> Any:
        """폴백 결과 생성 (서브클래스에서 구현)"""
        return f"FALLBACK_RESULT_FOR_{task_id}"

# ==================== 개선된 JSXTemplateAnalyzer ====================

class JSXTemplateAnalyzer(BaseAsyncAgent):
    """JSX 템플릿 분석기 (CrewAI 기반 로깅 시스템 통합, 복원력 강화)"""

    def __init__(self):
        super().__init__()  # BaseAsyncAgent 명시적 초기화
        self.llm = get_azure_llm()
        self.templates_cache = {}
        self.vector_manager = PDFVectorManager()
        self.result_manager = get_complete_data_manager()

        # 기존 변수명 유지 (호환성)
        self.crew_circuit_breaker = self.circuit_breaker  # 기존 코드와의 호환성
        self.vector_db_circuit_breaker = CircuitBreaker(failure_threshold=8, recovery_timeout=30.0)

        # 템플릿 분석 특화 타임아웃 설정
        self.timeouts.update({
            'template_analysis': 180.0,
            'crew_execution': 600.0,
            'vector_enhancement': 45.0,
            'single_template': 180.0
        })


        # CrewAI 에이전트들 생성 (기존 방식 유지)
        self.template_analysis_agent = self._create_template_analysis_agent()
        self.vector_enhancement_agent = self._create_vector_enhancement_agent()
        self.agent_result_integrator = self._create_agent_result_integrator()
        self.template_selector_agent = self._create_template_selector_agent()

        self.logger.info("JSXTemplateAnalyzer 초기화 완료")
        self.logger.info(f"타임아웃 설정: {self.timeouts}")


    def some_method(self):
        # 일반 로깅은 표준 로거 사용
        self.logger.info("JSX code generation started")


    def _get_fallback_result(self, task_id: str, context: Optional[Dict] = None) -> Any:
        """템플릿 분석 전용 폴백 결과 생성"""
        self.logger.warning(f"Generating fallback result for task_id: {task_id}")
        self.execution_stats["fallback_used"] += 1
        
        if "analyze_jsx_templates" in task_id:
            return {}
        if "get_best_template_for_content" in task_id:
            return "Section01.jsx"
        if context and "single_template_analysis" in task_id and "file_name" in context:
            return self._create_default_template_analysis(context["file_name"])
        return None

    # --- Helper for Resilient Execution (기존 메서드 유지하되 BaseAsyncAgent 활용) ---
    async def _execute_with_resilience(
        self,
        task_func: Callable,
        task_id: str,
        circuit_breaker: CircuitBreaker = None,
        timeout: float = 120.0,
        max_retries: int = 2,
        fallback_value: Any = None,
        *args,
        **kwargs
    ) -> Any:
        """기존 메서드 시그니처 유지하되 BaseAsyncAgent의 execute_with_resilience 활용"""
        try:
            # 기존 파라미터를 BaseAsyncAgent의 메서드로 전달
            return await super().execute_with_resilience(
                task_id=task_id,
                task_func=task_func,
                args=args,
                kwargs=kwargs,
                max_retries=max_retries,
                initial_timeout=timeout,
                circuit_breaker=circuit_breaker
            )
        except Exception as e:
            self.logger.warning(f"⚠️ 작업 {task_id} 모든 재시도 실패 - 폴백 값 반환: {e}")
            if fallback_value is not None:
                return fallback_value
            raise e

    # ==================== 기존 메서드들 (완전 보존) ====================

    def _create_template_analysis_agent(self):
        """템플릿 분석 전문 에이전트 (기존 메서드 완전 보존)"""
        return Agent(
            role="JSX 템플릿 구조 분석 전문가",
            goal="JSX 템플릿 파일들의 구조적 특성과 레이아웃 패턴을 정밀 분석하여 최적화된 분류 및 특성 정보를 제공",
            backstory="""당신은 12년간 React 및 JSX 생태계에서 컴포넌트 아키텍처 분석과 패턴 인식을 담당해온 전문가입니다. 다양한 JSX 템플릿의 구조적 특성을 분석하여 최적의 사용 시나리오를 도출하는 데 특화되어 있습니다.

**전문 영역:**
- JSX 컴포넌트 구조 분석
- Styled-components 패턴 인식
- 레이아웃 시스템 분류
- 템플릿 복잡도 평가

**분석 방법론:**
"모든 JSX 템플릿은 고유한 설계 철학과 사용 목적을 가지고 있으며, 이를 정확히 분석하여 최적의 콘텐츠 매칭을 가능하게 합니다."

**핵심 역량:**
- 컴포넌트명 및 Props 추출
- Styled-components 패턴 분석
- 레이아웃 타입 분류 (simple/hero/grid/gallery)
- 이미지 전략 및 텍스트 전략 평가
- 복잡도 수준 측정""",
            verbose=True,
            llm=self.llm,
            allow_delegation=False
        )

    def _create_vector_enhancement_agent(self):
        """벡터 데이터 강화 전문가 (기존 메서드 완전 보존)"""
        return Agent(
            role="PDF 벡터 데이터 기반 템플릿 강화 전문가",
            goal="PDF 벡터 데이터베이스와 템플릿 특성을 매칭하여 템플릿 분석 결과를 강화하고 최적화된 사용 권장사항을 제공",
            backstory="""당신은 10년간 벡터 데이터베이스와 유사도 검색 시스템을 활용한 템플릿 최적화를 담당해온 전문가입니다. Azure Cognitive Search와 PDF 벡터 데이터를 활용하여 템플릿의 잠재적 활용도를 극대화하는 데 특화되어 있습니다.

**기술 전문성:**
- 벡터 유사도 검색 및 매칭
- PDF 레이아웃 패턴 분석
- 템플릿-콘텐츠 호환성 평가
- 사용 시나리오 최적화

**강화 전략:**
"벡터 데이터의 풍부한 레이아웃 정보를 활용하여 각 템플릿의 최적 활용 시나리오를 식별하고 신뢰도를 향상시킵니다."

**출력 강화 요소:**
- 벡터 매칭 기반 신뢰도 계산
- 유사 레이아웃 기반 사용 권장
- PDF 소스 기반 용도 분류
- 레이아웃 패턴 최적화""",
            verbose=True,
            llm=self.llm,
            allow_delegation=False
        )

    def _create_agent_result_integrator(self):
        """에이전트 결과 통합 전문가 (기존 메서드 완전 보존)"""
        return Agent(
            role="에이전트 결과 통합 및 템플릿 강화 전문가",
            goal="BindingAgent와 OrgAgent의 실행 결과를 분석하여 템플릿 특성을 강화하고 최적화된 인사이트를 제공",
            backstory="""당신은 8년간 다중 에이전트 시스템의 결과 통합과 패턴 분석을 담당해온 전문가입니다. BindingAgent의 이미지 배치 전략과 OrgAgent의 텍스트 구조 분석 결과를 템플릿 특성 강화에 활용하는 데 특화되어 있습니다.

**통합 전문성:**
- BindingAgent 이미지 배치 인사이트 활용
- OrgAgent 텍스트 구조 분석 통합
- 에이전트 간 시너지 효과 극대화
- 템플릿 신뢰도 향상

**분석 방법론:**
"각 에이전트의 전문성을 템플릿 분석에 반영하여 단일 분석으로는 달성할 수 없는 수준의 정확도와 신뢰도를 확보합니다."

**강화 영역:**
- 그리드/갤러리 레이아웃 최적화
- 이미지 배치 전략 반영
- 텍스트 구조 복잡도 조정
- 매거진 스타일 최적화""",
            verbose=True,
            llm=self.llm,
            allow_delegation=False
        )

    def _create_template_selector_agent(self):
        """템플릿 선택 전문가 (기존 메서드 완전 보존)"""
        return Agent(
            role="콘텐츠 기반 최적 템플릿 선택 전문가",
            goal="콘텐츠 특성과 템플릿 분석 결과를 종합하여 가장 적합한 템플릿을 선택하고 선택 근거를 제공",
            backstory="""당신은 15년간 콘텐츠 관리 시스템과 템플릿 매칭 알고리즘을 설계해온 전문가입니다. 복잡한 콘텐츠 특성과 다양한 템플릿 옵션 중에서 최적의 조합을 찾아내는 데 특화되어 있습니다.

**선택 전문성:**
- 콘텐츠-템플릿 호환성 분석
- 다차원 점수 계산 시스템
- 벡터 데이터 기반 매칭
- 에이전트 인사이트 통합

**선택 철학:**
"완벽한 템플릿 선택은 콘텐츠의 본질적 특성과 템플릿의 구조적 강점이 완벽히 조화를 이루는 지점에서 이루어집니다."

**평가 기준:**
- 이미지 개수 및 전략 매칭
- 텍스트 길이 및 복잡도 적합성
- 벡터 데이터 기반 보너스
- 에이전트 인사이트 반영
- 감정 톤 및 용도 일치성""",
            verbose=True,
            llm=self.llm,
            allow_delegation=False
        )

    async def analyze_jsx_templates(self, templates_dir: str = "jsx_templates") -> Dict[str, Dict]:
        """jsx_templates 폴더의 모든 템플릿 분석 (개선된 RecursionError 처리)"""

        self.execution_stats["total_attempts"] += 1

        # 재귀 깊이 확인 및 동기 모드 전환
        if self._should_use_sync():
            print("🔄 템플릿 분석 동기 모드로 전환")
            return await self._analyze_jsx_templates_sync_mode(templates_dir)

        try:
            return await self._analyze_jsx_templates_batch_mode(templates_dir)
        except RecursionError as e:
            print(f"🔄 템플릿 분석 RecursionError 감지 - 동기 모드로 전환: {e}")
            self.fallback_to_sync = True
            return await self._analyze_jsx_templates_sync_mode(templates_dir)
        except CircuitBreakerOpenError as e:
            print(f"🔄 Circuit breaker open - 동기 모드로 전환: {e}")
            self.fallback_to_sync = True
            return await self._analyze_jsx_templates_sync_mode(templates_dir)
        except Exception as e:
            print(f"❌ 템플릿 분석 중 예외 발생: {e} - 동기 모드로 폴백 시도")
            self.fallback_to_sync = True
            return await self._analyze_jsx_templates_sync_mode(templates_dir)

    async def _analyze_jsx_templates_batch_mode(self, templates_dir: str) -> Dict[str, Dict]:
        """개선된 배치 기반 템플릿 분석"""
        print("📦 템플릿 분석 배치 모드 시작")

        # 이전 에이전트 결과 수집
        previous_results = await self._safe_collect_results()
        binding_results = [r for r in previous_results if "BindingAgent" in r.get('agent_name', '')]
        org_results = [r for r in previous_results if "OrgAgent" in r.get('agent_name', '')]

        print(f"📊 이전 에이전트 결과 수집: 전체 {len(previous_results)}개, BindingAgent {len(binding_results)}개, OrgAgent {len(org_results)}개")

        if not os.path.exists(templates_dir):
            print(f"❌ 템플릿 폴더 없음: {templates_dir}")
            raise FileNotFoundError(f"Template directory not found: {templates_dir}")

        jsx_files = [f for f in os.listdir(templates_dir) if f.endswith('.jsx')]
        if not jsx_files:
            print(f"❌ JSX 템플릿 파일 없음: {templates_dir}")
            raise FileNotFoundError(f"No .jsx files found in {templates_dir}")

        # CrewAI 실행
        crew_result = await self._execute_crew_safe(templates_dir, jsx_files, binding_results, org_results)

        # 개별 템플릿 분석 (배치 처리)
        analyzed_templates = await self._execute_template_analysis_via_queue(
            crew_result, templates_dir, jsx_files, binding_results, org_results
        )

        self.templates_cache.update(analyzed_templates)

        # 결과 로깅
        successful_analyses = sum(1 for t in analyzed_templates.values() if isinstance(t, dict) and t.get('analysis_success', False))
        await self._safe_store_result(
            "JSXTemplateAnalyzer",
            f"성공적으로 {successful_analyses}/{len(jsx_files)}개 템플릿 분석 완료",
            f"CrewAI 분석 ({'성공' if crew_result else '실패/폴백'}). 개별 파일 분석 완료.",
            analyzed_templates,
            {
                "total_templates": len(jsx_files),
                "successful_analyses": successful_analyses,
                "crewai_kickoff_successful": bool(crew_result),
                "resilient_execution": True
            }
        )

        self.execution_stats["successful_executions"] += 1
        print("✅ PDF 벡터 기반 배치 템플릿 분석 완료")
        return analyzed_templates

    async def _safe_collect_results(self) -> List[Dict]:
        """안전한 결과 수집"""
        try:
            return await asyncio.wait_for(
                self.result_manager.get_all_outputs(exclude_agent="JSXTemplateAnalyzer"),
                timeout=self.timeouts['result_collection']
            )
        except asyncio.TimeoutError:
            self.logger.warning("Result collection timeout, using empty results")
            return []
        except Exception as e:
            self.logger.error(f"Result collection failed: {e}")
            return []

    async def _execute_crew_safe(self, templates_dir: str, jsx_files: List[str], binding_results: List[Dict], org_results: List[Dict]):
        """안전한 CrewAI 분석 실행 (동기 메서드 올바른 처리)"""
        try:
            # 태스크 생성
            template_analysis_task = self._create_template_analysis_task(templates_dir, jsx_files)
            vector_enhancement_task = self._create_vector_enhancement_task()
            agent_integration_task = self._create_agent_integration_task(binding_results, org_results)

            # CrewAI Crew 생성
            analysis_crew = Crew(
                agents=[self.template_analysis_agent, self.vector_enhancement_agent, self.agent_result_integrator],
                tasks=[template_analysis_task, vector_enhancement_task, agent_integration_task],
                process=Process.sequential,
                verbose=True
            )

            # 올바른 CrewAI 실행 방식
            def _sync_crew_execution():
                return analysis_crew.kickoff()  # 동기 메서드 직접 호출

            # executor를 통한 안전한 비동기 실행
            loop = asyncio.get_event_loop()
            crew_result = await asyncio.wait_for(
                loop.run_in_executor(None, _sync_crew_execution),
                timeout=self.timeouts['crew_execution']
            )

            return crew_result

        except asyncio.TimeoutError as e:
            self.logger.warning(f"CrewAI execution timed out: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected CrewAI error: {e}")
            return None


    async def _execute_template_analysis_via_queue(self, crew_result, templates_dir: str, jsx_files: List[str],
                                                 binding_results: List[Dict], org_results: List[Dict]) -> Dict[str, Dict]:
        """큐를 통한 템플릿 분석 실행"""
        print(f"📁 {len(jsx_files)}개 템플릿 파일 배치 분석 시작")

        await self.work_queue.start()
        await self.work_queue.clear_results()

        # 작업 아이템 생성 및 큐에 추가
        submitted_ids = []
        for i, jsx_file in enumerate(jsx_files):
            work_item_id = f"analyze_{jsx_file}_{time.time_ns()}"
            submitted_ids.append(work_item_id)

            work_item = WorkItem(
                id=work_item_id,
                task_func=self._analyze_single_template_with_enhancements,
                args=(jsx_file, templates_dir, crew_result, binding_results, org_results),
                priority=i,
                timeout=180.0
            )

            await self.work_queue.enqueue_work(work_item)

        # 결과 수집
        raw_results = await self.work_queue.get_results(specific_ids=submitted_ids)

        processed_templates = {}
        for item_id in submitted_ids:
            jsx_filename = item_id.split('_')[1]  # "analyze_" 제거
            result_data = raw_results.get(item_id)
            
            if result_data and result_data.get("status") == "success":
                result = result_data["result"]
                if isinstance(result, dict) and result.get('analysis_success'):
                    processed_templates[jsx_filename] = result
                else:
                    print(f"⚠️ {jsx_filename} 분석 실패 - 기본 분석 사용")
                    processed_templates[jsx_filename] = self._create_default_template_analysis(jsx_filename)
            else:
                print(f"⚠️ {jsx_filename} 작업 실패 - 기본 분석 사용")
                processed_templates[jsx_filename] = self._create_default_template_analysis(jsx_filename)

        return processed_templates

    async def _safe_store_result(self, agent_name: str, final_answer: str, reasoning_process: str, 
                               raw_output: Any, performance_metrics: Dict):
        """안전한 결과 저장"""
        try:
            await asyncio.wait_for(
                self.result_manager.store_agent_output(
                    agent_name=agent_name,
                    agent_role="JSX 템플릿 분석기",
                    task_description=f"CrewAI 기반 JSX 템플릿 분석 (Resilient)",
                    final_answer=final_answer,
                    reasoning_process=reasoning_process,
                    execution_steps=[
                        "에이전트 결과 수집",
                        "CrewAI 분석 실행",
                        "개별 템플릿 분석",
                        "결과 통합"
                    ],
                    raw_input={},
                    raw_output=raw_output,
                    performance_metrics=performance_metrics
                ),
                timeout=5.0
            )
        except Exception as e:
            self.logger.error(f"Failed to store result: {e}")

    async def _analyze_single_template_with_enhancements(self, jsx_file: str, templates_dir: str,
                                                       crew_result, binding_results: List[Dict],
                                                       org_results: List[Dict]) -> Dict:
        """개별 템플릿 분석 (강화 포함)"""
        file_path = os.path.join(templates_dir, jsx_file)

        try:
            # 1. 기본 분석
            template_analysis = await self._analyze_single_template(file_path, jsx_file)
            if not template_analysis.get('analysis_success'):
                return template_analysis

            # 2. 벡터 데이터 강화
            template_analysis = await self._execute_with_resilience(
                task_func=self._enhance_with_vector_data_async,
                task_id=f"vector_enhance:{jsx_file}",
                circuit_breaker=self.vector_db_circuit_breaker,
                timeout=45.0,
                fallback_value=template_analysis.copy(),
                template_analysis=template_analysis.copy(),
                jsx_file=jsx_file
            )

            # 3. 에이전트 결과 강화
            template_analysis = await self._enhance_with_agent_results(template_analysis, binding_results, org_results)

            print(f"✅ {jsx_file} 복원력 있는 분석 완료: {template_analysis.get('layout_type', 'N/A')}")
            return template_analysis

        except Exception as e:
            print(f"❌ {jsx_file} 분석 중 오류: {e}")
            return self._create_default_template_analysis(jsx_file)

    async def _analyze_jsx_templates_sync_mode(self, templates_dir: str) -> Dict[str, Dict]:
        """동기 모드 템플릿 분석"""
        print("🔄 템플릿 분석 동기 모드 실행")

        # 이전 에이전트 결과 수집
        previous_results = await self._safe_collect_results()
        binding_results = [r for r in previous_results if "BindingAgent" in r.get('agent_name', '')]
        org_results = [r for r in previous_results if "OrgAgent" in r.get('agent_name', '')]

        if not os.path.exists(templates_dir):
            return {}

        jsx_files = [f for f in os.listdir(templates_dir) if f.endswith('.jsx')]
        if not jsx_files:
            return {}

        # 간소화된 분석
        analyzed_templates = {}
        for jsx_file in jsx_files:
            file_path = os.path.join(templates_dir, jsx_file)
            template_analysis = await self._analyze_single_template(file_path, jsx_file)
            template_analysis = await self._enhance_with_agent_results(template_analysis, binding_results, org_results)
            analyzed_templates[jsx_file] = template_analysis

        self.templates_cache.update(analyzed_templates)
        print("✅ 동기 모드 템플릿 분석 완료")
        return analyzed_templates

    async def get_best_template_for_content(self, content: Dict, analysis: Dict) -> str:
        """콘텐츠에 가장 적합한 템플릿 선택 (개선된 RecursionError 처리)"""
        content_title = content.get('title', 'untitled_content')
        operation_id = f"get_best_template_for_content:{content_title}"
        self.execution_stats["total_attempts"] += 1

        # 재귀 깊이 확인 및 동기 모드 전환
        if self._should_use_sync():
            print("🔄 템플릿 선택 동기 모드로 전환")
            return await self._get_best_template_for_content_sync_mode(content, analysis)

        try:
            return await self._get_best_template_for_content_batch_mode(content, analysis)
        except RecursionError as e:
            print(f"🔄 템플릿 선택 RecursionError 감지 - 동기 모드로 전환: {e}")
            self.fallback_to_sync = True
            return await self._get_best_template_for_content_sync_mode(content, analysis)
        except CircuitBreakerOpenError as e:
            print(f"🔄 Circuit breaker open - 동기 모드로 전환: {e}")
            self.fallback_to_sync = True
            return await self._get_best_template_for_content_sync_mode(content, analysis)
        except Exception as e:
            print(f"❌ 템플릿 선택 중 예외 발생: {e} - 동기 모드로 폴백 시도")
            self.fallback_to_sync = True
            return await self._get_best_template_for_content_sync_mode(content, analysis)

    async def _get_best_template_for_content_batch_mode(self, content: Dict, analysis: Dict) -> str:
        """배치 모드 템플릿 선택"""
        print("📦 템플릿 선택 배치 모드 시작")

        # 이전 에이전트 결과 수집
        previous_results = await self._safe_collect_results()
        binding_results = [r for r in previous_results if "BindingAgent" in r.get('agent_name', '')]
        org_results = [r for r in previous_results if "OrgAgent" in r.get('agent_name', '')]

        if not self.templates_cache:
            selected_template = "Section01.jsx"
            await self._safe_store_result(
                "JSXTemplateAnalyzer_Selector",
                selected_template,
                "템플릿 캐시 없어 기본 템플릿 선택",
                {"content": content, "analysis": analysis},
                {"fallback_selection_due_to_empty_cache": True}
            )
            return selected_template

        # CrewAI 실행
        crew_result = await self._execute_crew_selection_safe(content, analysis, previous_results)

        # 실제 선택 수행
        selected_template = await self._execute_template_selection_resilient(
            crew_result, content, analysis, previous_results, binding_results, org_results
        )

        self.execution_stats["successful_executions"] += 1
        return selected_template

    async def _execute_crew_selection_safe(self, content: Dict, analysis: Dict, previous_results: List[Dict]):
        """안전한 CrewAI 선택 실행 (동기 메서드 올바른 처리)"""
        try:
            template_selection_task = self._create_template_selection_task(content, analysis, previous_results)
            selection_crew = Crew(
                agents=[self.template_selector_agent],
                tasks=[template_selection_task],
                process=Process.sequential,
                verbose=True
            )

            # 올바른 CrewAI 실행 방식
            def _sync_crew_execution():
                return selection_crew.kickoff()  # 동기 메서드 직접 호출

            # executor를 통한 안전한 비동기 실행
            loop = asyncio.get_event_loop()
            crew_result = await asyncio.wait_for(
                loop.run_in_executor(None, _sync_crew_execution),
                timeout=180.0
            )

            return crew_result

        except asyncio.TimeoutError as e:
            self.logger.warning(f"CrewAI selection timed out: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected CrewAI selection error: {e}")
            return None


    async def _execute_template_selection_resilient(self, crew_result, content: Dict, analysis: Dict,
                                                  previous_results: List[Dict], binding_results: List[Dict],
                                                  org_results: List[Dict]) -> str:
        """복원력 있는 템플릿 선택 실행 (기존 메서드 완전 보존)"""
        image_count = len(content.get('images', []))
        text_length = len(content.get('body', ''))
        content_emotion = analysis.get('emotion_tone', 'neutral')

        # 콘텐츠 기반 벡터 검색
        content_query = f"{content.get('title', '')} {content.get('body', '')[:200]}"
        content_vectors = await self._execute_with_resilience(
            task_func=self.vector_manager.search_similar_layouts,
            task_id=f"vector_search:{content.get('title', 'untitled')}",
            circuit_breaker=self.vector_db_circuit_breaker,
            timeout=45.0,
            fallback_value=[],
            query=content_query,
            index_name="magazine_layout",
            top_k=5
        )

        best_template = None
        best_score = 0
        scoring_details = []

        for template_name, template_info in self.templates_cache.items():
            if not isinstance(template_info, dict):
                continue

            score = 0
            score_breakdown = {"template": template_name}

            # 기본 매칭 점수
            template_images = template_info.get('image_strategy', 0)
            if image_count == 0 and template_images == 0:
                score += 30
                score_breakdown["image_match"] = 30
            elif image_count == 1 and template_images == 1:
                score += 30
                score_breakdown["image_match"] = 30
            elif image_count > 1 and template_images > 1:
                score += 20
                score_breakdown["image_match"] = 20

            # 텍스트 길이 매칭
            layout_type = template_info.get('layout_type', 'simple')
            if text_length < 300 and layout_type in ['simple', 'hero']:
                score += 20
                score_breakdown["text_match"] = 20
            elif text_length > 500 and layout_type in ['grid', 'gallery']:
                score += 20
                score_breakdown["text_match"] = 20

            # 벡터 데이터 기반 보너스 점수
            if template_info.get('vector_matched', False):
                vector_bonus = template_info.get('layout_confidence', 0) * 30
                score += vector_bonus
                score_breakdown["vector_bonus"] = vector_bonus

            # 에이전트 결과 기반 보너스 점수
            if template_info.get('agent_enhanced', False):
                agent_bonus = 0

                # BindingAgent 인사이트 보너스
                binding_insights = template_info.get('binding_insights', [])
                if binding_insights:
                    if image_count > 1 and 'grid_layout_optimized' in binding_insights:
                        agent_bonus += 15
                    if image_count > 3 and 'gallery_layout_optimized' in binding_insights:
                        agent_bonus += 15
                    if 'professional_image_placement' in binding_insights:
                        agent_bonus += 10

                # OrgAgent 인사이트 보너스
                org_insights = template_info.get('org_insights', [])
                if org_insights:
                    if text_length > 500 and 'structured_text_layout' in org_insights:
                        agent_bonus += 15
                    if 'magazine_style_optimized' in org_insights:
                        agent_bonus += 20
                    if text_length > 800 and 'complex_content_support' in org_insights:
                        agent_bonus += 10

                score += agent_bonus
                score_breakdown["agent_bonus"] = agent_bonus

            # 콘텐츠 벡터와 템플릿 벡터 매칭
            template_vectors = template_info.get('similar_pdf_layouts', [])
            vector_match_bonus = await self._calculate_vector_content_match(content_vectors, template_vectors) * 20
            score += vector_match_bonus
            score_breakdown["content_vector_match"] = vector_match_bonus

            # 감정 톤 매칭
            recommended_usage = template_info.get('recommended_usage', 'general')
            if content_emotion == 'peaceful' and 'culture' in recommended_usage:
                score += 15
                score_breakdown["emotion_match"] = 15
            elif content_emotion == 'exciting' and 'travel' in recommended_usage:
                score += 15
                score_breakdown["emotion_match"] = 15

            score_breakdown["total_score"] = score
            scoring_details.append(score_breakdown)

            if score > best_score:
                best_score = score
                best_template = template_name

        selected_template = best_template or "Section01.jsx"
        selected_info = self.templates_cache.get(selected_template, {})

        # 선택 결과 로깅
        await self._safe_store_result(
            "JSXTemplateAnalyzer_Selector",
            selected_template,
            f"CrewAI 결과 ({'있음' if crew_result else '없음/실패'}), 벡터 검색 ({len(content_vectors)}개 유사 레이아웃) 기반 점수화. 최고 점수: {best_score}",
            {
                "selected_template": selected_template,
                "best_score": best_score,
                "scoring_details": scoring_details,
                "selected_info_summary": selected_info.get("layout_type")
            },
            {
                "templates_evaluated": len(self.templates_cache),
                "best_score": best_score,
                "vector_matched_for_selection": bool(content_vectors),
                "resilient_execution": True
            }
        )

        print(f"🎯 최종 선택된 템플릿 (Resilient): '{selected_template}' (점수: {best_score})")
        return selected_template

    async def _get_best_template_for_content_sync_mode(self, content: Dict, analysis: Dict) -> str:
        """동기 모드 템플릿 선택 (기존 메서드 완전 보존)"""
        print("🔄 템플릿 선택 동기 모드 실행")

        if not self.templates_cache:
            return "Section01.jsx"

        # 간소화된 선택 로직
        image_count = len(content.get('images', []))
        text_length = len(content.get('body', ''))

        best_template = None
        best_score = 0

        for template_name, template_info in self.templates_cache.items():
            if not isinstance(template_info, dict):
                continue

            score = 0
            template_images = template_info.get('image_strategy', 0)

            if image_count == template_images:
                score += 30
            elif abs(image_count - template_images) <= 1:
                score += 20

            layout_type = template_info.get('layout_type', 'simple')
            if text_length < 300 and layout_type in ['simple', 'hero']:
                score += 20
            elif text_length > 500 and layout_type in ['grid', 'gallery']:
                score += 20

            if score > best_score:
                best_score = score
                best_template = template_name

        selected_template = best_template or "Section01.jsx"
        print(f"🎯 동기 모드 선택된 템플릿: '{selected_template}' (점수: {best_score})")
        return selected_template

    # ==================== 기존 메서드들 (완전 보존) ====================

    def _create_template_analysis_task(self, templates_dir: str, jsx_files: List[str]) -> Task:
        """템플릿 분석 태스크 (기존 메서드 완전 보존)"""
        return Task(
            description=f"""
{templates_dir} 폴더의 {len(jsx_files)}개 JSX 템플릿 파일들을 체계적으로 분석하세요.

**분석 대상 파일들:**
{', '.join(jsx_files)}

**분석 요구사항:**
1. 각 JSX 파일의 구조적 특성 분석
2. 컴포넌트명 및 Props 추출
3. Styled-components 패턴 인식
4. 레이아웃 타입 분류 (simple/hero/grid/gallery/overlay)
5. 이미지 전략 및 텍스트 전략 평가
6. 복잡도 수준 측정 (simple/moderate/complex)

**분석 결과 구조:**
각 템플릿별로 다음 정보 포함:
- 기본 정보 (파일명, 컴포넌트명, props)
- 레이아웃 특성 (타입, 특징, 그리드 구조)
- 콘텐츠 전략 (이미지, 텍스트)
- 복잡도 및 사용 권장사항

모든 템플릿의 상세 분석 결과를 제공하세요.
""",
            expected_output="JSX 템플릿별 상세 분석 결과",
            agent=self.template_analysis_agent
        )

    def _create_vector_enhancement_task(self) -> Task:
        """벡터 강화 태스크 (기존 메서드 완전 보존)"""
        return Task(
            description="""
PDF 벡터 데이터베이스를 활용하여 템플릿 분석 결과를 강화하세요.

**강화 요구사항:**
1. 각 템플릿의 레이아웃 특성을 벡터 검색 쿼리로 변환
2. 유사한 매거진 레이아웃 패턴 검색 (top 3)
3. 벡터 매칭 기반 신뢰도 계산
4. PDF 소스 기반 사용 용도 분류

**강화 영역:**
- 레이아웃 신뢰도 향상
- 사용 시나리오 최적화
- 벡터 매칭 상태 표시
- 유사 레이아웃 정보 제공

**출력 요구사항:**
- 벡터 매칭 성공/실패 상태
- 신뢰도 점수 (0.0-1.0)
- 권장 사용 용도
- 유사 레이아웃 목록

이전 태스크의 분석 결과를 벡터 데이터로 강화하세요.
""",
            expected_output="벡터 데이터 기반 강화된 템플릿 분석 결과",
            agent=self.vector_enhancement_agent,
            context=[self._create_template_analysis_task("", [])]
        )

    def _create_agent_integration_task(self, binding_results: List[Dict], org_results: List[Dict]) -> Task:
        """에이전트 통합 태스크 (기존 메서드 완전 보존)"""
        return Task(
            description=f"""
BindingAgent와 OrgAgent의 실행 결과를 분석하여 템플릿 특성을 더욱 강화하세요.

**통합 대상:**
- BindingAgent 결과: {len(binding_results)}개
- OrgAgent 결과: {len(org_results)}개

**BindingAgent 인사이트 활용:**
1. 이미지 배치 전략 분석 (그리드/갤러리)
2. 시각적 일관성 평가 결과 반영
3. 전문적 이미지 배치 인사이트 통합

**OrgAgent 인사이트 활용:**
1. 텍스트 구조 복잡도 분석
2. 매거진 스타일 최적화 정보
3. 구조화된 레이아웃 인사이트

**강화 방법:**
- 템플릿 신뢰도 점수 향상
- 레이아웃 타입별 보너스 적용
- 사용 권장사항 정교화
- 에이전트 인사이트 메타데이터 추가

이전 태스크들의 결과에 에이전트 인사이트를 통합하여 최종 강화된 템플릿 분석을 완성하세요.
""",
            expected_output="에이전트 인사이트가 통합된 최종 템플릿 분석 결과",
            agent=self.agent_result_integrator,
            context=[self._create_template_analysis_task("", []), self._create_vector_enhancement_task()]
        )

    def _create_template_selection_task(self, content: Dict, analysis: Dict, previous_results: List[Dict]) -> Task:
        """템플릿 선택 태스크 (기존 메서드 완전 보존)"""
        return Task(
            description=f"""
콘텐츠 특성과 템플릿 분석 결과를 종합하여 가장 적합한 템플릿을 선택하세요.

**콘텐츠 특성:**
- 이미지 개수: {len(content.get('images', []))}개
- 텍스트 길이: {len(content.get('body', ''))} 문자
- 감정 톤: {analysis.get('emotion_tone', 'neutral')}
- 제목: {content.get('title', 'N/A')}

**이전 에이전트 결과:** {len(previous_results)}개

**선택 기준:**
1. 이미지 개수 및 전략 매칭 (30점)
2. 텍스
2. 텍스트 길이 및 복잡도 적합성 (20점)
3. 벡터 데이터 기반 보너스 (30점)
4. 에이전트 인사이트 반영 (20점)
5. 감정 톤 및 용도 일치성 (15점)

**평가 방법:**
- 각 템플릿별 점수 계산
- 다차원 매칭 분석
- 벡터 유사도 고려
- 에이전트 강화 요소 반영

**출력 요구사항:**
- 선택된 템플릿 파일명
- 선택 근거 및 점수
- 대안 템플릿 순위

가장 적합한 템플릿을 선택하고 상세한 근거를 제시하세요.
""",
            expected_output="선택된 최적 템플릿과 상세 근거",
            agent=self.template_selector_agent
        )

    async def _analyze_single_template(self, file_path: str, jsx_file: str) -> Dict:
        """개별 템플릿 분석 (기존 메서드 완전 보존)"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                jsx_content = f.read()

            # 기본 분석
            component_name = self._extract_component_name(jsx_content)
            layout_type = self._determine_layout_type(jsx_content)
            image_strategy = self._analyze_image_strategy(jsx_content)
            text_strategy = self._analyze_text_strategy(jsx_content)
            complexity = self._assess_complexity(jsx_content)

            return {
                'file_name': jsx_file,
                'component_name': component_name,
                'layout_type': layout_type,
                'image_strategy': image_strategy,
                'text_strategy': text_strategy,
                'complexity': complexity,
                'styled_components_count': jsx_content.count('styled.'),
                'has_props': 'props' in jsx_content,
                'analysis_success': True,
                'vector_matched': False,
                'agent_enhanced': False
            }

        except Exception as e:
            print(f"❌ {jsx_file} 분석 실패: {e}")
            return self._create_default_template_analysis(jsx_file)

    def _create_default_template_analysis(self, jsx_file: str) -> Dict:
        """기본 템플릿 분석 생성 (기존 메서드 완전 보존)"""
        return {
            'file_name': jsx_file,
            'component_name': jsx_file.replace('.jsx', ''),
            'layout_type': 'simple',
            'image_strategy': 1,
            'text_strategy': 'moderate',
            'complexity': 'simple',
            'styled_components_count': 0,
            'has_props': False,
            'analysis_success': False,
            'vector_matched': False,
            'agent_enhanced': False,
            'fallback_analysis': True
        }

    def _extract_component_name(self, jsx_content: str) -> str:
        """컴포넌트명 추출 (기존 메서드 완전 보존)"""
        match = re.search(r'export\s+const\s+(\w+)', jsx_content)
        return match.group(1) if match else 'UnknownComponent'

    def _determine_layout_type(self, jsx_content: str) -> str:
        """레이아웃 타입 결정 (기존 메서드 완전 보존)"""
        if 'grid' in jsx_content.lower():
            return 'grid'
        elif 'gallery' in jsx_content.lower():
            return 'gallery'
        elif 'hero' in jsx_content.lower():
            return 'hero'
        elif 'overlay' in jsx_content.lower():
            return 'overlay'
        else:
            return 'simple'

    def _analyze_image_strategy(self, jsx_content: str) -> int:
        """이미지 전략 분석 (기존 메서드 완전 보존)"""
        image_count = jsx_content.count('<img') + jsx_content.count('<Image')
        if image_count == 0:
            return 0
        elif image_count == 1:
            return 1
        elif image_count <= 4:
            return 3
        else:
            return 5

    def _analyze_text_strategy(self, jsx_content: str) -> str:
        """텍스트 전략 분석 (기존 메서드 완전 보존)"""
        text_elements = jsx_content.count('<p>') + jsx_content.count('<h') + jsx_content.count('<div>')
        if text_elements <= 3:
            return 'minimal'
        elif text_elements <= 8:
            return 'moderate'
        else:
            return 'rich'

    def _assess_complexity(self, jsx_content: str) -> str:
        """복잡도 평가 (기존 메서드 완전 보존)"""
        complexity_score = 0
        complexity_score += jsx_content.count('styled.') * 2
        complexity_score += jsx_content.count('useState') * 3
        complexity_score += jsx_content.count('useEffect') * 3
        complexity_score += jsx_content.count('props.') * 1

        if complexity_score <= 5:
            return 'simple'
        elif complexity_score <= 15:
            return 'moderate'
        else:
            return 'complex'

    async def _enhance_with_vector_data_async(self, template_analysis: Dict, jsx_file: str) -> Dict:
        """벡터 데이터로 비동기 강화 (기존 메서드 완전 보존)"""
        enhanced = template_analysis.copy()

        try:
            layout_type = template_analysis.get('layout_type', 'simple')
            query = f"{layout_type} layout magazine template"

            similar_layouts = self.vector_manager.search_similar_layouts(
                query=query,
                index_name="magazine_layout",
                top_k=3
            )

            if similar_layouts:
                enhanced['vector_matched'] = True
                enhanced['similar_pdf_layouts'] = similar_layouts
                enhanced['layout_confidence'] = min(sum(layout.get('score', 0) for layout in similar_layouts) / len(similar_layouts), 1.0)

                # PDF 소스 기반 용도 분류
                pdf_sources = [layout.get('pdf_name', '').lower() for layout in similar_layouts]
                if any('travel' in source for source in pdf_sources):
                    enhanced['recommended_usage'] = 'travel_content'
                elif any('culture' in source for source in pdf_sources):
                    enhanced['recommended_usage'] = 'culture_content'
                elif any('lifestyle' in source for source in pdf_sources):
                    enhanced['recommended_usage'] = 'lifestyle_content'
                else:
                    enhanced['recommended_usage'] = 'general_content'

                print(f"🔍 {jsx_file} 벡터 매칭 성공: {len(similar_layouts)}개 유사 레이아웃")
            else:
                enhanced['vector_matched'] = False
                enhanced['recommended_usage'] = 'general_content'

        except Exception as e:
            print(f"⚠️ {jsx_file} 벡터 강화 실패: {e}")
            enhanced['vector_matched'] = False

        return enhanced

    async def _enhance_with_agent_results(self, template_analysis: Dict, binding_results: List[Dict], org_results: List[Dict]) -> Dict:
        """에이전트 결과로 강화 (기존 메서드 완전 보존)"""
        enhanced = template_analysis.copy()

        if not binding_results and not org_results:
            return enhanced

        enhanced['agent_enhanced'] = True
        enhanced['binding_insights'] = []
        enhanced['org_insights'] = []

        # BindingAgent 결과 활용
        for binding_result in binding_results:
            binding_answer = binding_result.get('agent_final_answer', '')

            if '그리드' in binding_answer or 'grid' in binding_answer.lower():
                if template_analysis.get('layout_type') in ['grid', 'gallery']:
                    enhanced['binding_insights'].append('grid_layout_optimized')

            if '갤러리' in binding_answer or 'gallery' in binding_answer.lower():
                if template_analysis.get('layout_type') == 'gallery':
                    enhanced['binding_insights'].append('gallery_layout_optimized')

            if '전문적' in binding_answer or 'professional' in binding_answer.lower():
                enhanced['binding_insights'].append('professional_image_placement')

        # OrgAgent 결과 활용
        for org_result in org_results:
            org_answer = org_result.get('agent_final_answer', '')

            if '구조화' in org_answer or 'structured' in org_answer.lower():
                if template_analysis.get('complexity') in ['moderate', 'complex']:
                    enhanced['org_insights'].append('structured_text_layout')

            if '매거진' in org_answer or 'magazine' in org_answer.lower():
                enhanced['org_insights'].append('magazine_style_optimized')

            if '복잡' in org_answer or 'complex' in org_answer.lower():
                if template_analysis.get('complexity') == 'complex':
                    enhanced['org_insights'].append('complex_content_support')

        # 인사이트 기반 신뢰도 향상
        insight_count = len(enhanced['binding_insights']) + len(enhanced['org_insights'])
        if insight_count > 0:
            current_confidence = enhanced.get('layout_confidence', 0.5)
            enhanced['layout_confidence'] = min(current_confidence + (insight_count * 0.1), 1.0)

        return enhanced

    async def _calculate_vector_content_match(self, content_vectors: List[Dict], template_vectors: List[Dict]) -> float:
        """벡터 콘텐츠 매칭 계산 (기존 메서드 완전 보존)"""
        if not content_vectors or not template_vectors:
            return 0.0

        # 간단한 매칭 로직 (실제로는 더 복잡한 벡터 유사도 계산)
        content_sources = set(v.get('pdf_name', '').lower() for v in content_vectors)
        template_sources = set(v.get('pdf_name', '').lower() for v in template_vectors)

        intersection = content_sources.intersection(template_sources)
        union = content_sources.union(template_sources)

        if not union:
            return 0.0

        return len(intersection) / len(union)

    # 시스템 관리 메서드들
    def get_execution_statistics(self) -> Dict:
        """실행 통계 조회"""
        return {
            **self.execution_stats,
            "success_rate": (
                self.execution_stats["successful_executions"] / 
                max(self.execution_stats["total_attempts"], 1)
            ) * 100,
            "circuit_breaker_state": self.circuit_breaker.state.value,
            "templates_cached": len(self.templates_cache)
        }

    def reset_system_state(self) -> None:
        """시스템 상태 리셋"""
        self.circuit_breaker._reset_counts()
        self.circuit_breaker._state = CircuitBreakerState.CLOSED
        self.vector_db_circuit_breaker._reset_counts()
        self.vector_db_circuit_breaker._state = CircuitBreakerState.CLOSED
        self.fallback_to_sync = False
        self.templates_cache.clear()
        self.execution_stats = {
            "total_attempts": 0,
            "successful_executions": 0,
            "fallback_used": 0,
            "circuit_breaker_triggered": 0,
            "timeout_occurred": 0
        }

    def get_system_info(self) -> Dict:
        """시스템 정보 조회"""
        return {
            "class_name": self.__class__.__name__,
            "version": "2.0_standardized_resilient",
            "features": [
                "표준화된 인프라 클래스 사용",
                "개선된 RecursionError 처리",
                "통일된 Circuit Breaker 인터페이스",
                "안전한 CrewAI 동기 메서드 처리",
                "일관된 로깅 시스템",
                "벡터 데이터 강화",
                "에이전트 결과 통합"
            ],
            "execution_modes": ["batch_resilient", "sync_fallback"],
            "templates_cached": len(self.templates_cache)
        }

    async def cleanup_resources(self) -> None:
        """리소스 정리"""
        self.logger.info("🧹 JSXTemplateAnalyzer 리소스 정리 시작")

        try:
            # 작업 큐 정리 (graceful 파라미터 명시적 전달)
            await self.work_queue.stop(graceful=True)
            self.logger.info("✅ 리소스 정리 완료")
        except Exception as e:
            self.logger.error(f"⚠️ 리소스 정리 중 오류: {e}")

    # 기존 동기 버전 메서드들 (호환성 유지)
    def analyze_jsx_templates_sync(self, templates_dir: str = "jsx_templates") -> Dict[str, Dict]:
        """동기 버전 템플릿 분석 (호환성 유지)"""
        return asyncio.run(self.analyze_jsx_templates(templates_dir))

    def get_best_template_for_content_sync(self, content: Dict, analysis: Dict) -> str:
        """동기 버전 템플릿 선택 (호환성 유지)"""
        return asyncio.run(self.get_best_template_for_content(content, analysis))
