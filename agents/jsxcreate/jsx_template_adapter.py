import asyncio
import logging
from utils.hybridlogging import get_hybrid_logger
import time
import sys
import inspect
from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

from crewai import Agent, Task, Crew, Process
from custom_llm import get_azure_llm
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
        # 수정: 표준 로깅 사용
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

# ==================== 개선된 JSXTemplateAdapter ====================

class JSXTemplateAdapter(BaseAsyncAgent):
    """JSX 템플릿 어댑터 (CrewAI 기반 로깅 시스템 통합, 복원력 강화)"""

    def __init__(self):
        super().__init__()  # BaseAsyncAgent 명시적 초기화
        self.llm = get_azure_llm()
        self.result_manager = get_complete_data_manager()

        # 템플릿 어댑터 특화 타임아웃 설정
        self.timeouts.update({
            'template_adaptation': 120.0,
            'crew_execution': 100.0,
            'image_integration': 30.0,
            'structure_preservation': 20.0,
            'validation': 15.0
        })

        # 기존 방식으로 CrewAI 에이전트들 생성 (동기 메서드로 유지)
        self.template_adaptation_agent = self._create_template_adaptation_agent_sync()
        self.image_integration_agent = self._create_image_integration_agent_sync()
        self.structure_preservation_agent = self._create_structure_preservation_agent_sync()
        self.validation_agent = self._create_validation_agent_sync()

        # 기존 변수명 유지 (호환성)
        self.adapter_circuit_breaker = self.circuit_breaker  # 기존 코드와의 호환성
        self._force_sync_mode_global = self.fallback_to_sync  # 기존 코드와의 호환성

        self.logger.info("JSXTemplateAdapter 초기화 완료")
        self.logger.info(f"타임아웃 설정: {self.timeouts}")
    def some_method(self):
        # 일반 로깅은 표준 로거 사용
        self.logger.info("JSX code generation started")

    def _get_fallback_result(self, task_id: str, component_name: str = "FallbackComponent", content: Dict = None) -> str:
        """템플릿 어댑터 전용 폴백 결과 생성 (코루틴 검증 추가)"""
        self.logger.warning(f"Generating fallback result for task_id: {task_id}")
        self.execution_stats["fallback_used"] += 1
        
        if content:
            # 코루틴 검증 추가
            content = ensure_awaitable_result(content)
            if content:
                return self._create_fallback_adaptation_sync(
                    template_info={}, 
                    content=content, 
                    component_name=component_name
                )
        
        return f"""// Fallback for {component_name} due to error in task {task_id}
    import React from "react";

    export const {component_name} = () => {{
        return (
            <div className="fallback-component">
                <h2>Template Adaptation Error</h2>
                <p>Task ID: {task_id}</p>
                <p>Please check the logs for more details.</p>
            </div>
        );
    }};

    export default {component_name};"""


    # --- Helper for Resilient Execution (기존 메서드 유지하되 BaseAsyncAgent 활용) ---
    async def _execute_with_resilience(
        self,
        task_id: str,
        task_func: Callable,
        args: tuple = (),
        kwargs: dict = None,
        max_retries: int = 2,
        initial_timeout: float = 120.0,
        backoff_factor: float = 2.0,
        circuit_breaker: CircuitBreaker = None
    ) -> Any:
        """기존 메서드 시그니처 유지하되 BaseAsyncAgent의 execute_with_resilience 활용"""
        # 기존 파라미터를 BaseAsyncAgent의 메서드로 전달
        return await super().execute_with_resilience(
            task_id=task_id,
            task_func=task_func,
            args=args,
            kwargs=kwargs,
            max_retries=max_retries,
            initial_timeout=initial_timeout,
            backoff_factor=backoff_factor,
            circuit_breaker=circuit_breaker or self.adapter_circuit_breaker
        )

    # ==================== 기존 메서드들 (완전 보존) ====================

    def _create_template_adaptation_agent_sync(self):
        """템플릿 적응 에이전트 생성 (기존 메서드 완전 보존)"""
        return Agent(
            role="JSX 템플릿 적응 전문가",
            goal="선택된 템플릿을 콘텐츠 특성에 맞게 정밀하게 적응시켜 최적화된 JSX 구조를 생성",
            backstory="""당신은 10년간 React 및 JSX 템플릿 시스템을 설계하고 최적화해온 전문가입니다. 다양한 콘텐츠 유형에 맞는 템플릿 적응과 구조 최적화에 특화되어 있습니다.

**전문 분야:**
- JSX 템플릿 구조 분석 및 적응
- 콘텐츠 기반 레이아웃 최적화
- 반응형 디자인 구현
- 컴포넌트 재사용성 극대화

**적응 철학:**
"모든 템플릿은 콘텐츠의 본질을 존중하면서도 사용자 경험을 극대화하는 방향으로 적응되어야 합니다."

**품질 기준:**
- 콘텐츠와 템플릿의 완벽한 조화
- 반응형 디자인 보장
- 접근성 표준 준수
- 성능 최적화""",
            verbose=True,
            llm=self.llm,
            allow_delegation=False
        )

    def _create_image_integration_agent_sync(self):
        """이미지 통합 전문가 생성 (기존 메서드 완전 보존)"""
        return Agent(
            role="이미지 통합 및 최적화 전문가",
            goal="콘텐츠의 이미지를 템플릿에 최적으로 통합하고 시각적 일관성을 보장",
            backstory="""당신은 8년간 웹 디자인과 이미지 최적화 분야에서 활동해온 전문가입니다. 이미지와 텍스트의 조화로운 배치와 시각적 임팩트 극대화에 특화되어 있습니다.

**전문 영역:**
- 이미지 배치 및 크기 최적화
- 시각적 계층 구조 설계
- 반응형 이미지 처리
- 로딩 성능 최적화

**통합 원칙:**
"이미지는 단순한 장식이 아닌 콘텐츠의 핵심 메시지를 전달하는 중요한 요소입니다."

**최적화 기준:**
- 시각적 균형과 조화
- 빠른 로딩 속도
- 다양한 화면 크기 대응
- 접근성 고려""",
            verbose=True,
            llm=self.llm,
            allow_delegation=False
        )

    def _create_structure_preservation_agent_sync(self):
        """구조 보존 전문가 생성 (기존 메서드 완전 보존)"""
        return Agent(
            role="템플릿 구조 보존 및 최적화 전문가",
            goal="원본 템플릿의 핵심 구조를 보존하면서 콘텐츠에 맞는 최적화를 수행",
            backstory="""당신은 12년간 대규모 웹 프로젝트의 아키텍처 설계와 구조 최적화를 담당해온 전문가입니다. 템플릿의 본질적 특성을 유지하면서도 새로운 요구사항에 맞게 진화시키는 데 특화되어 있습니다.

**핵심 역량:**
- 템플릿 구조 분석 및 보존
- 컴포넌트 계층 구조 최적화
- CSS 및 스타일 일관성 유지
- 코드 품질 및 유지보수성 보장

**보존 철학:**
"좋은 템플릿의 DNA는 보존하되, 새로운 콘텐츠에 맞는 진화는 허용해야 합니다."

**최적화 영역:**
- 컴포넌트 재사용성
- 코드 가독성
- 성능 효율성
- 확장 가능성""",
            verbose=True,
            llm=self.llm,
            allow_delegation=False
        )

    def _create_validation_agent_sync(self):
        """검증 전문가 생성 (기존 메서드 완전 보존)"""
        return Agent(
            role="JSX 템플릿 검증 및 품질 보증 전문가",
            goal="적응된 템플릿의 품질을 종합적으로 검증하고 오류를 제거하여 완벽한 결과물을 보장",
            backstory="""당신은 10년간 React 프로젝트의 품질 보증과 코드 리뷰를 담당해온 전문가입니다. JSX 템플릿의 모든 측면을 검증하여 프로덕션 레벨의 품질을 보장하는 데 특화되어 있습니다.

**검증 전문성:**
- JSX 문법 및 구조 검증
- React 모범 사례 준수 확인
- 성능 및 접근성 검증
- 크로스 브라우저 호환성 테스트

**품질 철학:**
"완벽한 템플릿은 기능적 완성도와 코드 품질, 사용자 경험이 모두 조화를 이루는 결과물입니다."

**검증 프로세스:**
- 다단계 문법 검증
- 컴파일 가능성 테스트
- 성능 최적화 확인
- 최종 품질 승인""",
            verbose=True,
            llm=self.llm,
            allow_delegation=False
        )

    async def adapt_template_to_content(self, template_info: Dict, content: Dict, component_name: str) -> str:
        """템플릿을 콘텐츠에 맞게 적응 (개선된 RecursionError 처리)"""
        self.execution_stats["total_attempts"] += 1

        # 재귀 깊이 체크
        if self._should_use_sync():
            return await self._adapt_template_to_content_sync_mode(template_info, content, component_name)
        
        try:
            return await self._adapt_template_to_content_batch_mode(template_info, content, component_name)
        except RecursionError as e:
            self.logger.warning(f"RecursionError detected, switching to sync mode: {e}")
            self.fallback_to_sync = True
            return await self._adapt_template_to_content_sync_mode(template_info, content, component_name)
        except CircuitBreakerOpenError as e:
            self.logger.warning(f"Circuit breaker open, switching to sync mode: {e}")
            self.fallback_to_sync = True
            return await self._adapt_template_to_content_sync_mode(template_info, content, component_name)
        except Exception as e:
            self.logger.error(f"⚠️ 배치 모드 실패, 동기 모드로 폴백: {e}")
            return await self._adapt_template_to_content_sync_mode(template_info, content, component_name)

    async def _adapt_template_to_content_batch_mode(self, template_info: Dict, content: Dict, component_name: str) -> str:
        """배치 기반 안전한 템플릿 적응"""
        task_id = f"template_adaptation_{component_name}_{int(time.time())}"

        async def _safe_template_adaptation():
            return await self._execute_template_adaptation_pipeline(template_info, content, component_name)

        try:
            result = await self.execute_with_resilience(
                task_id=task_id,
                task_func=_safe_template_adaptation,
                initial_timeout=self.timeouts['template_adaptation'],
                max_retries=2
            )

            if result and not str(result).startswith("FALLBACK_RESULT"):
                return result
            else:
                self.logger.warning(f"Batch mode returned fallback for {component_name}, switching to sync mode")
                return await self._adapt_template_to_content_sync_mode(template_info, content, component_name)

        except Exception as e:
            self.logger.error(f"Batch mode failed for {component_name}: {e}")
            return await self._adapt_template_to_content_sync_mode(template_info, content, component_name)

    async def _adapt_template_to_content_sync_mode(self, template_info: Dict, content: Dict, component_name: str) -> str:
        """동기 모드 폴백 처리"""
        try:
            self.logger.info(f"Executing template adaptation in sync mode for {component_name}")
            
            # 간소화된 적응 수행
            adapted_template = self._create_fallback_adaptation_sync(template_info, content, component_name)
            
            # 간소화된 결과 저장
            await self._safe_store_result(adapted_template, template_info, content, component_name, mode="sync_fallback")
            
            self.logger.info(f"Sync mode template adaptation completed for {component_name}")
            return adapted_template

        except Exception as e:
            self.logger.error(f"Sync mode adaptation failed: {e}")
            return self._get_fallback_result(f"template_adaptation_{component_name}", component_name, content)

    async def _execute_template_adaptation_pipeline(self, template_info: Dict, content: Dict, component_name: str) -> str:
        """개선된 템플릿 적응 파이프라인"""
        # 1단계: CrewAI Task들 생성 (안전하게)
        tasks = await self._create_adaptation_tasks_safe(template_info, content, component_name)

        # 2단계: CrewAI Crew 실행 (Circuit Breaker 적용)
        crew_result = await self._execute_crew_safe(tasks)

        # 3단계: 결과 처리 및 적응 (타임아웃 적용)
        adapted_template = await self._process_crew_adaptation_result_safe(
            crew_result, template_info, content, component_name
        )

        # 4단계: 결과 저장
        await self._safe_store_result(adapted_template, template_info, content, component_name)

        self.logger.info(f"Template adaptation completed for {component_name}")
        self.execution_stats["successful_executions"] += 1
        return adapted_template

    async def _create_adaptation_tasks_safe(
        self,
        template_info: Dict,
        content: Dict,
        component_name: str
    ) -> List[Task]:
        """안전한 적응 태스크 생성"""
        try:
            adaptation_task = self._create_adaptation_task(template_info, content, component_name)
            image_integration_task = self._create_image_integration_task(content)
            structure_preservation_task = self._create_structure_preservation_task(template_info)
            validation_task = self._create_validation_task(component_name)

            return [adaptation_task, image_integration_task, structure_preservation_task, validation_task]
        except Exception as e:
            self.logger.error(f"Task creation failed: {e}")
            # 최소한의 기본 태스크 반환
            return [self._create_basic_adaptation_task(template_info, content, component_name)]

    async def _execute_crew_safe(self, agents: List[Agent], tasks: List[Task]) -> Any:
        """안전한 CrewAI 실행 (동기 메서드 올바른 처리)"""
        try:
            # CrewAI Crew 생성
            adaptation_crew = Crew(
                agents=agents,
                tasks=tasks,
                process=Process.sequential,
                verbose=True
            )

            # 올바른 CrewAI 실행 방식
            def _sync_crew_execution():
                return adaptation_crew.kickoff()  # 동기 메서드 직접 호출

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


    async def _process_crew_adaptation_result_safe(
        self,
        crew_result: Any,
        template_info: Dict,
        content: Dict,
        component_name: str
    ) -> str:
        """안전한 CrewAI 적응 결과 처리"""
        try:
            return await asyncio.wait_for(
                self._process_crew_adaptation_result(
                    crew_result, template_info, content, component_name
                ),
                timeout=self.timeouts['post_processing']
            )
        except asyncio.TimeoutError:
            self.logger.warning("Crew result processing timeout, using fallback")
            return self._create_fallback_adaptation_sync(template_info, content, component_name)
        except Exception as e:
            self.logger.error(f"Crew result processing failed: {e}")
            return self._create_fallback_adaptation_sync(template_info, content, component_name)

    async def _safe_store_result(
        self,
        adapted_template: str,
        template_info: Dict,
        content: Dict,
        component_name: str,
        mode: str = "batch"
    ):
        """안전한 결과 저장"""
        try:
            await asyncio.wait_for(
                self.result_manager.store_agent_output(
                    agent_name="JSXTemplateAdapter",
                    agent_role="JSX 템플릿 적응 전문가",
                    task_description=f"컴포넌트 {component_name} 템플릿 적응 ({mode} 모드)",
                    final_answer=adapted_template,
                    reasoning_process=f"템플릿 적응 및 콘텐츠 통합",
                    execution_steps=[
                        "템플릿 분석",
                        "콘텐츠 통합",
                        "구조 보존",
                        "검증 완료"
                    ],
                    raw_input={"template_info": template_info, "content": content, "component_name": component_name},
                    raw_output=adapted_template,
                    performance_metrics={
                        "component_name": component_name,
                        "template_adapted": True,
                        "execution_mode": mode
                    }
                ),
                timeout=5.0
            )
        except Exception as e:
            self.logger.error(f"Failed to store result: {e}")

    # ==================== 기존 메서드들 (완전 보존) ====================

    def _create_adaptation_task(self, template_info: Dict, content: Dict, component_name: str) -> Task:
        """적응 태스크 생성 (기존 메서드 완전 보존)"""
        return Task(
            description=f"""
선택된 템플릿을 콘텐츠 특성에 맞게 정밀하게 적응시키세요.

**템플릿 정보:**
- 템플릿 타입: {template_info.get('template_type', 'unknown')}
- 구조: {template_info.get('structure', 'default')}

**콘텐츠 정보:**
- 컴포넌트명: {component_name}
- 제목: {content.get('title', 'N/A')}
- 본문 길이: {len(content.get('body', ''))} 문자
- 이미지 수: {len(content.get('images', []))}개

**적응 요구사항:**
1. 템플릿의 핵심 구조 유지
2. 콘텐츠에 맞는 레이아웃 조정
3. 반응형 디자인 적용
4. 성능 최적화

**출력 형식:**
완전한 JSX 코드 (import문 포함)
""",
            expected_output="적응된 완전한 JSX 템플릿 코드",
            agent=self.template_adaptation_agent
        )

    def _create_image_integration_task(self, content: Dict) -> Task:
        """이미지 통합 태스크 생성 (기존 메서드 완전 보존)"""
        return Task(
            description=f"""
콘텐츠의 이미지를 템플릿에 최적으로 통합하세요.

**이미지 정보:**
- 이미지 수: {len(content.get('images', []))}개
- 이미지 URLs: {content.get('images', [])}

**통합 요구사항:**
1. 시각적 균형과 조화
2. 반응형 이미지 처리
3. 로딩 성능 최적화
4. 접근성 고려

**최적화 기준:**
- 적절한 크기 및 배치
- 빠른 로딩 속도
- 다양한 화면 크기 대응
""",
            expected_output="이미지 통합 최적화 방안",
            agent=self.image_integration_agent
        )

    def _create_structure_preservation_task(self, template_info: Dict) -> Task:
        """구조 보존 태스크 생성 (기존 메서드 완전 보존)"""
        return Task(
            description=f"""
원본 템플릿의 핵심 구조를 보존하면서 최적화를 수행하세요.

**템플릿 구조:**
- 타입: {template_info.get('template_type', 'unknown')}
- 주요 컴포넌트: {template_info.get('components', [])}

**보존 요구사항:**
1. 핵심 구조 유지
2. 컴포넌트 계층 보존
3. 스타일 일관성 유지
4. 코드 품질 보장

**최적화 영역:**
- 컴포넌트 재사용성
- 코드 가독성
- 성능 효율성
""",
            expected_output="구조 보존 및 최적화 결과",
            agent=self.structure_preservation_agent
        )

    def _create_validation_task(self, component_name: str) -> Task:
        """검증 태스크 생성 (기존 메서드 완전 보존)"""
        return Task(
            description=f"""
적응된 템플릿의 품질을 종합적으로 검증하세요.

**검증 대상:**
- 컴포넌트명: {component_name}

**검증 영역:**
1. JSX 문법 및 구조
2. React 모범 사례 준수
3. 성능 및 접근성
4. 크로스 브라우저 호환성

**품질 기준:**
- 문법 오류 제로
- 컴파일 가능성 보장
- 최적화된 성능
- 완벽한 사용자 경험

**최종 승인:**
모든 검증 항목 통과 시 승인
""",
            expected_output="검증 완료된 최종 템플릿",
            agent=self.validation_agent,
            context=[
                self._create_adaptation_task({}, {}, component_name),
                self._create_image_integration_task({}),
                self._create_structure_preservation_task({})
            ]
        )

    def _create_basic_adaptation_task(self, template_info: Dict, content: Dict, component_name: str) -> Task:
        """기본 적응 태스크 생성 (폴백용)"""
        return Task(
            description=f"""
기본 템플릿 적응을 수행하세요.

**컴포넌트:** {component_name}
**콘텐츠:** {content.get('title', 'N/A')}

기본적인 템플릿 적응을 수행하여 JSX 코드를 생성하세요.
""",
            agent=self.template_adaptation_agent,
            expected_output="기본 적응된 JSX 코드"
        )

    async def _process_crew_adaptation_result(self, crew_result, template_info: Dict, content: Dict, component_name: str) -> str:
        """CrewAI 적응 결과 처리 (기존 메서드 완전 보존)"""
        try:
            # CrewAI 결과에서 데이터 추출
            if hasattr(crew_result, 'raw') and crew_result.raw:
                result_text = crew_result.raw
            else:
                result_text = str(crew_result)

            # 기본 적응 수행
            adapted_template = self._create_fallback_adaptation_sync(template_info, content, component_name)

            # CrewAI 결과 통합 (가능한 경우)
            if result_text and len(result_text) > 100:
                # CrewAI 결과가 유의미한 경우 일부 적용
                adapted_template = self._integrate_crew_insights(adapted_template, result_text)

            return adapted_template

        except Exception as e:
            self.logger.error(f"CrewAI result processing failed: {e}")
            # 폴백: 기존 방식으로 처리
            return self._create_fallback_adaptation_sync(template_info, content, component_name)

    def _integrate_crew_insights(self, base_template: str, crew_insights: str) -> str:
        """CrewAI 인사이트를 기본 템플릿에 통합 (기존 메서드 완전 보존)"""
        # 간단한 통합 로직
        if "styled-components" in crew_insights.lower():
            # styled-components 사용 권장이 있으면 적용
            base_template = base_template.replace(
                'import React from "react";',
                'import React from "react";\nimport styled from "styled-components";'
            )
        
        return base_template

    def _create_fallback_adaptation_sync(self, template_info: Dict, content: Dict, component_name: str) -> str:
        """폴백 적응 생성 (동기 모드) (기존 메서드 완전 보존)"""
        title = content.get('title', 'Default Title')
        body = content.get('body', 'Default content body.')
        images = content.get('images', [])

        # 기본 JSX 템플릿 생성
        jsx_template = f'''import React from "react";
import styled from "styled-components";

const Container = styled.div`
  max-width: 1200px;
  margin: 0 auto;
  padding: 20px;
  background-color: #ffffff;
  border-radius: 8px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
`;

const Title = styled.h1`
  font-size: 2.5rem;
  color: #2c3e50;
  margin-bottom: 1rem;
  text-align: center;
`;

const Content = styled.div`
  font-size: 1rem;
  line-height: 1.6;
  color: #333;
  margin-bottom: 2rem;
`;

const ImageGallery = styled.div`
  display: flex;
  flex-wrap: wrap;
  gap: 15px;
  justify-content: center;
`;

const Image = styled.img`
  max-width: 300px;
  height: auto;
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
`;

export const {component_name} = () => {{
  return (
    <Container>
      <Title>{title}</Title>
      {images}
      <Content>{body}</Content>
    </Container>
  );
}};'''

        return jsx_template

    # 시스템 관리 메서드들
    def get_execution_statistics(self) -> Dict:
        """실행 통계 조회"""
        return {
            **self.execution_stats,
            "success_rate": (
                self.execution_stats["successful_executions"] / 
                max(self.execution_stats["total_attempts"], 1)
            ) * 100,
            "circuit_breaker_state": self.circuit_breaker.state.value
        }

    def reset_system_state(self) -> None:
        """시스템 상태 리셋"""
        self.circuit_breaker._reset_counts()
        self.circuit_breaker._state = CircuitBreakerState.CLOSED
        self.fallback_to_sync = False
        self._force_sync_mode_global = False  # 기존 변수도 리셋
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
                "일관된 로깅 시스템"
            ],
            "execution_modes": ["batch_resilient", "sync_fallback"]
        }

    async def cleanup_resources(self) -> None:
        """리소스 정리"""
        self.logger.info("🧹 JSXTemplateAdapter 리소스 정리 시작")

        try:
            # 작업 큐 정리 (graceful 파라미터 명시적 전달)
            await self.work_queue.stop(graceful=True)
            self.logger.info("✅ 리소스 정리 완료")
        except Exception as e:
            self.logger.error(f"⚠️ 리소스 정리 중 오류: {e}")

    # 기존 동기 버전 메서드 (호환성 유지)
    def adapt_template_to_content_sync(self, template_info: Dict, content: Dict, component_name: str) -> str:
        """동기 버전 템플릿 적응 (호환성 유지)"""
        return asyncio.run(self.adapt_template_to_content(template_info, content, component_name))
