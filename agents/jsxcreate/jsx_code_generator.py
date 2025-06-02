import asyncio
import time
import sys
import logging
from utils.hybridlogging import get_hybrid_logger
import inspect
from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import json
import re

from crewai import Agent, Task, Crew, Process
from custom_llm import get_azure_llm
from utils.agent_decision_logger import get_complete_data_manager




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


# ==================== 표준화된 기본 인프라 클래스들 ====================

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

        # 수정된 코드
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

# ==================== 개선된 JSXCodeGenerator ====================

class JSXCodeGenerator(BaseAsyncAgent):
    """JSX 코드 생성 전문 에이전트 (CrewAI 기반 에이전트 결과 데이터 통합)"""

    def __init__(self):
        super().__init__()  # BaseAsyncAgent의 하이브리드 로깅 자동 적용
        self.llm = get_azure_llm()
        self.result_manager = get_complete_data_manager()
        
        # JSX 코드 생성 특화 타임아웃 설정
        self.timeouts.update({
            'jsx_generation': 120.0,
            'crew_execution': 100.0,
            'agent_result_analysis': 30.0,
            'code_validation': 20.0
        })
        
        # CrewAI 에이전트들 생성 (기존 방식 유지)
        self.jsx_code_generation_agent = self._create_jsx_code_generation_agent()
        self.agent_result_integrator = self._create_agent_result_integrator()
        self.code_validation_agent = self._create_code_validation_agent()
        
        # 초기화 완료 로그
        self.logger.info("JSXCodeGenerator 초기화 완료")
        self.logger.info(f"타임아웃 설정: {self.timeouts}")

    def _create_jsx_code_generation_agent(self):
        """JSX 코드 생성 전문 에이전트 (기존 메서드 완전 보존)"""
        return Agent(
            role="JSX 코드 생성 전문가",
            goal="레이아웃 설계와 콘텐츠 분석 결과를 바탕으로 완벽하게 작동하는 JSX 코드를 생성",
            backstory="""당신은 12년간 React 및 JSX 개발 분야에서 활동해온 시니어 개발자입니다. 수천 개의 JSX 컴포넌트를 설계하고 구현한 경험을 바탕으로 오류 없는 고품질 코드를 생성하는 데 특화되어 있습니다.

**기술 전문성:**
- React 및 JSX 고급 패턴
- Styled-components 기반 디자인 시스템
- 반응형 웹 디자인 구현
- 컴포넌트 성능 최적화

**코드 생성 철학:**
"모든 JSX 코드는 기능적 완성도와 코드 품질, 사용자 경험이 완벽히 조화를 이루어야 합니다."

**품질 기준:**
- 문법 오류 제로
- 컴파일 가능성 보장
- 접근성 표준 준수
- 성능 최적화 적용
- 재사용 가능한 컴포넌트 구조
- ` 가 필요한 부분은 꼭 사용합니다!""",
            verbose=True,
            llm=self.llm,
            allow_delegation=False
        )

    def _create_agent_result_integrator(self):
        """에이전트 결과 통합 전문가 (기존 메서드 완전 보존)"""
        return Agent(
            role="에이전트 결과 통합 및 JSX 강화 전문가",
            goal="이전 에이전트들의 실행 결과를 분석하여 JSX 코드 생성에 필요한 인사이트를 도출하고 코드 품질을 강화",
            backstory="""당신은 8년간 다중 에이전트 시스템의 결과 통합과 패턴 분석을 담당해온 전문가입니다. BindingAgent의 이미지 배치 전략과 OrgAgent의 텍스트 구조 분석 결과를 JSX 코드 생성에 활용하는 데 특화되어 있습니다.

**통합 전문성:**
- BindingAgent 이미지 배치 인사이트 활용
- OrgAgent 텍스트 구조 분석 통합
- 에이전트 간 시너지 효과 극대화
- JSX 코드 품질 향상

**분석 방법론:**
"각 에이전트의 전문성을 JSX 코드 생성에 반영하여 단일 분석으로는 달성할 수 없는 수준의 정확도와 품질을 확보합니다."

**강화 영역:**
- 그리드/갤러리 레이아웃 최적화
- 이미지 배치 전략 반영
- 텍스트 구조 복잡도 조정
- 매거진 스타일 최적화""",
            verbose=True,
            llm=self.llm,
            allow_delegation=False
        )

    def _create_code_validation_agent(self):
        """코드 검증 전문가 (기존 메서드 완전 보존)"""
        return Agent(
            role="JSX 코드 검증 및 최적화 전문가",
            goal="생성된 JSX 코드의 문법적 정확성과 기능적 완성도를 검증하고 최적화하여 완벽한 코드를 보장",
            backstory="""당신은 10년간 React 프로젝트의 코드 리뷰와 품질 보증을 담당해온 전문가입니다. JSX 코드의 모든 측면을 검증하여 프로덕션 레벨의 품질을 보장하는 데 특화되어 있습니다.

**검증 전문성:**
- JSX 문법 및 구조 검증
- React 모범 사례 준수 확인
- 컴파일 가능성 테스트
- 성능 최적화 검증

**검증 철학:**
"완벽한 JSX 코드는 기능적 완성도와 코드 품질, 유지보수성이 모두 보장되는 결과물입니다."

**검증 프로세스:**
- 다단계 문법 검증
- 컴파일 가능성 테스트
- 성능 최적화 확인
- 최종 품질 승인""",
            verbose=True,
            llm=self.llm,
            allow_delegation=False
        )

    async def generate_jsx_code(self, content: Dict, design_result: Dict, component_name: str) -> str:
        """JSX 코드 생성 (개선된 RecursionError 처리)"""
        self.execution_stats["total_attempts"] += 1

        # 재귀 깊이 체크
        if self._should_use_sync():
            return await self._generate_jsx_code_sync_mode(content, design_result, component_name)
        
        try:
            return await self._generate_jsx_code_batch_mode(content, design_result, component_name)
        except RecursionError as e:
            self.logger.warning(f"RecursionError detected, switching to sync mode: {e}")
            self.fallback_to_sync = True
            return await self._generate_jsx_code_sync_mode(content, design_result, component_name)
        except CircuitBreakerOpenError as e:
            self.logger.warning(f"Circuit breaker open, switching to sync mode: {e}")
            self.fallback_to_sync = True
            return await self._generate_jsx_code_sync_mode(content, design_result, component_name)

    async def _generate_jsx_code_batch_mode(self, content: Dict, design_result: Dict, component_name: str) -> str:
        """배치 기반 안전한 JSX 코드 생성"""
        task_id = f"jsx_code_generation_{component_name}_{int(time.time())}"

        async def _safe_jsx_generation():
            return await self._execute_jsx_generation_pipeline(content, design_result, component_name)

        try:
            result = await self.execute_with_resilience(
                task_id=task_id,
                task_func=_safe_jsx_generation,
                initial_timeout=self.timeouts['jsx_generation'],
                max_retries=2
            )

            if result and not str(result).startswith("FALLBACK_RESULT"):
                return result
            else:
                self.logger.warning(f"Batch mode returned fallback for {component_name}, switching to sync mode")
                return await self._generate_jsx_code_sync_mode(content, design_result, component_name)

        except Exception as e:
            self.logger.error(f"Batch mode failed for {component_name}: {e}")
            return await self._generate_jsx_code_sync_mode(content, design_result, component_name)

    async def _generate_jsx_code_sync_mode(self, content: Dict, design_result: Dict, component_name: str) -> str:
        """동기 모드 폴백 처리"""
        try:
            self.logger.info(f"Executing JSX code generation in sync mode for {component_name}")
            
            # 안전한 결과 수집
            previous_results = await self._safe_collect_results()
            binding_results = [
                r for r in previous_results if "BindingAgent" in r.get('agent_name', '')]
            org_results = [
                r for r in previous_results if "OrgAgent" in r.get('agent_name', '')]

            self.logger.info(
                f"Sync mode result collection: Total {len(previous_results)}, BindingAgent {len(binding_results)}, OrgAgent {len(org_results)}")

            # 기본 JSX 생성 수행
            basic_jsx = self._create_default_jsx_code(content, design_result, component_name)

            # 에이전트 결과로 강화
            agent_enhanced_jsx = self._enhance_jsx_with_agent_results(
                basic_jsx, content, design_result, previous_results, binding_results, org_results
            )

            # 간단한 검증
            validated_jsx = self._safe_validate_jsx_code(agent_enhanced_jsx, component_name)

            # 결과 저장
            await self._safe_store_result(
                validated_jsx, content, design_result, component_name,
                len(previous_results), len(binding_results), len(org_results)
            )

            self.logger.info(f"Sync mode JSX code generation completed for {component_name}")
            return validated_jsx

        except Exception as e:
            self.logger.error(f"Sync mode generation failed: {e}")
            return self._get_fallback_result(f"jsx_generation_{component_name}")

    async def _execute_jsx_generation_pipeline(self, content: Dict, design_result: Dict, component_name: str) -> str:
        """개선된 JSX 생성 파이프라인"""
        # 1단계: 이전 에이전트 결과 수집 (타임아웃 적용)
        previous_results = await self._safe_collect_results()

        # BindingAgent와 OrgAgent 응답 특별 수집
        binding_results = [
            r for r in previous_results if "BindingAgent" in r.get('agent_name', '')]
        org_results = [
            r for r in previous_results if "OrgAgent" in r.get('agent_name', '')]

        self.logger.info(
            f"Previous results collected: Total {len(previous_results)}, BindingAgent {len(binding_results)}, OrgAgent {len(org_results)}")

        # 2단계: CrewAI Task들 생성 (안전하게)
        tasks = await self._create_generation_tasks_safe(content, design_result, component_name, previous_results, binding_results, org_results)

        # 3단계: CrewAI Crew 실행 (Circuit Breaker 적용)
        crew_result = await self._execute_crew_safe(tasks)

        # 4단계: 결과 처리 및 JSX 생성 (타임아웃 적용)
        jsx_code = await self._process_crew_generation_result_safe(
            crew_result, content, design_result, component_name, previous_results, binding_results, org_results
        )

        # 5단계: 결과 저장
        await self._safe_store_result(
            jsx_code, content, design_result, component_name,
            len(previous_results), len(binding_results), len(org_results)
        )

        self.logger.info(f"JSX code generation completed for {component_name} (CrewAI based agent data utilization: {len(previous_results)})")
        return jsx_code

    async def _safe_collect_results(self) -> List[Dict]:
        """안전한 결과 수집"""
        try:
            return await asyncio.wait_for(
                self.result_manager.get_all_outputs(
                    exclude_agent="JSXCodeGenerator"),
                timeout=self.timeouts['result_collection']
            )
        except asyncio.TimeoutError:
            self.logger.warning("Result collection timeout, using empty results")
            return []
        except Exception as e:
            self.logger.error(f"Result collection failed: {e}")
            return []

    async def _create_generation_tasks_safe(
        self,
        content: Dict,
        design_result: Dict,
        component_name: str,
        previous_results: List[Dict],
        binding_results: List[Dict],
        org_results: List[Dict]
    ) -> List[Task]:
        """안전한 생성 태스크 생성"""
        try:
            jsx_generation_task = self._create_jsx_generation_task(
                content, design_result, component_name)
            agent_integration_task = self._create_agent_integration_task(
                previous_results, binding_results, org_results)
            code_validation_task = self._create_code_validation_task(component_name)

            return [jsx_generation_task, agent_integration_task, code_validation_task]
        except Exception as e:
            self.logger.error(f"Task creation failed: {e}")
            # 최소한의 기본 태스크 반환
            return [self._create_jsx_generation_task(content, design_result, component_name)]

    async def _execute_crew_safe(self, tasks: List[Task]) -> Any:
        """안전한 CrewAI 실행 (동기 메서드 올바른 처리)"""
        try:
            # CrewAI Crew 생성
            generation_crew = Crew(
                agents=[self.jsx_code_generation_agent,
                    self.agent_result_integrator, self.code_validation_agent],
                tasks=tasks,
                process=Process.sequential,
                verbose=True
            )

            # 올바른 CrewAI 실행 방식
            def _sync_crew_execution():
                return generation_crew.kickoff()  # 동기 메서드 직접 호출

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

        except CircuitBreakerOpenError as e:
            self.logger.warning(f"CrewAI execution failed due to circuit breaker: {e}")
            return None
        except asyncio.TimeoutError as e:
            self.logger.warning(f"CrewAI execution timed out: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected CrewAI error: {e}")
            return None

    async def _process_crew_generation_result_safe(
        self,
        crew_result: Any,
        content: Dict,
        design_result: Dict,
        component_name: str,
        previous_results: List[Dict],
        binding_results: List[Dict],
        org_results: List[Dict]
    ) -> str:
        """안전한 CrewAI 생성 결과 처리"""
        try:
            return await asyncio.wait_for(
                self._process_crew_generation_result(
                    crew_result, content, design_result, component_name,
                    previous_results, binding_results, org_results
                ),
                timeout=self.timeouts['post_processing']
            )
        except asyncio.TimeoutError:
            self.logger.warning("Crew result processing timeout, using fallback")
            return await self._create_fallback_jsx_code(content, design_result, component_name, previous_results, binding_results, org_results)
        except Exception as e:
            self.logger.error(f"Crew result processing failed: {e}")
            return await self._create_fallback_jsx_code(content, design_result, component_name, previous_results, binding_results, org_results)

    def _safe_validate_jsx_code(self, jsx_code: str, component_name: str) -> str:
        """안전한 JSX 코드 검증"""
        try:
            return self._validate_jsx_code(jsx_code, component_name)
        except Exception as e:
            self.logger.error(f"JSX validation failed: {e}")
            return jsx_code  # 검증 실패 시 원본 반환

    async def _safe_store_result(
        self,
        jsx_code: str,
        content: Dict,
        design_result: Dict,
        component_name: str,
        agent_count: int,
        binding_count: int,
        org_count: int
    ):
        """안전한 결과 저장"""
        try:
            await asyncio.wait_for(
                self.result_manager.store_agent_output(
                    agent_name="JSXCodeGenerator",
                    agent_role="JSX 코드 생성 전문가",
                    task_description=f"컴포넌트 {component_name} JSX 코드 생성",
                    final_answer=jsx_code,
                    reasoning_process=f"CrewAI 기반 이전 {agent_count}개 에이전트 결과 분석 후 JSX 코드 생성",
                    execution_steps=[
                        "CrewAI 에이전트 생성",
                        "기본 JSX 코드 생성",
                        "에이전트 결과 통합",
                        "코드 검증 및 최적화",
                        "최종 JSX 코드 완성"
                    ],
                    raw_input={"content": content, "design_result": design_result, "component_name": component_name},
                    raw_output=jsx_code,
                    performance_metrics={
                        "component_name": component_name,
                        "jsx_code_length": len(jsx_code),
                        "agent_results_utilized": agent_count,
                        "binding_results_count": binding_count,
                        "org_results_count": org_count,
                        "crewai_enhanced": True,
                        "safe_mode_used": self.fallback_to_sync
                    }
                ),
                timeout=5.0
            )
        except Exception as e:
            self.logger.error(f"Failed to store result: {e}")

    async def _create_fallback_jsx_code(
        self,
        content: Dict,
        design_result: Dict,
        component_name: str,
        previous_results: List[Dict],
        binding_results: List[Dict],
        org_results: List[Dict]
    ) -> str:
        """폴백 JSX 코드 생성"""
        basic_jsx = self._create_default_jsx_code(content, design_result, component_name)
        
        # 에이전트 결과가 있다면 간단히 적용
        if previous_results:
            basic_jsx = self._enhance_jsx_with_agent_results(
                basic_jsx, content, design_result, previous_results, binding_results, org_results
            )

        return basic_jsx

    def _get_fallback_result(self, task_id: str) -> str:
        """JSX 코드 생성 전용 폴백 결과 생성"""
        component_name = "FallbackComponent"
        if "jsx_generation_" in task_id:
            try:
                component_name = task_id.split("jsx_generation_")[1].split("_")[0]
            except:
                pass

        return f'''import React from "react";
import styled from "styled-components";

const Container = styled.div`
  padding: 20px;
  margin: 10px;
  border: 1px solid #ddd;
  border-radius: 8px;
  background-color: #f9f9f9;
`;

export const {component_name} = () => {{
  return (
    <Container>
      <h2>Fallback Component</h2>
      <p>This component was generated in fallback mode due to system constraints.</p>
      <p><small>Task ID: {task_id}</small></p>
    </Container>
  );
}};'''

    # ==================== 기존 메서드들 (완전 보존) ====================

    def _create_jsx_generation_task(self, content: Dict, design_result: Dict, component_name: str) -> Task:
        """JSX 코드 생성 태스크 (기존 메서드 완전 보존)"""
        return Task(
            description=f"""
콘텐츠와 레이아웃 설계 결과를 바탕으로 완벽한 JSX 코드를 생성하세요.

**생성 대상:**
- 컴포넌트명: {component_name}
- 콘텐츠: {content.get('title', 'N/A')} (본문 {len(content.get('body', ''))} 문자)
- 레이아웃 타입: {design_result.get('layout_type', 'default')}

**설계 정보:**
- 색상 스키마: {design_result.get('color_scheme', {})}
- 타이포그래피: {design_result.get('typography_scale', {})}
- 스타일 컴포넌트: {design_result.get('styled_components', [])}

**생성 요구사항:**
1. React 및 JSX 문법 완벽 준수
2. Styled-components 활용한 스타일링
3. 반응형 디자인 적용
4. 접근성 표준 준수
5. 컴파일 가능한 완전한 코드

**코드 구조:**
- import 문 (React, styled-components)
- styled 컴포넌트 정의
- 메인 컴포넌트 함수
- export 문

**품질 기준:**
- 문법 오류 제로
- 실행 가능한 코드
- 최적화된 성능
- 재사용 가능한 구조
""",
            expected_output="완전하고 오류 없는 JSX 코드",
            agent=self.jsx_code_generation_agent
        )

    def _create_agent_integration_task(self, previous_results: List[Dict], binding_results: List[Dict], org_results: List[Dict]) -> Task:
        """에이전트 통합 태스크 (기존 메서드 완전 보존)"""
        return Task(
            description=f"""
이전 에이전트들의 실행 결과를 분석하여 JSX 코드 생성에 필요한 인사이트를 도출하세요.

**통합 대상:**
- 전체 에이전트 결과: {len(previous_results)}개
- BindingAgent 결과: {len(binding_results)}개 (이미지 배치 전략)
- OrgAgent 결과: {len(org_results)}개 (텍스트 구조)

**BindingAgent 인사이트 활용:**
1. 이미지 배치 전략 분석 (그리드/갤러리)
2. 시각적 일관성 평가 결과 반영
3. 전문적 이미지 배치 인사이트 통합

**OrgAgent 인사이트 활용:**
1. 텍스트 구조 복잡도 분석
2. 매거진 스타일 최적화 정보
3. 구조화된 레이아웃 인사이트

**JSX 코드 강화 방법:**
- 에이전트 인사이트 기반 스타일 조정
- 레이아웃 최적화 적용
- 품질 향상 전략 반영
- 성능 최적화 인사이트 통합

**출력 요구사항:**
- 에이전트별 인사이트 요약
- JSX 코드 강화 권장사항
- 품질 향상 전략
""",
            expected_output="에이전트 인사이트 기반 JSX 코드 강화 방안",
            agent=self.agent_result_integrator
        )

    def _create_code_validation_task(self, component_name: str) -> Task:
        """코드 검증 태스크 (기존 메서드 완전 보존)"""
        return Task(
            description=f"""
생성된 JSX 코드의 품질과 정확성을 종합적으로 검증하세요.

**검증 대상:**
- 컴포넌트명: {component_name}

**검증 영역:**
1. JSX 문법 및 구조 검증
   - import/export 문 정확성
   - 컴포넌트 함수 구조
   - JSX 요소 문법
2. React 모범 사례 준수 확인
   - 컴포넌트 명명 규칙
   - Props 사용법
   - 상태 관리 패턴
3. Styled-components 활용 검증
   - 스타일 정의 정확성
   - CSS 속성 유효성
   - 반응형 디자인 적용
4. 컴파일 가능성 테스트
   - 문법 오류 확인
   - 의존성 검증
   - 실행 가능성 보장

**최종 검증:**
- 모든 문법 오류 제거
- 컴파일 가능성 보장
- 성능 최적화 확인
- 접근성 표준 준수

**승인 기준:**
모든 검증 항목 통과 시 최종 승인
""",
            expected_output="검증 완료된 최종 JSX 코드",
            agent=self.code_validation_agent,
            context=[self._create_jsx_generation_task({}, {}, component_name), self._create_agent_integration_task([], [], [])]
        )

    async def _process_crew_generation_result(self, crew_result, content: Dict, design_result: Dict, component_name: str,
                                        previous_results: List[Dict], binding_results: List[Dict],
                                        org_results: List[Dict]) -> str:
        """CrewAI 생성 결과 처리 (result_text 활용 포함)"""
        try:
            # CrewAI 결과에서 데이터 추출
            if hasattr(crew_result, 'raw') and crew_result.raw:
                result_text = crew_result.raw
            else:
                result_text = str(crew_result)

            # result_text에서 유용한 정보 추출
            crew_insights = self._extract_crew_insights(result_text)
            
            # 기본 JSX 생성 수행 (crew_insights 포함)
            basic_jsx = self._create_default_jsx_code(content, design_result, component_name, crew_insights)

            # 에이전트 결과 데이터로 JSX 강화
            agent_enhanced_jsx = self._enhance_jsx_with_agent_results(
                basic_jsx, content, design_result, previous_results, binding_results, org_results
            )

            # CrewAI 결과 텍스트 기반 추가 강화
            crew_enhanced_jsx = self._enhance_jsx_with_crew_result(
                agent_enhanced_jsx, result_text, component_name
            )

            # JSX 코드 검증 및 정제
            validated_jsx = self._validate_jsx_code(crew_enhanced_jsx, component_name)

            return validated_jsx

        except Exception as e:
            self.logger.error(f"CrewAI result processing failed: {e}")
            # 폴백: 기존 방식으로 처리
            basic_jsx = self._create_default_jsx_code(content, design_result, component_name)
            agent_enhanced_jsx = self._enhance_jsx_with_agent_results(
                basic_jsx, content, design_result, previous_results, binding_results, org_results
            )
            return self._validate_jsx_code(agent_enhanced_jsx, component_name)

    def _extract_crew_insights(self, result_text: str) -> Dict:
        """CrewAI 결과 텍스트에서 인사이트 추출"""
        insights = {
            "style_preferences": [],
            "layout_suggestions": [],
            "component_recommendations": []
        }
        
        # 스타일 관련 키워드 추출
        if "modern" in result_text.lower():
            insights["style_preferences"].append("modern")
        if "minimal" in result_text.lower():
            insights["style_preferences"].append("minimal")
        if "responsive" in result_text.lower():
            insights["layout_suggestions"].append("responsive")
        
        return insights

    def _enhance_jsx_with_crew_result(self, jsx_code: str, result_text: str, component_name: str) -> str:
        """CrewAI 결과 텍스트를 기반으로 JSX 추가 강화"""
        enhanced_jsx = jsx_code
        
        # CrewAI가 제안한 스타일 적용
        if "gradient" in result_text.lower():
            enhanced_jsx = enhanced_jsx.replace(
                'background: #fff;',
                'background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);'
            )
        
        if "shadow" in result_text.lower():
            enhanced_jsx = enhanced_jsx.replace(
                'box-shadow: none;',
                'box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);'
            )
        
        return enhanced_jsx



    def _enhance_jsx_with_agent_results(self, basic_jsx: str, content: Dict, design_result: Dict,
                                  previous_results: List[Dict], binding_results: List[Dict],
                                  org_results: List[Dict]) -> str:
        """에이전트 결과 데이터로 JSX 강화 (content와 design_result 활용 포함)"""
        enhanced_jsx = basic_jsx

        if not previous_results:
            return enhanced_jsx

        # 1. content 데이터 기반 강화
        enhanced_jsx = self._enhance_jsx_with_content_data(enhanced_jsx, content)
        
        # 2. design_result 데이터 기반 강화
        enhanced_jsx = self._enhance_jsx_with_design_data(enhanced_jsx, design_result)
        
        # 3. BindingAgent 결과 특별 활용 (기존 로직 유지 + content/design_result 연계)
        if binding_results:
            enhanced_jsx = self._apply_binding_agent_enhancements(
                enhanced_jsx, binding_results, content, design_result
            )

        # 4. OrgAgent 결과 특별 활용 (기존 로직 유지 + content/design_result 연계)
        if org_results:
            enhanced_jsx = self._apply_org_agent_enhancements(
                enhanced_jsx, org_results, content, design_result
            )

        # 5. 전체 에이전트 결과 기반 품질 향상 (기존 로직 유지)
        enhanced_jsx = self._apply_quality_enhancements(enhanced_jsx, previous_results)
        
        # 6. content와 design_result 통합 최적화
        enhanced_jsx = self._optimize_content_design_integration(
            enhanced_jsx, content, design_result
        )

        return enhanced_jsx

    def _enhance_jsx_with_content_data(self, jsx_code: str, content: Dict) -> str:
        """content 데이터를 기반으로 JSX 강화"""
        if not content:
            return jsx_code
        
        enhanced_jsx = jsx_code
        
        # 콘텐츠 제목 길이에 따른 타이포그래피 조정
        title = content.get('title', '')
        if title:
            title_length = len(title)
            if title_length > 50:
                # 긴 제목: 작은 폰트, 좁은 줄간격
                enhanced_jsx = enhanced_jsx.replace(
                    'font-size: 2rem;',
                    'font-size: 1.6rem; line-height: 1.3; word-break: keep-all;'
                )
            elif title_length < 20:
                # 짧은 제목: 큰 폰트, 넓은 줄간격
                enhanced_jsx = enhanced_jsx.replace(
                    'font-size: 2rem;',
                    'font-size: 2.4rem; line-height: 1.2; letter-spacing: 1px;'
                )
        
        # 본문 내용 길이에 따른 레이아웃 조정
        body = content.get('body', '')
        if body:
            word_count = len(body.split())
            if word_count > 300:
                # 긴 본문: 다단 레이아웃
                enhanced_jsx = enhanced_jsx.replace(
                    'max-width: 100%;',
                    'max-width: 100%; column-count: 2; column-gap: 30px; column-rule: 1px solid #eee;'
                )
            elif word_count < 50:
                # 짧은 본문: 중앙 정렬, 큰 폰트
                enhanced_jsx = enhanced_jsx.replace(
                    'text-align: left;',
                    'text-align: center; font-size: 1.2rem; padding: 40px;'
                )
        
        # 태그라인 존재 여부에 따른 스타일 조정
        tagline = content.get('tagline', '')
        if tagline:
            enhanced_jsx = enhanced_jsx.replace(
                'margin-bottom: 20px;',
                'margin-bottom: 30px; position: relative;'
            )
            # 태그라인용 스타일 추가
            tagline_style = '''
    const TaglineStyle = styled.div`
    font-size: 0.9rem;
    color: #666;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 10px;
    border-left: 3px solid #007bff;
    padding-left: 15px;
    `;'''
            enhanced_jsx = enhanced_jsx.replace(
                'import styled from "styled-components";',
                f'import styled from "styled-components";{tagline_style}'
            )
        
        self.logger.info(f"Content-based enhancements applied: title_length={len(title)}, body_words={len(body.split()) if body else 0}")
        return enhanced_jsx

    def _enhance_jsx_with_design_data(self, jsx_code: str, design_result: Dict) -> str:
        """design_result 데이터를 기반으로 JSX 강화"""
        if not design_result:
            return jsx_code
        
        enhanced_jsx = jsx_code
        
        # 템플릿 타입에 따른 스타일 조정
        template = design_result.get('template', '')
        if template:
            if 'Cover' in template or 'Hero' in template:
                # 커버/히어로 템플릿: 풀스크린 스타일
                enhanced_jsx = enhanced_jsx.replace(
                    'min-height: 400px;',
                    'min-height: 100vh; background-attachment: fixed; background-size: cover;'
                )
            elif 'Gallery' in template:
                # 갤러리 템플릿: 그리드 최적화
                enhanced_jsx = enhanced_jsx.replace(
                    'display: flex;',
                    'display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 15px;'
                )
            elif 'Article' in template:
                # 아티클 템플릿: 읽기 최적화
                enhanced_jsx = enhanced_jsx.replace(
                    'max-width: 100%;',
                    'max-width: 800px; margin: 0 auto; font-family: "Georgia", serif;'
                )
        
        # 색상 스키마 적용
        color_scheme = design_result.get('color_scheme', {})
        if color_scheme:
            primary_color = color_scheme.get('primary', '#007bff')
            secondary_color = color_scheme.get('secondary', '#6c757d')
            
            enhanced_jsx = enhanced_jsx.replace(
                'color: #333;',
                f'color: {primary_color};'
            )
            enhanced_jsx = enhanced_jsx.replace(
                'border-color: #ddd;',
                f'border-color: {secondary_color};'
            )
        
        # 레이아웃 방향 적용
        layout_direction = design_result.get('layout_direction', 'vertical')
        if layout_direction == 'horizontal':
            enhanced_jsx = enhanced_jsx.replace(
                'flex-direction: column;',
                'flex-direction: row; align-items: center;'
            )
        
        # 반응형 설정 적용
        responsive_config = design_result.get('responsive', {})
        if responsive_config:
            breakpoints = responsive_config.get('breakpoints', {})
            if breakpoints:
                media_queries = self._generate_media_queries(breakpoints)
                enhanced_jsx = enhanced_jsx.replace(
                    '`;',
                    f'{media_queries}`;',
                    1  # 첫 번째 스타일 컴포넌트에만 적용
                )
        
        self.logger.info(f"Design-based enhancements applied: template={template}, color_scheme={bool(color_scheme)}")
        return enhanced_jsx

    def _apply_binding_agent_enhancements(self, jsx_code: str, binding_results: List[Dict], 
                                        content: Dict, design_result: Dict) -> str:
        """BindingAgent 결과와 content/design_result 연계 강화"""
        enhanced_jsx = jsx_code
        latest_binding = binding_results[-1]
        binding_answer = latest_binding.get('agent_final_answer', '')
        
        # 기존 로직 유지
        if '그리드' in binding_answer or 'grid' in binding_answer.lower():
            enhanced_jsx = enhanced_jsx.replace(
                'display: flex;',
                'display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px;'
            )
        elif '갤러리' in binding_answer or 'gallery' in binding_answer.lower():
            enhanced_jsx = enhanced_jsx.replace(
                'padding: 20px;',
                'padding: 40px; background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);'
            )
        
        # content와 design_result 연계 추가
        # 이미지 수에 따른 그리드 조정
        images = content.get('images', [])
        if images and len(images) > 0:
            image_count = len(images)
            if 'grid' in binding_answer.lower():
                if image_count <= 2:
                    grid_columns = 'repeat(2, 1fr)'
                elif image_count <= 4:
                    grid_columns = 'repeat(2, 1fr)'
                else:
                    grid_columns = 'repeat(3, 1fr)'
                
                enhanced_jsx = enhanced_jsx.replace(
                    'grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));',
                    f'grid-template-columns: {grid_columns};'
                )
        
        # design_result의 템플릿과 연계
        template = design_result.get('template', '')
        if 'Gallery' in template and 'gallery' in binding_answer.lower():
            # 갤러리 템플릿 + 갤러리 바인딩: 특별 강화
            enhanced_jsx = enhanced_jsx.replace(
                'border-radius: 8px;',
                'border-radius: 15px; overflow: hidden; transition: transform 0.3s ease;'
            )
            # 호버 효과 추가
            hover_effect = '''
    &:hover {
        transform: scale(1.05);
        box-shadow: 0 20px 40px rgba(0,0,0,0.15);
    }'''
            enhanced_jsx = enhanced_jsx.replace(
                '`;',
                f'{hover_effect}`;',
                1
            )
        
        self.logger.info("BindingAgent insights applied with content/design integration")
        return enhanced_jsx

    def _apply_org_agent_enhancements(self, jsx_code: str, org_results: List[Dict],
                                    content: Dict, design_result: Dict) -> str:
        """OrgAgent 결과와 content/design_result 연계 강화"""
        enhanced_jsx = jsx_code
        latest_org = org_results[-1]
        org_answer = latest_org.get('agent_final_answer', '')
        
        # 기존 로직 유지
        if '복잡' in org_answer or 'complex' in org_answer.lower():
            enhanced_jsx = enhanced_jsx.replace(
                'font-size: 1rem;',
                'font-size: 0.95rem; line-height: 1.6; letter-spacing: 0.5px;'
            )
        elif '단순' in org_answer or 'simple' in org_answer.lower():
            enhanced_jsx = enhanced_jsx.replace(
                'font-size: 1rem;',
                'font-size: 1.1rem; line-height: 1.8; font-weight: 300;'
            )
        
        # content의 실제 복잡도와 OrgAgent 분석 결과 연계
        body = content.get('body', '')
        if body:
            actual_complexity = self._calculate_content_complexity(body)
            
            if 'complex' in org_answer.lower() and actual_complexity > 0.7:
                # 실제로 복잡한 콘텐츠 + OrgAgent도 복잡하다고 판단
                enhanced_jsx = enhanced_jsx.replace(
                    'line-height: 1.6;',
                    'line-height: 1.8; margin-bottom: 1.5em; text-indent: 1em;'
                )
                # 읽기 도우미 추가
                reading_helper = '''
    const ReadingHelper = styled.div`
    background: #f8f9fa;
    border-left: 4px solid #007bff;
    padding: 15px;
    margin: 20px 0;
    font-style: italic;
    color: #666;
    `;'''
                enhanced_jsx = enhanced_jsx.replace(
                    'import styled from "styled-components";',
                    f'import styled from "styled-components";{reading_helper}'
                )
        
        # design_result의 템플릿과 연계
        template = design_result.get('template', '')
        if 'Article' in template and ('구조' in org_answer or 'structure' in org_answer.lower()):
            # 아티클 템플릿 + 구조 분석: 목차 스타일 추가
            toc_style = '''
    const TableOfContents = styled.nav`
    background: #f8f9fa;
    border: 1px solid #e9ecef;
    border-radius: 8px;
    padding: 20px;
    margin: 30px 0;
    
    h3 {
        margin-top: 0;
        color: #495057;
        font-size: 1.1rem;
    }
    
    ul {
        list-style: none;
        padding: 0;
        
        li {
        padding: 5px 0;
        border-bottom: 1px solid #e9ecef;
        
        &:last-child {
            border-bottom: none;
        }
        
        a {
            color: #007bff;
            text-decoration: none;
            
            &:hover {
            text-decoration: underline;
            }
        }
        }
    }
    `;'''
            enhanced_jsx = enhanced_jsx.replace(
                'import styled from "styled-components";',
                f'import styled from "styled-components";{toc_style}'
            )
        
        self.logger.info("OrgAgent insights applied with content/design integration")
        return enhanced_jsx

    def _apply_quality_enhancements(self, jsx_code: str, previous_results: List[Dict]) -> str:
        """전체 에이전트 결과 기반 품질 향상 (기존 로직 유지)"""
        enhanced_jsx = jsx_code
        
        success_count = 0
        for result in previous_results:
            performance_data = result.get('performance_data', {})
            if performance_data.get('success_rate', 0) > 0.8:
                success_count += 1

        if success_count >= 3:
            # 고품질 에이전트 결과가 많으면 프리미엄 스타일 적용
            enhanced_jsx = enhanced_jsx.replace(
                'border-radius: 8px;',
                'border-radius: 12px; box-shadow: 0 10px 30px rgba(0,0,0,0.1); backdrop-filter: blur(10px);'
            )
        
        return enhanced_jsx

    def _optimize_content_design_integration(self, jsx_code: str, content: Dict, design_result: Dict) -> str:
        """content와 design_result 통합 최적화"""
        enhanced_jsx = jsx_code
        
        # 콘텐츠 타입과 디자인 템플릿의 최적 조합 확인
        content_type = self._determine_content_type(content)
        template = design_result.get('template', '')
        
        # 콘텐츠 타입과 템플릿이 잘 맞는지 확인하고 조정
        if content_type == 'image_heavy' and 'Article' in template:
            # 이미지가 많은 콘텐츠인데 아티클 템플릿: 이미지 최적화
            enhanced_jsx = enhanced_jsx.replace(
                'img {',
                'img { width: 100%; height: auto; object-fit: cover; border-radius: 8px; margin: 15px 0;'
            )
        elif content_type == 'text_heavy' and 'Gallery' in template:
            # 텍스트가 많은 콘텐츠인데 갤러리 템플릿: 텍스트 영역 확대
            enhanced_jsx = enhanced_jsx.replace(
                'grid-template-columns:',
                'grid-template-columns: 2fr 1fr; /* 텍스트 영역 확대 */'
            )
        
        # 색상 조화 최적화
        if content.get('tagline') and design_result.get('color_scheme'):
            primary_color = design_result['color_scheme'].get('primary', '#007bff')
            enhanced_jsx = enhanced_jsx.replace(
                'border-left: 3px solid #007bff;',
                f'border-left: 3px solid {primary_color};'
            )
        
        self.logger.info(f"Content-design integration optimized: content_type={content_type}, template={template}")
        return enhanced_jsx

    def _calculate_content_complexity(self, text: str) -> float:
        """텍스트 복잡도 계산 (0.0 ~ 1.0)"""
        if not text:
            return 0.0
        
        # 간단한 복잡도 지표들
        word_count = len(text.split())
        sentence_count = text.count('.') + text.count('!') + text.count('?')
        avg_word_length = sum(len(word) for word in text.split()) / word_count if word_count > 0 else 0
        
        # 복잡도 점수 계산
        complexity_score = 0.0
        
        # 단어 수 기반 (300단어 이상이면 복잡)
        complexity_score += min(word_count / 300, 1.0) * 0.4
        
        # 평균 단어 길이 기반 (7글자 이상이면 복잡)
        complexity_score += min(avg_word_length / 7, 1.0) * 0.3
        
        # 문장 길이 기반
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else word_count
        complexity_score += min(avg_sentence_length / 20, 1.0) * 0.3
        
        return min(complexity_score, 1.0)

    def _determine_content_type(self, content: Dict) -> str:
        """콘텐츠 타입 결정"""
        images = content.get('images', [])
        body = content.get('body', '')
        
        image_count = len(images)
        word_count = len(body.split()) if body else 0
        
        if image_count >= 3 and word_count < 200:
            return 'image_heavy'
        elif image_count <= 1 and word_count > 300:
            return 'text_heavy'
        elif image_count >= 2 and word_count >= 200:
            return 'balanced'
        else:
            return 'minimal'

    def _generate_media_queries(self, breakpoints: Dict) -> str:
        """반응형 미디어 쿼리 생성"""
        media_queries = ""
        
        for device, width in breakpoints.items():
            if device == 'mobile':
                media_queries += f"""
    @media (max-width: {width}px) {{
        font-size: 0.9rem;
        padding: 15px;
        margin: 10px;
    }}"""
            elif device == 'tablet':
                media_queries += f"""
    @media (max-width: {width}px) {{
        font-size: 0.95rem;
        padding: 20px;
        grid-template-columns: 1fr;
    }}"""
        
        return media_queries


    def _create_default_jsx_code(self, content: Dict, design_result: Dict, component_name: str) -> str:
        """기본 JSX 코드 생성"""
        title = content.get('title', 'Default Title')
        body = content.get('body', 'Default content body.')
        subtitle = content.get('subtitle', '')
        images = content.get('images', [])

        layout_type = design_result.get('layout_type', 'simple')
        color_scheme = design_result.get('color_scheme', {})
        
        primary_color = color_scheme.get('primary', '#2c3e50')
        secondary_color = color_scheme.get('secondary', '#f8f9fa')
        accent_color = color_scheme.get('accent', '#3498db')

        jsx_template = f'''import React from "react";
    import styled from "styled-components";

    const Container = styled.div`
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
    background-color: {secondary_color};
    border-radius: 8px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    `;

    const Header = styled.header`
    text-align: center;
    margin-bottom: 2rem;
    `;

    const Title = styled.h1`
    font-size: 2.5rem;
    color: {primary_color};
    margin-bottom: 0.5rem;
    font-weight: 700;
    `;

    const Subtitle = styled.h2`
    font-size: 1.2rem;
    color: {accent_color};
    margin-bottom: 1rem;
    font-weight: 400;
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
    margin-top: 2rem;
    `;

    const Image = styled.img`
    max-width: 300px;
    height: auto;
    border-radius: 8px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
    `;

    export const {component_name} = () => {{
    const title = "{title}";
    const subtitle = "{subtitle}";
    const body = "{body}";
    const images = {images};

    return (
        <Container>
        <Header>
            <Title>{{title}}</Title>
            {{subtitle && <Subtitle>{{subtitle}}</Subtitle>}}
        </Header>
        <Content>
            <p>{{body}}</p>
        </Content>
        {{images.length > 0 && (
            <ImageGallery>
            {{images.map((imageUrl, index) => (
                <Image key={{index}} src={{imageUrl}} alt={{`Image ${{index + 1}}`}} />
            ))}}
            </ImageGallery>
        )}}
        </Container>
    );
    }};'''

        return jsx_template


    def _validate_jsx_code(self, jsx_code: str, component_name: str) -> str:
        """JSX 코드 검증 및 정제 (기존 메서드 완전 보존)"""
        # 1. 기본 import 문 확인
        if 'import React' not in jsx_code:
            jsx_code = 'import React from "react";\n' + jsx_code

        if 'import styled' not in jsx_code:
            jsx_code = jsx_code.replace(
                'import React from "react";',
                'import React from "react";\nimport styled from "styled-components";'
            )

        # 2. export 문 확인
        if f'export const {component_name}' not in jsx_code:
            jsx_code = re.sub(r'export const \w+', f'export const {component_name}', jsx_code)

        # 3. 기본 구조 보장
        if 'return (' not in jsx_code:
            jsx_code = jsx_code.replace(
                f'export const {component_name} = () => {{',
                f'export const {component_name} = () => {{\n  return (\n    <div>Component Content</div>\n  );\n}};'
            )

        # 4. 중괄호 균형 맞추기
        open_braces = jsx_code.count('{')
        close_braces = jsx_code.count('}')
        if open_braces > close_braces:
            jsx_code += '}' * (open_braces - close_braces)

        # 5. 괄호 균형 맞추기
        open_parens = jsx_code.count('(')
        close_parens = jsx_code.count(')')
        if open_parens > close_parens:
            jsx_code += ')' * (open_parens - close_parens)

        return jsx_code

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

    # 기존 동기 버전 메서드 (호환성 유지)
    def generate_jsx_code_sync(self, content: Dict, design_result: Dict, component_name: str) -> str:
        """동기 버전 JSX 코드 생성 (호환성 유지)"""
        return asyncio.run(self.generate_jsx_code(content, design_result, component_name))
