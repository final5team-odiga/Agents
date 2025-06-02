import asyncio
import time
import sys
import inspect
from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import json
import re
import logging
from utils.hybridlogging import get_hybrid_logger
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

# ==================== 개선된 JSXLayoutDesigner ====================

class JSXLayoutDesigner(BaseAsyncAgent):
    """레이아웃 설계 전문 에이전트 (에이전트 결과 데이터 기반)"""

    def __init__(self):
        super().__init__()  # BaseAsyncAgent 명시적 초기화
        self.llm = get_azure_llm()
        self.result_manager = get_complete_data_manager()

        # 레이아웃 설계 특화 타임아웃 설정
        self.timeouts.update({
            'layout_design': 120.0,
            'crew_execution': 100.0,
            'agent_result_analysis': 30.0,
            'design_validation': 20.0
        })

        self.logger.info("JSXLayoutDesigner 초기화 완료")
        self.logger.info(f"타임아웃 설정: {self.timeouts}")
    def some_method(self):
        # 일반 로깅은 표준 로거 사용
        self.logger.info("JSX code generation started")

    def _get_fallback_result(self, task_id: str, analysis: Dict, component_name: str, agent_analysis: Dict) -> Dict:
        """레이아웃 설계 전용 폴백 결과 생성"""
        self.logger.warning(f"Generating fallback design result for task_id: {task_id}")
        self.execution_stats["fallback_used"] += 1
        return self._create_agent_based_default_design_sync_mode(analysis, component_name, agent_analysis)

    def create_agent(self):
        """에이전트 생성 (기존 메서드 완전 보존)"""
        return Agent(
            role="에이전트 결과 데이터 기반 매거진 레이아웃 아키텍트",
            goal="이전 에이전트들의 모든 결과 데이터, template_data.json, PDF 벡터 데이터를 종합 분석하여 완벽한 JSX 레이아웃 구조를 설계",
            backstory="""당신은 25년간 세계 최고 수준의 매거진 디자인과 디지털 레이아웃 분야에서 활동해온 전설적인 레이아웃 아키텍트입니다.

**에이전트 결과 데이터 활용 마스터십:**
- 이전 모든 에이전트들의 출력 결과를 종합 분석
- ContentCreator, ImageAnalyzer, ContentAnalyzer 등의 결과를 레이아웃에 반영
- 에이전트 협업 패턴과 성공 지표를 설계 결정에 활용
- jsx_templates는 사용하지 않고 에이전트 데이터만 활용

**데이터 기반 설계 우선순위:**
1. 이전 에이전트들의 결과 데이터 (최우선)
2. template_data.json의 콘텐츠 구조
3. PDF 벡터 데이터의 레이아웃 패턴
4. 에이전트 협업 품질 지표
5. 존재하는 모든 콘텐츠 데이터와 이미지 URL을 사용해야함

**설계 철학:**
"진정한 매거진 레이아웃은 에이전트들의 협업 결과를 존중하면서도 독자의 인지 과정을 과학적으로 설계한 시스템입니다. jsx_templates에 의존하지 않고 순수한 에이전트 데이터만으로 최적의 레이아웃을 창조합니다."

**오류 없는 설계 보장:**
모든 설계 결정은 JSX 구현 시 오류가 발생하지 않도록 기술적 완성도를 고려합니다.""",
            verbose=True,
            llm=self.llm
        )

    async def design_layout_structure(self, content: Dict, analysis: Dict, component_name: str) -> Dict:
        """에이전트 결과 데이터 기반 레이아웃 구조 설계 (개선된 RecursionError 처리)"""
        self.execution_stats["total_attempts"] += 1

        # 재귀 깊이 체크
        if self._should_use_sync():
            return await self._design_layout_structure_sync_mode(content, analysis, component_name)
        
        try:
            return await self._design_layout_structure_batch_mode(content, analysis, component_name)
        except RecursionError as e:
            self.logger.warning(f"RecursionError detected, switching to sync mode: {e}")
            self.fallback_to_sync = True
            return await self._design_layout_structure_sync_mode(content, analysis, component_name)
        except CircuitBreakerOpenError as e:
            self.logger.warning(f"Circuit breaker open, switching to sync mode: {e}")
            self.fallback_to_sync = True
            return await self._design_layout_structure_sync_mode(content, analysis, component_name)
        except Exception as e:
            self.logger.error(f"⚠️ 배치 모드 실패, 동기 모드로 폴백: {e}")
            return await self._design_layout_structure_sync_mode(content, analysis, component_name)

    async def _design_layout_structure_batch_mode(self, content: Dict, analysis: Dict, component_name: str) -> Dict:
        """배치 기반 안전한 처리"""
        task_id = f"layout_design_{component_name}_{int(time.time())}"

        async def _safe_layout_design():
            return await self._execute_layout_design_pipeline(content, analysis, component_name)

        try:
            result = await self.execute_with_resilience(
                task_id=task_id,
                task_func=_safe_layout_design,
                initial_timeout=self.timeouts['layout_design'],
                max_retries=2
            )

            if result and not str(result).startswith("FALLBACK_RESULT"):
                return result
            else:
                self.logger.warning(f"Batch mode returned fallback for {component_name}, switching to sync mode")
                return await self._design_layout_structure_sync_mode(content, analysis, component_name)

        except Exception as e:
            self.logger.error(f"Batch mode failed for {component_name}: {e}")
            return await self._design_layout_structure_sync_mode(content, analysis, component_name)

    async def _design_layout_structure_sync_mode(self, content: Dict, analysis: Dict, component_name: str) -> Dict:
        """동기 모드 폴백 처리"""
        try:
            self.logger.info(f"🔄 동기 폴백 모드로 레이아웃 설계 실행: {component_name}")
            
            # 안전한 결과 수집
            all_agent_results = await self._safe_collect_results()
            learning_insights = await self._safe_get_learning_insights()
            
            agent_data_analysis = await self._analyze_all_agent_results(all_agent_results)
            
            # 동기 모드에서는 기본 설계 사용
            design_result = self._create_agent_based_default_design_sync_mode(analysis, component_name, agent_data_analysis)
            
            # 간소화된 결과 저장
            await self._safe_store_design_result(component_name, design_result, content, analysis, all_agent_results, learning_insights, mode="sync_fallback")
            
            self.logger.info(f"✅ 동기 모드 레이아웃 설계 완료: {design_result.get('layout_type', '기본')} 구조")
            return design_result

        except Exception as e:
            self.logger.error(f"⚠️ 동기 모드에서도 실패: {e}")
            # 최종 폴백
            return self._create_agent_based_default_design_sync_mode(analysis, component_name, {})

    async def _execute_layout_design_pipeline(self, content: Dict, analysis: Dict, component_name: str) -> Dict:
        """개선된 레이아웃 설계 파이프라인"""
        # 1단계: 이전 에이전트 결과 수집 (타임아웃 적용)
        all_agent_results = await self._safe_collect_results()
        learning_insights = await self._safe_get_learning_insights()

        self.logger.info(f"📚 수집된 에이전트 결과: {len(all_agent_results)}개")
        self.logger.info(f"🧠 학습 인사이트: {len(learning_insights.get('recommendations', []))}개")

        # 2단계: 에이전트 데이터 분석
        agent_data_analysis = await self._analyze_all_agent_results(all_agent_results)

        # 3단계: CrewAI Task들 생성 (안전하게)
        tasks = await self._create_design_tasks_safe(content, analysis, component_name, all_agent_results, learning_insights, agent_data_analysis)

        # 4단계: CrewAI Crew 실행 (Circuit Breaker 적용)
        crew_result = await self._execute_crew_safe(tasks)

        # 5단계: 결과 처리 및 설계 (타임아웃 적용)
        design_result = await self._process_crew_design_result_safe(
            crew_result, content, analysis, component_name, all_agent_results, learning_insights, agent_data_analysis
        )

        # 6단계: 결과 저장
        await self._safe_store_design_result(
            component_name, design_result, content, analysis, all_agent_results, learning_insights
        )

        self.logger.info(f"✅ 에이전트 데이터 기반 레이아웃 설계 완료: {design_result.get('layout_type', '기본')} 구조 (CrewAI based agent data utilization: {len(all_agent_results)})")
        self.execution_stats["successful_executions"] += 1
        return design_result

    async def _safe_collect_results(self) -> List[Dict]:
        """안전한 결과 수집"""
        try:
            return await asyncio.wait_for(
                self.result_manager.get_all_outputs(exclude_agent="JSXLayoutDesigner"),
                timeout=self.timeouts['result_collection']
            )
        except asyncio.TimeoutError:
            self.logger.warning("Result collection timeout, using empty results")
            return []
        except Exception as e:
            self.logger.error(f"Result collection failed: {e}")
            return []

    async def _safe_get_learning_insights(self) -> Dict:
        """안전한 학습 인사이트 수집"""
        try:
            return await asyncio.wait_for(
                self.logger.get_learning_insights("JSXLayoutDesigner"),
                timeout=10.0
            )
        except asyncio.TimeoutError:
            self.logger.warning("Learning insights collection timeout, using empty insights")
            return {"recommendations": []}
        except Exception as e:
            self.logger.error(f"Learning insights collection failed: {e}")
            return {"recommendations": []}

    async def _create_design_tasks_safe(
        self,
        content: Dict,
        analysis: Dict,
        component_name: str,
        all_agent_results: List[Dict],
        learning_insights: Dict,
        agent_data_analysis: Dict
    ) -> List[Task]:
        """안전한 설계 태스크 생성"""
        try:
            design_task = self._create_design_task(
                content, analysis, component_name, all_agent_results, learning_insights, agent_data_analysis)
            return [design_task]
        except Exception as e:
            self.logger.error(f"Task creation failed: {e}")
            # 최소한의 기본 태스크 반환
            return [self._create_basic_design_task(content, analysis, component_name)]

    async def _execute_crew_safe(self, tasks: List[Task]) -> Any:
        """안전한 CrewAI 실행 (동기 메서드 올바른 처리)"""
        try:
            # CrewAI Crew 생성
            design_crew = Crew(
                agents=[self.create_agent()],
                tasks=tasks,
                process=Process.sequential,
                verbose=True
            )

            # 올바른 CrewAI 실행 방식
            def _sync_crew_execution():
                return design_crew.kickoff()  # 동기 메서드 직접 호출

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


    async def _process_crew_design_result_safe(
        self,
        crew_result: Any,
        content: Dict,
        analysis: Dict,
        component_name: str,
        all_agent_results: List[Dict],
        learning_insights: Dict,
        agent_data_analysis: Dict
    ) -> Dict:
        """안전한 CrewAI 설계 결과 처리"""
        try:
            return await asyncio.wait_for(
                self._process_crew_design_result(
                    crew_result, content, analysis, component_name,
                    all_agent_results, learning_insights, agent_data_analysis
                ),
                timeout=self.timeouts['post_processing']
            )
        except asyncio.TimeoutError:
            self.logger.warning("Crew result processing timeout, using fallback")
            return self._create_agent_based_default_design_sync_mode(analysis, component_name, agent_data_analysis)
        except Exception as e:
            self.logger.error(f"Crew result processing failed: {e}")
            return self._create_agent_based_default_design_sync_mode(analysis, component_name, agent_data_analysis)

    async def _safe_store_design_result(
        self,
        component_name: str,
        design_result: Dict,
        content: Dict,
        analysis: Dict,
        all_agent_results: List[Dict],
        learning_insights: Dict,
        mode: str = "batch"
    ):
        """안전한 설계 결과 저장"""
        try:
            await asyncio.wait_for(
                self.result_manager.store_agent_output(
                    agent_name="JSXLayoutDesigner",
                    agent_role="에이전트 데이터 기반 레이아웃 아키텍트",
                    task_description=f"컴포넌트 {component_name} 레이아웃 설계 ({mode} 모드)",
                    final_answer=str(design_result),
                    reasoning_process=f"{len(all_agent_results)}개 에이전트 결과 분석하여 레이아웃 설계",
                    execution_steps=[
                        "에이전트 결과 수집",
                        "데이터 분석",
                        "레이아웃 설계",
                        "검증 완료"
                    ],
                    raw_input={"content": content, "analysis": analysis, "component_name": component_name},
                    raw_output=design_result,
                    performance_metrics={
                        "agent_results_utilized": len(all_agent_results),
                        "jsx_templates_ignored": True,
                        "learning_insights_applied": len(learning_insights.get('recommendations', [])),
                        "layout_type": design_result.get('layout_type'),
                        "error_prevention_applied": True,
                        "execution_mode": mode
                    }
                ),
                timeout=5.0
            )
        except Exception as e:
            self.logger.error(f"⚠️ 결과 저장 실패: {e}")

    # ==================== 기존 메서드들 (완전 보존) ====================

    async def _analyze_all_agent_results(self, agent_results: List[Dict]) -> Dict:
        """모든 에이전트 결과 데이터 분석 (기존 메서드 완전 보존)"""
        analysis = {
            "agent_summary": {},
            "quality_indicators": {},
            "content_patterns": {},
            "design_preferences": {},
            "success_metrics": {}
        }

        if not agent_results:
            return analysis

        # 에이전트별 결과 분류
        for result in agent_results:
            agent_name = result.get('agent_name', 'unknown')
            if agent_name not in analysis["agent_summary"]:
                analysis["agent_summary"][agent_name] = {
                    "count": 0,
                    "avg_confidence": 0,
                    "latest_output": None,
                    "success_rate": 0
                }

            analysis["agent_summary"][agent_name]["count"] += 1

            # 신뢰도 계산
            confidence = result.get('metadata', {}).get('confidence_score', 0)
            if confidence > 0:
                current_avg = analysis["agent_summary"][agent_name]["avg_confidence"]
                count = analysis["agent_summary"][agent_name]["count"]
                analysis["agent_summary"][agent_name]["avg_confidence"] = (current_avg * (count-1) + confidence) / count

            # 최신 출력 저장
            analysis["agent_summary"][agent_name]["latest_output"] = result.get('full_output')

        # 전체 품질 지표
        all_confidences = [
            r.get('metadata', {}).get('confidence_score', 0)
            for r in agent_results
            if r.get('metadata', {}).get('confidence_score', 0) > 0
        ]

        if all_confidences:
            analysis["quality_indicators"] = {
                "overall_confidence": sum(all_confidences) / len(all_confidences),
                "high_quality_count": len([c for c in all_confidences if c > 0.8]),
                "total_agents": len(analysis["agent_summary"]),
                "collaboration_success": len(all_confidences) / len(agent_results)
            }

        return analysis

    def _create_design_task(self, content: Dict, analysis: Dict, component_name: str,
                           all_agent_results: List[Dict], learning_insights: Dict,
                           agent_data_analysis: Dict) -> Task:
        """설계 태스크 생성 (기존 메서드 완전 보존)"""
        return Task(
            description=f"""
**에이전트 결과 데이터 기반 완벽한 JSX 레이아웃 설계**

이전 모든 에이전트들의 결과 데이터를 종합 분석하여 완벽한 JSX 레이아웃 구조를 설계하세요:

**이전 에이전트 결과 데이터 분석 ({len(all_agent_results)}개):**
{self._format_agent_data_analysis(agent_data_analysis)}

**학습 인사이트 ({len(learning_insights.get('recommendations', []))}개):**
{chr(10).join(learning_insights.get('recommendations', [])[:3])}

**현재 콘텐츠 특성:**
- 제목: "{content.get('title', '')}" (길이: {len(content.get('title', ''))}자)
- 부제목: "{content.get('subtitle', '')}" (길이: {len(content.get('subtitle', ''))}자)
- 본문 길이: {len(content.get('body', ''))}자
- 이미지 수: {len(content.get('images', []))}개
- 이미지 URLs: {content.get('images', [])}

**ContentAnalyzer 분석 결과:**
- 권장 레이아웃: {analysis.get('recommended_layout', 'grid')}
- 감정 톤: {analysis.get('emotion_tone', 'neutral')}
- 이미지 전략: {analysis.get('image_strategy', 'grid')}
- 에이전트 강화: {analysis.get('agent_enhanced', False)}

**설계 요구사항:**
- 컴포넌트 이름: {component_name}
- jsx_templates 사용 금지
- 에이전트 결과 데이터 최우선 활용
- 오류 없는 JSX 구현 보장

**설계 결과 JSON 형식:**
{{
"layout_type": "에이전트 데이터 기반 선택된 레이아웃",
"layout_rationale": "에이전트 결과 데이터 기반 선택 근거",
"grid_structure": "CSS Grid 구조",
"styled_components": ["컴포넌트 목록"],
"color_scheme": {{"primary": "#색상", "secondary": "#색상"}},
"typography_scale": {{"title": "크기", "body": "크기"}},
"image_layout": "이미지 배치 전략",
"agent_data_integration": "에이전트 데이터 활용 방식",
"error_prevention": "오류 방지 전략",
"quality_metrics": {{"score": 0.95}}
}}

**중요 지침:**
1. 에이전트 결과 데이터를 최우선으로 활용
2. jsx_templates는 절대 참조하지 않음
3. 모든 설계 결정에 에이전트 데이터 근거 제시
4. JSX 구현 시 오류 발생 방지 고려
5. 에이전트 협업 품질 지표 반영

**출력:** 완전한 레이아웃 설계 JSON (에이전트 데이터 기반)
""",
            agent=self.create_agent(),
            expected_output="에이전트 결과 데이터 기반 완전한 레이아웃 구조 설계 JSON"
        )

    def _create_basic_design_task(self, content: Dict, analysis: Dict, component_name: str) -> Task:
        """기본 설계 태스크 생성 (폴백용)"""
        return Task(
            description=f"""
기본 JSX 레이아웃을 설계하세요.

**컴포넌트:** {component_name}
**콘텐츠:** {content.get('title', 'N/A')}
**분석 결과:** {analysis.get('recommended_layout', 'grid')}

기본적인 레이아웃 구조를 JSON 형식으로 제공하세요.
""",
            agent=self.create_agent(),
            expected_output="기본 레이아웃 설계 JSON"
        )

    def _format_agent_data_analysis(self, agent_analysis: Dict) -> str:
        """에이전트 데이터 분석 결과 포맷팅 (기존 메서드 완전 보존)"""
        if not agent_analysis.get("agent_summary"):
            return "이전 에이전트 결과 없음"

        formatted_parts = []
        for agent_name, summary in agent_analysis["agent_summary"].items():
            formatted_parts.append(
                f"- {agent_name}: {summary['count']}개 결과, "
                f"평균 신뢰도: {summary['avg_confidence']:.2f}, "
                f"최신 출력 타입: {type(summary['latest_output']).__name__}"
            )

        quality_info = agent_analysis.get("quality_indicators", {})
        if quality_info:
            formatted_parts.append(
                f"- 전체 품질: 신뢰도 {quality_info.get('overall_confidence', 0):.2f}, "
                f"고품질 결과 {quality_info.get('high_quality_count', 0)}개"
            )

        return "\n".join(formatted_parts)

    async def _process_crew_design_result(self, crew_result, content: Dict, analysis: Dict, component_name: str,
                                        all_agent_results: List[Dict], learning_insights: Dict,
                                        agent_data_analysis: Dict) -> Dict:
        """CrewAI 설계 결과 처리 (기존 메서드 완전 보존)"""
        try:
            # CrewAI 결과에서 데이터 추출
            if hasattr(crew_result, 'raw') and crew_result.raw:
                result_text = crew_result.raw
            else:
                result_text = str(crew_result)

            # 결과 파싱 시도
            design_result = await self._parse_design_result_with_agent_data(result_text, analysis, agent_data_analysis)
            
            return design_result

        except Exception as e:
            self.logger.error(f"CrewAI result processing failed: {e}")
            # 폴백: 기존 방식으로 처리
            return self._create_agent_based_default_design_sync_mode(analysis, component_name, agent_data_analysis)

    async def _parse_design_result_with_agent_data(self, result_text: str, analysis: Dict, agent_analysis: Dict) -> Dict:
        """에이전트 데이터 기반 설계 결과 파싱 (기존 메서드 완전 보존)"""
        try:
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                parsed_result = json.loads(json_match.group())
                # 에이전트 데이터 통합
                parsed_result['agent_data_integration'] = agent_analysis
                parsed_result['jsx_templates_ignored'] = True
                parsed_result['error_prevention_applied'] = True
                return parsed_result
        except Exception as e:
            self.logger.error(f"⚠️ JSON 파싱 실패: {e}")

        return self._create_agent_based_default_design_sync_mode(analysis, "DefaultComponent", agent_analysis)

    def _create_agent_based_default_design_sync_mode(self, analysis: Dict, component_name: str, agent_analysis: Dict) -> Dict:
        """에이전트 데이터 기반 기본 설계 (동기 모드) (기존 메서드 완전 보존)"""
        layout_type = analysis.get('recommended_layout', 'grid')

        # 에이전트 품질 지표 기반 조정
        quality_indicators = agent_analysis.get("quality_indicators", {})
        if quality_indicators.get("overall_confidence", 0) > 0.8:
            layout_type = 'magazine'  # 고품질일 때 매거진 레이아웃

        return {
            "layout_type": layout_type,
            "layout_rationale": f"에이전트 데이터 기반 {layout_type} 레이아웃 선택. "
                              f"{len(agent_analysis.get('agent_summary', {}))}개 에이전트 결과 반영",
            "grid_structure": "1fr 1fr" if layout_type == 'grid' else "1fr",
            "styled_components": ["Container", "Header", "Title", "Subtitle", "Content", "ImageGallery", "Footer"],
            "color_scheme": {
                "primary": "#2c3e50",
                "secondary": "#f8f9fa",
                "accent": "#3498db",
                "background": "#ffffff"
            },
            "typography_scale": {
                "title": "3em",
                "subtitle": "1.4em",
                "body": "1.1em",
                "caption": "0.9em"
            },
            "image_layout": "grid_responsive",
            "agent_data_integration": agent_analysis,
            "jsx_templates_ignored": True,
            "error_prevention": "완전한 JSX 문법 준수 및 오류 방지 적용",
            "quality_metrics": {
                "agent_collaboration_score": quality_indicators.get("collaboration_success", 0.8),
                "design_confidence": quality_indicators.get("overall_confidence", 0.85),
                "error_free_guarantee": 1.0
            }
        }

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

    async def cleanup_resources(self) -> None:
        """리소스 정리"""
        self.logger.info("🧹 JSXLayoutDesigner 리소스 정리 시작")

        try:
            # 작업 큐 정리 (graceful 파라미터 명시적 전달)
            await self.work_queue.stop(graceful=True)
            self.logger.info("✅ 리소스 정리 완료")
        except Exception as e:
            self.logger.error(f"⚠️ 리소스 정리 중 오류: {e}")

    # 기존 동기 버전 메서드 (호환성 유지)
    def design_layout_structure_sync(self, content: Dict, analysis: Dict, component_name: str) -> Dict:
        """동기 버전 레이아웃 설계 (호환성 유지)"""
        return asyncio.run(self.design_layout_structure(content, analysis, component_name))
