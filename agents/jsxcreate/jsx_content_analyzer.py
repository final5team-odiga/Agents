import asyncio
from typing import Dict, List, Callable, Optional, Any
from dataclasses import dataclass, field
from crewai import Agent, Task, Crew, Process
from custom_llm import get_azure_llm
from utils.pdf_vector_manager import PDFVectorManager
from utils.agent_decision_logger import get_complete_data_manager
import time
import logging
from utils.hybridlogging import get_hybrid_logger
import sys
from enum import Enum
import inspect
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
    """작업 항목 정의"""
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
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"




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
    """표준화된 비동기 작업 큐 (배치 처리 특화)"""
    
    def __init__(self, max_workers: int = 2, max_queue_size: int = 50, batch_size: int = 3):
        # 표준화된 구조로 변경
        self._queue = asyncio.PriorityQueue(max_queue_size if max_queue_size > 0 else 0)
        self._workers: List[asyncio.Task] = []
        self._max_workers = max_workers
        self._running = False
        self._results: Dict[str, Any] = {}  # 표준화된 결과 저장 형식
        self.logger = logging.getLogger(self.__class__.__name__)
        # 배치 처리 특화 설정 유지
        self.batch_size = batch_size
        self.semaphore = asyncio.Semaphore(max_workers)
        self._batch_queue = asyncio.Queue()
        self._batch_processing = False
        
        # 표준 로깅 사용
        self._internal_logger = logging.getLogger(self.__class__.__name__)

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


    async def _collect_batch_from_priority_queue(self) -> List[WorkItem]:
        """우선순위 큐에서 배치 수집"""
        batch = []
        for _ in range(self.batch_size):
            try:
                item: WorkItem = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                batch.append(item)
            except asyncio.TimeoutError:
                break
        return batch

    async def _process_batch_with_semaphore(self, batch: List[WorkItem], worker_id: int):
        """세마포어를 사용한 배치 처리"""
        async with self.semaphore:
            tasks = [self._execute_work_item_safe(item, worker_id) for item in batch]
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _execute_work_item_safe(self, work_item: WorkItem, worker_id: int):
        """안전한 개별 작업 실행"""
        try:
            self._internal_logger.info(f"Worker {worker_id} processing task {work_item.id}")
            
            if asyncio.iscoroutinefunction(work_item.task_func):
                result = await asyncio.wait_for(
                    work_item.task_func(*work_item.args, **work_item.kwargs),
                    timeout=work_item.timeout
                )
            else:
                loop = asyncio.get_event_loop()
                result = await asyncio.wait_for(
                    loop.run_in_executor(None, lambda: work_item.task_func(*work_item.args, **work_item.kwargs)),
                    timeout=work_item.timeout
                )

            # 표준화된 결과 저장 형식
            self._results[work_item.id] = {"status": "success", "result": result}
            self._internal_logger.info(f"Task {work_item.id} completed successfully by worker {worker_id}.")
            
        except asyncio.TimeoutError:
            error_msg = f"Task {work_item.id} timed out"
            self._results[work_item.id] = {"status": "timeout", "error": error_msg}
            self._internal_logger.error(error_msg)
            
        except Exception as e:
            error_msg = f"Task {work_item.id} failed: {e}"
            self._results[work_item.id] = {"status": "error", "error": str(e)}
            self._internal_logger.error(error_msg)
        
        finally:
            self._queue.task_done()

    async def start(self):
        """표준화된 시작 메서드"""
        if not self._running:
            self._running = True
            self._internal_logger.info(f"Starting {self._max_workers} workers with batch processing.")
            self._workers = [asyncio.create_task(self._worker(i)) for i in range(self._max_workers)]

    async def stop(self, graceful=True):
        """표준화된 정지 메서드"""
        if self._running:
            self._internal_logger.info("Stopping work queue...")
            self._running = False
            
            if graceful:
                await self._queue.join()
            
            if self._workers:
                for worker_task in self._workers:
                    worker_task.cancel()
                await asyncio.gather(*self._workers, return_exceptions=True)
                self._workers.clear()
            
            self._internal_logger.info("Work queue stopped.")

    async def enqueue_work(self, item: WorkItem) -> bool:
        """표준화된 작업 추가 메서드"""
        if not self._running:
            await self.start()
        
        try:
            await self._queue.put(item)
            self._internal_logger.debug(f"Enqueued task {item.id}")
            return True
        except asyncio.QueueFull:
            self._internal_logger.warning(f"Queue is full. Could not enqueue task {item.id}")
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
        """표준화된 결과 정리"""
        self._results.clear()

    # 기존 메서드들 (호환성 유지)
    async def submit_work(self, work_item: WorkItem) -> str:
        """기존 호환성을 위한 메서드"""
        success = await self.enqueue_work(work_item)
        if success:
            return work_item.id
        else:
            raise Exception("Failed to submit work")

    async def clear_result(self, work_id: str):
        """기존 호환성을 위한 메서드"""
        if work_id in self._results:
            del self._results[work_id]


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

                result = await asyncio.wait_for(
                    actual_circuit_breaker.execute(task_func, *args, **kwargs),
                    timeout=current_timeout
                )
                
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
                raise e  # RecursionError는 즉시 상위로 전파하여 동기 모드 전환 유도
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

# ==================== 개선된 JSXContentAnalyzer ====================

class JSXContentAnalyzer(BaseAsyncAgent):
    """콘텐츠 분석 전문 에이전트 (CrewAI 기반 에이전트 결과 데이터 통합)"""

    def __init__(self):
        super().__init__()  # BaseAsyncAgent 명시적 초기화
        self.llm = get_azure_llm()
        self.vector_manager = PDFVectorManager()
        self.result_manager = get_complete_data_manager()
        
        
        # JSX 콘텐츠 분석 특화 타임아웃 설정
        self.timeouts.update({
            'content_analysis': 120.0,
            'crew_execution': 100.0,
            'agent_result_analysis': 30.0,
            'vector_enhancement': 20.0
        })

        # CrewAI 에이전트들 생성 (기존 방식 유지)
        self.content_analysis_agent = self._create_content_analysis_agent()
        self.agent_result_analyzer = self._create_agent_result_analyzer()
        self.vector_enhancement_agent = self._create_vector_enhancement_agent()

        self.logger.info("SXContentAnalyzer 초기화 완료")
        self.logger.info(f"타임아웃 설정: {self.timeouts}")

    def some_method(self):
        # 일반 로깅은 표준 로거 사용
        self.logger.info("JSX code generation started")
        
    def _create_content_analysis_agent(self):
        """콘텐츠 분석 전문 에이전트 (기존 메서드 완전 보존)"""
        return Agent(
            role="JSX 콘텐츠 분석 전문가",
            goal="JSX 생성을 위한 콘텐츠의 구조적 특성과 레이아웃 요구사항을 정밀 분석하여 최적화된 분석 결과를 제공",
            backstory="""당신은 10년간 React 및 JSX 기반 웹 개발 프로젝트에서 콘텐츠 분석을 담당해온 전문가입니다. 다양한 콘텐츠 유형에 대한 최적의 레이아웃과 디자인 패턴을 도출하는 데 특화되어 있습니다.

**전문 분야:**
- JSX 컴포넌트 구조 설계
- 콘텐츠 기반 레이아웃 최적화
- 사용자 경험 중심의 디자인 패턴 분석
- 반응형 웹 디자인 구조 설계

**분석 철학:**
"모든 콘텐츠는 고유한 특성을 가지며, 이를 정확히 분석하여 최적의 JSX 구조로 변환하는 것이 사용자 경험의 핵심입니다."

**출력 요구사항:**
- 콘텐츠 길이 및 복잡도 분석
- 감정 톤 및 분위기 파악
- 이미지 전략 및 배치 권장사항
- 레이아웃 복잡도 및 권장 구조
- 색상 팔레트 및 타이포그래피 스타일 제안""",
            verbose=True,
            llm=self.llm,
            allow_delegation=False
        )

    def _create_agent_result_analyzer(self):
        """에이전트 결과 분석 전문가 (기존 메서드 완전 보존)"""
        return Agent(
            role="에이전트 결과 데이터 분석 전문가",
            goal="이전 에이전트들의 실행 결과를 분석하여 성공 패턴과 최적화 인사이트를 도출하고 콘텐츠 분석에 반영",
            backstory="""당신은 8년간 다중 에이전트 시스템의 성능 분석과 최적화를 담당해온 전문가입니다. BindingAgent와 OrgAgent의 결과 패턴을 분석하여 JSX 생성 품질을 향상시키는 데 특화되어 있습니다.

**전문 영역:**
- 에이전트 실행 결과 패턴 분석
- 성공적인 레이아웃 전략 식별
- 에이전트 간 협업 최적화
- 품질 지표 기반 개선 방안 도출

**분석 방법론:**
"이전 에이전트들의 성공과 실패 패턴을 체계적으로 분석하여 현재 작업에 최적화된 인사이트를 제공합니다."

**특별 처리 대상:**
- BindingAgent: 이미지 배치 전략 및 시각적 일관성
- OrgAgent: 텍스트 구조 및 레이아웃 복잡도
- 성능 메트릭: 신뢰도 점수 및 품질 지표""",
            verbose=True,
            llm=self.llm,
            allow_delegation=False
        )

    def _create_vector_enhancement_agent(self):
        """벡터 데이터 강화 전문가 (기존 메서드 완전 보존)"""
        return Agent(
            role="PDF 벡터 데이터 기반 분석 강화 전문가",
            goal="PDF 벡터 데이터베이스에서 유사한 레이아웃 패턴을 검색하여 콘텐츠 분석 결과를 강화하고 최적화된 디자인 권장사항을 제공",
            backstory="""당신은 12년간 벡터 데이터베이스와 유사도 검색 시스템을 활용한 콘텐츠 최적화를 담당해온 전문가입니다. Azure Cognitive Search와 PDF 벡터 데이터를 활용한 레이아웃 패턴 분석에 특화되어 있습니다.

**기술 전문성:**
- 벡터 유사도 검색 및 패턴 매칭
- PDF 레이아웃 구조 분석
- 콘텐츠 기반 디자인 패턴 추출
- 색상 팔레트 및 타이포그래피 최적화

**강화 전략:**
"벡터 데이터베이스의 풍부한 레이아웃 정보를 활용하여 현재 콘텐츠에 가장 적합한 디자인 패턴을 식별하고 적용합니다."

**출력 강화 요소:**
- 유사 레이아웃 기반 구조 권장
- 벡터 신뢰도 기반 품질 향상
- PDF 소스 기반 색상 팔레트 최적화
- 타이포그래피 스타일 정교화""",
            verbose=True,
            llm=self.llm,
            allow_delegation=False
        )

    async def analyze_content_for_jsx(self, content: Dict, section_index: int, total_sections: int) -> Dict:
        """JSX 생성을 위한 콘텐츠 분석 (개선된 RecursionError 처리)"""
        # 재귀 깊이 체크
        if self._should_use_sync():
            return await self._analyze_content_for_jsx_sync_mode(content, section_index, total_sections)
        
        try:
            return await self._analyze_content_for_jsx_batch_mode(content, section_index, total_sections)
        except RecursionError as e:
            self.logger.warning(f"RecursionError detected, switching to sync mode: {e}")
            self.fallback_to_sync = True
            return await self._analyze_content_for_jsx_sync_mode(content, section_index, total_sections)
        except CircuitBreakerOpenError as e:
            self.logger.warning(f"Circuit breaker open, switching to sync mode: {e}")
            self.fallback_to_sync = True
            return await self._analyze_content_for_jsx_sync_mode(content, section_index, total_sections)

    async def _analyze_content_for_jsx_batch_mode(self, content: Dict, section_index: int, total_sections: int) -> Dict:
        """배치 기반 안전한 콘텐츠 분석"""
        task_id = f"content_analysis_{section_index}_{int(time.time())}"

        async def _safe_content_analysis():
            return await self._execute_content_analysis_pipeline(content, section_index, total_sections)

        try:
            result = await self.execute_with_resilience(
                task_id,
                _safe_content_analysis,
                timeout=self.timeouts['content_analysis'],
                max_retries=2
            )


            if result and not str(result).startswith("FALLBACK_RESULT"):
                return result
            else:
                self.logger.warning(f"Batch mode returned fallback for section {section_index}, switching to sync mode")
                return await self._analyze_content_for_jsx_sync_mode(content, section_index, total_sections)

        except Exception as e:
            self.logger.error(f"Batch mode failed for section {section_index}: {e}")
            return await self._analyze_content_for_jsx_sync_mode(content, section_index, total_sections)

    async def _analyze_content_for_jsx_sync_mode(self, content: Dict, section_index: int, total_sections: int) -> Dict:
        """동기 모드 폴백 처리"""
        try:
            self.logger.info(f"Executing content analysis in sync mode for section {section_index}/{total_sections}")
            
            # 안전한 결과 수집
            previous_results = await self._safe_collect_results()
            binding_results = [
                r for r in previous_results if "BindingAgent" in r.get('agent_name', '')]
            org_results = [
                r for r in previous_results if "OrgAgent" in r.get('agent_name', '')]

            self.logger.info(
                f"Sync mode result collection: Total {len(previous_results)}, BindingAgent {len(binding_results)}, OrgAgent {len(org_results)}")

            # 기본 분석 수행
            basic_analysis = self._create_default_analysis(content, section_index)

            # 에이전트 결과로 강화
            agent_enhanced_analysis = self._enhance_analysis_with_agent_results(
                content, basic_analysis, previous_results, binding_results, org_results
            )

            # 간단한 벡터 강화
            vector_enhanced_analysis = await self._safe_enhance_analysis_with_vectors(content, agent_enhanced_analysis)

            # 결과 저장
            await self._safe_store_result(
                vector_enhanced_analysis, content, section_index, total_sections,
                len(previous_results), len(binding_results), len(org_results)
            )

            self.logger.info(
                f"Sync mode content analysis completed: {vector_enhanced_analysis.get('recommended_layout', 'default')} layout")

            return vector_enhanced_analysis

        except Exception as e:
            self.logger.error(f"Sync mode analysis failed: {e}")
            return self._get_fallback_result(f"content_analysis_{section_index}")

    async def _execute_content_analysis_pipeline(self, content: Dict, section_index: int, total_sections: int) -> Dict:
        """개선된 콘텐츠 분석 파이프라인"""
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
        tasks = await self._create_analysis_tasks_safe(content, section_index, total_sections, previous_results, binding_results, org_results)

        # 3단계: CrewAI Crew 실행 (Circuit Breaker 적용)
        crew_result = await self._execute_crew_safe(tasks)

        # 4단계: 결과 처리 및 통합 (타임아웃 적용)
        vector_enhanced_analysis = await self._process_crew_analysis_result_safe(
            crew_result, content, section_index, previous_results, binding_results, org_results
        )

        # 5단계: 결과 저장
        await self._safe_store_result(
            vector_enhanced_analysis, content, section_index, total_sections,
            len(previous_results), len(binding_results), len(org_results)
        )

        self.logger.info(
            f"Content analysis completed: {vector_enhanced_analysis.get('recommended_layout', 'default')} layout recommended (CrewAI based agent data utilization: {len(previous_results)})")

        return vector_enhanced_analysis

    async def _safe_collect_results(self) -> List[Dict]:
        """안전한 결과 수집"""
        try:
            return await asyncio.wait_for(
                self.result_manager.get_all_outputs(
                    exclude_agent="JSXContentAnalyzer"),
                timeout=self.timeouts['result_collection']
            )
        except asyncio.TimeoutError:
            self.logger.warning("Result collection timeout, using empty results")
            return []
        except Exception as e:
            self.logger.error(f"Result collection failed: {e}")
            return []

    async def _create_analysis_tasks_safe(
        self,
        content: Dict,
        section_index: int,
        total_sections: int,
        previous_results: List[Dict],
        binding_results: List[Dict],
        org_results: List[Dict]
    ) -> List[Task]:
        """안전한 분석 태스크 생성"""
        try:
            content_analysis_task = self._create_content_analysis_task(
                content, section_index, total_sections)
            agent_result_analysis_task = self._create_agent_result_analysis_task(
                previous_results, binding_results, org_results)
            vector_enhancement_task = self._create_vector_enhancement_task(content)

            return [content_analysis_task, agent_result_analysis_task, vector_enhancement_task]
        except Exception as e:
            self.logger.error(f"Task creation failed: {e}")
            # 최소한의 기본 태스크 반환
            return [self._create_content_analysis_task(content, section_index, total_sections)]

    async def _execute_crew_safe(self, tasks: List[Task]) -> Any:
        """안전한 CrewAI 실행 (동기 메서드 올바른 처리)"""
        try:
            # CrewAI Crew 생성
            analysis_crew = Crew(
                agents=[self.content_analysis_agent,
                    self.agent_result_analyzer, self.vector_enhancement_agent],
                tasks=tasks,
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


    async def _process_crew_analysis_result_safe(
        self,
        crew_result: Any,
        content: Dict,
        section_index: int,
        previous_results: List[Dict],
        binding_results: List[Dict],
        org_results: List[Dict]
    ) -> Dict:
        """안전한 CrewAI 분석 결과 처리"""
        try:
            return await asyncio.wait_for(
                self._process_crew_analysis_result(
                    crew_result, content, section_index,
                    previous_results, binding_results, org_results
                ),
                timeout=self.timeouts['post_processing']
            )
        except asyncio.TimeoutError:
            self.logger.warning("Crew result processing timeout, using fallback")
            return await self._create_fallback_analysis(content, section_index, previous_results, binding_results, org_results)
        except Exception as e:
            self.logger.error(f"Crew result processing failed: {e}")
            return await self._create_fallback_analysis(content, section_index, previous_results, binding_results, org_results)

    async def _safe_enhance_analysis_with_vectors(self, content: Dict, basic_analysis: Dict) -> Dict:
        """안전한 벡터 데이터 강화"""
        try:
            return await asyncio.wait_for(
                self._enhance_analysis_with_vectors(content, basic_analysis),
                timeout=self.timeouts['vector_enhancement']
            )
        except asyncio.TimeoutError:
            self.logger.warning("Vector enhancement timeout, using basic analysis")
            basic_analysis['vector_enhanced'] = False
            return basic_analysis
        except Exception as e:
            self.logger.error(f"Vector enhancement failed: {e}")
            basic_analysis['vector_enhanced'] = False
            return basic_analysis

    async def _safe_store_result(
        self,
        analysis_result: Dict,
        content: Dict,
        section_index: int,
        total_sections: int,
        agent_count: int,
        binding_count: int,
        org_count: int
    ):
        """안전한 결과 저장"""
        try:
            await asyncio.wait_for(
                self.result_manager.store_agent_output(
                    agent_name="JSXContentAnalyzer",
                    agent_role="콘텐츠 분석 전문가",
                    task_description=f"섹션 {section_index+1}/{total_sections} JSX 콘텐츠 분석",
                    final_answer=str(analysis_result),
                    reasoning_process=f"CrewAI 기반 이전 {agent_count}개 에이전트 결과 분석 후 벡터 데이터 강화 적용",
                    execution_steps=[
                        "CrewAI 에이전트 생성",
                        "기본 콘텐츠 분석 수행",
                        "에이전트 결과 통합",
                        "벡터 데이터 강화",
                        "최종 분석 완료"
                    ],
                    raw_input=content,
                    raw_output=analysis_result,
                    performance_metrics={
                        "section_index": section_index,
                        "total_sections": total_sections,
                        "agent_results_utilized": agent_count,
                        "binding_results_count": binding_count,
                        "org_results_count": org_count,
                        "vector_enhanced": analysis_result.get('vector_enhanced', False),
                        "crewai_enhanced": True,
                        "safe_mode_used": self.fallback_to_sync
                    }
                ),
                timeout=5.0
            )
        except Exception as e:
            self.logger.error(f"Failed to store result: {e}")

    async def _create_fallback_analysis(
        self,
        content: Dict,
        section_index: int,
        previous_results: List[Dict],
        binding_results: List[Dict],
        org_results: List[Dict]
    ) -> Dict:
        """폴백 분석 결과 생성"""
        basic_analysis = self._create_default_analysis(content, section_index)
        
        # 에이전트 결과가 있다면 간단히 적용
        if previous_results:
            basic_analysis = self._enhance_analysis_with_agent_results(
                content, basic_analysis, previous_results, binding_results, org_results
            )

        basic_analysis.update({
            'fallback_mode': True,
            'crewai_enhanced': False,
            'vector_enhanced': False,
            'agent_results_count': len(previous_results)
        })

        return basic_analysis

    def _get_fallback_result(self, task_id: str) -> Dict:
        """JSX 콘텐츠 분석 전용 폴백 결과 생성"""
        section_index = 0
        if "content_analysis_" in task_id:
            try:
                section_index = int(task_id.split("_")[2])
            except:
                pass

        return {
            "text_length": "보통",
            "emotion_tone": "peaceful",
            "image_strategy": "그리드",
            "layout_complexity": "보통",
            "recommended_layout": "grid",
            "color_palette": "안전 모드 블루",
            "typography_style": "기본 모던",
            "section_index": section_index,
            "fallback_mode": True,
            "agent_enhanced": False,
            "vector_enhanced": False,
            "crewai_enhanced": False,
            "safe_mode_reason": "시스템 제약으로 인한 안전 모드 실행"
        }

    # ==================== 기존 메서드들 (완전 보존) ====================

    def _create_content_analysis_task(self, content: Dict, section_index: int, total_sections: int) -> Task:
        """기본 콘텐츠 분석 태스크 (기존 메서드 완전 보존)"""
        return Task(
            description=f"""
섹션 {section_index+1}/{total_sections}의 콘텐츠를 분석하여 JSX 생성에 필요한 기본 분석 결과를 제공하세요.

**분석 대상 콘텐츠:**
- 제목: {content.get('title', 'N/A')}
- 본문 길이: {len(content.get('body', ''))} 문자
- 이미지 개수: {len(content.get('images', []))}개

**분석 요구사항:**
1. 텍스트 길이 분석 (짧음/보통/긺)
2. 감정 톤 파악 (peaceful/energetic/professional 등)
3. 이미지 전략 권장 (단일/그리드/갤러리)
4. 레이아웃 복잡도 평가 (단순/보통/복잡)
5. 권장 레이아웃 타입 (minimal/hero/grid/magazine)
6. 색상 팔레트 제안
7. 타이포그래피 스타일 권장

**출력 형식:**
JSON 형태로 분석 결과를 구조화하여 제공하세요.
""",
            expected_output="JSX 생성을 위한 기본 콘텐츠 분석 결과 (JSON 형식)",
            agent=self.content_analysis_agent
        )

    def _create_agent_result_analysis_task(self, previous_results: List[Dict], binding_results: List[Dict], org_results: List[Dict]) -> Task:
        """에이전트 결과 분석 태스크 (기존 메서드 완전 보존)"""
        return Task(
            description=f"""
이전 에이전트들의 실행 결과를 분석하여 성공 패턴과 최적화 인사이트를 도출하세요.

**분석 대상:**
- 전체 에이전트 결과: {len(previous_results)}개
- BindingAgent 결과: {len(binding_results)}개 (이미지 배치 전략)
- OrgAgent 결과: {len(org_results)}개 (텍스트 구조)

**특별 분석 요구사항:**
1. BindingAgent 결과에서 이미지 배치 전략 추출
   - 그리드/갤러리 패턴 식별
   - 시각적 일관성 평가
2. OrgAgent 결과에서 텍스트 구조 분석
   - 레이아웃 복잡도 평가
   - 타이포그래피 스타일 추출
3. 성공 패턴 학습
   - 높은 신뢰도를 보인 접근법 식별
   - 레이아웃 권장사항 도출
   - 품질 향상 전략 제안

**출력 요구사항:**
- 에이전트별 인사이트 요약
- 성공적인 레이아웃 패턴
- 품질 향상 권장사항
""",
            expected_output="에이전트 결과 분석 및 최적화 인사이트 (구조화된 데이터)",
            agent=self.agent_result_analyzer
        )

    def _create_vector_enhancement_task(self, content: Dict) -> Task:
        """벡터 데이터 강화 태스크 (기존 메서드 완전 보존)"""
        return Task(
            description=f"""
PDF 벡터 데이터베이스를 활용하여 콘텐츠 분석 결과를 강화하세요.

**검색 쿼리 생성:**
- 콘텐츠 제목: {content.get('title', '')}
- 본문 일부: {content.get('body', '')[:300]}

**벡터 검색 및 분석:**
1. 유사한 레이아웃 패턴 검색 (top 5)
2. 레이아웃 타입 분석 및 권장사항 도출
3. 벡터 신뢰도 기반 품질 점수 계산
4. PDF 소스 기반 색상 팔레트 최적화
5. 타이포그래피 스타일 정교화

**강화 요소:**
- 벡터 기반 레이아웃 권장
- 신뢰도 점수 계산
- 색상 팔레트 최적화
- 타이포그래피 스타일 개선

**실패 처리:**
벡터 검색 실패 시 기본 분석 결과 유지
""",
            expected_output="벡터 데이터 기반 강화된 분석 결과",
            agent=self.vector_enhancement_agent,
            context=[self._create_content_analysis_task(
                content, 0, 1), self._create_agent_result_analysis_task([], [], [])]
        )

    async def _process_crew_analysis_result(self, crew_result, content: Dict, section_index: int,
                                          previous_results: List[Dict], binding_results: List[Dict],
                                          org_results: List[Dict]) -> Dict:
        """CrewAI 분석 결과 처리 (기존 메서드 완전 보존)"""
        try:
            # CrewAI 결과에서 데이터 추출
            if hasattr(crew_result, 'raw') and crew_result.raw:
                result_text = crew_result.raw
            else:
                result_text = str(crew_result)

            # 기본 분석 수행
            basic_analysis = self._create_default_analysis(content, section_index)

            # 에이전트 결과 데이터로 분석 강화
            agent_enhanced_analysis = self._enhance_analysis_with_agent_results(
                content, basic_analysis, previous_results, binding_results, org_results
            )

            # 벡터 데이터로 추가 강화
            vector_enhanced_analysis = await self._enhance_analysis_with_vectors(content, agent_enhanced_analysis)

            # CrewAI 결과 통합
            vector_enhanced_analysis['crewai_enhanced'] = True
            vector_enhanced_analysis['crew_result_length'] = len(result_text)

            return vector_enhanced_analysis

        except Exception as e:
            self.logger.error(f"CrewAI result processing failed: {e}")
            # 폴백: 기존 방식으로 처리
            basic_analysis = self._create_default_analysis(content, section_index)
            agent_enhanced_analysis = self._enhance_analysis_with_agent_results(
                content, basic_analysis, previous_results, binding_results, org_results
            )
            return await self._enhance_analysis_with_vectors(content, agent_enhanced_analysis)

    def _enhance_analysis_with_agent_results(self, content: Dict, basic_analysis: Dict,
                                           previous_results: List[Dict], binding_results: List[Dict],
                                           org_results: List[Dict]) -> Dict:
        """에이전트 결과 데이터로 분석 강화 (기존 메서드 완전 보존)"""
        enhanced_analysis = basic_analysis.copy()
        enhanced_analysis['agent_results_count'] = len(previous_results)
        enhanced_analysis['binding_results_count'] = len(binding_results)
        enhanced_analysis['org_results_count'] = len(org_results)

        if not previous_results:
            enhanced_analysis['agent_enhanced'] = False
            return enhanced_analysis

        enhanced_analysis['agent_enhanced'] = True

        # 이전 분석 결과 패턴 학습
        layout_recommendations = []
        confidence_scores = []

        for result in previous_results:
            final_answer = result.get('agent_final_answer', '')
            if 'layout' in final_answer.lower():
                if 'grid' in final_answer.lower():
                    layout_recommendations.append('grid')
                elif 'hero' in final_answer.lower():
                    layout_recommendations.append('hero')
                elif 'magazine' in final_answer.lower():
                    layout_recommendations.append('magazine')

            # 성능 메트릭에서 신뢰도 추출
            performance_data = result.get('performance_data', {})
            if isinstance(performance_data, dict):
                confidence = performance_data.get('confidence_score', 0)
                if confidence > 0:
                    confidence_scores.append(confidence)

        # BindingAgent 결과 특별 활용
        if binding_results:
            latest_binding = binding_results[-1]
            binding_answer = latest_binding.get('agent_final_answer', '')

            # 이미지 배치 전략에서 레이아웃 힌트 추출
            if '그리드' in binding_answer or 'grid' in binding_answer.lower():
                enhanced_analysis['image_strategy'] = '그리드'
                enhanced_analysis['recommended_layout'] = 'grid'
            elif '갤러리' in binding_answer or 'gallery' in binding_answer.lower():
                enhanced_analysis['image_strategy'] = '갤러리'
                enhanced_analysis['recommended_layout'] = 'gallery'

            enhanced_analysis['binding_insights_applied'] = True
            self.logger.info("BindingAgent insights applied: image strategy adjusted")

        # OrgAgent 결과 특별 활용
        if org_results:
            latest_org = org_results[-1]
            org_answer = latest_org.get('agent_final_answer', '')

            # 텍스트 구조에서 레이아웃 힌트 추출
            if '복잡' in org_answer or 'complex' in org_answer.lower():
                enhanced_analysis['layout_complexity'] = '복잡'
                enhanced_analysis['typography_style'] = '정보 집약형'
            elif '단순' in org_answer or 'simple' in org_answer.lower():
                enhanced_analysis['layout_complexity'] = '단순'
                enhanced_analysis['typography_style'] = '미니멀 모던'

            enhanced_analysis['org_insights_applied'] = True
            self.logger.info("OrgAgent insights applied: text structure adjusted")

        # 가장 성공적인 레이아웃 패턴 적용
        if layout_recommendations:
            most_common_layout = max(
                set(layout_recommendations), key=layout_recommendations.count)
            if layout_recommendations.count(most_common_layout) >= 2:
                enhanced_analysis['recommended_layout'] = most_common_layout
                enhanced_analysis['layout_confidence'] = 'high'

        # 평균 신뢰도 기반 조정
        if confidence_scores:
            avg_confidence = sum(confidence_scores) / len(confidence_scores)
            if avg_confidence > 0.8:
                enhanced_analysis['quality_boost'] = True
                enhanced_analysis['color_palette'] = '프리미엄 블루'
                enhanced_analysis['typography_style'] = '고급 모던'

        return enhanced_analysis

    async def _enhance_analysis_with_vectors(self, content: Dict, basic_analysis: Dict) -> Dict:
        """벡터 데이터로 분석 강화 (기존 메서드 완전 보존)"""
        try:
            content_query = f"{content.get('title', '')} {content.get('body', '')[:300]}"
            similar_layouts = self.vector_manager.search_similar_layouts(
                content_query,
                "magazine_layout",
                top_k=5
            )

            if similar_layouts:
                enhanced_analysis = basic_analysis.copy()
                enhanced_analysis['vector_enhanced'] = True
                enhanced_analysis['similar_layouts'] = similar_layouts

                vector_layout_recommendation = await self._get_vector_layout_recommendation(similar_layouts)
                if vector_layout_recommendation:
                    enhanced_analysis['recommended_layout'] = vector_layout_recommendation

                enhanced_analysis['layout_confidence'] = self._calculate_vector_confidence(similar_layouts)
                enhanced_analysis['vector_color_palette'] = self._get_vector_color_palette(similar_layouts)
                enhanced_analysis['vector_typography'] = self._get_vector_typography_style(similar_layouts)

                return enhanced_analysis
            else:
                basic_analysis['vector_enhanced'] = False
                return basic_analysis

        except Exception as e:
            self.logger.error(f"Vector data analysis enhancement failed: {e}")
            basic_analysis['vector_enhanced'] = False
            return basic_analysis

    async def _get_vector_layout_recommendation(self, similar_layouts: List[Dict]) -> str:
        """벡터 데이터 기반 레이아웃 추천 (기존 메서드 완전 보존)"""
        layout_types = []
        for layout in similar_layouts:
            layout_info = layout.get('layout_info', {})
            text_blocks = len(layout_info.get('text_blocks', []))
            images = len(layout_info.get('images', []))

            if images == 0:
                layout_types.append('minimal')
            elif images == 1 and text_blocks <= 3:
                layout_types.append('hero')
            elif images <= 3 and text_blocks <= 6:
                layout_types.append('grid')
            elif images > 3:
                layout_types.append('gallery')
            else:
                layout_types.append('magazine')

        if layout_types:
            return max(set(layout_types), key=layout_types.count)
        return None

    def _calculate_vector_confidence(self, similar_layouts: List[Dict]) -> float:
        """벡터 기반 신뢰도 계산 (기존 메서드 완전 보존)"""
        if not similar_layouts:
            return 0.5

        scores = [layout.get('score', 0) for layout in similar_layouts]
        avg_score = sum(scores) / len(scores)

        layout_consistency = len(set(self._get_vector_layout_recommendation(
            [layout]) for layout in similar_layouts))
        consistency_bonus = 0.2 if layout_consistency <= 2 else 0.1

        return min(avg_score + consistency_bonus, 1.0)

    def _get_vector_color_palette(self, similar_layouts: List[Dict]) -> str:
        """벡터 데이터 기반 색상 팔레트 (기존 메서드 완전 보존)"""
        pdf_sources = [layout.get('pdf_name', '').lower()
                      for layout in similar_layouts]

        if any('travel' in source for source in pdf_sources):
            return "여행 블루 팔레트"
        elif any('culture' in source for source in pdf_sources):
            return "문화 브라운 팔레트"
        elif any('lifestyle' in source for source in pdf_sources):
            return "라이프스타일 핑크 팔레트"
        elif any('nature' in source for source in pdf_sources):
            return "자연 그린 팔레트"
        else:
            return "클래식 그레이 팔레트"

    def _get_vector_typography_style(self, similar_layouts: List[Dict]) -> str:
        """벡터 데이터 기반 타이포그래피 스타일 (기존 메서드 완전 보존)"""
        total_text_blocks = sum(len(layout.get('layout_info', {}).get(
            'text_blocks', [])) for layout in similar_layouts)
        avg_text_blocks = total_text_blocks / \
            len(similar_layouts) if similar_layouts else 0

        if avg_text_blocks > 8:
            return "정보 집약형"
        elif avg_text_blocks > 5:
            return "균형잡힌 편집형"
        elif avg_text_blocks > 2:
            return "미니멀 모던"
        else:
            return "대형 타이틀 중심"

    def _create_default_analysis(self, content: Dict, section_index: int) -> Dict:
        """기본 분석 결과 생성 (기존 메서드 완전 보존)"""
        body_length = len(content.get('body', ''))
        image_count = len(content.get('images', []))

        if body_length < 300:
            recommended_layout = "minimal"
        elif image_count == 0:
            recommended_layout = "minimal"
        elif image_count == 1:
            recommended_layout = "hero"
        elif image_count <= 4:
            recommended_layout = "grid"
        else:
            recommended_layout = "magazine"

        return {
            "text_length": "보통" if body_length < 500 else "긺",
            "emotion_tone": "peaceful",
            "image_strategy": "그리드" if image_count > 1 else "단일",
            "layout_complexity": "보통",
            "recommended_layout": recommended_layout,
            "color_palette": "차분한 블루",
            "typography_style": "모던"
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
        self.circuit_breaker.failure_count = 0
        self.circuit_breaker.state = CircuitBreakerState.CLOSED
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
    def analyze_content_for_jsx_sync(self, content: Dict, section_index: int, total_sections: int) -> Dict:
        """동기 버전 콘텐츠 분석 (호환성 유지)"""
        return asyncio.run(self.analyze_content_for_jsx(content, section_index, total_sections))
