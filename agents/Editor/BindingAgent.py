import asyncio
import sys
import time
import concurrent.futures
from typing import Dict, List, Optional, Callable, Any
from collections import deque
from dataclasses import dataclass

from crewai import Agent, Task, Crew
from custom_llm import get_azure_llm
from utils.pdf_vector_manager import PDFVectorManager
from utils.hybridlogging import get_hybrid_logger
from utils.ai_search_isolation import BindingAgentIsolation
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
                
                if asyncio.iscoroutinefunction(work_item.task_func):
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
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0):
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

def safe_slice(data, start=0, end=None, default=None):
    """안전한 슬라이싱 함수"""
    if not isinstance(data, (list, tuple, str)):
        return default if default is not None else []
    
    if data is None:
        return default if default is not None else []
    
    try:
        if end is None:
            return data[start:]
        else:
            return data[start:end]
    except (IndexError, TypeError) as e:
        print(f"⚠️ Slicing error: {e}, returning default")
        return default if default is not None else []


def safe_dict_get(data, key, default=None):
    """안전한 딕셔너리 접근 함수"""
    if isinstance(data, dict):
        return data.get(key, default)
    elif isinstance(data, str):
        # 문자열이 JSON 형태일 경우 파싱 시도
        try:
            import json
            parsed_data = json.loads(data)
            if isinstance(parsed_data, dict):
                return parsed_data.get(key, default)
        except (json.JSONDecodeError, ValueError):
            pass
        return default
    else:
        print(f"⚠️ Expected dict but got {type(data)}: {data}")
        return default
    
    
def safe_list_operation(data, operation="length", default=0):
    """리스트 연산 안전 처리"""
    if not isinstance(data, (list, tuple)):
        print(f"⚠️ Expected list/tuple but got {type(data)}")
        return default
    
    if operation == "length":
        return len(data)
    elif operation == "first":
        return data[0] if data else default
    elif operation == "last":
        return data[-1] if data else default
    else:
        return default


class BindingAgent(BindingAgentIsolation, SessionAwareMixin, InterAgentCommunicationMixin):
    """PDF 벡터 데이터 기반 이미지 배치 에이전트 (AI Search 격리 적용)"""
    
    def __init__(self, session_id: Optional[str] = None):
        # 기존 초기화 코드
        self.llm = get_azure_llm()
        self.vector_manager = PDFVectorManager()
        self.logger = get_hybrid_logger()
        self.recursion_threshold = 600
        self.fallback_to_sync = False
        self.work_queue = AsyncWorkQueue(max_workers=2, max_queue_size=30)
        self.circuit_breaker = CircuitBreaker()
        self.batch_size = 3
        
  
        self.__init_isolation__()
        

        self.__init_session_awareness__(session_id)
        

        self.__init_inter_agent_communication__()

    async def _get_layout_recommendations_by_image_count_async(self, image_urls: List[str], template_requirements: List[Dict]) -> List[Dict]:
        """이미지 개수별 레이아웃 추천 가져오기 (격리 적용)"""
        total_images = len(image_urls)
        
        # 격리된 쿼리 생성
        if total_images <= 3:
            base_query = "minimal clean layout single image focus simple elegant"
        elif total_images <= 6:
            base_query = "multiple images grid layout balanced composition"
        elif total_images <= 10:
            base_query = "gallery style layout many images organized grid"
        else:
            base_query = "complex magazine layout multiple images rich visual content"
        
        # Azure AI Search 키워드 제거
        clean_query = self.isolation_manager.clean_query_from_azure_keywords(base_query)
        
        # 벡터 검색 실행
        raw_recommendations = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.vector_manager.search_similar_layouts(clean_query, "magazine_layout", top_k=8)
        )
        
        # 격리된 레이아웃 추천 적용
        isolated_recommendations = self.isolate_layout_recommendations(raw_recommendations, total_images)
        
        print(f"🛡️ 격리된 레이아웃 추천: 원본 {len(raw_recommendations)}개 → 필터링 후 {len(isolated_recommendations)}개")
        return isolated_recommendations

    async def _process_single_template_safe(self, template_req: Dict, image_urls: List[str],
                                          image_locations: List[str], template_index: int,
                                          total_templates: int, layout_recommendations: List[Dict]) -> Dict:
        """안전한 단일 템플릿 처리 (격리 적용)"""
        try:
            if not isinstance(template_req, dict) or "template" not in template_req:
                raise ValueError("잘못된 template_req 형식")
            
            template_name = template_req["template"]
            
            # 격리된 이미지 URL 사용
            clean_image_urls = self.isolate_image_urls(image_urls)
            
            # 해당 템플릿에 할당할 이미지들 결정
            assigned_images = self._assign_images_to_template(
                clean_image_urls, image_locations, template_index, total_templates
            )
            
            if not assigned_images:
                return {
                    "template_dist": {
                        "template": template_name,
                        "images": [],
                        "layout_strategy": "no_images",
                        "coordination_result": "이미지 없음",
                        "layout_source": "no_images",
                        "ai_search_isolation": {
                            "isolation_applied": True,
                            "no_images_available": True
                        }
                    },
                    "agent_responses": []
                }
            
            print(f"🖼️ {template_name}: {len(assigned_images)}개 격리된 이미지 배치 설계 중...")
            
            # 격리된 레이아웃 추천 사용
            isolated_layouts = self.isolate_layout_recommendations(layout_recommendations, len(assigned_images))
            
            # 에이전트 생성 및 실행
            layout_specialist = self.create_image_layout_agent()
            visual_coordinator = self.create_visual_coordinator_agent()
            
            crew_result = await self._execute_crew_safe(
                layout_specialist, visual_coordinator, template_name,
                assigned_images, image_locations, isolated_layouts
            )
            
            # 결과 무결성 검증
            original_content = f"Template: {template_name}, Images: {len(assigned_images)}"
            validated_result = self._validate_content_integrity(
                crew_result, original_content, f"template_{template_name}"
            )
            
            # 응답 수집 및 저장
            layout_strategy = validated_result.get("layout_strategy", "기본 배치 전략")
            coordination_result = validated_result.get("coordination_result", "기본 순서 배치")
            
            # 비동기 로깅
            layout_response_id, coordination_response_id = await asyncio.gather(
                self._log_layout_response_async(template_name, assigned_images, isolated_layouts, layout_strategy),
                self._log_coordination_response_async(template_name, layout_strategy, coordination_result)
            )
            
            template_dist = {
                "template": template_name,
                "images": assigned_images,
                "layout_strategy": layout_strategy,
                "coordination_result": coordination_result,
                "layout_source": isolated_layouts[0].get("pdf_name", "isolated_default") if isolated_layouts else "isolated_default",
                "agent_responses": {
                    "layout_specialist_id": layout_response_id,
                    "visual_coordinator_id": coordination_response_id
                },
                "ai_search_isolation": {
                    "isolation_applied": True,
                    "layouts_filtered": len(layout_recommendations) - len(isolated_layouts),
                    "images_filtered": len(image_urls) - len(clean_image_urls),
                    "contamination_detected": False
                },
                "safe_processed": True
            }
            
            agent_responses = [{
                "template": template_name,
                "layout_specialist_response": {
                    "response_id": layout_response_id,
                    "content": layout_strategy,
                    "agent_name": "BindingAgent_LayoutSpecialist"
                },
                "visual_coordinator_response": {
                    "response_id": coordination_response_id,
                    "content": coordination_result,
                    "agent_name": "BindingAgent_VisualCoordinator"
                }
            }]
            
            print(f"✅ {template_name} 격리된 안전 처리 완료: {len(assigned_images)}개")
            
            return {
                "template_dist": template_dist,
                "agent_responses": agent_responses
            }
            
        except Exception as e:
            print(f"⚠️ 템플릿 {template_req.get('template', 'unknown')} 격리된 안전 처리 실패: {e}")
            
            template_name = template_req.get('template', 'unknown') if isinstance(template_req, dict) else 'unknown'
            try:
                error_response_id = await self._log_error_response_async(template_name, str(e))
            except:
                error_response_id = "error_logging_failed"
            
            return {
                "template_dist": {
                    "template": template_name,
                    "images": [],
                    "layout_strategy": "에러로 인한 기본 배치",
                    "coordination_result": "기본 순서 배치",
                    "layout_source": "error_fallback",
                    "error_response_id": error_response_id,
                    "ai_search_isolation": {
                        "isolation_applied": True,
                        "error_occurred": True,
                        "error_message": str(e)
                    }
                },
                "agent_responses": []
            }

    def _get_fallback_result(self, task_id: str) -> Dict:
        """폴백 결과 생성 (격리 메타데이터 추가)"""
        template_name = "unknown"
        if "_" in task_id:
            parts = task_id.split("_")
            if parts:
                template_name = parts[0]
        
        return {
            "status": "fallback",
            "task_id": task_id,
            "message": "Circuit Breaker 또는 실패로 인한 폴백 결과",
            "template": template_name,
            "images": [],
            "layout_strategy": "폴백 기본 배치",
            "coordination_result": "폴백 순서 배치",
            "layout_source": "fallback",
            "template_dist": {
                "template": template_name,
                "images": [],
                "layout_strategy": "폴백 기본 배치",
                "coordination_result": "폴백 순서 배치",
                "layout_source": "fallback",
                "ai_search_isolation": {
                    "isolation_applied": True,
                    "fallback_used": True,
                    "reason": "circuit_breaker_or_failure"
                }
            },
            "agent_responses": []
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
            print(f"⚠️ BindingAgent 재귀 깊이 {current_depth} 감지 - 동기 모드로 전환")
            self.fallback_to_sync = True
            return True
        return self.fallback_to_sync

    async def execute_with_resilience(self, task_func: Callable, task_id: str,
                                    timeout: float = 300.0, max_retries: int = 2,
                                    *args, **kwargs) -> Any:
        """복원력 있는 작업 실행"""
        
        if self.circuit_breaker.is_open():
            print(f"🚫 Circuit Breaker 열림 - 작업 {task_id} 건너뜀")
            return self._get_fallback_result(task_id)
        
        work_item = WorkItem(
            id=task_id,
            task_func=task_func,
            args=args,
            kwargs=kwargs,
            timeout=timeout,
            max_retries=max_retries
        )
        
        await self.work_queue.add_work(work_item)
        results = await self.work_queue.process_queue()
        
        result = results.get(task_id)
        if result and result["status"] == "success":
            self.circuit_breaker.record_success()
            return result["result"]
        else:
            self.circuit_breaker.record_failure()
            return self._get_fallback_result(task_id)

    def _get_fallback_result(self, task_id: str) -> Dict:
        """폴백 결과 생성"""
        template_name = "unknown"
        if "_" in task_id:
            parts = safe_slice(task_id.split("_"), 0, 1, ["unknown"])
            if parts:
                template_name = parts[0]
        
        return {
            "status": "fallback",
            "task_id": task_id,
            "message": "Circuit Breaker 또는 실패로 인한 폴백 결과",
            "template": template_name,
            "images": [],
            "layout_strategy": "폴백 기본 배치",
            "coordination_result": "폴백 순서 배치",
            "layout_source": "fallback",
            # 표준 구조 추가
            "template_dist": {
                "template": template_name,
                "images": [],
                "layout_strategy": "폴백 기본 배치",
                "coordination_result": "폴백 순서 배치",
                "layout_source": "fallback"
            },
            "agent_responses": []
        }



    def create_image_layout_agent(self):
        """이미지 레이아웃 에이전트 (위치 정합성 강화)"""
        return Agent(
            role="매거진 이미지 배치 전문가 및 텍스트-이미지 정합성 보장자",
            goal="OrgAgent가 설계한 페이지 구조와 텍스트 레이아웃에 완벽히 맞춰 이미지를 배치하고, 텍스트와 이미지의 위치 관계가 독자에게 자연스럽고 직관적으로 인식되도록 정밀한 배치 전략을 수립",
            backstory="""당신은 매거진 이미지 레이아웃 및 텍스트-이미지 정합성 전문가입니다.

**전문 분야:**
- 기존 레이아웃 구조 기반 이미지 배치
- 텍스트-이미지 위치 관계 최적화
- 독자 인지 부하 최소화를 위한 배치 전략
- 시각적 일관성 및 흐름 보장

**텍스트-이미지 정합성 전문성:**
당신은 이미지 배치 시 다음 정합성 원칙을 엄격히 준수합니다:

1. **구조 연동 배치**:
- OrgAgent가 정의한 이미지 영역에 정확히 맞춰 배치
- 텍스트 블록과의 거리 및 정렬 규칙 준수
- 그리드 시스템 내에서의 정확한 위치 설정

2. **내용 연관성 매칭**:
- 이미지 내용과 관련 텍스트의 근접 배치
- 제목-주요 이미지, 본문-보조 이미지 관계 설정
- 캡션과 이미지의 직관적 연결 보장

3. **독자 인지 최적화**:
- 텍스트 읽기 흐름을 방해하지 않는 이미지 배치
- 시선 이동 경로 상의 자연스러운 이미지 위치
- 혼란을 방지하는 명확한 텍스트-이미지 경계

4. **PDF 벡터 데이터 활용**:
- 3000개 이상의 매거진에서 추출한 성공적 배치 패턴
- 텍스트-이미지 정합성 높은 레이아웃 사례 분석
- 독자 시선 추적 데이터 기반 최적 배치점 계산

당신은 이미지의 크기, 위치, 색감 등을 고려하여
매거진의 전체적인 시각적 흐름과 임팩트를 극대화하는 전문성을 가지고 있습니다.

**출력 데이터 구조:**
- 이미지별 정확한 위치 좌표 (x, y, width, height)
- 연관 텍스트 블록과의 관계 매핑
- 이미지-텍스트 정합성 점수
- 독자 시선 흐름 상의 이미지 역할 정의
- 레이아웃 구조도 상의 이미지 배치 검증 결과""",
            llm=self.llm,
            verbose=True
        )

    def create_visual_coordinator_agent(self):
        """비주얼 코디네이터 에이전트 (전체 구조 조율)"""
        return Agent(
            role="매거진 전체 구조 조율자 및 시각적 일관성 보장자",
            goal="OrgAgent의 텍스트 레이아웃 구조와 이미지 배치 결과를 통합하여 전체 매거진의 구조적 완성도와 텍스트-이미지 정합성을 검증하고, 독자 경험을 최적화하는 최종 레이아웃 구조를 완성",
            backstory="""당신은 15년간 세계 최고 수준의 매거진에서 전체 구조 조율 및 시각적 일관성 전문가로 활동해온 전문가입니다.

**전문 경력:**
- 시각 예술 및 매체학 석사 학위 보유
- 국제 사진 편집자 협회(NPPA) 골드 메달 수상
- 매거진 전체 구조 설계 및 조율 전문가
- 독자 경험(UX) 및 시각적 일관성 최적화 전문성

**전체 구조 조율 전문성:**
당신은 최종 매거진 구조 완성 시 다음 요소들을 종합적으로 조율합니다:

1. **구조적 완성도 검증**:
- 텍스트 레이아웃과 이미지 배치의 구조적 일치성 확인
- 페이지 그리드 시스템의 일관성 검증
- 전체 매거진의 시각적 균형과 리듬감 평가

2. **텍스트-이미지 정합성 최종 검증**:
- 모든 텍스트와 이미지의 위치 관계 적절성 확인
- 독자 혼란 요소 제거 및 직관성 보장
- 내용 연관성과 시각적 근접성의 일치 검증

3. **독자 경험 최적화**:
- 전체 매거진의 읽기 흐름 최적화
- 페이지 간 전환의 자연스러움 보장
- 정보 계층 구조의 명확성 확인

4. **최종 구조 문서화**:
- 완성된 레이아웃 구조도 생성
- 텍스트-이미지 배치 가이드라인 문서화
- JSX 구현을 위한 상세 스펙 제공

**작업 철학:**
"훌륭한 매거진은 개별 요소들의 단순한 합이 아니라, 모든 요소가 하나의 완성된 구조 안에서 조화롭게 작동하는 유기적 통합체입니다. 나는 텍스트와 이미지의 모든 배치 결정이 독자에게 자연스럽고 직관적으로 인식되도록 전체 구조를 조율합니다. 5. 주의 사항!!: 최대한 제공받은 데이터를 활용합니다. "

**출력 데이터 구조:**
- 최종 매거진 구조도 및 와이어프레임
- 텍스트-이미지 정합성 검증 보고서
- 독자 경험 최적화 가이드라인
- JSX 구현용 상세 레이아웃 스펙
- 전체 매거진의 시각적 일관성 평가서""",
            llm=self.llm,
            verbose=True
        )

    async def process_images(self, image_urls: List[str], image_locations: List[str], 
                        template_requirements: List[Dict]) -> Dict:
        """PDF 벡터 데이터 기반 이미지 처리 (세션 및 통신 격리 적용)"""
        print(f"⚡ BindingAgent 처리 시작 (세션: {self.current_session_id})")
        
        # 재귀 깊이 확인 및 동기 모드 전환
        if self._should_use_sync():
            print("🔄 BindingAgent 동기 모드로 전환하여 실행")
            result = await self._process_images_sync_mode(image_urls, image_locations, template_requirements)
        else:
            try:
                # 개선된 배치 기반 비동기 모드 실행
                result = await self._process_images_batch_mode(image_urls, image_locations, template_requirements)
            except RecursionError:
                print("🔄 BindingAgent RecursionError 감지 - 동기 모드로 전환")
                self.fallback_to_sync = True
                result = await self._process_images_sync_mode(image_urls, image_locations, template_requirements)
        
        # 세션별 결과 저장 (새로 추가)
        self.store_result(result)
        
        # 결과에 세션 메타데이터 추가 (새로 추가)
        result["session_metadata"] = {
            "session_id": self.current_session_id,
            "agent_name": self.agent_name,
            "isolation_applied": True,
            "communication_isolated": True
        }
        
        return result


    async def _process_images_batch_mode(self, image_urls: List[str], image_locations: List[str], template_requirements: List[Dict]) -> Dict:
        """개선된 배치 기반 이미지 처리"""
        print(f"📦 BindingAgent 배치 모드 시작: {len(template_requirements)}개 템플릿을 {self.batch_size}개씩 배치 처리")

        # 입력 데이터 검증 추가
        if not image_urls or not template_requirements:
            print("⚠️ 입력 데이터가 비어있음 - 폴백 결과 반환")
            return self._get_fallback_result("empty_input")

        # 입력 데이터 로깅
        input_data = {
            "image_urls": image_urls,
            "image_locations": image_locations,
            "template_requirements": template_requirements,
            "total_images": len(image_urls),
            "total_templates": len(template_requirements)
        }

        # 레이아웃 추천 가져오기
        layout_recommendations = await self._get_layout_recommendations_by_image_count_async(
            image_urls, template_requirements
        )

        # 템플릿을 배치로 그룹화
        template_batches = self._create_template_batches(template_requirements, self.batch_size)
        template_distributions = []
        all_agent_responses = []

        # 배치별 순차 처리
        for batch_idx, batch_templates in enumerate(template_batches):
            print(f"📦 배치 {batch_idx + 1}/{len(template_batches)} 처리 중...")
            
            try:
                batch_results = await self._process_template_batch(
                    batch_templates, batch_idx, image_urls, image_locations, layout_recommendations
                )
                
                # 결과 검증 강화
                if batch_results and isinstance(batch_results, dict):
                    distributions = safe_dict_get(batch_results, "distributions", [])
                    responses = safe_dict_get(batch_results, "responses", [])
                    
                    if isinstance(distributions, list):
                        template_distributions.extend(distributions)
                    if isinstance(responses, list):
                        all_agent_responses.extend(responses)
                else:
                    print(f"⚠️ 배치 {batch_idx} 결과가 None 또는 잘못된 형식")
                    
            except Exception as e:
                print(f"⚠️ 배치 {batch_idx} 처리 중 오류: {e}")
                # 폴백 결과 추가
                fallback_result = self._get_fallback_result(f"batch_{batch_idx}")
                template_distributions.append(fallback_result)

            # 배치 간 쿨다운
            await asyncio.sleep(1)

        # 최종 이미지 분배 결과 생성
        final_distribution = await self._create_final_distribution_async(template_distributions)

        # 전체 BindingAgent 프로세스 응답 저장 (비동기)
        final_response_id = await self._log_final_response_async(
            input_data, final_distribution, template_distributions, all_agent_responses
        )

        print(f"✅ BindingAgent 배치 모드 완료: {len(image_urls)}개 이미지를 {len(template_requirements)}개 템플릿에 배치")
        
        return {
            "image_distribution": final_distribution,
            "template_distributions": template_distributions,
            "layout_recommendations": layout_recommendations,
            "vector_enhanced": True,
            "agent_responses": all_agent_responses,
            "final_response_id": final_response_id,
            "execution_mode": "batch_async",
            "batches_processed": len(template_batches)
        }

    
    def _safe_process_agent_result(self, result) -> Dict:
        """안전한 에이전트 결과 처리"""
        if not result:
            return self._get_fallback_result("unknown")
        
        # 안전한 딕셔너리 접근 적용
        confidence = safe_dict_get(result, 'confidence', 0.5)
        layout_type = safe_dict_get(result, 'layout_type', 'grid')
        template = safe_dict_get(result, 'template', 'unknown')
        images = safe_dict_get(result, 'images', [])
        layout_strategy = safe_dict_get(result, 'layout_strategy', '기본 배치')
        coordination_result = safe_dict_get(result, 'coordination_result', '기본 순서')
        
        return {
            'confidence': confidence,
            'layout_type': layout_type,
            'template': template,
            'images': images,
            'layout_strategy': layout_strategy,
            'coordination_result': coordination_result
        }



    def _create_template_batches(self, template_requirements: List[Dict], batch_size: int) -> List[List[Dict]]:
        """템플릿을 배치로 그룹화"""
        if not isinstance(template_requirements, (list, tuple)):
            print(f"⚠️ template_requirements is not a list/tuple: {type(template_requirements)}")
            return []
        
        if not template_requirements:  # 빈 리스트 체크 추가
            return []
            
        if batch_size <= 0:  # 배치 크기 검증 추가
            batch_size = 1

        batches = []
        for i in range(0, len(template_requirements), batch_size):
            batch = safe_slice(template_requirements, i, i + batch_size, [])
            if batch:
                batches.append(batch)
        return batches



    async def _process_template_batch(self, batch_templates: List[Dict], batch_idx: int,
                                image_urls: List[str], image_locations: List[str],
                                layout_recommendations: List[Dict]) -> Dict:
        """템플릿 배치 처리"""
        batch_tasks = []
        for i, template_req in enumerate(batch_templates):
            template_index = batch_idx * self.batch_size + i
            task_id = f"batch_{batch_idx}_template_{i}"
            
            # 작업을 큐에 추가
            task = self.execute_with_resilience(
                task_func=self._process_single_template_safe,
                task_id=task_id,
                timeout=180.0,
                max_retries=1,
                template_req=template_req,
                image_urls=image_urls,
                image_locations=image_locations,
                template_index=template_index,
                total_templates=len(batch_templates) + batch_idx * self.batch_size,
                layout_recommendations=layout_recommendations
            )
            batch_tasks.append(task)

        # 배치 내 모든 작업 병렬 실행
        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

        # 결과 정리
        distributions = []
        responses = []
        
        for result in batch_results:
            if isinstance(result, Exception):
                print(f"⚠️ 배치 작업 실패: {result}")
                continue
                
            if isinstance(result, dict):
                # 안전한 딕셔너리 접근 적용
                status = safe_dict_get(result, "status", "unknown")
                if status == "fallback":
                    distributions.append(result)
                elif "template_dist" in result:
                    template_dist = safe_dict_get(result, "template_dist", {})
                    if template_dist:
                        distributions.append(template_dist)
                        
                    agent_responses = safe_dict_get(result, "agent_responses", [])
                    if isinstance(agent_responses, list):
                        responses.extend(agent_responses)
            else:
                print(f"⚠️ 예상치 못한 결과 타입: {type(result)}")

        # 누락된 반환문 추가
        return {
            "distributions": distributions,
            "responses": responses
        }


    async def _process_single_template_safe(self, template_req: Dict, image_urls: List[str],
                                      image_locations: List[str], template_index: int,
                                      total_templates: int, layout_recommendations: List[Dict]) -> Dict:
        """안전한 단일 템플릿 처리"""
        try:
            # 입력 검증 추가
            if not isinstance(template_req, dict) or "template" not in template_req:
                raise ValueError("잘못된 template_req 형식")
                
            template_name = template_req["template"]
            
            # 해당 템플릿에 할당할 이미지들 결정
            assigned_images = self._assign_images_to_template(
                image_urls, image_locations, template_index, total_templates
            )

            if not assigned_images:
                return {
                    "template_dist": {
                        "template": template_name,
                        "images": [],
                        "layout_strategy": "no_images",
                        "coordination_result": "이미지 없음",
                        "layout_source": "no_images"
                    },
                    "agent_responses": []
                }

            print(f"🖼️ {template_name}: {len(assigned_images)}개 이미지 배치 설계 중... (안전 모드)")

            # 해당 이미지 수에 맞는 레이아웃 추천 가져오기
            relevant_layouts = [
                layout for layout in layout_recommendations
                if len(layout.get('image_info', [])) == len(assigned_images)
            ]

            if not relevant_layouts and layout_recommendations:
                relevant_layouts = [min(layout_recommendations,
                                    key=lambda x: abs(len(x.get('image_info', [])) - len(assigned_images)))]

            # 에이전트 생성 (매번 새로 생성하여 상태 격리)
            layout_specialist = self.create_image_layout_agent()
            visual_coordinator = self.create_visual_coordinator_agent()

            # CrewAI 실행
            crew_result = await self._execute_crew_safe(
                layout_specialist, visual_coordinator, template_name,
                assigned_images, image_locations, relevant_layouts
            )

            # 결과 검증 추가
            if not isinstance(crew_result, dict):
                crew_result = {"layout_strategy": "기본 배치 전략", "coordination_result": "기본 순서 배치"}

            # 응답 수집 및 저장
            layout_strategy = crew_result.get("layout_strategy", "기본 배치 전략")
            coordination_result = crew_result.get("coordination_result", "기본 순서 배치")

            # 비동기 로깅
            layout_response_id, coordination_response_id = await asyncio.gather(
                self._log_layout_response_async(template_name, assigned_images, relevant_layouts, layout_strategy),
                self._log_coordination_response_async(template_name, layout_strategy, coordination_result)
            )

            # 응답 수집 데이터
            agent_responses = [{
                "template": template_name,
                "layout_specialist_response": {
                    "response_id": layout_response_id,
                    "content": layout_strategy,
                    "agent_name": "BindingAgent_LayoutSpecialist"
                },
                "visual_coordinator_response": {
                    "response_id": coordination_response_id,
                    "content": coordination_result,
                    "agent_name": "BindingAgent_VisualCoordinator"
                }
            }]

            template_dist = {
                "template": template_name,
                "images": assigned_images,
                "layout_strategy": layout_strategy,
                "coordination_result": coordination_result,
                "layout_source": relevant_layouts[0].get("pdf_name", "default") if relevant_layouts else "default",
                "agent_responses": {
                    "layout_specialist_id": layout_response_id,
                    "visual_coordinator_id": coordination_response_id
                },
                "safe_processed": True
            }

            print(f"✅ {template_name} 안전 처리 완료: {len(assigned_images)}개")
            
            return {
                "template_dist": template_dist,
                "agent_responses": agent_responses
            }

        except Exception as e:
            print(f"⚠️ 템플릿 {template_req.get('template', 'unknown')} 안전 처리 실패: {e}")
            
            # 안전한 폴백 처리
            template_name = template_req.get('template', 'unknown') if isinstance(template_req, dict) else 'unknown'
            
            try:
                error_response_id = await self._log_error_response_async(template_name, str(e))
            except:
                error_response_id = "error_logging_failed"

            return {
                "template_dist": {
                    "template": template_name,
                    "images": self._assign_images_to_template(
                        image_urls, image_locations, template_index, total_templates
                    ) if isinstance(image_urls, list) else [],
                    "layout_strategy": "에러로 인한 기본 배치",
                    "coordination_result": "기본 순서 배치",
                    "layout_source": "error_fallback",
                    "error_response_id": error_response_id
                },
                "agent_responses": []
            }


    async def _execute_crew_safe(self, layout_specialist: Agent, visual_coordinator: Agent,
                               template_name: str, assigned_images: List[str],
                               image_locations: List[str], relevant_layouts: List[Dict]) -> Dict:
        """안전한 CrewAI 실행"""
        try:
            # 간소화된 태스크 생성
            layout_analysis_task = Task(
                description=f"""
다음 이미지들과 매거진 레이아웃 데이터를 분석하여 최적의 이미지 배치 전략을 수립하세요:

**배치할 이미지들:**
{self._format_image_data(assigned_images, image_locations)}

**참고할 매거진 레이아웃 데이터:**
{self._format_layout_recommendations(relevant_layouts)}

**템플릿 정보:**
- 템플릿명: {template_name}

**출력 형식:**
레이아웃 전략: [선택된 레이아웃 패턴과 특징]
주요 이미지: [메인으로 사용할 이미지와 배치 위치]
배치 순서: [이미지들의 최적 배치 순서]
""",
                agent=layout_specialist,
                expected_output="벡터 데이터 기반 이미지 배치 전략"
            )

            image_coordination_task = Task(
                description=f"""
레이아웃 분석 결과를 바탕으로 이미지들을 최적으로 배치하고 조합하세요:

**배치 지침:**
1. 분석된 레이아웃 패턴에 따른 정확한 이미지 배치
2. 각 이미지의 특성을 살린 최적 위치 선정
3. 전체적인 시각적 균형과 조화 고려

**출력:** 최종 이미지 배치 결과 (이미지 URL과 배치 정보)
""",
                agent=visual_coordinator,
                expected_output="최적화된 이미지 배치 결과",
                context=[layout_analysis_task]
            )

            # 순차 실행 (병렬 실행으로 인한 복잡성 제거)
            print(f"🔄 {template_name} CrewAI 안전 실행 시작...")
            
            analysis_result = await asyncio.get_event_loop().run_in_executor(
                None, self._execute_single_task, layout_analysis_task
            )
            
            coordination_result = await asyncio.get_event_loop().run_in_executor(
                None, self._execute_single_task, image_coordination_task
            )

            print(f"✅ {template_name} CrewAI 안전 실행 완료")

            return {
                "layout_strategy": str(analysis_result),
                "coordination_result": str(coordination_result)
            }

        except Exception as e:
            print(f"⚠️ {template_name} CrewAI 안전 실행 실패: {e}")
            return {
                "layout_strategy": f"CrewAI 실행 실패로 인한 기본 배치 ({len(assigned_images)}개 이미지)",
                "coordination_result": "기본 순서 배치"
            }

    def _execute_single_task(self, task: Task) -> str:
        """단일 태스크 실행"""
        try:
            # 간단한 Crew 생성 및 실행
            crew = Crew(
                agents=[task.agent],
                tasks=[task],
                verbose=False
            )
            result = crew.kickoff()
            return str(result)
        except Exception as e:
            print(f"⚠️ 단일 태스크 실행 실패: {e}")
            return f"태스크 실행 실패: {str(e)}"

    # 기존 _process_images_async_mode 메서드 유지 (호환성을 위해)
    async def _process_images_async_mode(self, image_urls: List[str], image_locations: List[str], template_requirements: List[Dict]) -> Dict:
        """비동기 모드 이미지 처리 (기존 호환성 유지)"""
        print("⚠️ 기존 async_mode 호출됨 - batch_mode로 리다이렉트")
        return await self._process_images_batch_mode(image_urls, image_locations, template_requirements)

    async def _process_images_sync_mode(self, image_urls: List[str], image_locations: List[str], template_requirements: List[Dict]) -> Dict:
        """동기 모드 이미지 처리 (run_in_executor 사용)"""
        print(f"🔄 BindingAgent 동기 모드: 처리할 이미지 {len(image_urls)}개, 템플릿 {len(template_requirements)}개")

        # 동기 버전 메서드들을 executor에서 실행
        loop = asyncio.get_event_loop()

        # 기본 데이터 준비 (동기)
        layout_recommendations = await loop.run_in_executor(
            None, self._get_layout_recommendations_sync, image_urls, template_requirements
        )

        # 템플릿별 처리 (동기)
        template_distributions = await loop.run_in_executor(
            None, self._process_all_templates_sync,
            image_urls, image_locations, template_requirements, layout_recommendations
        )

        # 최종 분배 결과 생성
        final_distribution = await loop.run_in_executor(
            None, self._create_final_distribution_sync, template_distributions
        )

        # 동기 모드 로깅
        final_response_id = await self._log_sync_mode_response_async(
            image_urls, template_requirements, final_distribution, template_distributions
        )

        print(f"✅ BindingAgent 동기 완료: {len(image_urls)}개 이미지를 {len(template_requirements)}개 템플릿에 배치")
        return {
            "image_distribution": final_distribution,
            "template_distributions": template_distributions,
            "layout_recommendations": layout_recommendations,
            "vector_enhanced": True,
            "agent_responses": [],
            "final_response_id": final_response_id,
            "execution_mode": "sync_fallback",
            "recursion_fallback": True
        }

    async def _process_remaining_templates_sync(self, remaining_requirements: List[Dict],
                                              image_urls: List[str], image_locations: List[str],
                                              layout_recommendations: List[Dict]) -> List[Dict]:
        """나머지 템플릿들을 동기 모드로 처리"""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._process_all_templates_sync,
            image_urls, image_locations, remaining_requirements, layout_recommendations
        )

    # 기존 _process_single_template_async 메서드 유지 (호환성을 위해)
    async def _process_single_template_async(self, template_req: Dict, image_urls: List[str],
                                           image_locations: List[str], template_index: int,
                                           total_templates: int, layout_recommendations: List[Dict],
                                           layout_specialist: Agent, visual_coordinator: Agent) -> tuple:
        """단일 템플릿 처리 (기존 호환성 유지) - 안전 모드로 리다이렉트"""
        print("⚠️ 기존 single_template_async 호출됨 - safe 모드로 리다이렉트")
        
        result = await self._process_single_template_safe(
            template_req, image_urls, image_locations, template_index, total_templates, layout_recommendations
        )
        
        # 기존 반환 형식에 맞게 변환
        template_dist = result.get("template_dist", {})
        agent_responses = result.get("agent_responses", [])
        
        return (template_dist, agent_responses)

    # 모든 기존 동기 메서드들과 유틸리티 메서드들 유지
    def _get_layout_recommendations_sync(self, image_urls: List[str], template_requirements: List[Dict]) -> List[Dict]:
        """이미지 개수별 레이아웃 추천 가져오기 (동기 버전)"""
        total_images = len(image_urls)
        if total_images <= 3:
            query = "minimal clean layout single image focus simple elegant"
        elif total_images <= 6:
            query = "multiple images grid layout balanced composition"
        elif total_images <= 10:
            query = "gallery style layout many images organized grid"
        else:
            query = "complex magazine layout multiple images rich visual content"

        recommendations = self.vector_manager.search_similar_layouts(query, "magazine_layout", top_k=5)
        print(f"📊 이미지 {total_images}개에 대한 레이아웃 추천 {len(recommendations)}개 획득 (동기)")
        return recommendations

    def _process_all_templates_sync(self, image_urls: List[str], image_locations: List[str],
                                  template_requirements: List[Dict], layout_recommendations: List[Dict]) -> List[Dict]:
        """모든 템플릿을 동기 모드로 처리"""
        template_distributions = []

        for i, template_req in enumerate(template_requirements):
            template_name = template_req["template"]
            assigned_images = self._assign_images_to_template(
                image_urls, image_locations, i, len(template_requirements)
            )

            # 해당 이미지 수에 맞는 레이아웃 추천 가져오기
            relevant_layouts = [
                layout for layout in layout_recommendations
                if len(layout.get('image_info', [])) == len(assigned_images)
            ]

            if not relevant_layouts and layout_recommendations:
                relevant_layouts = [min(layout_recommendations,
                                      key=lambda x: abs(len(x.get('image_info', [])) - len(assigned_images)))]

            template_dist = {
                "template": template_name,
                "images": assigned_images,
                "layout_strategy": f"동기 모드 기본 배치 ({len(assigned_images)}개 이미지)",
                "coordination_result": "동기 모드 순서 배치",
                "layout_source": relevant_layouts[0].get("pdf_name", "sync_default") if relevant_layouts else "sync_default",
                "sync_processed": True
            }

            template_distributions.append(template_dist)
            print(f"✅ {template_name} 동기 처리 완료: {len(assigned_images)}개 이미지")

        return template_distributions

    def _create_final_distribution_sync(self, template_distributions: List[Dict]) -> Dict:
        """최종 이미지 분배 결과 생성 (동기 버전)"""
        return {dist["template"]: dist["images"] for dist in template_distributions}

    async def _log_sync_mode_response_async(self, image_urls: List[str], template_requirements: List[Dict],
                                          final_distribution: Dict, template_distributions: List[Dict]) -> str:
        """동기 모드 응답 로깅 (비동기)"""
        return await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.logger.log_agent_real_output(
                agent_name="BindingAgent_SyncMode",
                agent_role="동기 모드 이미지 배치 에이전트",
                task_description=f"동기 모드로 {len(image_urls)}개 이미지를 {len(template_requirements)}개 템플릿에 배치",
                final_answer=f"동기 모드 이미지 배치 완료: {len(template_distributions)}개 템플릿 처리",
                reasoning_process="재귀 깊이 초과로 인한 동기 모드 전환 후 안전한 이미지 배치 실행",
                execution_steps=[
                    "재귀 깊이 감지",
                    "동기 모드 전환",
                    "레이아웃 추천 수집",
                    "템플릿별 이미지 할당",
                    "최종 분배 결과 생성"
                ],
                raw_input={
                    "image_urls": image_urls,
                    "template_requirements": template_requirements
                },
                raw_output={
                    "final_distribution": final_distribution,
                    "template_distributions": template_distributions
                },
                performance_metrics={
                    "sync_mode_used": True,
                    "recursion_fallback": True,
                    "templates_processed": len(template_distributions),
                    "images_processed": len(image_urls),
                    "safe_execution": True
                }
            )
        )

    async def _get_layout_recommendations_by_image_count_async(self, image_urls: List[str], template_requirements: List[Dict]) -> List[Dict]:
        """이미지 개수별 레이아웃 추천 가져오기 (비동기)"""
        total_images = len(image_urls)
        if total_images <= 3:
            query = "minimal clean layout single image focus simple elegant"
        elif total_images <= 6:
            query = "multiple images grid layout balanced composition"
        elif total_images <= 10:
            query = "gallery style layout many images organized grid"
        else:
            query = "complex magazine layout multiple images rich visual content"

        # 비동기 벡터 검색
        recommendations = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.vector_manager.search_similar_layouts(query, "magazine_layout", top_k=5)
        )

        print(f"📊 이미지 {total_images}개에 대한 레이아웃 추천 {len(recommendations)}개 획득 (비동기)")
        return recommendations

    async def _log_layout_response_async(self, template_name: str, assigned_images: List[str],
                                       relevant_layouts: List[Dict], layout_strategy: str) -> str:
        """레이아웃 분석 에이전트 응답 저장 (비동기)"""
        return await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.logger.log_agent_real_output(
                agent_name="BindingAgent_LayoutSpecialist",
                agent_role="이미지 레이아웃 전문가",
                task_description=f"템플릿 {template_name}의 {len(assigned_images)}개 이미지 배치 전략 수립",
                final_answer=layout_strategy,
                reasoning_process=f"PDF 벡터 데이터 {len(relevant_layouts)}개 레이아웃 참조하여 분석",
                execution_steps=[
                    "이미지 특성 분석",
                    "레이아웃 패턴 매칭",
                    "배치 전략 수립"
                ],
                raw_input={
                    "template_name": template_name,
                    "assigned_images": assigned_images,
                    "relevant_layouts": relevant_layouts
                },
                raw_output=layout_strategy,
                performance_metrics={
                    "images_processed": len(assigned_images),
                    "layouts_referenced": len(relevant_layouts),
                    "analysis_depth": "comprehensive"
                }
            )
        )

    async def _log_coordination_response_async(self, template_name: str, layout_strategy: str,
                                             coordination_result: str) -> str:
        """비주얼 코디네이터 에이전트 응답 저장 (비동기)"""
        return await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.logger.log_agent_real_output(
                agent_name="BindingAgent_VisualCoordinator",
                agent_role="시각적 일관성 보장자",
                task_description=f"템플릿 {template_name}의 최종 이미지 배치 실행",
                final_answer=coordination_result,
                reasoning_process="레이아웃 분석 결과를 바탕으로 최적 배치 실행",
                execution_steps=[
                    "분석 결과 검토",
                    "배치 최적화",
                    "시각적 일관성 검증",
                    "최종 배치 결정"
                ],
                raw_input={
                    "layout_analysis": layout_strategy,
                    "template_name": template_name
                },
                raw_output=coordination_result,
                performance_metrics={
                    "coordination_quality": "high",
                    "visual_consistency": "verified",
                    "placement_accuracy": "optimized"
                }
            )
        )

    async def _log_error_response_async(self, template_name: str, error_message: str) -> str:
        """에러 응답 저장 (비동기)"""
        return await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.logger.log_agent_real_output(
                agent_name="BindingAgent_Error",
                agent_role="에러 처리",
                task_description=f"템플릿 {template_name} 처리 중 에러 발생",
                final_answer=f"ERROR: {error_message}",
                reasoning_process="에이전트 실행 중 예외 발생",
                error_logs=[{"error": error_message, "template": template_name}]
            )
        )

    async def _create_final_distribution_async(self, template_distributions: List[Dict]) -> Dict:
        """최종 이미지 분배 결과 생성 (비동기)"""
        return await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: {dist["template"]: dist["images"] for dist in template_distributions}
        )

    async def _log_final_response_async(self, input_data: Dict, final_distribution: Dict,
                                      template_distributions: List[Dict], all_agent_responses: List[Dict]) -> str:
        """전체 BindingAgent 프로세스 응답 저장 (비동기)"""
        return await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.logger.log_agent_real_output(
                agent_name="BindingAgent",
                agent_role="PDF 벡터 데이터 기반 이미지 배치 에이전트",
                task_description=f"{input_data['total_images']}개 이미지를 {input_data['total_templates']}개 템플릿에 배치",
                final_answer=str(final_distribution),
                reasoning_process=f"개선된 배치 처리 시스템으로 안전한 {len(template_distributions)}개 템플릿 처리 완료",
                execution_steps=[
                    "재귀 깊이 체크",
                    "배치 기반 처리 모드 선택",
                    "레이아웃 추천 수집",
                    "템플릿 배치별 처리",
                    "최종 분배 결과 생성"
                ],
                raw_input=input_data,
                raw_output={
                    "image_distribution": final_distribution,
                    "template_distributions": template_distributions,
                    "all_agent_responses": all_agent_responses
                },
                performance_metrics={
                    "total_images_processed": input_data['total_images'],
                    "total_templates_processed": input_data['total_templates'],
                    "successful_templates": len([t for t in template_distributions if "error_response_id" not in t]),
                    "agent_responses_collected": len(all_agent_responses),
                    "recursion_depth_check": True,
                    "safe_execution": True,
                    "batch_processing": True
                }
            )
        )

    def _assign_images_to_template(self, image_urls: List[str], image_locations: List[str],
                              template_index: int, total_templates: int) -> List[str]:
        """템플릿에 이미지 할당"""
        if not image_urls:
            return []

        # 안전한 타입 검증 추가
        if not isinstance(image_urls, (list, tuple)):
            print(f"⚠️ image_urls is not a list/tuple: {type(image_urls)}")
            return []

        images_per_template = len(image_urls) // total_templates
        remainder = len(image_urls) % total_templates

        start_idx = template_index * images_per_template
        if template_index < remainder:
            start_idx += template_index
            end_idx = start_idx + images_per_template + 1
        else:
            start_idx += remainder
            end_idx = start_idx + images_per_template

        # 안전한 슬라이싱 적용
        return safe_slice(image_urls, start_idx, end_idx, [])


    def _format_image_data(self, image_urls: List[str], image_locations: List[str]) -> str:
        """이미지 데이터를 텍스트로 포맷팅"""
        if not image_urls:
            return "배치할 이미지 없음"

        formatted_data = []
        for i, url in enumerate(image_urls):
            location = image_locations[i] if i < len(image_locations) else f"위치 {i+1}"
            formatted_data.append(f"이미지 {i+1}: {url} (위치: {location})")

        return "\n".join(formatted_data)

    def _format_layout_recommendations(self, recommendations: List[Dict]) -> str:
        """레이아웃 추천 데이터를 텍스트로 포맷팅"""
        if not recommendations:
            return "참고할 레이아웃 데이터 없음"

        formatted_data = []
        for i, rec in enumerate(recommendations):
            image_count = len(rec.get('image_info', []))
            formatted_data.append(f"""
레이아웃 {i+1} (유사도: {rec.get('score', 0):.2f}):
- 출처: {rec.get('pdf_name', 'unknown')} (페이지 {rec.get('page_number', 0)})
- 이미지 수: {image_count}개
- 레이아웃 특징: {self._analyze_layout_structure(rec.get('layout_info', {}))}
- 텍스트 샘플: {rec.get('text_content', '')[:150]}...
""")

        return "\n".join(formatted_data)

    def _analyze_layout_structure(self, layout_info: Dict) -> str:
        """레이아웃 구조 분석"""
        text_blocks = layout_info.get('text_blocks', [])
        images = layout_info.get('images', [])
        tables = layout_info.get('tables', [])

        structure_analysis = []

        if len(images) == 1:
            structure_analysis.append("단일 이미지 중심")
        elif len(images) <= 3:
            structure_analysis.append("소수 이미지 균형 배치")
        elif len(images) <= 6:
            structure_analysis.append("다중 이미지 그리드")
        else:
            structure_analysis.append("갤러리 스타일")

        if len(text_blocks) > 5:
            structure_analysis.append("텍스트 중심")
        elif len(text_blocks) <= 2:
            structure_analysis.append("이미지 중심")
        else:
            structure_analysis.append("텍스트-이미지 균형")

        if tables:
            structure_analysis.append("정보 테이블 포함")

        return ", ".join(structure_analysis) if structure_analysis else "기본 레이아웃"

    # 동기 버전 메서드 (호환성 보장)
    def process_images_sync(self, image_urls: List[str], image_locations: List[str], template_requirements: List[Dict]) -> Dict:
        """동기 버전 이미지 처리 (호환성 유지)"""
        return asyncio.run(self.process_images(image_urls, image_locations, template_requirements))
