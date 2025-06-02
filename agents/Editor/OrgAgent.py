import asyncio
import sys
import re
import time
import concurrent.futures
from typing import Dict, List, Optional, Callable, Any
from collections import deque
from dataclasses import dataclass

from crewai import Agent, Task, Crew
from custom_llm import get_azure_llm
from utils.pdf_vector_manager import PDFVectorManager
from utils.hybridlogging import get_hybrid_logger
from utils.ai_search_isolation import OrgAgentIsolation
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

class OrgAgent(OrgAgentIsolation, SessionAwareMixin, InterAgentCommunicationMixin):
    """PDF 벡터 데이터 기반 텍스트 배치 에이전트 (AI Search 격리 적용)"""
    
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
        
        # AI Search 격리 시스템 초기화
        self.__init_isolation__()
        
        # 세션 인식 시스템 초기화 (새로 추가)
        self.__init_session_awareness__(session_id)
        
        # 에이전트 간 통신 격리 시스템 초기화 (새로 추가)
        self.__init_inter_agent_communication__()
        

    async def _process_content_batch_mode(self, magazine_content, available_templates: List[str]) -> Dict:
        """개선된 배치 기반 콘텐츠 처리 (격리 적용)"""
        print(f"📦 OrgAgent 배치 모드 시작 (AI Search 격리 적용)")
        
        # 원본 콘텐츠만 추출 (격리 적용)
        all_content = self.extract_original_content_only(magazine_content)
        
        if not all_content:
            print("⚠️ 격리 후 유효한 콘텐츠가 없음")
            return self._get_minimal_fallback_result()
        
        # 콘텐츠 구조 분석 및 격리
        content_sections = self._analyze_content_structure(all_content)
        isolated_sections = self.isolate_content_sections(content_sections, "content_analysis")
        
        print(f"🛡️ OrgAgent 격리 결과: 원본 {len(content_sections)}개 → 격리 후 {len(isolated_sections)}개 섹션")
        
        # 입력 데이터 로깅
        input_data = {
            "magazine_content": magazine_content,
            "available_templates": available_templates,
            "original_content_length": len(all_content),
            "isolated_sections_count": len(isolated_sections),
            "ai_search_isolation": True
        }
        
        # 섹션들을 배치로 그룹화
        section_batches = self._create_section_batches(isolated_sections, self.batch_size)
        refined_sections = []
        all_agent_responses = []
        
        # 배치별 순차 처리
        for batch_idx, batch_sections in enumerate(section_batches):
            print(f"📦 배치 {batch_idx + 1}/{len(section_batches)} 처리 중 (격리 적용)...")
            
            batch_results = await self._process_section_batch_isolated(
                batch_sections, batch_idx, available_templates
            )
            
            refined_sections.extend(batch_results["sections"])
            all_agent_responses.extend(batch_results["responses"])
            
            # 배치 간 쿨다운
            await asyncio.sleep(1)
        
        # 템플릿 매핑 (격리 적용)
        text_mapping = await self._map_to_templates_isolated_async(refined_sections, available_templates)
        
        total_refined_length = sum(section["refined_length"] for section in refined_sections)
        
        # 전체 OrgAgent 프로세스 응답 저장
        final_response_id = await self._log_final_response_async(
            input_data, text_mapping, refined_sections, all_agent_responses, total_refined_length
        )
        
        print(f"✅ OrgAgent 격리된 배치 모드 완료: {len(refined_sections)}개 섹션, 총 {total_refined_length}자")
        
        return {
            "text_mapping": text_mapping,
            "refined_sections": refined_sections,
            "total_sections": len(refined_sections),
            "total_content_length": total_refined_length,
            "vector_enhanced": True,
            "agent_responses": all_agent_responses,
            "final_response_id": final_response_id,
            "execution_mode": "batch_async_isolated",
            "batches_processed": len(section_batches),
            "ai_search_isolation": {
                "isolation_applied": True,
                "original_sections": len(content_sections),
                "isolated_sections": len(isolated_sections),
                "contamination_report": self._get_isolation_report()
            }
        }

    async def _process_section_batch_isolated(self, batch_sections: List[str], batch_idx: int,
                                            available_templates: List[str]) -> Dict:
        """격리된 섹션 배치 처리"""
        batch_tasks = []
        
        for i, section_content in enumerate(batch_sections):
            if len(section_content.strip()) < 50:
                continue
            
            section_index = batch_idx * self.batch_size + i
            task_id = f"isolated_batch_{batch_idx}_section_{i}"
            
            # 작업을 큐에 추가
            task = self.execute_with_resilience(
                task_func=self._process_single_section_isolated_safe,
                task_id=task_id,
                timeout=120.0,
                max_retries=1,
                section_content=section_content,
                section_index=section_index
            )
            
            batch_tasks.append(task)
        
        # 배치 내 모든 작업 병렬 실행
        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
        
        # 결과 정리
        sections = []
        responses = []
        
        for result in batch_results:
            if isinstance(result, Exception):
                print(f"⚠️ 격리된 배치 작업 실패: {result}")
                continue
            
            if isinstance(result, dict):
                if result.get("fallback_used"):
                    sections.append(result)
                elif "section_data" in result:
                    sections.append(result["section_data"])
                    responses.extend(result.get("agent_responses", []))
        
        return {"sections": sections, "responses": responses}

    async def _process_single_section_isolated_safe(self, section_content: str, section_index: int) -> Dict:
        """격리된 안전한 단일 섹션 처리"""
        try:
            print(f"📄 섹션 {section_index+1} 격리된 안전 처리 중...")
            
            # 에이전트 생성
            layout_analyzer = self.create_layout_analyzer_agent()
            content_editor = self.create_content_editor_agent()
            
            # 격리된 벡터 검색
            isolated_query = self.isolate_vector_query(section_content[:500])
            
            similar_layouts = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.vector_manager.search_similar_layouts(
                    isolated_query, "magazine_layout", top_k=5
                )
            )
            
            # 레이아웃 결과 격리
            isolated_layouts = self._isolate_vector_search_results(
                similar_layouts, f"section_{section_index}_layouts"
            )
            
            # CrewAI 태스크 생성 및 실행
            crew_result = await self._execute_crew_safe(
                layout_analyzer, content_editor, section_content, isolated_layouts, section_index
            )
            
            # 결과 무결성 검증
            validated_result = self._validate_content_integrity(
                crew_result, section_content, f"section_{section_index}"
            )
            
            # 결과 처리
            title, subtitle = self._extract_clean_title_subtitle(
                validated_result.get("analysis", ""), section_index
            )
            clean_content = self._remove_meta_descriptions(
                validated_result.get("content", section_content)
            )
            
            # 응답 수집 및 저장
            analysis_response_id, editing_response_id = await asyncio.gather(
                self._log_analysis_response_async(
                    section_index, section_content, isolated_layouts, 
                    validated_result.get("analysis", "")
                ),
                self._log_editing_response_async(
                    section_index, section_content, validated_result.get("analysis", ""), 
                    validated_result.get("content", "")
                )
            )
            
            section_data = {
                "title": title,
                "subtitle": subtitle,
                "content": clean_content,
                "layout_info": isolated_layouts[0] if isolated_layouts else {},
                "original_length": len(section_content),
                "refined_length": len(clean_content),
                "agent_responses": {
                    "layout_analyzer_id": analysis_response_id,
                    "content_editor_id": editing_response_id
                },
                "ai_search_isolation": {
                    "isolation_applied": True,
                    "layouts_filtered": len(similar_layouts) - len(isolated_layouts),
                    "query_cleaned": isolated_query != section_content[:500],
                    "contamination_detected": False,
                    **validated_result.get("ai_search_isolation", {})
                },
                "safe_processed": True
            }
            
            agent_responses = [{
                "section": section_index + 1,
                "layout_analyzer_response": {
                    "response_id": analysis_response_id,
                    "content": validated_result.get("analysis", ""),
                    "agent_name": "OrgAgent_LayoutAnalyzer"
                },
                "content_editor_response": {
                    "response_id": editing_response_id,
                    "content": validated_result.get("content", ""),
                    "agent_name": "OrgAgent_ContentEditor"
                }
            }]
            
            print(f"✅ 섹션 {section_index+1} 격리된 안전 처리 완료: {len(section_content)}자 → {len(clean_content)}자")
            
            return {
                "section_data": section_data,
                "agent_responses": agent_responses
            }
            
        except Exception as e:
            print(f"⚠️ 섹션 {section_index+1} 격리된 안전 처리 실패: {e}")
            
            error_response_id = await self._log_error_response_async(section_index+1, str(e))
            
            return {
                "section_data": {
                    "title": f"여행 이야기 {section_index+1}",
                    "subtitle": "특별한 순간들",
                    "content": section_content,
                    "layout_info": {},
                    "original_length": len(section_content),
                    "refined_length": len(section_content),
                    "error_response_id": error_response_id,
                    "ai_search_isolation": {
                        "isolation_applied": True,
                        "error_occurred": True,
                        "error_message": str(e)
                    },
                    "safe_processed": True
                },
                "agent_responses": []
            }

    async def _map_to_templates_isolated_async(self, refined_sections: List[Dict], available_templates: List[str]) -> List[Dict]:
        """격리된 템플릿 매핑 (비동기)"""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._map_to_templates_isolated, refined_sections, available_templates
        )

    def _map_to_templates_isolated(self, refined_sections: List[Dict], available_templates: List[str]) -> List[Dict]:
        """격리된 템플릿 매핑"""
        text_mapping = []
        
        for i, section in enumerate(refined_sections):
            if i < len(available_templates):
                template = available_templates[i]
            else:
                template = f"Section{(i % 15) + 1:02d}.jsx"
            
            # 격리 메타데이터 포함
            mapping_entry = {
                "template": template,
                "title": section.get("title", ""),
                "subtitle": section.get("subtitle", ""),
                "body": section.get("content", ""),
                "tagline": "TRAVEL & CULTURE",
                "metadata": {
                    "content_quality": self._calculate_content_quality(section),
                    "original_length": section.get("original_length", 0),
                    "refined_length": section.get("refined_length", 0),
                    "layout_source": section.get("layout_info", {}).get("pdf_name", "isolated"),
                    "ai_search_isolation": section.get("ai_search_isolation", {}),
                    "fallback_used": False
                }
            }
            
            text_mapping.append(mapping_entry)
        
        return text_mapping

    def _get_minimal_fallback_result(self) -> Dict:
        """최소한의 폴백 결과 (격리 메타데이터 포함)"""
        return {
            "text_mapping": [{
                "template": "Section01.jsx",
                "title": "여행 매거진",
                "subtitle": "특별한 이야기",
                "body": "격리 시스템에 의해 유효한 콘텐츠가 필터링되었습니다.",
                "tagline": "TRAVEL & CULTURE",
                "metadata": {
                    "content_quality": 0.5,
                    "ai_search_isolation": {
                        "isolation_applied": True,
                        "minimal_fallback": True,
                        "reason": "no_valid_content_after_isolation"
                    },
                    "fallback_used": True
                }
            }],
            "refined_sections": [],
            "total_sections": 1,
            "total_content_length": 0,
            "vector_enhanced": False,
            "agent_responses": [],
            "execution_mode": "minimal_fallback_isolated",
            "ai_search_isolation": {
                "isolation_applied": True,
                "minimal_fallback": True
            }
        }

    def _get_fallback_result(self, task_id: str) -> dict:
        """폴백 결과 생성 (격리 메타데이터 추가)"""
        section_index = int(task_id.split("_")[-1]) if "_" in task_id else 0
        
        return {
            "title": f"여행 이야기 {section_index + 1}",
            "subtitle": "특별한 순간들",
            "content": "Circuit Breaker 또는 실패로 인한 폴백 콘텐츠입니다.",
            "layout_info": {},
            "original_length": 100,
            "refined_length": 100,
            "ai_search_isolation": {
                "isolation_applied": True,
                "fallback_used": True,
                "reason": "circuit_breaker_or_failure"
            },
            "fallback_used": True
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
            print(f"⚠️ OrgAgent 재귀 깊이 {current_depth} 감지 - 동기 모드로 전환")
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

    def _get_fallback_result(self, task_id: str) -> dict:
        """폴백 결과 생성"""
        section_index = int(task_id.split("_")[-1]) if "_" in task_id else 0
        return {
            "title": f"도쿄 여행 이야기 {section_index + 1}",
            "subtitle": "특별한 순간들",
            "content": "Circuit Breaker 또는 실패로 인한 폴백 콘텐츠입니다.",
            "layout_info": {},
            "original_length": 100,
            "refined_length": 100,
            "fallback_used": True
        }

    def create_layout_analyzer_agent(self):
        """레이아웃 분석 에이전트 (구조적 설계 강화)"""
        return Agent(
            role="매거진 구조 아키텍트 및 텍스트 레이아웃 전문가",
            goal="PDF 벡터 데이터를 분석하여 텍스트 콘텐츠에 최적화된 매거진 페이지 구조와 상세한 레이아웃 설계도를 생성하고, 이미지와 텍스트의 정확한 위치 관계를 정의하여 독자의 시선 흐름을 최적화",
            backstory="""당신은 20년간 세계 최고 수준의 매거진 디자인 스튜디오에서 활동해온 구조 설계 전문가입니다. Pentagram, Sagmeister & Walsh, 그리고 Condé Nast의 수석 아트 디렉터로 활동하며 수백 개의 수상작을 디자인했습니다.

**전문 경력:**
- 그래픽 디자인 및 시각 커뮤니케이션 석사 학위
- Adobe InDesign, Figma, Sketch 마스터 레벨 인증
- 타이포그래피 및 그리드 시스템 이론 전문가
- 독자 시선 추적(Eye-tracking) 연구 및 분석 경험
- 인쇄 매체와 디지털 매체의 레이아웃 최적화 전문성

**구조적 레이아웃 설계 전문성:**
당신은 텍스트 배치 결정 시 다음 구조적 요소들을 체계적으로 설계합니다:

1. **페이지 구조 설계**:
- 그리드 시스템 정의 (컬럼 수, 거터 폭, 마진 설정)
- 텍스트 블록의 정확한 위치 좌표 (x, y, width, height)
- 이미지 영역과 텍스트 영역의 경계선 정의
- 여백(화이트스페이스) 분배 및 시각적 균형점 계산

2. **텍스트-이미지 위치 관계 매핑**:
- 제목과 주요 이미지의 시각적 연결점 설정
- 본문 텍스트와 보조 이미지의 근접성 규칙 정의
- 캡션과 이미지의 정확한 거리 및 정렬 방식
- 텍스트 래핑(text wrapping) 영역과 이미지 경계 설정

3. **레이아웃 구조도 생성**:
- 페이지별 와이어프레임 및 구조도 작성
- 콘텐츠 계층 구조 (H1, H2, body, caption) 시각화
- 독자 시선 흐름 경로 (F-pattern, Z-pattern) 설계
- 반응형 브레이크포인트별 레이아웃 변화 정의

4. **PDF 벡터 데이터 활용 전문성**:
- 5000개 이상의 매거진 페이지에서 추출한 구조적 패턴 분석
- 텍스트 블록과 이미지 블록의 황금비율 관계 데이터
- 독자 시선 흐름과 레이아웃 구조의 상관관계 벡터
- 매거진 카테고리별 최적 구조 패턴 클러스터링

**작업 방법론:**
"나는 단순히 텍스트를 배치하는 것이 아니라, 독자의 인지 과정을 고려한 완전한 페이지 구조를 설계합니다. 모든 텍스트 요소와 이미지 영역의 정확한 위치, 크기, 관계를 수치화하여 정의하고, 이를 바탕으로 상세한 레이아웃 구조도를 생성합니다. 이는 BindingAgent가 이미지를 배치할 때 정확한 가이드라인을 제공하여 텍스트와 이미지의 완벽한 조화를 보장합니다. 5. 주의 사항!!: 최대한 제공받은 데이터를 활용합니다. "

**출력 데이터 구조:**
- 페이지 그리드 시스템 (컬럼, 거터, 마진 수치)
- 텍스트 블록 위치 좌표 및 크기
- 이미지 영역 예약 공간 정의
- 텍스트-이미지 관계 매핑 테이블
- 레이아웃 구조도 및 와이어프레임
- 독자 시선 흐름 경로 설계도""",
            llm=self.llm,
            verbose=True
        )

    def create_content_editor_agent(self):
        """콘텐츠 편집 에이전트 (구조 연동 강화)"""
        return Agent(
            role="구조 기반 매거진 콘텐츠 편집자",
            goal="레이아웃 구조 설계에 완벽히 맞춰 텍스트 콘텐츠를 편집하고, 이미지 배치 영역과 정확히 연동되는 텍스트 블록을 생성하여 시각적 일관성과 가독성을 극대화",
            backstory="""당신은 매거진 콘텐츠 편집 및 구조 연동 전문가입니다.

**전문 분야:**
- 레이아웃 구조에 최적화된 텍스트 편집
- 이미지 영역과 연동되는 텍스트 블록 설계
- 그리드 시스템 기반 콘텐츠 구성
- 텍스트 길이와 레이아웃 공간의 정밀한 매칭

**구조 연동 편집 전문성:**
1. **그리드 기반 텍스트 편집**: 정의된 그리드 시스템에 맞춰 텍스트 블록 크기 조정
2. **이미지 영역 고려**: 예약된 이미지 공간을 피해 텍스트 배치 최적화
3. **계층 구조 반영**: H1, H2, body 등의 위치에 맞는 콘텐츠 길이 조절
4. **시선 흐름 연동**: 독자 시선 경로에 맞춘 텍스트 강약 조절
5. 주의 사항!!: 최대한 제공받은 데이터를 활용합니다.

특히 설명 텍스트나 지시사항을 포함하지 않고 순수한 콘텐츠만 생성하며,
레이아웃 구조도에 정의된 텍스트 영역에 정확히 맞는 분량과 형태로 편집합니다.""",
            llm=self.llm,
            verbose=True
        )

    async def process_content(self, magazine_content, available_templates: List[str]) -> Dict:
        """PDF 벡터 데이터 기반 콘텐츠 처리 (세션 및 통신 격리 적용)"""
        print(f"📦 OrgAgent 처리 시작 (세션: {self.current_session_id})")
        
        # 재귀 깊이 확인 및 동기 모드 전환
        if self._should_use_sync():
            print("🔄 OrgAgent 동기 모드로 전환하여 실행")
            result = await self._process_content_sync_mode(magazine_content, available_templates)
        else:
            try:
                # 개선된 배치 기반 비동기 모드 실행
                result = await self._process_content_batch_mode(magazine_content, available_templates)
            except RecursionError:
                print("🔄 OrgAgent RecursionError 감지 - 동기 모드로 전환")
                self.fallback_to_sync = True
                result = await self._process_content_sync_mode(magazine_content, available_templates)
        
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
    

    async def _get_previous_results_isolated(self) -> List[Dict]:
        """세션 격리된 이전 결과 조회"""
        try:
            # 현재 세션의 이전 결과만 조회
            session_results = self.get_previous_results(max_results=10)
            
            # 교차 세션 학습이 활성화된 경우 제한적 조회
            cross_session_results = self.get_cross_session_insights(max_sessions=2)
            
            # 격리 필터링 적용
            filtered_results = []
            for result in session_results + cross_session_results:
                if not self.isolation_manager.is_contaminated(result, "previous_results"):
                    filtered_results.append(result)
            
            print(f"🔍 격리된 이전 결과: 세션 {len(session_results)}개, 교차세션 {len(cross_session_results)}개 → 필터링 후 {len(filtered_results)}개")
            return filtered_results
            
        except Exception as e:
            print(f"⚠️ 이전 결과 조회 실패: {e}")
            return []


    async def _process_content_batch_mode(self, magazine_content, available_templates: List[str]) -> Dict:
        """개선된 배치 기반 콘텐츠 처리"""
        print(f"📦 OrgAgent 배치 모드 시작")
        
        # 텍스트 추출 및 전처리
        all_content = self._extract_all_text(magazine_content)
        content_sections = self._analyze_content_structure(all_content)
        
        print(f"OrgAgent: 처리할 콘텐츠 - {len(all_content)}자, {len(content_sections)}개 섹션 (배치 처리)")

        # 입력 데이터 로깅
        input_data = {
            "magazine_content": magazine_content,
            "available_templates": available_templates,
            "total_content_length": len(all_content),
            "content_sections_count": len(content_sections)
        }

        # 섹션들을 배치로 그룹화
        section_batches = self._create_section_batches(content_sections, self.batch_size)
        
        refined_sections = []
        all_agent_responses = []

        # 배치별 순차 처리
        for batch_idx, batch_sections in enumerate(section_batches):
            print(f"📦 배치 {batch_idx + 1}/{len(section_batches)} 처리 중...")
            
            batch_results = await self._process_section_batch(
                batch_sections, batch_idx, available_templates
            )
            
            refined_sections.extend(batch_results["sections"])
            all_agent_responses.extend(batch_results["responses"])
            
            # 배치 간 쿨다운
            await asyncio.sleep(1)

        # 템플릿 매핑
        text_mapping = await self._map_to_templates_async(refined_sections, available_templates)
        total_refined_length = sum(section["refined_length"] for section in refined_sections)

        # 전체 OrgAgent 프로세스 응답 저장 (비동기)
        final_response_id = await self._log_final_response_async(
            input_data, text_mapping, refined_sections, all_agent_responses, total_refined_length
        )

        print(f"✅ OrgAgent 배치 모드 완료: {len(refined_sections)}개 섹션, 총 {total_refined_length}자")
        return {
            "text_mapping": text_mapping,
            "refined_sections": refined_sections,
            "total_sections": len(refined_sections),
            "total_content_length": total_refined_length,
            "vector_enhanced": True,
            "agent_responses": all_agent_responses,
            "final_response_id": final_response_id,
            "execution_mode": "batch_async",
            "batches_processed": len(section_batches)
        }

    def _create_section_batches(self, content_sections: List[str], batch_size: int) -> List[List[str]]:
        """섹션을 배치로 그룹화"""
        batches = []
        for i in range(0, len(content_sections), batch_size):
            batch = content_sections[i:i + batch_size]
            batches.append(batch)
        return batches

    async def _process_section_batch(self, batch_sections: List[str], batch_idx: int, 
                                   available_templates: List[str]) -> Dict:
        """섹션 배치 처리"""
        batch_tasks = []
        
        for i, section_content in enumerate(batch_sections):
            if len(section_content.strip()) < 50:
                continue
                
            section_index = batch_idx * self.batch_size + i
            task_id = f"batch_{batch_idx}_section_{i}"
            
            # 작업을 큐에 추가
            task = self.execute_with_resilience(
                task_func=self._process_single_section_safe,
                task_id=task_id,
                timeout=120.0,  # 2분 타임아웃
                max_retries=1,
                section_content=section_content,
                section_index=section_index
            )
            batch_tasks.append(task)
        
        # 배치 내 모든 작업 병렬 실행
        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
        
        # 결과 정리
        sections = []
        responses = []
        
        for result in batch_results:
            if isinstance(result, Exception):
                print(f"⚠️ 배치 작업 실패: {result}")
                continue
            
            if isinstance(result, dict):
                if result.get("fallback_used"):
                    sections.append(result)
                elif "section_data" in result:
                    sections.append(result["section_data"])
                    responses.extend(result.get("agent_responses", []))
        
        return {"sections": sections, "responses": responses}

    async def _process_single_section_safe(self, section_content: str, section_index: int) -> Dict:
        """안전한 단일 섹션 처리"""
        try:
            print(f"📄 섹션 {section_index+1} 안전 처리 중...")

            # 에이전트 생성 (매번 새로 생성하여 상태 격리)
            layout_analyzer = self.create_layout_analyzer_agent()
            content_editor = self.create_content_editor_agent()

            # 벡터 검색
            similar_layouts = await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: self.vector_manager.search_similar_layouts(
                    section_content[:500], "magazine_layout", top_k=3
                )
            )

            # CrewAI 태스크 생성 및 실행
            crew_result = await self._execute_crew_safe(
                layout_analyzer, content_editor, section_content, similar_layouts, section_index
            )

            # 결과 처리
            title, subtitle = self._extract_clean_title_subtitle(crew_result.get("analysis", ""), section_index)
            clean_content = self._remove_meta_descriptions(crew_result.get("content", section_content))

            # 응답 수집 및 저장
            analysis_response_id, editing_response_id = await asyncio.gather(
                self._log_analysis_response_async(section_index, section_content, similar_layouts, crew_result.get("analysis", "")),
                self._log_editing_response_async(section_index, section_content, crew_result.get("analysis", ""), crew_result.get("content", ""))
            )

            section_data = {
                "title": title,
                "subtitle": subtitle,
                "content": clean_content,
                "layout_info": similar_layouts[0] if similar_layouts else {},
                "original_length": len(section_content),
                "refined_length": len(clean_content),
                "agent_responses": {
                    "layout_analyzer_id": analysis_response_id,
                    "content_editor_id": editing_response_id
                },
                "safe_processed": True
            }

            agent_responses = [{
                "section": section_index + 1,
                "layout_analyzer_response": {
                    "response_id": analysis_response_id,
                    "content": crew_result.get("analysis", ""),
                    "agent_name": "OrgAgent_LayoutAnalyzer"
                },
                "content_editor_response": {
                    "response_id": editing_response_id,
                    "content": crew_result.get("content", ""),
                    "agent_name": "OrgAgent_ContentEditor"
                }
            }]

            print(f"✅ 섹션 {section_index+1} 안전 처리 완료: {len(section_content)}자 → {len(clean_content)}자")
            return {
                "section_data": section_data,
                "agent_responses": agent_responses
            }

        except Exception as e:
            print(f"⚠️ 섹션 {section_index+1} 안전 처리 실패: {e}")
            error_response_id = await self._log_error_response_async(section_index+1, str(e))
            
            return {
                "section_data": {
                    "title": f"도쿄 여행 이야기 {section_index+1}",
                    "subtitle": "특별한 순간들",
                    "content": section_content,
                    "layout_info": {},
                    "original_length": len(section_content),
                    "refined_length": len(section_content),
                    "error_response_id": error_response_id,
                    "safe_processed": True
                },
                "agent_responses": []
            }

    async def _execute_crew_safe(self, layout_analyzer: Agent, content_editor: Agent,
                               section_content: str, similar_layouts: List[Dict], section_index: int) -> Dict:
        """안전한 CrewAI 실행"""
        try:
            # 간소화된 태스크 생성
            layout_analysis_task = Task(
                description=f"""
다음 텍스트 콘텐츠와 유사한 매거진 레이아웃을 분석하여 최적의 텍스트 배치 전략을 수립하세요:

**분석할 콘텐츠:**
{section_content}

**유사한 매거진 레이아웃 데이터:**
{self._format_layout_data(similar_layouts)}

**출력 형식:**
제목: [구체적이고 매력적인 제목]
부제목: [간결하고 흥미로운 부제목]
편집방향: [전체적인 편집 방향성]
""",
                agent=layout_analyzer,
                expected_output="벡터 데이터 기반 레이아웃 분석 및 편집 전략"
            )

            content_editing_task = Task(
                description=f"""
레이아웃 분석 결과를 바탕으로 다음 콘텐츠를 전문 매거진 수준으로 편집하세요:

**원본 콘텐츠:**
{section_content}

**출력:** 매거진 레이아웃에 최적화된 편집 콘텐츠
""",
                agent=content_editor,
                expected_output="매거진 스타일 레이아웃에 최적화된 전문 콘텐츠",
                context=[layout_analysis_task]
            )

            # 순차 실행 (병렬 실행으로 인한 복잡성 제거)
            analysis_result = await asyncio.get_event_loop().run_in_executor(
                None, self._execute_single_task, layout_analysis_task
            )
            
            editing_result = await asyncio.get_event_loop().run_in_executor(
                None, self._execute_single_task, content_editing_task
            )

            return {
                "analysis": str(analysis_result),
                "content": str(editing_result)
            }

        except Exception as e:
            print(f"⚠️ 섹션 {section_index+1} CrewAI 안전 실행 실패: {e}")
            return {
                "analysis": "",
                "content": section_content
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

    # 기존 _process_content_async_mode 메서드 유지 (호환성을 위해)
    async def _process_content_async_mode(self, magazine_content, available_templates: List[str]) -> Dict:
        """비동기 모드 콘텐츠 처리 (기존 호환성 유지)"""
        print("⚠️ 기존 async_mode 호출됨 - batch_mode로 리다이렉트")
        return await self._process_content_batch_mode(magazine_content, available_templates)

    async def _process_content_sync_mode(self, magazine_content, available_templates: List[str]) -> Dict:
        """동기 모드 콘텐츠 처리 (run_in_executor 사용)"""
        print("🔄 OrgAgent 동기 모드 실행")
        
        # 동기 버전 메서드들을 executor에서 실행
        loop = asyncio.get_event_loop()
        
        # 기본 데이터 준비 (동기)
        all_content = self._extract_all_text(magazine_content)
        content_sections = self._analyze_content_structure(all_content)
        
        print(f"OrgAgent: 처리할 콘텐츠 - {len(all_content)}자, {len(content_sections)}개 섹션 (동기 처리)")

        # 섹션별 처리 (동기)
        refined_sections = await loop.run_in_executor(
            None, self._process_all_sections_sync, content_sections
        )

        # 템플릿 매핑 (동기)
        text_mapping = await loop.run_in_executor(
            None, self._map_to_templates, refined_sections, available_templates
        )

        total_refined_length = sum(section["refined_length"] for section in refined_sections)

        # 동기 모드 로깅
        final_response_id = await self._log_sync_mode_response_async(
            magazine_content, available_templates, text_mapping, refined_sections, total_refined_length
        )

        print(f"✅ OrgAgent 동기 완료: {len(refined_sections)}개 섹션, 총 {total_refined_length}자")
        return {
            "text_mapping": text_mapping,
            "refined_sections": refined_sections,
            "total_sections": len(refined_sections),
            "total_content_length": total_refined_length,
            "vector_enhanced": True,
            "agent_responses": [],
            "final_response_id": final_response_id,
            "execution_mode": "sync_fallback",
            "recursion_fallback": True
        }

    async def _process_remaining_sections_sync(self, remaining_sections: List[str],
                                             layout_analyzer: Agent, content_editor: Agent,
                                             start_index: int) -> List[Dict]:
        """나머지 섹션들을 동기 모드로 처리"""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._process_sections_sync_batch, remaining_sections, start_index
        )

    # 기존 _process_single_section_async 메서드 유지 (호환성을 위해)
    async def _process_single_section_async(self, section_content: str, section_index: int,
                                          layout_analyzer: Agent, content_editor: Agent) -> tuple:
        """단일 섹션 처리 (기존 호환성 유지) - 안전 모드로 리다이렉트"""
        print("⚠️ 기존 single_section_async 호출됨 - safe 모드로 리다이렉트")
        
        result = await self._process_single_section_safe(section_content, section_index)
        
        # 기존 반환 형식에 맞게 변환
        section_data = result.get("section_data", {})
        agent_responses = result.get("agent_responses", [])
        
        return (section_data, agent_responses)

    # 모든 기존 동기 메서드들과 유틸리티 메서드들 유지
    def _process_all_sections_sync(self, content_sections: List[str]) -> List[Dict]:
        """모든 섹션을 동기 모드로 처리"""
        refined_sections = []
        for i, section_content in enumerate(content_sections):
            if len(section_content.strip()) < 50:
                continue

            # 기본 처리
            title, subtitle = self._extract_basic_title_subtitle(section_content, i)
            clean_content = self._basic_content_cleanup(section_content)

            section_data = {
                "title": title,
                "subtitle": subtitle,
                "content": clean_content,
                "layout_info": {},
                "original_length": len(section_content),
                "refined_length": len(clean_content),
                "sync_processed": True
            }

            refined_sections.append(section_data)
            print(f"✅ 섹션 {i+1} 동기 처리 완료: {len(section_content)}자 → {len(clean_content)}자")

        return refined_sections

    def _process_sections_sync_batch(self, sections: List[str], start_index: int) -> List[Dict]:
        """섹션 배치를 동기 모드로 처리"""
        refined_sections = []
        for i, section_content in enumerate(sections):
            if len(section_content.strip()) < 50:
                continue

            actual_index = start_index + i
            title, subtitle = self._extract_basic_title_subtitle(section_content, actual_index)
            clean_content = self._basic_content_cleanup(section_content)

            section_data = {
                "title": title,
                "subtitle": subtitle,
                "content": clean_content,
                "layout_info": {},
                "original_length": len(section_content),
                "refined_length": len(clean_content),
                "sync_processed": True
            }

            refined_sections.append(section_data)
            print(f"✅ 섹션 {actual_index+1} 동기 처리 완료: {len(section_content)}자 → {len(clean_content)}자")

        return refined_sections

    def _extract_basic_title_subtitle(self, content: str, index: int) -> tuple:
        """기본 제목과 부제목 추출"""
        lines = content.split('\n')
        title = f"도쿄 여행 이야기 {index + 1}"
        subtitle = "특별한 순간들"

        # 첫 번째 줄이 제목으로 적합한지 확인
        if lines and len(lines[0].strip()) > 5 and len(lines[0].strip()) < 100:
            title = lines[0].strip()[:50]

        # 두 번째 줄이 부제목으로 적합한지 확인
        if len(lines) > 1 and len(lines[1].strip()) > 3 and len(lines[1].strip()) < 80:
            subtitle = lines[1].strip()[:40]

        return title, subtitle

    def _basic_content_cleanup(self, content: str) -> str:
        """기본 콘텐츠 정리"""
        # 연속된 줄바꿈 정리
        cleaned = re.sub(r'\n{3,}', '\n\n', content)
        # 앞뒤 공백 제거
        cleaned = cleaned.strip()
        return cleaned

    async def _log_sync_mode_response_async(self, magazine_content, available_templates: List[str],
                                          text_mapping: Dict, refined_sections: List[Dict],
                                          total_refined_length: int) -> str:
        """동기 모드 응답 로깅 (비동기)"""
        return await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.logger.log_agent_real_output(
                agent_name="OrgAgent_SyncMode",
                agent_role="동기 모드 텍스트 배치 에이전트",
                task_description=f"동기 모드로 {len(refined_sections)}개 섹션을 {len(available_templates)}개 템플릿에 매핑",
                final_answer=str(text_mapping),
                reasoning_process="재귀 깊이 초과로 인한 동기 모드 전환 후 안전한 콘텐츠 처리 실행",
                execution_steps=[
                    "재귀 깊이 감지",
                    "동기 모드 전환",
                    "콘텐츠 추출 및 분석",
                    "섹션별 기본 처리",
                    "템플릿 매핑"
                ],
                raw_input={
                    "magazine_content": str(magazine_content)[:500],
                    "available_templates": available_templates
                },
                raw_output={
                    "text_mapping": text_mapping,
                    "refined_sections": refined_sections
                },
                performance_metrics={
                    "sync_mode_used": True,
                    "recursion_fallback": True,
                    "total_sections_processed": len(refined_sections),
                    "total_content_length": total_refined_length,
                    "safe_execution": True
                }
            )
        )

    # 기존 비동기 메서드들 유지
    async def _get_similar_layouts_async(self, section_content: str) -> List[Dict]:
        """유사한 레이아웃 비동기 검색"""
        return await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.vector_manager.search_similar_layouts(
                section_content[:500], "magazine_layout", top_k=3
            )
        )

    async def _log_analysis_response_async(self, section_index: int, section_content: str,
                                         similar_layouts: List[Dict], analysis_result: str) -> str:
        """레이아웃 분석 에이전트 응답 저장 (비동기)"""
        return await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.logger.log_agent_real_output(
                agent_name="OrgAgent_LayoutAnalyzer",
                agent_role="매거진 구조 아키텍트",
                task_description=f"섹션 {section_index+1} 텍스트 레이아웃 분석 및 편집 전략 수립",
                final_answer=analysis_result,
                reasoning_process=f"PDF 벡터 데이터 {len(similar_layouts)}개 레이아웃 참조하여 분석",
                execution_steps=[
                    "콘텐츠 특성 분석",
                    "유사 레이아웃 매칭",
                    "편집 전략 수립"
                ],
                raw_input={
                    "section_content": section_content[:500],
                    "similar_layouts": similar_layouts,
                    "section_index": section_index
                },
                raw_output=analysis_result,
                performance_metrics={
                    "content_length": len(section_content),
                    "layouts_referenced": len(similar_layouts),
                    "analysis_depth": "comprehensive"
                }
            )
        )

    async def _log_editing_response_async(self, section_index: int, section_content: str,
                                        analysis_result: str, edited_content: str) -> str:
        """콘텐츠 편집 에이전트 응답 저장 (비동기)"""
        return await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.logger.log_agent_real_output(
                agent_name="OrgAgent_ContentEditor",
                agent_role="구조 기반 매거진 콘텐츠 편집자",
                task_description=f"섹션 {section_index+1} 매거진 스타일 콘텐츠 편집",
                final_answer=edited_content,
                reasoning_process="레이아웃 분석 결과를 바탕으로 매거진 수준 편집 실행",
                execution_steps=[
                    "분석 결과 검토",
                    "텍스트 구조화",
                    "매거진 스타일 적용",
                    "최종 편집 완료"
                ],
                raw_input={
                    "original_content": section_content,
                    "analysis_result": analysis_result
                },
                raw_output=edited_content,
                performance_metrics={
                    "original_length": len(section_content),
                    "edited_length": len(edited_content),
                    "editing_quality": "professional"
                }
            )
        )

    async def _log_error_response_async(self, section_number: int, error_message: str) -> str:
        """에러 응답 저장 (비동기)"""
        return await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.logger.log_agent_real_output(
                agent_name="OrgAgent_Error",
                agent_role="에러 처리",
                task_description=f"섹션 {section_number} 처리 중 에러 발생",
                final_answer=f"ERROR: {error_message}",
                reasoning_process="에이전트 실행 중 예외 발생",
                error_logs=[{"error": error_message, "section": section_number}]
            )
        )

    async def _map_to_templates_async(self, refined_sections: List[Dict], available_templates: List[str]) -> Dict:
        """섹션을 템플릿에 매핑 (비동기)"""
        return await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self._map_to_templates(refined_sections, available_templates)
        )

    async def _log_final_response_async(self, input_data: Dict, text_mapping: Dict,
                                      refined_sections: List[Dict], all_agent_responses: List[Dict],
                                      total_refined_length: int) -> str:
        """전체 OrgAgent 프로세스 응답 저장 (비동기)"""
        return await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.logger.log_agent_real_output(
                agent_name="OrgAgent",
                agent_role="PDF 벡터 데이터 기반 텍스트 배치 에이전트",
                task_description=f"{input_data['content_sections_count']}개 콘텐츠 섹션을 {len(input_data['available_templates'])}개 템플릿에 매핑",
                final_answer=str(text_mapping),
                reasoning_process=f"개선된 배치 처리 시스템으로 안전한 {len(refined_sections)}개 섹션 처리 완료",
                execution_steps=[
                    "재귀 깊이 체크",
                    "배치 기반 처리 모드 선택",
                    "콘텐츠 추출 및 분석",
                    "섹션 배치별 처리",
                    "템플릿 매핑"
                ],
                raw_input=input_data,
                raw_output={
                    "text_mapping": text_mapping,
                    "refined_sections": refined_sections,
                    "all_agent_responses": all_agent_responses
                },
                performance_metrics={
                    "total_sections_processed": len(refined_sections),
                    "total_content_length": total_refined_length,
                    "successful_sections": len([s for s in refined_sections if "error_response_id" not in s]),
                    "agent_responses_collected": len(all_agent_responses),
                    "recursion_depth_check": True,
                    "safe_execution": True,
                    "batch_processing": True
                }
            )
        )

    # 기존 동기 메서드들 유지 (호환성 보장)
    def _extract_clean_title_subtitle(self, analysis_result: str, index: int) -> tuple:
        """분석 결과에서 깨끗한 제목과 부제목 추출"""
        title_pattern = r'제목[:\s]*([^\n]+)'
        subtitle_pattern = r'부제목[:\s]*([^\n]+)'

        title_match = re.search(title_pattern, analysis_result)
        subtitle_match = re.search(subtitle_pattern, analysis_result)

        title = title_match.group(1).strip() if title_match else f"도쿄 여행 이야기 {index + 1}"
        subtitle = subtitle_match.group(1).strip() if subtitle_match else "특별한 순간들"

        # 설명 텍스트 제거
        title = self._clean_title_from_descriptions(title)
        subtitle = self._clean_title_from_descriptions(subtitle)

        # 제목 길이 조정
        if len(title) > 40:
            title = title[:37] + "..."
        if len(subtitle) > 30:
            subtitle = subtitle[:27] + "..."

        return title, subtitle

    def _clean_title_from_descriptions(self, text: str) -> str:
        """제목에서 설명 텍스트 제거"""
        patterns_to_remove = [
            r'\(헤드라인\)', r'\(섹션 타이틀\)', r'및 부.*?배치.*?있음',
            r'필자 정보.*?있음', r'포토 크레딧.*?있음', r'계층적.*?있음',
            r'과 본문.*?관계', r'배치.*?관계', r'상단.*?배치',
            r'좌상단.*?배치', r'혹은.*?배치', r'없이.*?집중',
            r'그 아래로.*?있습니다'
        ]

        clean_text = text
        for pattern in patterns_to_remove:
            clean_text = re.sub(pattern, '', clean_text, flags=re.IGNORECASE | re.DOTALL)

        # 연속된 공백과 특수문자 정리
        clean_text = re.sub(r'\s+', ' ', clean_text)
        clean_text = re.sub(r'^[,\s:]+|[,\s:]+$', '', clean_text)

        return clean_text.strip() if clean_text.strip() else "도쿄 여행 이야기"

    def _remove_meta_descriptions(self, content: str) -> str:
        """콘텐츠에서 메타 설명 제거"""
        patterns_to_remove = [
            r'\*이 페이지에는.*?살렸습니다\.\*',
            r'블록은 균형.*?줄여줍니다',
            r'\(사진 캡션\)',
            r'시각적 리듬과.*?살렸습니다',
            r'충분한 여백.*?완성합니다',
            r'사진은 본문.*?완성합니다',
            r'이 콘텐츠는.*?디자인되었습니다'
        ]

        clean_content = content
        for pattern in patterns_to_remove:
            clean_content = re.sub(pattern, '', clean_content, flags=re.IGNORECASE | re.DOTALL)

        return clean_content.strip()

    def _format_layout_data(self, similar_layouts: List[Dict]) -> str:
        """레이아웃 데이터를 텍스트로 포맷팅"""
        if not similar_layouts:
            return "유사한 레이아웃 데이터 없음"

        formatted_data = []
        for i, layout in enumerate(similar_layouts):
            formatted_data.append(f"""
레이아웃 {i+1} (유사도: {layout.get('score', 0):.2f}):
- 출처: {layout.get('pdf_name', 'unknown')} (페이지 {layout.get('page_number', 0)})
- 텍스트 샘플: {layout.get('text_content', '')[:200]}...
- 이미지 수: {len(layout.get('image_info', []))}개
- 레이아웃 특징: {self._summarize_layout_info(layout.get('layout_info', {}))}
""")

        return "\n".join(formatted_data)

    def _summarize_layout_info(self, layout_info: Dict) -> str:
        """레이아웃 정보 요약"""
        text_blocks = layout_info.get('text_blocks', [])
        images = layout_info.get('images', [])
        tables = layout_info.get('tables', [])

        summary = []
        if text_blocks:
            summary.append(f"텍스트 블록 {len(text_blocks)}개")
        if images:
            summary.append(f"이미지 {len(images)}개")
        if tables:
            summary.append(f"테이블 {len(tables)}개")

        return ", ".join(summary) if summary else "기본 레이아웃"

    def _extract_all_text(self, magazine_content) -> str:
        """모든 텍스트 추출"""
        if isinstance(magazine_content, dict):
            all_text = ""
            # 우선순위에 따른 텍스트 추출
            priority_fields = [
                "integrated_content", "essay_content", "interview_content",
                "sections", "content", "body", "text"
            ]

            for field in priority_fields:
                if field in magazine_content:
                    value = magazine_content[field]
                    if isinstance(value, str) and value.strip():
                        all_text += value + "\n\n"
                    elif isinstance(value, dict):
                        for sub_key, sub_value in value.items():
                            if isinstance(sub_value, str) and sub_value.strip():
                                all_text += sub_value + "\n\n"
                    elif isinstance(value, list):
                        for item in value:
                            if isinstance(item, dict):
                                for sub_key, sub_value in item.items():
                                    if isinstance(sub_value, str) and sub_value.strip():
                                        all_text += sub_value + "\n\n"
                            elif isinstance(item, str) and item.strip():
                                all_text += item + "\n\n"

            return all_text.strip()
        else:
            return str(magazine_content)

    def _analyze_content_structure(self, content: str) -> List[str]:
        """콘텐츠 구조 분석 및 지능적 분할"""
        if not content:
            return []

        sections = []

        # 1. 헤더 기반 분할
        header_sections = self._split_by_headers(content)
        if len(header_sections) >= 3:
            sections.extend(header_sections)

        # 2. 문단 기반 분할
        if len(sections) < 5:
            paragraph_sections = self._split_by_paragraphs(content)
            sections.extend(paragraph_sections)

        # 3. 의미 기반 분할
        if len(sections) < 6:
            semantic_sections = self._split_by_semantics(content)
            sections.extend(semantic_sections)

        # 중복 제거 및 길이 필터링
        unique_sections = []
        seen_content = set()
        for section in sections:
            section_clean = re.sub(r'\s+', ' ', section.strip())
            if len(section_clean) >= 100 and section_clean not in seen_content:
                unique_sections.append(section)
                seen_content.add(section_clean)

        return unique_sections[:8]  # 최대 8개 섹션

    def _split_by_headers(self, content: str) -> List[str]:
        """헤더 기반 분할"""
        sections = []
        header_pattern = r'^(#{1,3})\s+(.+?)$'
        current_section = ""
        lines = content.split('\n')

        for line in lines:
            if re.match(header_pattern, line.strip()):
                if current_section.strip():
                    sections.append(current_section.strip())
                current_section = line + "\n"
            else:
                current_section += line + "\n"

        if current_section.strip():
            sections.append(current_section.strip())

        return [s for s in sections if len(s) >= 100]

    def _split_by_paragraphs(self, content: str) -> List[str]:
        """문단 기반 분할"""
        paragraphs = content.split('\n\n')
        sections = []
        current_section = ""

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            if len(current_section + paragraph) > 800:
                if current_section:
                    sections.append(current_section.strip())
                current_section = paragraph + "\n\n"
            else:
                current_section += paragraph + "\n\n"

        if current_section.strip():
            sections.append(current_section.strip())

        return [s for s in sections if len(s) >= 100]

    def _split_by_semantics(self, content: str) -> List[str]:
        """의미 기반 분할"""
        sentences = re.split(r'[.!?]\s+', content)
        sections = []
        current_section = ""

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            if len(current_section + sentence) > 600:
                if current_section:
                    sections.append(current_section.strip())
                current_section = sentence + ". "
            else:
                current_section += sentence + ". "

        if current_section.strip():
            sections.append(current_section.strip())

        return [s for s in sections if len(s) >= 100]

    def _map_to_templates(self, refined_sections: List[Dict], available_templates: List[str]) -> Dict:
        """섹션을 템플릿에 매핑"""
        text_mapping = []

        for i, section in enumerate(refined_sections):
            template_index = i % len(available_templates) if available_templates else 0
            template_name = available_templates[template_index] if available_templates else f"Section{i+1:02d}.jsx"

            text_mapping.append({
                "template": template_name,
                "title": section["title"],
                "subtitle": section["subtitle"],
                "body": section["content"],
                "tagline": "TRAVEL & CULTURE",
                "layout_source": section.get("layout_info", {}).get("pdf_name", "default"),
                "agent_responses": section.get("agent_responses", {})
            })

        return {"text_mapping": text_mapping}

    # 동기 버전 메서드 (호환성 보장)
    def process_content_sync(self, magazine_content, available_templates: List[str]) -> Dict:
        """동기 버전 콘텐츠 처리 (호환성 유지)"""
        return asyncio.run(self.process_content(magazine_content, available_templates))
