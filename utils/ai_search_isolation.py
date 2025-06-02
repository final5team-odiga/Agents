"""
AI Search 격리 모듈
모든 에이전트에서 Azure AI Search 데이터 오염을 차단하고 원본 데이터 무결성을 보장
"""

import re
import json
import os
from typing import Any, Dict, List, Optional
from dataclasses import dataclass


@dataclass
class IsolationConfig:
    """격리 설정 클래스"""
    azure_search_keywords: List[str]
    ai_generated_patterns: List[str]
    trusted_domains: List[str]
    preservation_threshold: float
    max_images_per_section: int
    enable_logging: bool

class AISearchIsolationManager:
    """AI Search 격리 관리자 - 모든 에이전트에서 공통 사용"""
    
    def __init__(self, config: Optional[IsolationConfig] = None):
        self.config = config or self._get_default_config()
        self.contamination_log = []
        
    def _get_default_config(self) -> IsolationConfig:
        """기본 격리 설정"""
        return IsolationConfig(
            azure_search_keywords=[
                "도시의 미학", "골목길의 재발견", "아티스트 인터뷰", "친환경 도시",
                "도심 속 자연", "빛과 그림자", "새로운 시선", "편집장의 글",
                "특집:", "포토 에세이", "트렌드:", "프로파일 하이라이트",
                "도시 계획", "건축의 미학", "문화 탐방", "라이프스타일"
            ],
            ai_generated_patterns=[
                "특별한 이야기를 담고 있습니다",
                "자세한 이야기를 담고 있습니다", 
                "새로운 관점에서",
                "독특한 매력을",
                "흥미로운 경험을",
                "다양한 이야기가 펼쳐집니다",
                "특별한 경험을 선사합니다"
            ],
            trusted_domains=[
                "blob.core.windows.net",
                "your-trusted-cdn.com"
            ],
            preservation_threshold=0.3,
            max_images_per_section=3,
            enable_logging=True
        )
    
    def is_contaminated(self, data: Any, context: str = "") -> bool:
        """데이터 오염 여부 검사"""
        if isinstance(data, str):
            return self._check_text_contamination(data, context)
        elif isinstance(data, dict):
            return self._check_dict_contamination(data, context)
        elif isinstance(data, list):
            return any(self.is_contaminated(item, f"{context}[{i}]") for i, item in enumerate(data))
        return False
    
    def _check_text_contamination(self, text: str, context: str = "") -> bool:
        """텍스트 오염 검사"""
        if not text or not isinstance(text, str):
            return False
            
        text_lower = text.lower()
        
        # Azure AI Search 키워드 검사
        for keyword in self.config.azure_search_keywords:
            if keyword.lower() in text_lower:
                self._log_contamination("azure_keyword", keyword, context)
                return True
        
        # AI 생성 패턴 검사
        for pattern in self.config.ai_generated_patterns:
            if pattern.lower() in text_lower:
                self._log_contamination("ai_pattern", pattern, context)
                return True
                
        return False
    
    def _check_dict_contamination(self, data: dict, context: str = "") -> bool:
        """딕셔너리 오염 검사"""
        # 폴백 데이터 검사
        if data.get("fallback_used") or data.get("metadata", {}).get("fallback_used"):
            self._log_contamination("fallback_data", "fallback_used=True", context)
            return True
            
        # 텍스트 필드 검사
        text_fields = ["title", "subtitle", "body", "content", "final_answer", "description"]
        for field in text_fields:
            if field in data and self._check_text_contamination(str(data[field]), f"{context}.{field}"):
                return True
                
        return False
    
    def _log_contamination(self, contamination_type: str, detected_content: str, context: str):
        """오염 감지 로깅"""
        if self.config.enable_logging:
            log_entry = {
                "type": contamination_type,
                "content": detected_content[:100],
                "context": context,
                "timestamp": __import__("time").time()
            }
            self.contamination_log.append(log_entry)
            print(f"🚫 AI Search 오염 감지 [{contamination_type}]: {detected_content[:50]}... (위치: {context})")
    
    def filter_contaminated_data(self, data_list: List[Any], context: str = "") -> List[Any]:
        """오염된 데이터 필터링"""
        if not isinstance(data_list, list):
            return data_list
            
        clean_data = []
        contaminated_count = 0
        
        for i, item in enumerate(data_list):
            if not self.is_contaminated(item, f"{context}[{i}]"):
                clean_data.append(item)
            else:
                contaminated_count += 1
        
        if contaminated_count > 0:
            print(f"🛡️ AI Search 격리: {contaminated_count}개 오염 데이터 제거, {len(clean_data)}개 정화 데이터 유지")
            
        return clean_data
    
    def validate_original_preservation(self, result: Any, original: str, context: str = "") -> Dict[str, Any]:
        """원본 데이터 보존 검증"""
        if not isinstance(result, dict) or not original:
            return {"preservation_rate": 0.0, "contamination_detected": True}
        
        # 원본 키워드 추출
        original_words = set(re.findall(r'\w+', original.lower()))
        
        # 결과에서 키워드 추출
        result_text = ""
        for key in ['title', 'subtitle', 'content', 'body', 'final_answer']:
            if key in result:
                result_text += str(result[key]) + " "
        
        result_words = set(re.findall(r'\w+', result_text.lower()))
        
        # 보존율 계산
        if original_words:
            preserved = original_words.intersection(result_words)
            preservation_rate = len(preserved) / len(original_words)
        else:
            preservation_rate = 0.0
        
        # 오염 검사
        contamination_detected = self.is_contaminated(result, context)
        
        return {
            "preservation_rate": preservation_rate,
            "original_keywords": len(original_words),
            "preserved_keywords": len(preserved),
            "contamination_detected": contamination_detected,
            "meets_threshold": preservation_rate >= self.config.preservation_threshold,
            "context": context
        }
    
    def clean_query_from_azure_keywords(self, query: str) -> str:
        """쿼리에서 Azure AI Search 키워드 제거"""
        if not query:
            return "magazine layout design structure"
            
        clean_query = query
        for keyword in self.config.azure_search_keywords:
            clean_query = clean_query.replace(keyword, "")
        
        # 빈 쿼리 방지
        clean_query = clean_query.strip()
        if len(clean_query) < 10:
            clean_query = "magazine layout design structure"
            
        return clean_query
    
    def is_trusted_image_url(self, url: str) -> bool:
        """신뢰할 수 있는 이미지 URL 검증"""
        if not url or not isinstance(url, str):
            return False
        
        # 예시 URL이나 플레이스홀더 제외
        excluded_patterns = ['example.com', 'placeholder', 'sample', 'demo']
        for pattern in excluded_patterns:
            if pattern in url.lower():
                return False
        
        # 신뢰할 수 있는 도메인 확인
        for domain in self.config.trusted_domains:
            if domain in url:
                return True
                
        return False
    
    def restore_original_content(self, original_data: Dict[str, Any], exclude_keys: Optional[List[str]] = None) -> Dict[str, Any]:
        """원본 콘텐츠 복원"""
        if not original_data:
            return {}
        
        exclude_keys = exclude_keys or ['template', 'template_data', 'jsx_template']
        restored_data = {}
        
        try:
            for key, value in original_data.items():
                if key.lower() not in [k.lower() for k in exclude_keys]:
                    if isinstance(value, dict):
                        restored_value = {}
                        for nested_key, nested_value in value.items():
                            if nested_key.lower() not in [k.lower() for k in exclude_keys]:
                                restored_value[nested_key] = self._deep_copy_value(nested_value)
                        restored_data[key] = restored_value
                    else:
                        restored_data[key] = self._deep_copy_value(value)
            
            return restored_data
            
        except Exception as e:
            print(f"⚠️ 원본 콘텐츠 복원 중 오류: {e}")
            return original_data
    
    def _deep_copy_value(self, value: Any) -> Any:
        """값 깊은 복사"""
        try:
            if isinstance(value, dict):
                return {k: self._deep_copy_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [self._deep_copy_value(item) for item in value]
            elif isinstance(value, tuple):
                return tuple(self._deep_copy_value(item) for item in value)
            else:
                return value
        except Exception:
            return value
    
    def get_contamination_report(self) -> Dict[str, Any]:
        """오염 감지 보고서 생성"""
        if not self.contamination_log:
            return {"total_contaminations": 0, "types": {}, "recent_detections": []}
        
        types_count = {}
        for entry in self.contamination_log:
            contamination_type = entry["type"]
            types_count[contamination_type] = types_count.get(contamination_type, 0) + 1
        
        return {
            "total_contaminations": len(self.contamination_log),
            "types": types_count,
            "recent_detections": self.contamination_log[-10:],  # 최근 10개
            "config": {
                "keywords_count": len(self.config.azure_search_keywords),
                "patterns_count": len(self.config.ai_generated_patterns),
                "preservation_threshold": self.config.preservation_threshold
            }
        }
    
    def reset_contamination_log(self):
        """오염 로그 초기화"""
        self.contamination_log.clear()
        print("🧹 AI Search 격리 로그 초기화 완료")

class AgentIsolationMixin:
    """에이전트 격리 기능 믹스인 클래스"""
    
    def __init_isolation__(self, config: Optional[IsolationConfig] = None):
        """격리 시스템 초기화"""
        self.isolation_manager = AISearchIsolationManager(config)
        self.ai_search_isolation = True
        print(f"🛡️ {self.__class__.__name__} AI Search 격리 시스템 활성화")
    
    def _isolate_vector_search_results(self, results: List[Dict], context: str = "") -> List[Dict]:
        """벡터 검색 결과 격리"""
        return self.isolation_manager.filter_contaminated_data(results, f"{context}_vector_search")
    
    def _isolate_agent_responses(self, responses: List[Dict], context: str = "") -> List[Dict]:
        """에이전트 응답 격리"""
        return self.isolation_manager.filter_contaminated_data(responses, f"{context}_agent_responses")
    
    def _validate_content_integrity(self, result: Dict, original_content: str, context: str = "") -> Dict:
        """콘텐츠 무결성 검증"""
        validation_result = self.isolation_manager.validate_original_preservation(
            result, original_content, context
        )
        
        # 보존율이 낮으면 원본 사용
        if not validation_result["meets_threshold"]:
            print(f"⚠️ {context} 원본 보존율 낮음 ({validation_result['preservation_rate']:.2f}), 원본 사용")
            if isinstance(result, dict):
                result["content"] = original_content
                result["preservation_fallback"] = True
        
        # 메타데이터 추가
        if isinstance(result, dict):
            result["ai_search_isolation"] = {
                **validation_result,
                "isolation_applied": True
            }
        
        return result
    
    def _get_isolation_report(self) -> Dict[str, Any]:
        """격리 보고서 반환"""
        return self.isolation_manager.get_contamination_report()

# 에이전트별 특화 격리 클래스들

class BindingAgentIsolation(AgentIsolationMixin):
    """BindingAgent 전용 격리 기능"""
    
    def isolate_layout_recommendations(self, recommendations: List[Dict], image_count: int) -> List[Dict]:
        """레이아웃 추천 격리"""
        # 1차: 기본 오염 필터링
        clean_recommendations = self.isolation_manager.filter_contaminated_data(
            recommendations, "layout_recommendations"
        )
        
        # 2차: 이미지 수 기반 필터링
        relevant_layouts = []
        for layout in clean_recommendations:
            layout_image_count = len(layout.get('image_info', []))
            if abs(layout_image_count - image_count) <= 2:  # 이미지 수 차이 2개 이하
                relevant_layouts.append(layout)
        
        # 3차: 우선순위 적용 (원본 데이터 우선)
        prioritized = self._prioritize_original_layouts(relevant_layouts)
        
        print(f"🛡️ 레이아웃 추천 격리: {len(recommendations)} → {len(prioritized)}개")
        return prioritized[:3]  # 최대 3개
    
    def _prioritize_original_layouts(self, layouts: List[Dict]) -> List[Dict]:
        """원본 레이아웃 우선순위 적용"""
        original_sources = ['image_analysis_json', 'user_uploaded', 'direct_input']
        prioritized = []
        
        for layout in layouts:
            source = layout.get('source', 'unknown')
            if any(original_source in source for original_source in original_sources):
                layout['priority'] = 1
                prioritized.insert(0, layout)
            else:
                layout['priority'] = 2
                prioritized.append(layout)
        
        return prioritized
    
    def isolate_image_urls(self, image_urls: List[str]) -> List[str]:
        """이미지 URL 격리"""
        clean_urls = []
        for url in image_urls:
            if self.isolation_manager.is_trusted_image_url(url):
                clean_urls.append(url)
            else:
                print(f"🚫 신뢰할 수 없는 이미지 URL 제외: {url[:50]}...")
        
        return clean_urls

class OrgAgentIsolation(AgentIsolationMixin):
    """OrgAgent 전용 격리 기능"""
    
    def isolate_content_sections(self, sections: List[str], context: str = "content_sections") -> List[str]:
        """콘텐츠 섹션 격리"""
        clean_sections = []
        
        for i, section in enumerate(sections):
            if not self.isolation_manager.is_contaminated(section, f"{context}[{i}]"):
                clean_sections.append(section)
        
        print(f"🛡️ 콘텐츠 섹션 격리: {len(sections)} → {len(clean_sections)}개")
        return clean_sections
    
    def isolate_vector_query(self, query: str) -> str:
        """벡터 검색 쿼리 격리"""
        return self.isolation_manager.clean_query_from_azure_keywords(query)
    
    def extract_original_content_only(self, magazine_content: Any) -> str:
        """원본 콘텐츠만 추출"""
        if isinstance(magazine_content, dict):
            sections = magazine_content.get("sections", [])
            original_text = []
            
            for section in sections:
                if isinstance(section, dict):
                    title = section.get("title", "")
                    content = section.get("content", "")
                    combined_text = title + " " + content
                    
                    if not self.isolation_manager.is_contaminated(combined_text, "magazine_content_section"):
                        original_text.append(combined_text)
            
            return "\n\n".join(original_text)
        
        elif isinstance(magazine_content, str):
            if not self.isolation_manager.is_contaminated(magazine_content, "magazine_content_string"):
                return magazine_content
        
        return ""

class CoordinatorAgentIsolation(AgentIsolationMixin):
    """CoordinatorAgent 전용 격리 기능"""
    
    def block_azure_search_influence(self, crew_result: Any) -> Dict:
        """Azure AI Search 영향 차단"""
        try:
            if hasattr(crew_result, 'raw'):
                result_text = crew_result.raw
            else:
                result_text = str(crew_result)
            
            if self.isolation_manager.is_contaminated(result_text, "crew_result"):
                print("🚫 Azure AI Search 영향 감지, 원본 데이터로 복원")
                return self._restore_from_magazine_content()
            
            return self._extract_json_from_text(result_text)
            
        except Exception as e:
            print(f"⚠️ Azure Search 영향 차단 실패: {e}")
            return self._restore_from_magazine_content()
    
    def _restore_from_magazine_content(self) -> Dict:
        """magazine_content.json에서 복원"""
        try:
            magazine_content_path = "./output/magazine_content.json"
            if os.path.exists(magazine_content_path):
                with open(magazine_content_path, 'r', encoding='utf-8') as f:
                    original_data = json.load(f)
                
                return self.isolation_manager.restore_original_content(
                    original_data, exclude_keys=['template', 'template_data']
                )
        except Exception as e:
            print(f"⚠️ magazine_content.json 복원 실패: {e}")
        
        return {"selected_templates": [], "content_sections": []}
    
    def _extract_json_from_text(self, text: str) -> Dict:
        """텍스트에서 JSON 추출"""
        try:
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            print(f"⚠️ JSON 파싱 실패: {e}")
        
        return {"selected_templates": [], "content_sections": []}
    
    def validate_content_authenticity(self, final_result: Dict) -> Dict:
        """콘텐츠 진정성 검증"""
        try:
            content_sections = final_result.get("content_sections", [])
            magazine_content_path = "./output/magazine_content.json"
            
            if os.path.exists(magazine_content_path):
                with open(magazine_content_path, 'r', encoding='utf-8') as f:
                    original_data = json.load(f)
                
                original_sections = original_data.get("sections", [])
                
                if len(content_sections) == len(original_sections):
                    corrected_count = 0
                    
                    for i, (generated, original) in enumerate(zip(content_sections, original_sections)):
                        if not self._is_content_similar(
                            generated.get("title", ""), 
                            original.get("title", "")
                        ):
                            print(f"🔄 섹션 {i+1} 원본 데이터로 교체")
                            content_sections[i].update({
                                "title": original.get("title", ""),
                                "subtitle": original.get("subtitle", ""),
                                "body": original.get("content", original.get("body", "")),
                                "metadata": {
                                    **content_sections[i].get("metadata", {}),
                                    "source": "magazine_content_json_corrected",
                                    "azure_search_influence": "corrected",
                                    "original_content_preserved": True
                                }
                            })
                            corrected_count += 1
                    
                    if corrected_count > 0:
                        print(f"✅ {corrected_count}개 섹션이 원본 데이터로 교정됨")
                        final_result["integration_metadata"] = {
                            **final_result.get("integration_metadata", {}),
                            "content_corrections_applied": corrected_count,
                            "azure_search_influence": "corrected"
                        }
            
            return final_result
            
        except Exception as e:
            print(f"⚠️ 콘텐츠 진정성 검증 실패: {e}")
            return final_result
    
    def _is_content_similar(self, text1: str, text2: str) -> bool:
        """콘텐츠 유사성 검사"""
        if not text1 or not text2:
            return False
        
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return False
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        similarity = len(intersection) / len(union) if union else 0
        
        return similarity > self.isolation_manager.config.preservation_threshold

# 유틸리티 함수들

def create_isolation_manager(agent_type: str = "default") -> AISearchIsolationManager:
    """에이전트 타입별 격리 매니저 생성"""
    configs = {
        "binding": IsolationConfig(
            azure_search_keywords=[
                "도시의 미학", "골목길의 재발견", "아티스트 인터뷰", "친환경 도시",
                "도심 속 자연", "빛과 그림자", "새로운 시선", "편집장의 글"
            ],
            ai_generated_patterns=[
                "특별한 이야기를 담고 있습니다",
                "자세한 이야기를 담고 있습니다"
            ],
            trusted_domains=["blob.core.windows.net"],
            preservation_threshold=0.3,
            max_images_per_section=3,
            enable_logging=True
        ),
        "org": IsolationConfig(
            azure_search_keywords=[
                "도시의 미학", "골목길", "도시 계획", "친환경 도시",
                "도심 속 자연", "빛과 그림자", "아티스트 인터뷰"
            ],
            ai_generated_patterns=[
                "특별한 이야기를 담고 있습니다",
                "새로운 관점에서", "독특한 매력을"
            ],
            trusted_domains=["blob.core.windows.net"],
            preservation_threshold=0.3,
            max_images_per_section=3,
            enable_logging=True
        ),
        "coordinator": IsolationConfig(
            azure_search_keywords=[
                "도시의 미학", "골목길", "도시 계획", "친환경 도시",
                "도심 속 자연", "빛과 그림자", "아티스트 인터뷰",
                "특집:", "포토 에세이", "트렌드:"
            ],
            ai_generated_patterns=[
                "특별한 이야기를 담고 있습니다",
                "자세한 이야기를 담고 있습니다",
                "새로운 관점에서", "독특한 매력을"
            ],
            trusted_domains=["blob.core.windows.net"],
            preservation_threshold=0.3,
            max_images_per_section=3,
            enable_logging=True
        )
    }
    
    config = configs.get(agent_type, configs["default"])
    return AISearchIsolationManager(config)

def test_isolation_system():
    """격리 시스템 테스트"""
    print("🧪 AI Search 격리 시스템 테스트 시작")
    
    # 테스트 데이터
    contaminated_text = "도시의 미학을 담은 특별한 이야기를 담고 있습니다"
    clean_text = "독일 여행에서 만난 아름다운 순간들"
    
    manager = AISearchIsolationManager()
    
    # 오염 검사 테스트
    assert manager.is_contaminated(contaminated_text), "오염된 텍스트 감지 실패"
    assert not manager.is_contaminated(clean_text), "깨끗한 텍스트 오탐지"
    
    # 필터링 테스트
    test_data = [clean_text, contaminated_text, "또 다른 깨끗한 텍스트"]
    filtered = manager.filter_contaminated_data(test_data)
    assert len(filtered) == 2, "필터링 결과 불일치"
    
    print("✅ AI Search 격리 시스템 테스트 통과")
    return True

if __name__ == "__main__":
    test_isolation_system()
    print("🛡️ AI Search 격리 모듈 로드 완료")
