"""
에이전트 간 데이터 전파 격리 시스템
에이전트 간 데이터 교환 시 오염 방지 및 격리 적용
"""
import time
from typing import Dict, Any
from dataclasses import dataclass
from utils.ai_search_isolation import AISearchIsolationManager
from utils.session_isolation import SessionManager

@dataclass
class DataTransferRequest:
    """데이터 전송 요청"""
    source_agent: str
    target_agent: str
    data: Any
    transfer_type: str
    session_id: str
    timestamp: float

class AgentCommunicationIsolator:
    """에이전트 간 통신 격리자"""
    
    def __init__(self):
        self.isolation_manager = AISearchIsolationManager()
        self.session_manager = SessionManager()
        self.transfer_log = []
        self.blocked_transfers = []
        
    def transfer_data(self, request: DataTransferRequest) -> Dict[str, Any]:
        """격리된 데이터 전송"""
        print(f"🔄 데이터 전송: {request.source_agent} → {request.target_agent}")
        
        # 1. 소스 데이터 오염 검사
        if self.isolation_manager.is_contaminated(
            request.data, f"{request.source_agent}_to_{request.target_agent}"
        ):
            print(f"🚫 오염된 데이터 전송 차단: {request.source_agent} → {request.target_agent}")
            self.blocked_transfers.append({
                "request": request,
                "reason": "contamination_detected",
                "timestamp": time.time()
            })
            return {
                "success": False,
                "reason": "contamination_detected",
                "cleaned_data": self._get_clean_fallback_data(request)
            }
        
        # 2. 세션 격리 검증
        if not self._validate_session_isolation(request):
            print(f"🚫 세션 격리 위반: {request.source_agent} → {request.target_agent}")
            return {
                "success": False,
                "reason": "session_isolation_violation",
                "cleaned_data": self._get_session_isolated_data(request)
            }
        
        # 3. 데이터 정화 및 전송
        cleaned_data = self._clean_transfer_data(request.data, request)
        
        # 4. 전송 로그 기록
        self.transfer_log.append({
            "source_agent": request.source_agent,
            "target_agent": request.target_agent,
            "transfer_type": request.transfer_type,
            "session_id": request.session_id,
            "data_size": len(str(cleaned_data)),
            "timestamp": request.timestamp,
            "isolation_applied": True
        })
        
        print(f"✅ 격리된 데이터 전송 완료: {request.source_agent} → {request.target_agent}")
        return {
            "success": True,
            "cleaned_data": cleaned_data,
            "isolation_metadata": {
                "contamination_filtered": True,
                "session_isolated": True,
                "transfer_id": len(self.transfer_log)
            }
        }
    
    def _validate_session_isolation(self, request: DataTransferRequest) -> bool:
        """세션 격리 검증"""
        # 같은 세션 내 에이전트 간 통신만 허용
        source_session = self._get_agent_session(request.source_agent)
        target_session = self._get_agent_session(request.target_agent)
        
        if source_session != target_session:
            print(f"⚠️ 세션 불일치: {request.source_agent}({source_session}) → {request.target_agent}({target_session})")
            return False
        
        return True
    
    def _clean_transfer_data(self, data: Any, request: DataTransferRequest) -> Any:
        """전송 데이터 정화"""
        if isinstance(data, dict):
            cleaned_data = {}
            for key, value in data.items():
                if not self.isolation_manager.is_contaminated(value, f"transfer_{key}"):
                    cleaned_data[key] = self._deep_clean_value(value, request)
                else:
                    print(f"🧹 오염된 키 제거: {key}")
            
            # 메타데이터 추가
            cleaned_data["_isolation_metadata"] = {
                "source_agent": request.source_agent,
                "target_agent": request.target_agent,
                "transfer_timestamp": request.timestamp,
                "isolation_applied": True,
                "session_id": request.session_id
            }
            
            return cleaned_data
        
        elif isinstance(data, list):
            return [
                self._deep_clean_value(item, request) 
                for item in data 
                if not self.isolation_manager.is_contaminated(item, "transfer_list_item")
            ]
        
        else:
            return data if not self.isolation_manager.is_contaminated(data, "transfer_value") else None
    
    def _deep_clean_value(self, value: Any, request: DataTransferRequest) -> Any:
        """값 깊은 정화"""
        if isinstance(value, dict):
            return {
                k: self._deep_clean_value(v, request) 
                for k, v in value.items() 
                if not self.isolation_manager.is_contaminated(v, f"deep_clean_{k}")
            }
        elif isinstance(value, list):
            return [
                self._deep_clean_value(item, request) 
                for item in value 
                if not self.isolation_manager.is_contaminated(item, "deep_clean_list")
            ]
        else:
            return value
    
    def _get_clean_fallback_data(self, request: DataTransferRequest) -> Any:
        """정화된 폴백 데이터"""
        fallback_data = {
            "source_agent": request.source_agent,
            "target_agent": request.target_agent,
            "fallback_reason": "contamination_detected",
            "session_id": request.session_id,
            "timestamp": request.timestamp,
            "isolation_applied": True
        }
        
        # 에이전트별 특화 폴백 데이터
        if "OrgAgent" in request.target_agent:
            fallback_data.update({
                "text_mapping": [],
                "refined_sections": [],
                "total_sections": 0
            })
        elif "BindingAgent" in request.target_agent:
            fallback_data.update({
                "image_distribution": {},
                "template_distributions": [],
                "layout_recommendations": []
            })
        elif "CoordinatorAgent" in request.target_agent:
            fallback_data.update({
                "selected_templates": [],
                "content_sections": [],
                "integration_metadata": {}
            })
        
        return fallback_data
    
    def _get_session_isolated_data(self, request: DataTransferRequest) -> Any:
        """세션 격리된 데이터"""
        return {
            "session_isolation_violation": True,
            "allowed_session": request.session_id,
            "fallback_data": self._get_clean_fallback_data(request)
        }
    
    def _get_agent_session(self, agent_name: str) -> str:
        """에이전트의 세션 ID 조회"""
        # 현재 스레드의 세션 ID 반환
        import threading
        return getattr(threading.current_thread(), 'session_id', 'default_session')
    
    def get_transfer_statistics(self) -> Dict[str, Any]:
        """전송 통계"""
        total_transfers = len(self.transfer_log)
        blocked_transfers = len(self.blocked_transfers)
        
        return {
            "total_transfers": total_transfers,
            "successful_transfers": total_transfers,
            "blocked_transfers": blocked_transfers,
            "success_rate": (total_transfers / (total_transfers + blocked_transfers) * 100) if (total_transfers + blocked_transfers) > 0 else 0,
            "isolation_effectiveness": (blocked_transfers / (total_transfers + blocked_transfers) * 100) if (total_transfers + blocked_transfers) > 0 else 0
        }

class InterAgentCommunicationMixin:
    """에이전트 간 통신 믹스인"""
    
    def __init_inter_agent_communication__(self):
        """에이전트 간 통신 시스템 초기화"""
        self.communication_isolator = AgentCommunicationIsolator()
        self.agent_name = self.__class__.__name__
        print(f"📡 {self.agent_name} 에이전트 간 통신 격리 시스템 활성화")
    
    def send_data_to_agent(self, target_agent: str, data: Any, 
                          transfer_type: str = "result", session_id: str = None) -> Dict[str, Any]:
        """다른 에이전트로 데이터 전송 (격리 적용)"""
        if session_id is None:
            session_id = getattr(self, 'current_session_id', 'default_session')
        
        request = DataTransferRequest(
            source_agent=self.agent_name,
            target_agent=target_agent,
            data=data,
            transfer_type=transfer_type,
            session_id=session_id,
            timestamp=time.time()
        )
        
        return self.communication_isolator.transfer_data(request)
    
    def receive_data_from_agent(self, source_agent: str, data: Any) -> Any:
        """다른 에이전트로부터 데이터 수신 (격리 검증)"""
        # 수신 데이터 오염 검사
        if self.communication_isolator.isolation_manager.is_contaminated(
            data, f"{source_agent}_to_{self.agent_name}_receive"
        ):
            print(f"🚫 {self.agent_name}: {source_agent}로부터 오염된 데이터 수신 거부")
            return None
        
        print(f"📨 {self.agent_name}: {source_agent}로부터 정화된 데이터 수신")
        return data
    
    def get_communication_stats(self) -> Dict[str, Any]:
        """통신 통계 조회"""
        return self.communication_isolator.get_transfer_statistics()

# 전역 통신 격리자
communication_isolator = AgentCommunicationIsolator()
