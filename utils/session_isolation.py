"""
세션 격리 시스템
각 세션마다 독립적인 데이터 공간을 제공하여 세션 간 데이터 오염 방지
"""

import os
import json
import time
import uuid
import threading
from typing import List, Optional, Any
from dataclasses import dataclass
from pathlib import Path
from utils.ai_search_isolation import AISearchIsolationManager

@dataclass
class SessionConfig:
    """세션 설정"""
    session_id: str
    isolation_level: str  # "strict", "moderate", "minimal"
    data_retention_hours: int
    enable_cross_session_learning: bool
    vector_index_isolation: bool

class SessionManager:
    """세션 관리자 - 세션별 독립적인 데이터 공간 제공"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.sessions = {}
            self.session_data = {}
            self.session_locks = {}
            self.isolation_manager = AISearchIsolationManager()
            self.initialized = True
            print("🔒 SessionManager 초기화 완료")
    
    def create_session(self, isolation_level: str = "strict") -> str:
        """새 세션 생성"""
        session_id = f"session_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        
        config = SessionConfig(
            session_id=session_id,
            isolation_level=isolation_level,
            data_retention_hours=24,
            enable_cross_session_learning=False,
            vector_index_isolation=True
        )
        
        with self._lock:
            self.sessions[session_id] = config
            self.session_data[session_id] = {
                "created_at": time.time(),
                "agent_results": {},
                "vector_data": {},
                "shared_state": {},
                "contamination_log": []
            }
            self.session_locks[session_id] = threading.Lock()
        
        # 세션별 디렉토리 생성
        self._create_session_directory(session_id)
        
        print(f"🆕 새 세션 생성: {session_id} (격리 수준: {isolation_level})")
        return session_id
    
    def get_session_data_path(self, session_id: str) -> str:
        """세션별 데이터 경로 반환"""
        return f"./output/sessions/{session_id}"
    
    def get_isolated_vector_index(self, session_id: str) -> str:
        """세션별 격리된 벡터 인덱스명 반환"""
        return f"magazine-vector-{session_id}"
    
    def store_agent_result(self, session_id: str, agent_name: str, result: Any):
        """세션별 에이전트 결과 저장"""
        if session_id not in self.sessions:
            raise ValueError(f"세션 {session_id}가 존재하지 않습니다")
        
        with self.session_locks[session_id]:
            # AI Search 오염 검사
            if self.isolation_manager.is_contaminated(result, f"{agent_name}_result"):
                print(f"🚫 세션 {session_id}: {agent_name} 결과에서 오염 감지")
                self.session_data[session_id]["contamination_log"].append({
                    "agent": agent_name,
                    "timestamp": time.time(),
                    "contamination_type": "agent_result"
                })
                return False
            
            if agent_name not in self.session_data[session_id]["agent_results"]:
                self.session_data[session_id]["agent_results"][agent_name] = []
            
            self.session_data[session_id]["agent_results"][agent_name].append({
                "timestamp": time.time(),
                "result": result,
                "isolation_verified": True
            })
            
            # 세션별 파일로 저장
            self._save_session_data(session_id)
            return True
    
    def get_agent_results(self, session_id: str, agent_name: str) -> List[Any]:
        """세션별 에이전트 결과 조회 (격리 적용)"""
        if session_id not in self.sessions:
            return []
        
        with self.session_locks[session_id]:
            results = self.session_data[session_id]["agent_results"].get(agent_name, [])
            
            # 격리 수준에 따른 필터링
            config = self.sessions[session_id]
            if config.isolation_level == "strict":
                # 현재 세션 결과만 반환
                return [r["result"] for r in results]
            elif config.isolation_level == "moderate":
                # 최근 결과만 반환
                recent_results = [r for r in results if time.time() - r["timestamp"] < 3600]
                return [r["result"] for r in recent_results]
            else:
                # 모든 결과 반환
                return [r["result"] for r in results]
    
    def get_cross_session_data(self, current_session_id: str, agent_name: str, 
                              max_sessions: int = 3) -> List[Any]:
        """교차 세션 데이터 조회 (격리 적용)"""
        config = self.sessions.get(current_session_id)
        if not config or not config.enable_cross_session_learning:
            return []
        
        cross_session_results = []
        session_count = 0
        
        for session_id, session_config in self.sessions.items():
            if session_id == current_session_id or session_count >= max_sessions:
                continue
            
            results = self.get_agent_results(session_id, agent_name)
            for result in results:
                if not self.isolation_manager.is_contaminated(result, f"cross_session_{agent_name}"):
                    cross_session_results.append(result)
            
            session_count += 1
        
        print(f"🔄 교차 세션 데이터: {len(cross_session_results)}개 결과 (세션 {current_session_id})")
        return cross_session_results
    
    def cleanup_expired_sessions(self):
        """만료된 세션 정리"""
        current_time = time.time()
        expired_sessions = []
        
        for session_id, config in self.sessions.items():
            session_age = current_time - self.session_data[session_id]["created_at"]
            if session_age > config.data_retention_hours * 3600:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            self._cleanup_session(session_id)
            print(f"🗑️ 만료된 세션 정리: {session_id}")
    
    def _create_session_directory(self, session_id: str):
        """세션별 디렉토리 생성"""
        session_path = Path(self.get_session_data_path(session_id))
        session_path.mkdir(parents=True, exist_ok=True)
        
        # 세션 설정 파일 생성
        config_path = session_path / "session_config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump({
                "session_id": session_id,
                "created_at": time.time(),
                "isolation_level": self.sessions[session_id].isolation_level
            }, f, indent=2)
    
    def _save_session_data(self, session_id: str):
        """세션 데이터 파일 저장"""
        session_path = Path(self.get_session_data_path(session_id))
        data_path = session_path / "session_data.json"
        
        try:
            with open(data_path, 'w', encoding='utf-8') as f:
                json.dump(self.session_data[session_id], f, indent=2, default=str)
        except Exception as e:
            print(f"⚠️ 세션 데이터 저장 실패 {session_id}: {e}")
    
    def _cleanup_session(self, session_id: str):
        """세션 정리"""
        with self._lock:
            if session_id in self.sessions:
                del self.sessions[session_id]
            if session_id in self.session_data:
                del self.session_data[session_id]
            if session_id in self.session_locks:
                del self.session_locks[session_id]
        
        # 세션 디렉토리 정리
        session_path = Path(self.get_session_data_path(session_id))
        if session_path.exists():
            import shutil
            shutil.rmtree(session_path)

class SessionAwareMixin:
    """세션 인식 믹스인 클래스"""
    
    def __init_session_awareness__(self, session_id: Optional[str] = None):
        """세션 인식 시스템 초기화"""
        self.session_manager = SessionManager()
        self.current_session_id = session_id or self.session_manager.create_session()
        self.agent_name = self.__class__.__name__
        print(f"🔒 {self.agent_name} 세션 인식 시스템 활성화: {self.current_session_id}")
    
    def store_result(self, result: Any) -> bool:
        """세션별 결과 저장"""
        return self.session_manager.store_agent_result(
            self.current_session_id, self.agent_name, result
        )
    
    def get_previous_results(self, max_results: int = 10) -> List[Any]:
        """이전 결과 조회 (세션 격리 적용)"""
        results = self.session_manager.get_agent_results(
            self.current_session_id, self.agent_name
        )
        return results[-max_results:] if results else []
    
    def get_cross_session_insights(self, max_sessions: int = 3) -> List[Any]:
        """교차 세션 인사이트 조회"""
        return self.session_manager.get_cross_session_data(
            self.current_session_id, self.agent_name, max_sessions
        )
    
    def get_session_isolated_path(self, filename: str) -> str:
        """세션별 격리된 파일 경로"""
        session_path = self.session_manager.get_session_data_path(self.current_session_id)
        return os.path.join(session_path, filename)

# 전역 세션 관리자 인스턴스
session_manager = SessionManager()

def get_current_session() -> str:
    """현재 활성 세션 ID 반환"""
    return getattr(threading.current_thread(), 'session_id', 
                   session_manager.create_session())

def set_current_session(session_id: str):
    """현재 스레드의 세션 ID 설정"""
    threading.current_thread().session_id = session_id
