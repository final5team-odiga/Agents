import os
import json
from typing import List, Dict
from openai import AzureOpenAI
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.models import VectorizedQuery
from azure.search.documents.indexes.models import (
    SearchIndex, SimpleField, SearchFieldDataType, VectorSearch,
    VectorSearchProfile, HnswAlgorithmConfiguration, SearchField
)
from utils.pdf_splitter import PDFSplitter
from dotenv import load_dotenv
from pathlib import Path

# AI Search 격리 시스템 import 추가
try:
    from utils.ai_search_isolation import AISearchIsolationManager
    ISOLATION_AVAILABLE = True
except ImportError:
    print("⚠️ AI Search 격리 모듈을 찾을 수 없습니다. 기본 모드로 실행됩니다.")
    ISOLATION_AVAILABLE = False

# 환경 변수 로드
dotenv_path = Path(r'C:\Users\EL0021\Desktop\odiga_agent\.env')
load_dotenv(dotenv_path=dotenv_path, override=True)

class PDFVectorManager:
    """PDF 벡터 데이터 관리자 - Azure 서비스 활용 (AI Search 격리 적용)"""
    
    def __init__(self, isolation_enabled=True):
        # 기존 초기화 코드
        self.form_recognizer_endpoint = os.getenv("AZURE_FORM_RECOGNIZER_ENDPOINT")
        self.form_recognizer_key = os.getenv("AZURE_FORM_RECOGNIZER_KEY")
        self.form_recognizer_client = DocumentAnalysisClient(
            endpoint=self.form_recognizer_endpoint,
            credential=AzureKeyCredential(self.form_recognizer_key)
        )

        self.search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
        self.search_key = os.getenv("AZURE_SEARCH_KEY")
        self.search_index_name = "magazine-vector-index"
        self.search_client = SearchClient(
            endpoint=self.search_endpoint,
            index_name=self.search_index_name,
            credential=AzureKeyCredential(self.search_key)
        )

        self.search_index_client = SearchIndexClient(
            endpoint=self.search_endpoint,
            credential=AzureKeyCredential(self.search_key)
        )

        self.openai_client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )

        self.embedding_model = "text-embedding-ada-002"
        self.pdf_splitter = PDFSplitter(max_size_mb=20.0)
        
        # AI Search 격리 시스템 초기화 (새로 추가)
        self.isolation_enabled = isolation_enabled and ISOLATION_AVAILABLE
        if self.isolation_enabled:
            self.isolation_manager = AISearchIsolationManager()
            print("🛡️ PDFVectorManager AI Search 격리 시스템 활성화")
        else:
            self.isolation_manager = None
            print("⚠️ PDFVectorManager AI Search 격리 시스템 비활성화")
    
        
    def initialize_search_index(self):
        """Azure Cognitive Search 인덱스 초기화 - 데이터 존재 여부까지 확인"""
        try:
            # 1. 기존 인덱스 존재 여부 확인
            try:
                existing_index = self.search_index_client.get_index(self.search_index_name)
                print(f"✅ 기존 인덱스 '{self.search_index_name}' 발견")
                
                # 인덱스에 실제 데이터가 있는지 확인
                if self._check_index_has_data():
                    print(f"✅ 인덱스에 데이터 존재 ({self._get_document_count()}개 문서) - 벡터 처리 생략")
                    return True  # 초기화 완료, PDF 처리 불필요
                else:
                    print(f"⚠️ 인덱스는 있지만 데이터 없음 - PDF 처리 필요")
                    return False  # 데이터 처리 필요
                    
            except Exception as e:
                print(f"📄 기존 인덱스 없음 - 새로 생성: {e}")
            
            # 2. 새 인덱스 생성 (기존 인덱스가 없는 경우만)
            vector_search = VectorSearch(
                profiles=[
                    VectorSearchProfile(
                        name="magazine-profile",
                        algorithm_configuration_name="magazine-algorithm"
                    )
                ],
                algorithms=[
                    HnswAlgorithmConfiguration(
                        name="magazine-algorithm",
                        parameters={
                            "m": 4,
                            "efConstruction": 400,
                            "efSearch": 500,
                            "metric": "cosine"
                        }
                    )
                ]
            )
            
            fields = [
                SimpleField(name="id", type=SearchFieldDataType.String, key=True),
                SimpleField(name="pdf_name", type=SearchFieldDataType.String, filterable=True),
                SimpleField(name="page_number", type=SearchFieldDataType.Int32, filterable=True),
                SimpleField(name="content_type", type=SearchFieldDataType.String, filterable=True),
                SimpleField(name="text_content", type=SearchFieldDataType.String, searchable=True),
                SimpleField(name="layout_info", type=SearchFieldDataType.String),
                SimpleField(name="image_info", type=SearchFieldDataType.String),
                SearchField(
                    name="content_vector",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    searchable=True,
                    vector_search_dimensions=1536,
                    vector_search_profile_name="magazine-profile"
                )
            ]
            
            index = SearchIndex(
                name=self.search_index_name,
                fields=fields,
                vector_search=vector_search
            )
            
            self.search_index_client.create_index(index)
            print(f"✅ 새 인덱스 '{self.search_index_name}' 생성 완료")
            return False  # 데이터 처리 필요
            
        except Exception as e:
            print(f"❌ 인덱스 초기화 실패: {e}")
            return False

    def _check_index_has_data(self) -> bool:
        """인덱스에 실제 데이터가 있는지 확인 - 강화된 버전"""
        try:
            print("🔍 인덱스 데이터 존재 여부 확인 중...")
            
            # 와일드카드 검색으로 모든 문서 확인 (검색 결과 [8] 참조)
            results = self.search_client.search(
                search_text="*",  # 와일드카드로 모든 문서 검색
                top=1,            # 1개만 확인하면 충분
                include_total_count=True  # 전체 개수 포함
            )
            
            # 결과가 있는지 확인
            document_count = 0
            for result in results:
                document_count += 1
                break  # 첫 번째 결과만 확인하면 충분
            
            # 전체 개수도 확인 (가능한 경우)
            total_count = getattr(results, 'get_count', lambda: None)()
            
            if document_count > 0:
                print(f"✅ 데이터 확인됨: 최소 {document_count}개 문서 존재")
                if total_count:
                    print(f"   총 문서 수: {total_count}개")
                return True
            else:
                print(f"❌ 데이터 없음: 인덱스가 비어있음")
                return False
            
        except Exception as e:
            print(f"❌ 데이터 확인 중 오류: {e}")
            return False

    def _get_document_count(self) -> int:
        """인덱스의 총 문서 수 반환"""
        try:
            results = self.search_client.search(
                search_text="*",
                top=0,  # 문서는 가져오지 않고 개수만
                include_total_count=True
            )
            
            total_count = getattr(results, 'get_count', lambda: 0)()
            return total_count if total_count else 0
            
        except Exception as e:
            print(f"문서 수 확인 중 오류: {e}")
            return 0

    def process_pdf_templates(self, template_folder: str = "templates"):
        """템플릿 폴더의 PDF 파일들을 벡터화하여 인덱스에 저장 - 데이터 존재 여부 확인"""
        
        if not os.path.exists(template_folder):
            print(f"❌ 템플릿 폴더를 찾을 수 없습니다: {template_folder}")
            return
        
        # 인덱스 초기화 및 데이터 존재 여부 확인
        has_data = self.initialize_search_index()
        
        if has_data:
            print("🎉 기존 벡터 데이터 사용 - PDF 처리 완전 생략")
            return
        
        print("📄 PDF 처리 시작 - 인덱스에 데이터가 없거나 새로운 인덱스")
        
        # PDF 분할 기능 테스트
        self.pdf_splitter.test_split_functionality(template_folder)
        
        # 1단계: 큰 PDF 파일들 분할
        print("\n🔪 PDF 분할 단계")
        available_pdf_files = self.pdf_splitter.split_large_pdfs(template_folder)
        
        if not available_pdf_files:
            print(f"❌ 처리할 PDF 파일을 찾을 수 없습니다: {template_folder}")
            return
        
        # 기존에 처리된 PDF 확인 (인덱스에 데이터가 있는 경우만)
        processed_pdfs = self._get_processed_pdfs()
        new_pdfs = [pdf for pdf in available_pdf_files if pdf not in processed_pdfs]
        
        if not new_pdfs and processed_pdfs:
            print("✅ 모든 PDF가 이미 처리됨 - 추가 처리 불필요")
            return
        
        # 처리할 PDF가 없으면 모든 PDF 처리
        pdfs_to_process = new_pdfs if new_pdfs else available_pdf_files
        
        print(f"\n📁 {len(pdfs_to_process)}개 PDF 파일 처리 시작")
        
        all_documents = []
        
        for pdf_file in pdfs_to_process:
            pdf_path = os.path.join(template_folder, pdf_file)
            print(f"📄 처리 중: {pdf_file}")
            
            try:
                # PDF 분석
                pdf_documents = self._analyze_pdf(pdf_path, pdf_file)
                all_documents.extend(pdf_documents)
                print(f"✅ {pdf_file}: {len(pdf_documents)}개 문서 생성")
                
            except Exception as e:
                print(f"❌ {pdf_file} 처리 실패: {e}")
        
        # 벡터 인덱스에 업로드
        if all_documents:
            self._upload_to_search_index(all_documents)
            print(f"✅ 총 {len(all_documents)}개 문서가 벡터 인덱스에 저장됨")
            
            # 최종 확인
            final_count = self._get_document_count()
            print(f"🎯 최종 인덱스 문서 수: {final_count}개")
        else:
            print("⚠️ 처리된 문서가 없습니다")


    def _get_processed_pdfs(self) -> List[str]:
        """이미 처리된 PDF 목록 가져오기"""
        try:
            # 인덱스에서 고유한 PDF 이름들 가져오기
            results = self.search_client.search(
                "*",
                select=["pdf_name"],
                top=1000  # 충분한 수
            )
            
            processed_pdfs = set()
            for result in results:
                pdf_name = result.get("pdf_name")
                if pdf_name:
                    processed_pdfs.add(pdf_name)
            
            return list(processed_pdfs)
            
        except Exception as e:
            print(f"처리된 PDF 확인 중 오류: {e}")
            return []

    
    def _analyze_pdf(self, pdf_path: str, pdf_name: str) -> List[Dict]:
        """Azure Form Recognizer로 PDF 분석 - 안전한 키 생성"""
        
        documents = []
        
        with open(pdf_path, "rb") as pdf_file:
            # Form Recognizer로 레이아웃 분석
            poller = self.form_recognizer_client.begin_analyze_document(
                "prebuilt-layout", pdf_file
            )
            result = poller.result()
            
            # 페이지별 처리
            for page_idx, page in enumerate(result.pages):
                page_content = {
                    "text_blocks": [],
                    "images": [],
                    "tables": [],
                    "layout_structure": []
                }
                
                # 텍스트 블록 추출
                if result.paragraphs:
                    for para in result.paragraphs:
                        if para.bounding_regions and para.bounding_regions[0].page_number == page_idx + 1:
                            page_content["text_blocks"].append({
                                "content": para.content,
                                "bounding_box": self._extract_bounding_box(para.bounding_regions[0]),
                                "role": getattr(para, 'role', 'paragraph')
                            })
                
                # 이미지 정보 추출
                if hasattr(page, 'images'):
                    for img in page.images:
                        page_content["images"].append({
                            "bounding_box": self._extract_bounding_box(img),
                            "confidence": getattr(img, 'confidence', 0.0)
                        })
                
                # 테이블 정보 추출
                if result.tables:
                    for table in result.tables:
                        if table.bounding_regions and table.bounding_regions[0].page_number == page_idx + 1:
                            page_content["tables"].append({
                                "row_count": table.row_count,
                                "column_count": table.column_count,
                                "bounding_box": self._extract_bounding_box(table.bounding_regions[0])
                            })
                
                # 전체 텍스트 추출
                page_text = ""
                for line in page.lines:
                    page_text += line.content + "\n"
                
                # 문서 생성 - 안전한 키 생성 (핵심 수정 부분)
                if page_text.strip():
                    # PDF 파일명에서 안전한 키 생성
                    safe_pdf_name = self._create_safe_document_key(pdf_name)
                    doc_id = f"{safe_pdf_name}_page_{page_idx + 1}"
                    
                    documents.append({
                        "id": doc_id,
                        "pdf_name": pdf_name,  # 원본 파일명은 그대로 유지
                        "page_number": page_idx + 1,
                        "content_type": "magazine_layout",
                        "text_content": page_text.strip(),
                        "layout_info": json.dumps(page_content),
                        "image_info": json.dumps(page_content["images"])
                    })
        
        return documents

    def _create_safe_document_key(self, pdf_name: str) -> str:
        """Azure Cognitive Search에 안전한 문서 키 생성"""
        import re
        
        # .pdf 확장자 제거
        safe_name = pdf_name.replace('.pdf', '').replace('.PDF', '')
        
        # 허용되지 않는 문자를 언더스코어로 변경
        # 허용: 영문자, 숫자, 언더스코어(_), 대시(-), 등호(=)
        safe_name = re.sub(r'[^a-zA-Z0-9_\-=]', '_', safe_name)
        
        # 연속된 언더스코어를 하나로 변경
        safe_name = re.sub(r'_+', '_', safe_name)
        
        # 시작과 끝의 언더스코어 제거
        safe_name = safe_name.strip('_')
        
        # 길이 제한 (Azure Search 키는 최대 1024자, 안전하게 100자로 제한)
        if len(safe_name) > 100:
            safe_name = safe_name[:100]
        
        return safe_name

    
    def _extract_bounding_box(self, region) -> Dict:
        """바운딩 박스 정보 추출"""
        if hasattr(region, 'polygon'):
            points = [(point.x, point.y) for point in region.polygon]
            return {
                "points": points,
                "x": min(p[0] for p in points),
                "y": min(p[1] for p in points),
                "width": max(p[0] for p in points) - min(p[0] for p in points),
                "height": max(p[1] for p in points) - min(p[1] for p in points)
            }
        return {}
    
    def _create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Azure OpenAI Embeddings API로 벡터 생성 - v1.x 방식"""
        embeddings = []
        
        print(f"📊 {len(texts)}개 텍스트에 대한 임베딩 생성 중...")
        
        for i, text in enumerate(texts):
            try:
                # Azure OpenAI v1.x API 사용
                response = self.openai_client.embeddings.create(
                    input=text,
                    model=self.embedding_model
                )
                embeddings.append(response.data[0].embedding)
                
                if (i + 1) % 10 == 0:
                    print(f"   진행률: {i + 1}/{len(texts)} 완료")
                
            except Exception as e:
                print(f"❌ 임베딩 생성 실패 (텍스트 {i+1}): {e}")
                # 기본 벡터 (1536 차원)
                embeddings.append([0.0] * 1536)
        
        print(f"✅ 임베딩 생성 완료: {len(embeddings)}개")
        return embeddings
    
    def _create_embeddings_batch(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """배치 처리로 임베딩 생성 - 효율성 향상"""
        all_embeddings = []
        
        print(f"📊 배치 처리로 {len(texts)}개 텍스트 임베딩 생성 중...")
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            try:
                response = self.openai_client.embeddings.create(
                    input=batch_texts,  # 여러 텍스트 한 번에 처리
                    model=self.embedding_model
                )
                
                batch_embeddings = [data.embedding for data in response.data]
                all_embeddings.extend(batch_embeddings)
                
                print(f"✅ 배치 {i//batch_size + 1} 완료: {len(batch_texts)}개 임베딩")
                
            except Exception as e:
                print(f"❌ 배치 {i//batch_size + 1} 실패: {e}")
                # 실패한 배치는 기본 벡터로 채움
                for _ in batch_texts:
                    all_embeddings.append([0.0] * 1536)
        
        return all_embeddings
    
    def _upload_to_search_index(self, documents: List[Dict]):
        """Azure Cognitive Search 인덱스에 문서 업로드"""
        
        # 텍스트 추출 및 임베딩 생성
        texts = [doc["text_content"] for doc in documents]
        
        # 배치 처리로 임베딩 생성 (효율성 향상)
        if len(texts) > 50:
            embeddings = self._create_embeddings_batch(texts)
        else:
            embeddings = self._create_embeddings(texts)
        
        # 문서에 벡터 추가
        for i, doc in enumerate(documents):
            doc["content_vector"] = embeddings[i]
        
        # 배치 업로드
        try:
            result = self.search_client.upload_documents(documents)
            print(f"✅ {len(documents)}개 문서 업로드 완료")
            
            # 업로드 결과 확인
            failed_count = 0
            for res in result:
                if not res.succeeded:
                    failed_count += 1
                    print(f"❌ 업로드 실패: {res.key} - {res.error_message}")
            
            if failed_count == 0:
                print(f"🎉 모든 문서 업로드 성공!")
            else:
                print(f"⚠️ {failed_count}개 문서 업로드 실패")
                    
        except Exception as e:
            print(f"❌ 문서 업로드 실패: {e}")
    
    def search_similar_layouts(self, query_text: str, content_type: str = None, top_k: int = 5) -> List[Dict]:
        """유사한 레이아웃 검색 (AI Search 격리 적용)"""
        try:
            # 1. 쿼리 격리 (AI Search 키워드 제거)
            if self.isolation_enabled:
                clean_query = self.isolation_manager.clean_query_from_azure_keywords(query_text)
                print(f"🛡️ 쿼리 격리: '{query_text[:50]}...' → '{clean_query[:50]}...'")
            else:
                clean_query = query_text

            # 2. 쿼리 텍스트를 벡터로 변환
            query_embeddings = self._create_embeddings([clean_query])
            query_vector = query_embeddings[0]

            # 3. 벡터 검색 쿼리 생성
            vector_query = VectorizedQuery(
                vector=query_vector,
                k_nearest_neighbors=top_k * 2,  # 격리 필터링을 위해 더 많이 검색
                fields="content_vector"
            )

            # 4. 검색 실행
            search_params = {
                "vector_queries": [vector_query],
                "top": top_k * 2,  # 격리 필터링을 위해 더 많이 검색
                "select": ["id", "pdf_name", "page_number", "content_type", "text_content", "layout_info", "image_info"]
            }

            if content_type:
                search_params["filter"] = f"content_type eq '{content_type}'"

            raw_results = self.search_client.search(**search_params)

            # 5. 원시 결과 수집
            raw_layouts = []
            for result in raw_results:
                layout_info = json.loads(result.get("layout_info", "{}"))
                image_info = json.loads(result.get("image_info", "[]"))
                
                layout_data = {
                    "id": result["id"],
                    "pdf_name": result["pdf_name"],
                    "page_number": result["page_number"],
                    "text_content": result["text_content"],
                    "layout_info": layout_info,
                    "image_info": image_info,
                    "score": result.get("@search.score", 0.0),
                    "source": "pdf_vector_search"
                }
                raw_layouts.append(layout_data)

            # 6. AI Search 격리 필터링
            if self.isolation_enabled:
                filtered_layouts = self.isolation_manager.filter_contaminated_data(
                    raw_layouts, "pdf_vector_search"
                )
                
                # 7. 원본 데이터 우선순위 적용
                prioritized_layouts = self._prioritize_original_layouts(filtered_layouts)
                
                print(f"🛡️ 레이아웃 검색 격리: {len(raw_layouts)} → {len(prioritized_layouts)}개")
                return prioritized_layouts[:top_k]
            else:
                return raw_layouts[:top_k]

        except Exception as e:
            print(f"❌ 벡터 검색 실패: {e}")
            if self.isolation_enabled:
                print("🛡️ 격리된 폴백 결과 반환")
                return self._get_isolated_fallback_layouts()
            return []

    def _prioritize_original_layouts(self, layouts: List[Dict]) -> List[Dict]:
        """원본 레이아웃 우선순위 적용"""
        if not self.isolation_enabled:
            return layouts
            
        original_sources = ['user_uploaded', 'direct_input', 'verified_template']
        prioritized = []
        
        for layout in layouts:
            source = layout.get('source', 'unknown')
            pdf_name = layout.get('pdf_name', '').lower()
            
            # 원본 소스 우선
            if any(original_source in source for original_source in original_sources):
                layout['priority'] = 1
                prioritized.insert(0, layout)
            # 신뢰할 수 있는 PDF 이름 패턴
            elif any(pattern in pdf_name for pattern in ['template', 'layout', 'design']):
                layout['priority'] = 2
                prioritized.append(layout)
            # 기타
            else:
                layout['priority'] = 3
                prioritized.append(layout)
        
        return prioritized

    def _get_isolated_fallback_layouts(self) -> List[Dict]:
        """격리된 폴백 레이아웃 반환"""
        return [{
            "id": "isolated_fallback_1",
            "pdf_name": "isolated_default_layout",
            "page_number": 1,
            "text_content": "기본 매거진 레이아웃",
            "layout_info": {
                "text_blocks": [],
                "images": [],
                "layout_structure": ["single_column"]
            },
            "image_info": [],
            "score": 0.5,
            "source": "isolated_fallback",
            "priority": 1
        }]

    
    def get_layout_recommendations(self, content_description: str, image_count: int) -> List[Dict]:
        """콘텐츠 설명과 이미지 수를 바탕으로 레이아웃 추천 (AI Search 격리 적용)"""
        # 이미지 수에 따른 검색 쿼리 조정
        if image_count <= 1:
            base_query = f"single image layout simple clean {content_description}"
        elif image_count <= 3:
            base_query = f"multiple images grid layout {content_description}"
        else:
            base_query = f"many images gallery layout complex {content_description}"
        
        # AI Search 격리 적용
        if self.isolation_enabled:
            clean_query = self.isolation_manager.clean_query_from_azure_keywords(base_query)
            print(f"🛡️ 레이아웃 추천 쿼리 격리 적용")
        else:
            clean_query = base_query
        
        return self.search_similar_layouts(clean_query, "magazine_layout", top_k=3)

    
    def check_pdf_compatibility(self, template_folder: str):
        """PDF 파일 호환성 체크"""
        
        if not os.path.exists(template_folder):
            return
            
        pdf_files = [f for f in os.listdir(template_folder) if f.endswith('.pdf')]
        
        print("📋 PDF 파일 호환성 체크:")
        large_files = []
        
        for pdf_file in pdf_files:
            pdf_path = os.path.join(template_folder, pdf_file)
            file_size = os.path.getsize(pdf_path) / (1024 * 1024)  # MB
            
            if file_size > 50:
                status = "❌ 매우 큼"
                large_files.append(pdf_file)
            elif file_size > 20:
                status = "⚠️ 큰 파일 (분할 예정)"
                large_files.append(pdf_file)
            else:
                status = "✅ 적합"
            
            print(f"   {pdf_file}: {file_size:.2f}MB - {status}")
        
        if large_files:
            print(f"\n🔪 분할 예정 파일: {len(large_files)}개")
            print(f"   → 자동으로 20MB 이하로 분할됩니다")
            print(f"   → 원본 파일은 backup_large_pdfs/ 폴더로 이동됩니다")
