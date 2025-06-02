import os
from typing import List
from PyPDF2 import PdfReader, PdfWriter

class PDFSplitter:
    """PDF 파일 분할 유틸리티 - 수정된 버전"""
    
    def __init__(self, max_size_mb: float = 20.0):
        self.max_size_mb = max_size_mb
        self.max_size_bytes = max_size_mb * 1024 * 1024
    
    def split_large_pdfs(self, template_folder: str) -> List[str]:
        """큰 PDF 파일들을 분할하고 분할된 파일 목록 반환"""
        
        if not os.path.exists(template_folder):
            print(f"❌ 템플릿 폴더를 찾을 수 없습니다: {template_folder}")
            return []
        
        pdf_files = [f for f in os.listdir(template_folder) if f.endswith('.pdf')]
        split_files = []
        
        print(f"📋 PDF 분할 체크: {len(pdf_files)}개 파일")
        
        for pdf_file in pdf_files:
            pdf_path = os.path.join(template_folder, pdf_file)
            
            try:
                file_size = os.path.getsize(pdf_path)
                file_size_mb = file_size / (1024*1024)
                
                print(f"📄 {pdf_file}: {file_size_mb:.2f}MB")
                
                if file_size > self.max_size_bytes:
                    print(f"   🔪 크기 초과 - 분할 시작")
                    split_result = self._split_pdf_by_size(pdf_path, template_folder)
                    
                    if split_result:
                        split_files.extend(split_result)
                        # 원본 파일을 백업 폴더로 이동
                        self._backup_original_file(pdf_path, template_folder)
                        print(f"   ✅ 분할 완료: {len(split_result)}개 파일")
                    else:
                        print(f"   ❌ 분할 실패 - 원본 파일 유지")
                        split_files.append(pdf_file)
                else:
                    print(f"   ✅ 크기 적합 - 분할 불필요")
                    split_files.append(pdf_file)
                    
            except Exception as e:
                print(f"   ❌ {pdf_file} 처리 중 오류: {e}")
                split_files.append(pdf_file)  # 오류 시 원본 파일 유지
        
        print(f"📊 분할 결과: {len(split_files)}개 파일 준비됨")
        return split_files
    
    def _split_pdf_by_size(self, pdf_path: str, output_folder: str) -> List[str]:
        """PDF를 크기 기준으로 분할 - 수정된 버전"""
        
        try:
            print(f"   📖 PDF 읽기 시도: {os.path.basename(pdf_path)}")
            
            # PyPDF2 신버전 API 사용
            with open(pdf_path, 'rb') as file:
                reader = PdfReader(file)
                total_pages = len(reader.pages)
                
                print(f"   📑 총 {total_pages}페이지 감지")
                
                if total_pages == 0:
                    print(f"   ❌ 페이지가 없는 PDF")
                    return []
                
                # 페이지당 예상 크기 계산
                file_size = os.path.getsize(pdf_path)
                avg_page_size = file_size / total_pages
                pages_per_split = max(1, int(self.max_size_bytes / avg_page_size * 0.8))  # 80% 여유
                
                print(f"   📊 분할당 예상 페이지: {pages_per_split}페이지")
                
                split_files = []
                base_name = os.path.splitext(os.path.basename(pdf_path))[0]
                
                for start_page in range(0, total_pages, pages_per_split):
                    end_page = min(start_page + pages_per_split, total_pages)
                    
                    # 분할된 PDF 생성
                    writer = PdfWriter()
                    
                    # 페이지 추가
                    for page_num in range(start_page, end_page):
                        try:
                            writer.add_page(reader.pages[page_num])
                        except Exception as e:
                            print(f"   ⚠️ 페이지 {page_num+1} 추가 실패: {e}")
                            continue
                    
                    # 분할 파일명 생성
                    split_filename = f"{base_name}_part{len(split_files)+1:02d}.pdf"
                    split_path = os.path.join(output_folder, split_filename)
                    
                    # 분할 파일 저장
                    try:
                        with open(split_path, 'wb') as output_file:
                            writer.write(output_file)
                        
                        # 파일 크기 확인
                        if os.path.exists(split_path):
                            split_size = os.path.getsize(split_path) / (1024*1024)
                            print(f"   ✅ {split_filename}: {split_size:.2f}MB ({start_page+1}-{end_page}페이지)")
                            split_files.append(split_filename)
                        else:
                            print(f"   ❌ {split_filename} 생성 실패")
                            
                    except Exception as e:
                        print(f"   ❌ {split_filename} 저장 실패: {e}")
                        continue
                
                print(f"   🎉 분할 완료: {len(split_files)}개 파일 생성")
                return split_files
                
        except Exception as e:
            print(f"   ❌ PDF 분할 실패: {e}")
            return []
    
    def _backup_original_file(self, pdf_path: str, template_folder: str):
        """원본 파일을 백업 폴더로 이동"""
        
        backup_folder = os.path.join(template_folder, "backup_large_pdfs")
        os.makedirs(backup_folder, exist_ok=True)
        
        filename = os.path.basename(pdf_path)
        backup_path = os.path.join(backup_folder, filename)
        
        try:
            # 파일 이동 대신 복사 후 삭제 (더 안전)
            import shutil
            shutil.move(pdf_path, backup_path)
            print(f"   📦 원본 파일 백업: backup_large_pdfs/{filename}")
        except Exception as e:
            print(f"   ⚠️ 백업 실패: {e}")
    
    def test_split_functionality(self, template_folder: str):
        """PDF 분할 기능 테스트"""
        print("🧪 PDF 분할 기능 테스트")
        
        pdf_files = [f for f in os.listdir(template_folder) if f.endswith('.pdf')]
        
        if not pdf_files:
            print("❌ 테스트할 PDF 파일이 없습니다")
            return
        
        test_file = pdf_files[0]
        test_path = os.path.join(template_folder, test_file)
        
        try:
            with open(test_path, 'rb') as file:
                reader = PdfReader(file)
                pages = len(reader.pages)
                print(f"✅ {test_file}: {pages}페이지 읽기 성공")
                
                # 첫 페이지 추출 테스트
                if pages > 0:
                    writer = PdfWriter()
                    writer.add_page(reader.pages[0])
                    
                    test_output = os.path.join(template_folder, "test_split.pdf")
                    with open(test_output, 'wb') as output:
                        writer.write(output)
                    
                    if os.path.exists(test_output):
                        print("✅ PDF 분할 기능 정상 작동")
                        os.remove(test_output)  # 테스트 파일 삭제
                    else:
                        print("❌ PDF 분할 기능 오류")
                        
        except Exception as e:
            print(f"❌ 테스트 실패: {e}")
