# 1_build_vector_db.py
import os
import pymysql
import pandas as pd
import warnings
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

warnings.filterwarnings("ignore")

# 1. 환경 변수 로드
load_dotenv()

MYSQL_HOST = os.getenv("MYSQL_HOST")
MYSQL_USER = os.getenv("MYSQL_USER")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
MYSQL_DB = os.getenv("MYSQL_DB")

def build_vector_db():
    print("MySQL 데이터베이스 연결 중...")
    
    # 2. MySQL 연결 및 데이터 추출
    try:
        # DB 연결
        conn = pymysql.connect(
            host=MYSQL_HOST,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            db=MYSQL_DB,
            charset='utf8mb4' # 한글 깨짐 방지
        )
        
        # 🚨 이 부분의 쿼리를 본인의 실제 테이블명과 컬럼에 맞게 수정해야 합니다!
        # 예시: store DB의 products 테이블에서 아이디, 이름, 상세설명을 가져옴
        query = "SELECT product_UPC, name, brand, category, package_type, size, price FROM product"
        
        df = pd.read_sql(query, conn)
        conn.close()
        print(f"MySQL에서 {len(df)}개 데이터 로드 완료")
        
    except Exception as e:
        print("MySQL 연결 또는 쿼리 실행 실패:", e)
        print("팁: MySQL 서버(서비스)가 켜져 있는지 확인")
        return

    # 3. 데이터프레임을 LangChain Document 객체로 변환
    documents = []
    for index, row in df.iterrows():
        # 데이터가 비어있으면 건너뛰기
        if pd.isna(row['product_UPC']) or str(row['product_UPC']).strip() == "":
            continue
            
        # 메타데이터와 본문 구성 (본인 테이블 컬럼명에 맞춰 수정)
        content = f"[상품명: {row['name']}]\n{row['category']}"
        
        doc = Document(
            page_content=content, 
            metadata={
                "id": str(row['product_UPC']), 
                "title": str(row['name']),
                "brand": str(row['brand']),
                "category": str(row['category']),
                "package_type": str(row['package_type']),
                "size": str(row['size']),
                "price": str(row['price'])
            }
        )
        documents.append(doc)

    if not documents:
        print("텍스트 데이터가 없습니다. 쿼리를 다시 확인해 주세요.")
        return

    # 4. 텍스트 청킹 (Chunking)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    print(f"데이터를 {len(chunks)}개의 조각(Chunk)으로 나누었습니다.")

    # 5. 임베딩 및 로컬 Chroma DB에 저장
    print("HuggingFace 로컬 임베딩을 생성하고 Chroma DB에 저장합니다...")
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={'device': 'cpu'}, 
        encode_kwargs={'normalize_embeddings': True}
    )
    
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    print("MySQL 연동 벡터 데이터베이스 구축이 완료되었습니다!")

if __name__ == "__main__":
    build_vector_db()