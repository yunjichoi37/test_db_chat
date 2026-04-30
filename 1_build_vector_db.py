# 1_build_vector_db.py (CSV 더미 데이터 버전)
import pandas as pd
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import warnings

warnings.filterwarnings("ignore")

def build_vector_db():
    print("더미 CSV Loading")
    
    # 1. CSV 파일 읽기
    try:
        df = pd.read_csv('dummy_cases.csv')
        print(f"{len(df)}개 데이터 로드 완료")
    except FileNotFoundError:
        print("NotFound CSV")
        return

    # 2. DataFrame을 LangChain Document 객체로 변환    
    documents = []
    for index, row in df.iterrows():
        raw_desc = str(row['description'])
        customer_info = raw_desc.split(']')[0].replace('[고객명: ', '').split(',')[0].strip()
        enhanced_title = f"{customer_info} - {row['title']}"
        
        # 본문 구성
        content = (
            f"고객사: {customer_info}\n"
            f"제목: {enhanced_title}\n"
            f"내용: {raw_desc}"
        )
        
        doc = Document(
            page_content=content, 
            metadata={
                "id": str(row['id']), 
                "title": enhanced_title
            }
        )
        documents.append(doc)

    # 3. 텍스트 청킹 (Chunking)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    print(f"데이터를 {len(chunks)}개로 Chunking...")

    # 4. 임베딩 및 로컬 Chroma DB에 저장
    print("HuggingFace 로컬 임베딩 생성 및 Chroma DB 저장")
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
    print("Finished building vector DB and saved to './chroma_db'")

if __name__ == "__main__":
    build_vector_db()