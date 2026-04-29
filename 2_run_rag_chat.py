# 2_run_rag_chat.py
import os
import warnings
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from operator import itemgetter # 추가된 부분: 딕셔너리에서 특정 키의 값만 뽑아주는 역할

# HuggingFace 관련 경고 메시지 숨기기
warnings.filterwarnings("ignore")

# 1. 환경 변수 로드
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def run_chat():
    print("로컬 벡터 DB를 불러오는 중...")
    
    # 2. 로컬에 저장된 Chroma DB 로드
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    if not os.path.exists("./chroma_db"):
        print("NotFound Chroma DB")
        return

    vectorstore = Chroma(
        persist_directory="./chroma_db", 
        embedding_function=embeddings
    )
    
    # Retriever 설정: 가장 유사한 문맥 3개를 가져오도록 설정
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 3. LLM 설정
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model_name="llama-3.1-8b-instant", # llama 3.1 사용
        temperature=0
    )
    
    # 4. 프롬프트 설정
    prompt = ChatPromptTemplate.from_template("""
    당신은 Dynamics 365 CRM 데이터를 기반으로 답변하는 친절하고 정확한 비즈니스 어시스턴트입니다.
    아래에 제공된 CRM 데이터의 문맥(Context)을 바탕으로 사용자의 질문에 한국어로 답변해 주세요.
    만약 문맥에서 답을 찾을 수 없다면, 지어내지 말고 "해당 정보는 CRM 데이터에서 찾을 수 없습니다"라고 솔직하게 대답해 주세요.

    Context: {context}

    Question: {input}
    
    Answer:
    """)

    # 5. RAG 체인 조립 (최신 LCEL 방식)
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # itemgetter("input")을 사용해 딕셔너리에서 질문 문자열만 뽑아 retriever에 넘김
    retrieval_setup = RunnableParallel(
        {
            "context": itemgetter("input") | retriever, 
            "input": RunnablePassthrough()
        }
    )

    # 2단계: 문서를 문자열로 변환 후 -> 프롬프트 -> LLM -> 문자열로 출력
    answer_chain = (
        {
            "context": lambda x: format_docs(x["context"]),
            "input": lambda x: x["input"]
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    # 3단계: 1단계의 결과물에 'answer'라는 키로 2단계의 최종 답변을 추가하여 반환
    rag_chain = retrieval_setup.assign(answer=answer_chain)

    # 6. 터미널 채팅 루프
    print("\n" + "="*55)
    print("Dynamics 365 CRM 기반 RAG 시스템이 시작되었습니다.")
    print("종료하려면 'exit' 입력)")
    print("="*55 + "\n")

    while True:
        user_input = input("질문: ")
        
        if user_input.lower() in ['exit', 'quit']:
            print("채팅을 종료합니다.")
            break
            
        if not user_input.strip():
            continue
            
        # RAG 체인 실행
        response = rag_chain.invoke({"input": user_input})
        
        print("\n답변:", response["answer"])
        
        # 참고한 데이터 출처(메타데이터) 출력
        print("\n[참고한 CRM 데이터 출처]")
        for doc in response["context"]:
            print(f"- 레코드 제목: {doc.metadata.get('title', 'N/A')} (ID: {doc.metadata.get('id', 'N/A')})")
        print("-" * 55 + "\n")

if __name__ == "__main__":
    run_chat()