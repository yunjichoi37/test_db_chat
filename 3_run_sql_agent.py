# 3_run_sql_agent.py
import os
import warnings
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI


warnings.filterwarnings("ignore")
load_dotenv()

MYSQL_HOST = os.getenv("MYSQL_HOST")
MYSQL_USER = os.getenv("MYSQL_USER")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
MYSQL_DB = os.getenv("MYSQL_DB")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def run_sql_agent():
    print("MySQL 데이터베이스 구조 스캔")
    
    # 1. DB 연결 및 스키마 자동 파악
    # SQLAlchemy의 URI 형식을 사용하여 연결합니다.
    db_uri = f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}/{MYSQL_DB}"
    
    try:
        db = SQLDatabase.from_uri(db_uri)
        # db.get_usable_table_names()를 통해 챗봇이 파악한 테이블 목록을 확인할 수 있습니다.
        print(f"테이블들: {db.get_usable_table_names()}")
    except Exception as e:
        print("DB 연결 실패:", e)
        return

    # 2. LLM 설정 (Llama 3.1 사용)
    # 똑똑한 추론 능력이 필요하므로 온도를 0으로 설정합니다.
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model_name="llama-3.3-70b-versatile",
        temperature=0
    )
    # llm = ChatGoogleGenerativeAI(
    # model="gemini-2.0-flash",
    # google_api_key=os.getenv("GOOGLE_API_KEY"),
    # temperature=0
    # )

    # 3. SQL Agent 생성: 스키마 보고 쿼리 짜는 역할
    agent_executor = create_sql_agent(
        llm=llm,
        db=db,
        agent_type="tool-calling",
        verbose=True, # AI의 생각 과정을 보려면 True, 결과만 보려면 False
        prefix="""You are a SQL expert connected to a real production database.

                Available tables: customer, sales_transaction, product, transaction_product, store, inventory, vendor, customer_email, customer_phone, product_supply.

                Rules:
                1. Always call list_tables first before writing any query.
                2. Only use the tables listed above. Never assume or invent table/column names.
                3. Always verify column names with sql_db_schema before writing a query.
                4. Report query results as facts. Do NOT add disclaimers or caveats like 'this may not reflect actual data'.
                5. The query results ARE the actual data."""
    )

    # 4. 터미널 채팅 루프
    print("\n" + "="*60)
    print("SQL 챗봇 시작 (종료: 'exit')")
    print("="*60 + "\n")

    while True:
        user_input = input("질문: ")
        
        if user_input.lower() in ['exit', 'quit']:
            print("채팅 종료")
            break
            
        if not user_input.strip():
            continue
            
        # Agent 실행
        try:
            response = agent_executor.invoke({"input": user_input})
            print(f"\n답변: {response['output']}\n")
            print("-" * 60)
        except Exception as e:
            print(f"\n에러 발생 (쿼리 작성 실패 등): {e}\n")

if __name__ == "__main__":
    run_sql_agent()