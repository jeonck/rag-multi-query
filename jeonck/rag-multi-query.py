from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import dotenv

dotenv.load_dotenv()

# 프롬프트 템플릿을 정의합니다.(5개의 질문을 생성하도록 프롬프트를 작성하였습니다)
prompt = PromptTemplate.from_template(
    """You are an AI language model assistant. 
Your task is to generate 5 different versions of the given user question to retrieve relevant documents from a vector database. 
By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations of the distance-based similarity search. 
Your response should be a list of values separated by new lines, eg: `foo\nbar\nbaz\n`

#ORIGINAL QUESTION: 
{question}
"""
)

# 언어 모델 인스턴스를 생성합니다.
llm = ChatOpenAI(temperature=0)

# LLMChain을 생성합니다.
chain = {"question": RunnablePassthrough()} | prompt | llm | StrOutputParser()

# 질문을 정의합니다.
question = "OpenAI Assistant API의 Functions 사용법에 대해 알려주세요."

# 체인을 실행하여 생성된 다중 쿼리를 확인합니다.
multi_queries = chain.invoke({"question": question})

# 결과를 확인합니다.(5개 질문 생성)
print(multi_queries)

