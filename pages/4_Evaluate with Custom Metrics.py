
import streamlit as st
from dotenv import load_dotenv, find_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
#from langchain.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
# import os
import openai
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langsmith import Client
from trulens_eval.feedback.provider import OpenAI
from trulens_eval.utils.generated import re_0_10_rating
from trulens_eval.feedback import prompts
from trulens_eval import Feedback, Select, Tru, TruChain
from trulens_eval.app import App

import os

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "LLM Eval with Trulens"

template = """Answer the question based only on the following context:
{context}
If you don't know the answer, just say out of scope, don't try to make up an answer.
Question: {question}
"""

persist_directory = "data_docs"
prompt=ChatPromptTemplate.from_template(template)
embeddings = OpenAIEmbeddings()
model=ChatOpenAI(model_name="gpt-4-turbo-preview",temperature=0)
output_parser=StrOutputParser()
def format_docs(docs):
    format_D="\n\n".join([d.page_content for d in docs])
    return format_D

db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k":10})
chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
    )


st.set_page_config(
    page_title="Evaluate with Trulens",
    page_icon="👨‍💻",
    layout="wide",
    initial_sidebar_state="collapsed"
    
)


st.markdown(
    """
<style>
    [data-testid="collapsedControl"] {
        display: none
    }
</style>
""",
    unsafe_allow_html=True,
)

if st.button('Homepage', key='backend_button', type="primary", use_container_width=True, help="Go to Homepage"):
    st.switch_page("1_Homepage.py")

st.title("Q&A with Docuemnt")

ans = None
ques = None
cont = None
    
st.subheader("Check the Groundtruth",divider=False)
answer = st.checkbox("Answer")
question = st.checkbox("Question")
context = st.checkbox("Context")
prompt = st.checkbox("Prompt")
if prompt:
    st.text_input(placeholder='Please Enter the Prompt', key = 'givenPrompt')
submitted_btn = st.button("Evaluate with Custom Metrics", use_container_width=True, type="secondary")


if submitted_btn: 
    if answer:
        ans = 'ok'
    if question:    
        ques = 'ok'
    if context:
        cont = 'ok'
    if prompt:
        prompt = st.session_state.givenPrompt
        
    st.write("Answer: ", ans)
    st.write("Question: ", ques)
    st.write("Context: ", cont)
    st.write("Prompt: ", prompt)
    

st.write("")
st.write("")
st.write("") 
        