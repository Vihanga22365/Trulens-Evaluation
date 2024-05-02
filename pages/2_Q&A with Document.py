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
import os

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

col1, col2 = st.columns(2)

with col1:
    if st.button('Homepage', key='backend_button', type="primary", use_container_width=True, help="Go to Homepage"):
        st.switch_page("1_Homepage.py")

with col2:
    if st.button('Evaluate with Trulens', key='frontend_button', type="primary", use_container_width=True, help="Click for Evaluate with Trulens"):
        st.switch_page("pages/3_Evaluation with Trulens.py")

st.title("Q&A with Docuemnt")
    
st.subheader("Ask the Question",divider=False)
with st.form('qa_form'):
    st.text_input('Enter the Question', placeholder='Please Enter the Question', key = 'question')
    submitted_btn = st.form_submit_button("Generate the Answer", use_container_width=True, type="secondary")
    

st.write("")
st.write("")
st.write("") 
    
if submitted_btn:
    question = st.session_state.question
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k":10})
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
        )
    response = chain.invoke(question)
    st.subheader("Answer",divider=False)
    st.write(response)
