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

col1, col2 = st.columns(2)

with col1:
    if st.button('Homepage', key='backend_button', type="primary", use_container_width=True, help="Go to Homepage"):
        st.switch_page("1_Homepage.py")

with col2:
    if st.button('Q&A with Docuemnt', key='qa_btn', type="primary", use_container_width=True, help="Click for Q&A with Docuemnt"):
        st.switch_page("pages/2_Q&A with Document.py")

st.title("Evaluate with Trulens")

st.subheader("Ask the Question",divider=False)
with st.form('qa_form'):
    st.text_input('Enter the Question', placeholder='Please Enter the Question', key = 'question')
    submitted_btn = st.form_submit_button("Evaluate", use_container_width=True, type="secondary")
    
st.write("")
st.write("")
st.write("") 

from trulens_eval.feedback.provider import OpenAI
from trulens_eval import Feedback
import numpy as np
from trulens_eval import TruChain, Tru

tru=Tru()

# Initialize provider class
provider = OpenAI()

# select context to be used in feedback. the location of context is app specific.
from trulens_eval.app import App
context = App.select_context(chain)

from trulens_eval.feedback import Groundedness
grounded = Groundedness(groundedness_provider=OpenAI())
# Define a groundedness feedback function
f_groundedness = (
    Feedback(grounded.groundedness_measure_with_cot_reasons)
    .on(context.collect()) # collect context chunks into a list
    .on_output()
    .aggregate(grounded.grounded_statements_aggregator)
)

# Question/answer relevance between overall question and answer.
f_answer_relevance = (
    Feedback(provider.relevance)
    .on_input_output()
)
# Question/statement relevance between question and each context chunk.
f_context_relevance = (
    Feedback(provider.context_relevance_with_cot_reasons)
    .on_input()
    .on(context)
    .aggregate(np.mean)
)

tru_recorder = TruChain(chain,
    app_id='Chain1_ChatApplication',
    feedbacks=[f_answer_relevance, f_context_relevance, f_groundedness])

    
if submitted_btn:
    question = st.session_state.question
    with tru_recorder as recording:
        llm_response = chain.invoke(question)
    records, feedback = tru.get_records_and_feedback(app_ids=[])
    records.head(20)
    rec = recording.get()
    for feedback, feedback_result in rec.wait_for_feedback_results().items():
        st.write(feedback.name, feedback_result.result)