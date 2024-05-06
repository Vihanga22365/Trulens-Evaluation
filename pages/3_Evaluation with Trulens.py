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
from trulens_eval.feedback.provider import OpenAI
from trulens_eval import Feedback
import numpy as np
from trulens_eval import TruChain, Tru
import pandas as pd
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

# Start Trulens
# Start Trulens
# Start Trulens

tru=Tru()

# Initialize provider class
provider = OpenAI()

# select context to be used in feedback. the location of context is app specific.
from trulens_eval.app import App
context = App.select_context(chain)

from trulens_eval.feedback import Groundedness,GroundTruthAgreement
grounded = Groundedness(groundedness_provider=OpenAI())
# Define a groundedness feedback function



def get_evaluation_report(golden_set):
    
    f_groundtruth = Feedback(
        GroundTruthAgreement(golden_set).agreement_measure, name="Answer Correctness"
    ).on_input_output()

    tru_recorder = TruChain(chain,
        app_id='ground_truth',
        feedbacks=[f_groundtruth])
    
    with tru_recorder as recording:
        for q in golden_set:
            res=chain.invoke(q['query'])
        
        
    records, feedback = tru.get_records_and_feedback(app_ids=[])
    
    recs = recording.records
    
    final_result = tru.get_leaderboard(app_ids=[tru_recorder.app_id])
    
    return final_result["Answer Correctness"]["ground_truth"]

# End Trulens
# End Trulens
# End Trulens


def get_response(question):
    response = chain.invoke(question)
    return response
    

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
    
st.subheader("Check the Groundtruth",divider=False)
with st.form('qa_form'):
    # st.text_input('Enter the Question', placeholder='Please Enter the Question', key = 'question')
    uploaded_excel_file = st.file_uploader("Choose a Excel file")
    submitted_btn = st.form_submit_button("Generate the Answer", use_container_width=True, type="secondary")
    

st.write("")
st.write("")
st.write("") 
    
if submitted_btn:
    # question = st.session_state.question
    # st.subheader("Answer",divider=False)
    # st.markdown(get_response(question))
    st.subheader("Evaluation Details",divider=False)
    if uploaded_excel_file is not None:
        qa_df = pd.read_csv(uploaded_excel_file)
        golden_set = [{"query": item["Question"], "response": item["Answer"]} for index, item in qa_df.iterrows()]
        last_answer_g_t = get_evaluation_report(golden_set)
        st.markdown(last_answer_g_t)
        