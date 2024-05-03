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

def get_evaluation_report(user_question):
    with tru_recorder as recording:
        llm_response = chain.invoke(user_question)
    records, feedback = tru.get_records_and_feedback(app_ids=[])
    records.head(20)
    rec = recording.get()
    return rec
    # for feedback, feedback_result in rec.wait_for_feedback_results().items():
    #     st.write(feedback.name, feedback_result.result)

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
    
st.subheader("Ask the Question",divider=False)
with st.form('qa_form'):
    st.text_input('Enter the Question', placeholder='Please Enter the Question', key = 'question')
    submitted_btn = st.form_submit_button("Generate the Answer", use_container_width=True, type="secondary")
    

st.write("")
st.write("")
st.write("") 
    
if submitted_btn:
    question = st.session_state.question
    st.subheader("Answer",divider=False)
    st.markdown(get_response(question))
    results = get_evaluation_report(question)
    
    st.write("")
    st.write("")
    st.write("") 
    
    st.subheader("Evaluation Results",divider=False)
    for feedback, feedback_result in results.wait_for_feedback_results().items():
        if feedback.name == "relevance":
            st.write("Answer Relevance")
            st.text("How relevant is the final generated answer to the question?")
            st.text(f"Answer Relevance: {feedback_result.result}")
            st.divider()
        elif feedback.name == "context_relevance_with_cot_reasons":
            st.write("Context Relevance")
            st.text("How relevant are the retrieved text chucks to the question?")
            st.text(f"Context Relevance: {feedback_result.result}")
            st.divider()
        elif feedback.name == "groundedness_measure_with_cot_reasons":
            st.write("Groundedness")
            st.text("How factually accurate is the final generated answer?")
            st.text(f"Groundedness: {feedback_result.result}")
            st.divider()
            