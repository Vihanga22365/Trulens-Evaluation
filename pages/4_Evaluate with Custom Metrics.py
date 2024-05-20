
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

def check_cstom_metric(self, *args, **kwargs) -> float:
    """
    Custom feedback function to evaluate RAG using custom metric.

    Args:
        *args: Any number of positional arguments.
        **kwargs: Any number of keyword arguments.
        

    Returns:
        float: A value between 0 and 1 only. 0 being "not related to the formatted_prompt" and 1 being "related to the formatted_prompt".
    """

    answer = kwargs.get('answer', '')
    question = kwargs.get('question', '')
    context = kwargs.get('context', '')
    global promptSub

    
    formatted_prompt =  f"Professional Prompt: {promptSub}\n"\
                    f"where 0 is not at all related and 10 is extremely related: \n\n" \
                    f"Return only a score between  0 to 1. do not return minus values\n"\
                    f"Answer: {answer}\n" \
                    f"Question: {question}\n" \
                    f"Context: {context}\n" \
                    

    
    #professional_prompt = str.format("Check up to which extent answer data is related to.",{global prompt}," where 0 is not at all related and 10 is extremely related: \n\n Answer: {}\n Question: {}\ncontext:{}\n",answer, question,context)
    return self.generate_score_and_reasons(system_prompt=formatted_prompt)

def assign_variables(ans, ques, cont, prompt, promptSub):
    returned_ans = ans
    returned_ques = ques
    returned_cont = cont
    promptSub = promptSub

    # Check and define f_custom_function based on variable values
    if returned_ans is not None and returned_ques is not None and returned_cont is not None:
        f_custom_function = (
            Feedback(check_cstom_metric)
            .on(answer=Select.RecordOutput)
            .on(question=Select.RecordInput)
            .on(context)
        )
    elif returned_ans is None and returned_ques is None and returned_cont is not None:
        f_custom_function = (
            Feedback(check_cstom_metric)
            .on(context)
        )
    elif returned_ans is None and returned_ques is not None and returned_cont is None:
        f_custom_function = (
            Feedback(check_cstom_metric)
            .on(question=Select.RecordInput)
        )
    elif returned_ans is not None and returned_ques is None and returned_cont is None:
        f_custom_function = (
            Feedback(check_cstom_metric)
            .on(answer=Select.RecordOutput)
        )
    elif returned_ans is None and returned_ques is not None and returned_cont is not None:
        f_custom_function = (
            Feedback(check_cstom_metric)
            .on(question=Select.RecordInput)
            .on(context)
        )
    elif returned_ans is not None and returned_ques is None and returned_cont is not None:
        f_custom_function = (
            Feedback(check_cstom_metric)
            .on(answer=Select.RecordOutput)
            .on(context)
        )
        
    elif returned_ans is not None and returned_ques is not None and returned_cont is None:
        f_custom_function = (
            Feedback(check_cstom_metric)
            .on(answer=Select.RecordOutput)
            .on(question=Select.RecordInput)
        )
        
    tru_recorder = TruChain(chain,
    app_id='C',
    feedbacks=[f_custom_function])

    with tru_recorder as recording:
        llm_response = chain.invoke(prompt)


    tru=Tru()
    records, feedback = tru.get_records_and_feedback(app_ids=[])

    rec = recording.get()
    
    return rec

    # for feedback, feedback_result in rec.wait_for_feedback_results().items():
    #     print(feedback.name, feedback_result.result)






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
prompt = ""
    
st.subheader("Check the Groundtruth",divider=False)
answer = st.checkbox("Answer")
question = st.checkbox("Question")
context = st.checkbox("Context")
promptSub = st.checkbox("Prompt")
if promptSub:
    st.text_input("Prompt",placeholder='Please Enter the Prompt', key = 'givenPrompt')
mainPrompt = st.text_input("Main Prompt",placeholder='Please Enter the Prompt', key = 'mainPrompt')

submitted_btn = st.button("Evaluate with Custom Metrics", use_container_width=True, type="secondary")


if submitted_btn: 
    if answer:
        ans = 'ok'
    if question:    
        ques = 'ok'
    if context:
        cont = 'ok'
    if promptSub:
        promptSub = st.session_state.givenPrompt
        
    prompt = st.session_state.mainPrompt
        
        
    rec = assign_variables(ans, ques, cont, prompt, promptSub)
    
    st.write("aaa", rec)
    
    # for feedback, feedback_result in rec.wait_for_feedback_results().items():
    #     st.write(feedback.name, feedback_result.result)
        
    st.write("Answer: ", ans)
    st.write("Question: ", ques)
    st.write("Context: ", cont)
    st.write("Sub Prompt: ", promptSub)
    st.write("Main Prompt: ", prompt)
    
    

st.write("")
st.write("")
st.write("") 
        