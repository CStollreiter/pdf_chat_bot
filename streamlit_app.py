import streamlit as st

from decouple import config
import os
import uuid

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
# from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler

from PDFChatBot import PDFChatBot


st.title('PDF Chat Bot')


# load variables from .env file if they are not already set as environment variables
if 'OLLAMA_API_BASE_URL' not in os.environ:
    os.environ["OPENAI_API_KEY"] = config('OPENAI_API_KEY')
OLLAMA_API_BASE_URL = os.environ['OLLAMA_API_BASE_URL'] if 'OLLAMA_API_BASE_URL' in os.environ else config('OLLAMA_API_BASE_URL')   
LLM = os.environ['LLM'] if 'LLM' in os.environ else config('LLM')   
EMBEDDING_MODEL = os.environ['EMBEDDING_MODEL'] if 'EMBEDDING_MODEL' in os.environ else config('EMBEDDING_MODEL')  

if 'chat_bot' not in st.session_state:

    print(f'Loading embedding model {EMBEDDING_MODEL}')
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print(f'Loading LLM: {LLM}')
    llm = ChatOllama(
        base_url=OLLAMA_API_BASE_URL, 
        model=LLM
    )

    st.session_state.chat_bot = PDFChatBot('/Users/stolli/IT/Designing Data-Intensive Applications.pdf', embedding_model, llm)
    st.session_state.session_id = str(uuid.uuid4()).replace('-', '_')
    print(st.session_state.session_id)

with st.form('my_form'):
    question = st.text_area(
        'Enter text:',
        '',
    )
    submitted = st.form_submit_button('Submit')
    if submitted:
        response = st.session_state.chat_bot.get_response(question, session_id=st.session_state.session_id)
        st.info(response)

