import streamlit as st

import logging
from datetime import datetime
from decouple import config
import os
import uuid

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

from PDFChatBot import PDFChatBot


def format_response_stream(response_stream):
    for chunk in response_stream:
        if 'context' in chunk:
            context = chunk['context']
        elif 'answer' in chunk:
            yield chunk['answer']

    yield '\n\n#### Sources'
    # remove duplicates
    unique_documents = []
    for entry in context:
        if entry not in unique_documents:
            unique_documents.append(entry)
    for document in unique_documents:
        yield f'''
**Source:** {document.metadata["source"]}\n
**Page:** {document.metadata["page"]}\n
**Content:** "{document.page_content}"
        '''

# initialize logger
timestamp = datetime.now()
logging.basicConfig(
    filename=f'logs/app_run_{timestamp.strftime("%Y-%m-%d_%H-%M-%s")}.log', 
    level=logging.INFO,
    format="%(levelname)s:%(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

st.title('PDF Chat Bot')


with st.spinner('Loading models and PDF file ...'):
# load variables from .env file if they are not already set as environment variables
    if 'OLLAMA_API_BASE_URL' not in os.environ:
        os.environ["OPENAI_API_KEY"] = config('OPENAI_API_KEY')
    OLLAMA_API_BASE_URL = os.environ['OLLAMA_API_BASE_URL'] if 'OLLAMA_API_BASE_URL' in os.environ else config('OLLAMA_API_BASE_URL')   
    LLM = os.environ['LLM'] if 'LLM' in os.environ else config('LLM')   
    EMBEDDING_MODEL = os.environ['EMBEDDING_MODEL'] if 'EMBEDDING_MODEL' in os.environ else config('EMBEDDING_MODEL')  

    if 'chat_bot' not in st.session_state:

        print(f'Loading embedding model {EMBEDDING_MODEL}')
        logger.info(f'Loading embedding model {EMBEDDING_MODEL}')
        embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

        print(f'Loading LLM: {LLM}')
        logger.info(f'Loading LLM: {LLM}')
        llm = ChatOllama(
            base_url=OLLAMA_API_BASE_URL, 
            model=LLM
        )

        st.session_state.chat_bot = PDFChatBot('/Users/stolli/IT/Designing Data-Intensive Applications.pdf', embedding_model, llm, use_logging=True)
        st.session_state.session_id = str(uuid.uuid4()).replace('-', '_')


with st.form('my_form'):
    question = st.text_area(
        'Enter text:',
        '',
    )
    submitted = st.form_submit_button('Submit')
    if submitted:
        # response = st.session_state.chat_bot.get_response(question, session_id=st.session_state.session_id)
        # st.info(response)
        st.write_stream(format_response_stream(st.session_state.chat_bot.stream_response(question, session_id=st.session_state.session_id)))

