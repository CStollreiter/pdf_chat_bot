import streamlit as st

import sys
import os
from datetime import datetime
import logging
from uuid import uuid4
from decouple import config

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

from PDFChatBot import PDFChatBot


# initialize logger
print_log_file = False
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s: %(message)s"
)
logger = logging.getLogger(__name__)
if print_log_file:
    timestamp = datetime.now()
    file_handler = logging.FileHandler(f'logs/app_run_{timestamp.strftime("%Y-%m-%d_%H-%M-%s")}.log')
    logger.addHandler(file_handler)
else:
    file_handler = None


def format_response_stream(response_stream):
    for chunk in response_stream:
        if 'context' in chunk:
            context = chunk['context']
        elif 'answer' in chunk:
            yield chunk['answer']

    yield '\n\n#### Sources'
    # remove duplicates
    unique_documents = []
    for document in context:
        if document.page_content not in [doc.page_content for doc in unique_documents]:
            unique_documents.append(document)
    unique_documents.sort(key = lambda document: document.metadata["page"])
    for document in unique_documents:
        source = document.metadata["source"]
        source_title = source[(source.rfind('/')+1):]
        yield f'''
**Source:** {source_title}\n
**Page:** {document.metadata["page"]}\n
**Content:** "{document.page_content}"
            '''


st.title('PDF Chat Bot')


if 'chat_bot' not in st.session_state:
    with st.spinner('Initializing PDF bot ...'):
        OLLAMA_API_BASE_URL = os.environ['OLLAMA_API_BASE_URL'] if 'OLLAMA_API_BASE_URL' in os.environ else config('OLLAMA_API_BASE_URL')   
        
        EMBEDDING_MODEL = os.environ['EMBEDDING_MODEL'] if 'EMBEDDING_MODEL' in os.environ else config('EMBEDDING_MODEL')  
        logger.info(f'Loading embedding model: {EMBEDDING_MODEL}')
        embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL
        )

        LLM = os.environ['LLM'] if 'LLM' in os.environ else config('LLM')   
        logger.info(f'Loading LLM: {LLM}')
        llm = ChatOllama(
            base_url=OLLAMA_API_BASE_URL, 
            model=LLM
        )

        st.session_state.chat_bot = PDFChatBot(
            embedding_model, 
            llm, 
            logging_file_handler=file_handler
        )
        st.session_state.session_id = str(uuid4()).replace('-', '_')
        st.session_state.processed_files = []

uploaded_pdf_files = st.file_uploader("Please upload one or multiple PDF files", type="pdf", accept_multiple_files=True)  
for file_id in st.session_state.processed_files:
    if file_id not in [file.file_id for file in uploaded_pdf_files]:
        st.session_state.chat_bot.remove_pdf_data(file_id)
        st.session_state.processed_files.remove(file_id)  

if len(uploaded_pdf_files) > 0:
    for file in uploaded_pdf_files:
        if file.file_id not in st.session_state.processed_files:
            with st.spinner('Processing PDF ...'):
                st.session_state.chat_bot.add_pdf_data(pdf_file=file, file_id=file.file_id)
                st.session_state.processed_files.append(file.file_id)
    with st.form('chat_interface'):
        question = st.text_area(
            'Enter text:',
            '',
        )
        submitted = st.form_submit_button('Submit')
        if submitted:
            with st.spinner('Processing ...'):
                st.write_stream(format_response_stream(st.session_state.chat_bot.stream_response(question, session_id=st.session_state.session_id)))

