import sys
import os
from datetime import datetime
import logging
import tempfile
from uuid import uuid4

import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


class PDFChatBot:
    def __init__(self, embedding_model, llm, vectorstore_persist_directory='chroma_db', logging_file_handler=None):
        self._logger = logging.getLogger(__name__)
        if logging_file_handler: 
            self._logger.addHandler(logging_file_handler)

        self._logger.info('Initializing PDF Chatbot ...')

        self._logger.info('- Initializing vector database')
        self._chroma_client = chromadb.PersistentClient(path=vectorstore_persist_directory, settings=Settings(allow_reset=True)) 
        self._chroma_client.reset()
        self._chroma_client.create_collection(name="pdf_files")
        self._vectorstore = Chroma(
            client=self._chroma_client,  
            collection_name="pdf_files",
            embedding_function=embedding_model,
        )
        # create dictionnary for processed files: key=file_id, values=vectorstore_ids
        self.processed_files = {}
        
        self._logger.info('- Initializing history aware retriever')
        self.chat_history = {}
        retriever_prompt = """
Given a chat history and the latest user question which might reference context in the chat history, 
formulate a standalone question which can be understood without the chat history. Do NOT answer the question, "
just reformulate it if needed and otherwise return it as is.
        """
        self._retriever_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", retriever_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        self._retriever = create_history_aware_retriever(
            llm, 
            self._vectorstore.as_retriever(), 
            self._retriever_prompt
        )

        self._logger.info('- Initializing Q & A chain')
        qa_prompt = """
You are an assistant for answering questions about PDF files. 
Use the chat history and the PDF files in the context to answer the question. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.
If you find the answer, write the answer in a concise way. 
Context: {context}
        """
        self._qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", qa_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        qa_chain = create_stuff_documents_chain(llm, self._qa_prompt)

        self._logger.info('- Initializing RAG chain')
        rag_chain = create_retrieval_chain(self._retriever, qa_chain)
        self._chain = RunnableWithMessageHistory(
            rag_chain,
            self._get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

    def add_pdf_data(self, *, pdf_file_path=None, pdf_file=None, file_id=str(uuid4())):
        self._logger.info(f'Adding PDF data to vectorstore ({file_id=})')
        if pdf_file_path:
            pdf_data = self._load_pdf_file(pdf_file_path)
        elif pdf_file:
            pdf_data = self._process_pdf_file(pdf_file)
        else:
            raise ValueError("No PDF file provided")
        document_ids = self._vectorstore.add_documents(pdf_data)
        self.processed_files[file_id] = document_ids
    
    def remove_pdf_data(self, file_id):
        self._logger.info(f'Removing PDF data from vectorstore ({file_id=})')
        self._vectorstore.delete(ids=self.processed_files[file_id])

    def _load_pdf_file(self, pdf_file_path, use_splitter=True):
        pdf_data_loader = PyPDFLoader(pdf_file_path)
        if use_splitter:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            return pdf_data_loader.load_and_split(text_splitter)
        else:
            return pdf_data_loader.load()

    def _process_pdf_file(self, pdf_file, use_splitter=True):
        temp_dir = tempfile.TemporaryDirectory()
        temp_filepath = os.path.join(temp_dir.name, pdf_file.name)
        with open(temp_filepath, "wb") as f:
            f.write(pdf_file.getvalue())
        return self._load_pdf_file(temp_filepath, use_splitter)
    
    def _get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.chat_history:
            self._logger.info(f'Session ID {session_id} added to chat history store')
            self.chat_history[session_id] = ChatMessageHistory()
        return self.chat_history[session_id]

    def get_response(self, question, session_id):
        self._logger.info(f'Generating response for question "{question}" and session ID {session_id}')
        return self._chain.invoke(
            {"input": question},
            config={
                "configurable": {"session_id": session_id}
            }
        )

    def stream_response(self, question, session_id):
        self._logger.info(f'Streaming response for question "{question}" and session ID {session_id}')
        return self._chain.stream(
            {"input": question},
            config={
                "configurable": {"session_id": session_id}
            }
        )
            
