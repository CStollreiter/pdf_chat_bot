import logging
from datetime import datetime

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


class PDFChatBot:
    def __init__(self, pdf_path, embedding_model, llm, vectorstore_persist_directory='chroma_db', use_logging=False):
        if use_logging: 
            self._logger = logging.getLogger(__name__)

        print('Initializing PDF Chatbot ...')
        if self._logger: self._logger.info('Initializing PDF Chatbot ...')
        
        print('- Loading and vectorizing PDF file')
        if self._logger: self._logger.info("- Loading and vectorizing PDF file")
        pdf_data = self._load_pdf_data(pdf_path)
        self._vectorstore = Chroma.from_documents(pdf_data, embedding=embedding_model, persist_directory=vectorstore_persist_directory)

        print('- Initializing history aware retriever')
        if self._logger: self._logger.info('- Initializing history aware retriever')
        self.chat_history = {}
        retriever_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
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

        print('- Initializing Q & A chain')
        if self._logger: self._logger.info('- Initializing Q & A chain')
        qa_prompt = """You are an assistant for question-answering tasks. 
            Use the chat history and the following pieces of retrieved context to answer the question. 
            If you don't know the answer, just say that you don't know, don't try to make up an answer.
            If you find the answer, write the answer in a concise way. 
            Context: {context}"""
        self._qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", qa_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        qa_chain = create_stuff_documents_chain(llm, self._qa_prompt)

        print('- Initializing RAG chain')
        if self._logger: self._logger.info('- Initializing RAG chain')
        rag_chain = create_retrieval_chain(self._retriever, qa_chain)
        self._chain = RunnableWithMessageHistory(
            rag_chain,
            self._get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

    def _load_pdf_data(self, file_path, use_splitter=True):
        loader = PyPDFLoader(file_path)
        if use_splitter:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            return loader.load_and_split(text_splitter)
        else:
            return loader.load()
    
    def _get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.chat_history:
            print(f'Session ID: {session_id}')
            self.chat_history[session_id] = ChatMessageHistory()
        return self.chat_history[session_id]

    def get_response(self, question, session_id):
        print(f'Generating response for question "{question}"')
        if self._logger: self._logger.info(f'Generating response for question "{question}" and session ID {session_id}')
        return self._chain.invoke(
            {"input": question},
            config={
                "configurable": {"session_id": session_id}
            }
        )

    def stream_response(self, question, session_id):
        print(f'Streaming response for question: {question}')
        if self._logger: self._logger.info(f'Streaming response for question "{question}" and session ID {session_id}')
        return self._chain.stream(
            {"input": question},
            config={
                "configurable": {"session_id": session_id}
            }
        )
            
