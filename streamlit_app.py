import streamlit as st

from langchain_openai import OpenAI
from langchain_community.llms import Ollama
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler

from decouple import config
import os
# from dotenv import load_dotenv

st.title("LLM App")

# add OpenAi API key from .env file to environment variables
os.environ["OPENAI_API_KEY"] = config('OPENAI_API_KEY')


def generate_response(input_text):
    model = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0)
    '''
    model = Ollama(
        base_url="http://localhost:11434", 
        model="deepseek-coder-v2", 
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
    )
    '''
    st.info(model.invoke(input_text))


with st.form("my_form"):
    text = st.text_area(
        "Enter text:",
        "What are the three key pieces of advice for learning how to code?",
    )
    submitted = st.form_submit_button("Submit")
    if submitted:
        generate_response(text)

