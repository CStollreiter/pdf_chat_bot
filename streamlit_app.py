import streamlit as st

# from langchain_openai import OpenAI
from langchain_community.llms import Ollama
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler

from decouple import config
import os

st.title("LLM App")

# load variables from .env file if they are not already set as environment variables
if 'OLLAMA_API_BASE_URL' not in os.environ:
    os.environ["OPENAI_API_KEY"] = config('OPENAI_API_KEY')
OLLAMA_API_BASE_URL = os.environ['OLLAMA_API_BASE_URL'] if 'OLLAMA_API_BASE_URL' in os.environ else config('OLLAMA_API_BASE_URL')   
MODEL = os.environ['MODEL'] if 'MODEL' in os.environ else config('MODEL')   



def generate_response(input_text):
    # model = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0)
    model = Ollama(
        base_url=OLLAMA_API_BASE_URL, 
        model=MODEL, 
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
    )
    st.info(model.invoke(input_text))


with st.form("my_form"):
    text = st.text_area(
        "Enter text:",
        "What are the three key pieces of advice for learning how to code?",
    )
    submitted = st.form_submit_button("Submit")
    if submitted:
        generate_response(text)

