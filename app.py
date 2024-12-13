import os
from dotenv import load_dotenv
load_dotenv()

from langchain_ollama import OllamaLLM
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


## LangSmith Tracking
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACING_V2'] = os.getenv('LANGCHAIN_TRACING_V2')
os.environ['LANGCHAIN_PROJECT'] = os.getenv('LANGCHAIN_PROJECT')


## Design PromptTemplate
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the question asked"),
        ("user", "Question: {question}")
    ]
)

## Streamlit framework
st.title("LangChain Demo with Gemma 2B")
text_input = st.text_input("Your query: ")


## Ollama Gemma 2B model
llm = OllamaLLM(model='gemma2:2b')
output_parser = StrOutputParser()
chain = prompt|llm|output_parser

if text_input:
    st.write(chain.invoke({'question': text_input}))