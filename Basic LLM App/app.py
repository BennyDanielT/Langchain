from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from dotenv import load_dotenv
import os
import streamlit as st

load_dotenv(dotenv_path='./.env')
openai_api_key = os.getenv('OPENAI_API_KEY')

def retrieve_model_response(query):
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-3.5-turbo", temperature=0.9)
    human_message = HumanMessage(content=query)
    response = llm([human_message])
    return response.content

##initialize our streamlit app

st.set_page_config(page_title="Q&A Demo")

st.header("Langchain Application")

input=st.text_input("Input: ",key="input")

submit=st.button("Ask the question")

## If ask button is clicked

if submit:
    response=retrieve_model_response(input)

    st.subheader("The Response is")
    st.write(response)