from dotenv import load_dotenv
import os
from langchain_community.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain, SimpleSequentialChain
from langchain.memory import ConversationBufferMemory
import streamlit as st

load_dotenv()


# Read the value of OPENAI_API_KEY from the .env file
openai_api_key = os.getenv('OPENAI_API_KEY')

#Set the title for the page
st.title('A simple Langchain Application')

#Set an input box
input_text = st.text_input('Enter a topic...')

#I/P Prompt

topic_prompt = PromptTemplate(
    input_variables=['topic'],
    template='Write an essay on {topic}.'
)

details_prompt = PromptTemplate(input_variables=['context'], template='Identify and enumerate the individuals involved (who), locations mentioned (where) and reasons or motivations outlined (why) within the given essay: {context}')

coordinates_prompt = PromptTemplate(input_variables=['details'], template='Generate the coordinates of the locations in {details}')



#LLM

llm = OpenAI(temperature=0.7, openai_api_key=openai_api_key)

#chain
topic_chain = LLMChain(llm=llm, prompt=topic_prompt, verbose=True, output_key='context')
details_chain = LLMChain(llm=llm, prompt=details_prompt, verbose=True, output_key='details')
coordinates_chain = LLMChain(llm=llm, prompt=coordinates_prompt, verbose=True, output_key='coordinates')

main_chain = SequentialChain(chains=[topic_chain, details_chain, coordinates_chain], verbose=False, input_variables=['topic'], output_variables=['context','details','coordinates'])

if input_text:
    st.write(main_chain({'topic':input_text}))
