from dotenv import load_dotenv
import os
from langchain_community.llms import OpenAI
from langchain import PromptTemplate
from langchain.prompts import FewShotPromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain, SimpleSequentialChain
from langchain.memory import ConversationBufferMemory
import streamlit as st

load_dotenv()


# Read the value of OPENAI_API_KEY from the .env file
openai_api_key = os.getenv('OPENAI_API_KEY')

#Set the title for the page
st.title('An example of FewShotPrompt in a  Langchain Application')

#Set an input box
input_text = st.text_input('Ask a question...')

#Example Prompt - 1
demo_template='''I want you to assume the role of an acting financial advisor and accountant for people.
In an easy way, explain the basics of {financial_concept}.'''

prompt=PromptTemplate(
    input_variables=['financial_concept'],
    template=demo_template
    )

prompt.format(financial_concept='income tax') #Assigning a default value here

llm = OpenAI(temperature=0.7, openai_api_key=openai_api_key)
chain1 = LLMChain(llm=llm, prompt=prompt)

# Example 2 - FewShotPromptTemplate
template = "Question: {Question}\n Answer: {Answer}"
example_prompt = PromptTemplate(
    input_variables=["Question", "Answer"],
    template=template,
)


examples = [{"Question":"What is the Capital of Thailand?","Answer":"Bangkok"},{"Question":"Where's the deepest part of the ocean located?", "Answer":"Mariana Trench"}]
few_shot_prompt = FewShotPromptTemplate(examples=examples, suffix="Question:{Question}\n", prefix="Answer any question that's addressed to you.", input_variables=['Question'], example_prompt=example_prompt,example_separator="\n")

few_shot_chain = LLMChain(llm=llm, prompt=few_shot_prompt)

if input_text:
    st.write(few_shot_chain({'Question':input_text}))

#FewShotPromptTemplate
    

