import gradio as gr
import pandas as pd
from io import StringIO
import os,re
from langchain.llms import OpenAI
import pandas as pd

from langchain.document_loaders import UnstructuredPDFLoader

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.prompts import PromptTemplate

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def predict(file_obj):
    loader = UnstructuredPDFLoader(file_obj.orig_name)
    data = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(data)    
  
    embeddings = OpenAIEmbeddings()
    docsearch = Chroma.from_documents(texts, embeddings)
    qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="map_reduce", retriever=docsearch.as_retriever())
    
    prompt_template = """Ignore all previous instructions. You are the world's best interviewer now. I am going to give you a resume of a candidate. Analyze the resume in 4 categories: Education, Work Experience, Projects and Internships, Others including interests, skills etc. Be simple, direct and commanding. Start with greeting the candidate with a 2 line relatable introduction emphasizing your superiority. Ask the candidate if they have a particular company and a role that they want to apply for. 
If the candidate mentions either the company or the role, then ensure all questions that would be asked will are related to it.
If they don't mention either the company or role clearly, then ignore this and move to the next step. 
Then, give a one line response acknowledging the candidate or if they are not clear about the company or the role then acknowledge positively that you would ask practice interview questions. Then ask the candidate topic would they like to start with. There are 4 categories of questions: educational background related, role related or technical questions, behavioral questions and HR or culture related questions. Here, the candidate will have to put an input. 
Now you will have to ask interview questions. Ensure the questions are good have test the candidate's knowledge. You can choose between longer case based questions, hypothetical questions or academic questions etc. as you deem fit. 
If the candidate types educational background related, ask it 3-4 most relevant questions related to their education based on their resume which are relevant for the role or the company. 
If the candidate types role related or technical related then ask accordingly. Here you can even ask a coding question or test their technical understanding etc. 
Similarly, do it for behavioral questions and HR or culture related questions. You can also be creative, funny, or show emotions at time.
{context}
Question: {question}
Answer in possible questions for interview:"""
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    chain_type_kwargs = {"prompt": PROMPT}
    qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=docsearch.as_retriever(), chain_type_kwargs=chain_type_kwargs)
    response = []
    category = ["Technical", "Education Background", "Behaviour", "Project Specific"]
    for value in category:

        response.append({value:ai(qa, value)})

    html_output = ""
    for obj in response:
    # Loop through the key-value pairs in the object
        for key, value in obj.items():
            value = re.sub(r'[\d\.]+', '', value)
            value_list = value.strip().split('\n')
            value_html = "<ol>"
            for item in value_list:
                value_html += "<li>{}</li>".format(item.strip())
            value_html += "</ol>"
            html_output += "<h2>{}</h2>".format(key)
            html_output += value_html

    
    return html_output
    
def ai(qa,category):
    query = "please suggest "+ category +" interview questions"
    data = list(filter(None, qa.run(query).split('\n')))
    results = list(filter(lambda x: x != ' ', data))
    results = "\n".join(results)
    
    return results
    
title = "Resume-to-Interview (R2I): Personalized Interview Questions Based on Your Resume"

description = "Our app uses the latest technology to scan your resume, and generates personalized interview questions based on your qualifications and experience. With Resume-to-Interview, you can focus on answering questions that are tailored to your specific skills, rather than wasting time memorizing generic interview questions."

article = """As students prepare to enter the job market, it's important to make sure they are well-equipped to handle interviews. However, many students find themselves struggling to remember all the possible questions they might be asked. This is where Resume-to-Interview comes in. Our app makes it easy for students to prepare for interviews by generating personalized interview questions based on their resume.

Using cutting-edge technology, our app analyzes the key details in your resume and generates a list of questions that are tailored to your unique skills and qualifications. This means that you can spend your time preparing for the questions that are most likely to come up during your interview.

The questions are divided into four categories: Technical, Education Background, Behaviour, and Project Specific. This allows you to focus on the areas where you are strongest, and provides you with a clear roadmap for preparing for your interview.

We believe that our app can help students to feel more confident and prepared as they enter the job market. By using Resume-to-Interview, you can be sure that you are fully prepared for your interview, and that you have the knowledge and skills needed to succeed in your chosen career.

Overall, we think that Resume-to-Interview is a valuable tool for anyone who is looking to enter the job market. It's easy to use, highly effective, and provides students with the personalized guidance they need to succeed in their interviews."""


iface = gr.Interface(fn =predict,
  inputs = [gr.File(type="file",label="upload resume pdf file only",file_types=[".pdf"])],
  outputs = [
          gr.HTML()
,
          ],allow_flagging="never", title=title, description=description, article=article)

iface.launch(debug=True)
