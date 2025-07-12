import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import CohereEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Cohere
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import AIMessage, HumanMessage
import os
from dotenv import load_dotenv


load_dotenv() 
api_key = os.environ["COHERE_API_KEY"]


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_embeddings():
    return CohereEmbeddings(cohere_api_key=api_key,user_agent="langchain")


def get_conversational_chain(vectorstore):
    llm = Cohere()
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True
        )
    conversational_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever = vectorstore.as_retriever(),
        memory=memory
    )
    return conversational_chain


def handle_user_input(user_question):
    response = st.session_state.conversational_chain({"question":user_question})
    st.session_state.chat_history = response['chat_history']
    
    for message in st.session_state.chat_history:
        if isinstance(message,HumanMessage):
            st.write(':main_in_tuxedo',message.content)
        elif isinstance(message,AIMessage):
            st.write(':robot_face',message.content)
            
        
def main():
    st.set_page_config(page_title="Chat with Documents",page_icon=":books:")
    stl = f"""
    <style>
    .stTextInput{{
            position:fixed;
            bottom:3rem;}}
    </style>
    """
    st.markdown(stl,unsafe_allow_html=True)
    st.title("AI Chatbot for Your Documents:books:")
    
    if "conversational_chain" not in st.session_state:
        st.session_state.conversational_chain = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
        
    user_input = st.text_input("Upload files, hit submit button, then ask question. :blue[- Saddar U Din]:sunglasses:")
    if user_input:
        handle_user_input(user_input)
    
    with st.sidebar:
        pdf_docs = st.file_uploader("Upload your documents here", accept_multiple_files=True)
        if st.button("Submit"):
            with st.spinner("Processing..."):
                
                #Get PDF Data
                raw_text = get_pdf_text(pdf_docs)
                
                #Divide the data into chunks
                chunked_array = get_text_chunks(raw_text)
                
                #Create Embeddings
                embeddings = get_embeddings()
                
                #create vector store
                vectorstore = FAISS.from_texts(chunked_array,embeddings)
                
                #create conversation chain
                st.session_state.conversational_chain=get_conversational_chain(vectorstore)
            
            
if __name__== "__main__":
    main()