import streamlit as st
from langchain_helper import create_vectorstore, get_QA_Chain

st.title("Chatbot Agent with LangChain")

btn = st.button("Create Knowledge Base")

if btn:
    st.success("Creating vectorstore...")
    # create_vectorstore()

question = st.text_input("Enter your question:")

if question:
    chain = get_QA_Chain()
    answer = chain.invoke({"input":question})
    st.header("Answer:")
    st.write(answer["answer"])