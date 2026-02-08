from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import CSVLoader
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
# from prompt_toolkit import prompt

load_dotenv()

llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", temperature=0.7)


instruct_embedding = HuggingFaceInstructEmbeddings()

file_path = "faiss_index"

def create_vectorstore():
    loader = CSVLoader(file_path='D:/Repos/AIAgentProjects/Project2(ChatbotAgent)/FAQs.csv',source_column='prompt')
    data = loader.load()

    vectorstore = FAISS.from_documents(data, instruct_embedding)
    vectorstore.save_local(file_path)

def get_QA_Chain():
    # Load the vectorstore
    vectore_db = FAISS.load_local(file_path, instruct_embedding,allow_dangerous_deserialization=True)
    # Create retriever for querying the vectorstore
    retriever = vectore_db.as_retriever()

    prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the user's question based on the retrieved context:\n\n{context} and if you don't know the answer, just say that you don't know, don't try to make up an answer."),
    ("human", "{input}")
        ])

    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    chain = create_retrieval_chain(retriever, combine_docs_chain)

    return chain

if __name__ == "__main__":
    chain = get_QA_Chain()