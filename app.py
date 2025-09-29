import asyncio
import os

import streamlit as st
from dotenv import load_dotenv
from langchain.agents import AgentType, initialize_agent
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools import Tool
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# Load environment variables (e.g., API keys)
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")


try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())


# Ensure the API key is set
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY is not set. Check your .env file.")

st.title("CBC Chatbot")

# Load the document
loader = PyPDFLoader("5.-nyaboke-et-al.-kenya-155-169.pdf")
data = loader.load()

# Split the document into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
docs = text_splitter.split_documents(data)

# Create embeddings using GoogleGenerativeAIEmbeddings
# embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Store the document embeddings in FAISS
vectorstore = FAISS.from_documents(docs, embeddings)

# Set up a retriever for similarity search
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

# Initialize the Google Gemini model for the LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro", temperature=0, max_tokens=None, timeout=None
)

# Add Memory to retain conversation history
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Input query from user
query = st.text_input("Ask me Anything: ")
prompt = query

# Define system prompt for the chatbot
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

# Create the chat prompt template
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)


# Define a tool for document retrieval
def retrieve_documents(query):
    docs = retriever.get_relevant_documents(query)
    return docs


retrieval_tool = Tool(
    name="Document Retrieval",
    func=retrieve_documents,
    description="Retrieves relevant documents based on the user's query",
)

# Create an agent that can use retrieval as a tool
agent = initialize_agent(
    tools=[retrieval_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    memory=memory,
    verbose=True,
)

# Run the RAG (Retrieval-Augmented Generation) process if query is provided
if query:
    question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # Invoke the RAG chain and get the response
    response = rag_chain.invoke({"input": query})

    # Store conversation history
    memory.save_context({"input": query}, {"output": response["answer"]})

    # Display the answer in Streamlit
    st.write(response["answer"])
