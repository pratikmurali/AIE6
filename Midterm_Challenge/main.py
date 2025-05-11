import os
import chainlit as cl
from typing import TypedDict, Annotated, List
from dotenv import load_dotenv
import asyncio

# Import LangChain components
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, BaseMessage
from langchain.schema.output_parser import StrOutputParser
from operator import itemgetter

# Import LangGraph components
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages

# Load environment variables
load_dotenv()

# Tokenizer function for text splitter
import tiktoken
def tiktoken_len(text):
    tokens = tiktoken.encoding_for_model("gpt-4o-mini").encode(text)
    return len(tokens)

# Create a text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=0,
    length_function=tiktoken_len,
)

# Function to read PDF documents
def read_pdf_documents(docs_dir: str = "docs") -> List:
    """
    Read PDF documents from the specified directory and extract their text content.
    
    Args:
        docs_dir (str): Path to directory containing PDF documents
        
    Returns:
        List: List of extracted text content from PDFs
    """
    documents = []
    
    # Check if directory exists
    if not os.path.exists(docs_dir):
        print(f"Directory {docs_dir} does not exist.")
        return documents
    
    # Iterate through files in docs directory
    for filename in os.listdir(docs_dir):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(docs_dir, filename)
            
            try:
                temp_doc = PyMuPDFLoader(pdf_path).load()
                documents.extend(temp_doc)
                print(f"Loaded doc: {filename}")
                
            except Exception as e:
                print(f"Error reading {filename}: {str(e)}")
                
    return documents

# Global variables to store vectorstores, chains, and model
qdrant_vectorstore = None
cyber_security_qdrant_vectorstore = None
rag_chain = None
cyber_security_rag_chain = None
compiled_regulatory_framework_graph = None
model = None

# Create the FDA regulatory information retrieval tool
@tool
def retrieve_fda_information(
    query: Annotated[str, "query to ask the retrieve information tool"]
    ):
    """You are an FDA Regulatory expert who can provide information on the following regulatory topics:
    1. AI and ML topics related to software as a medical device 
    2. Premarket software functions guidance
    3. Quality Management (QMS) systems 
    4. Reporting gaps in FDA-approved AI medical devices
    """
    return rag_chain.invoke({"question": query})

# Create the cybersecurity information retrieval tool
@tool
def retrieve_cybersecurity_information(
    query: Annotated[str, "query to ask the retrieve information tool"]
    ):
    """You are an FDA Cybersecurity expert who can provide information on the following regulatory topics:
    1. Cybersecurity requirements for FDA submissions
    2. Cybersecurity requirements for FDA-approved AI medical devices
    3. Pre-Market Cybersecurity requirements for FDA-approved software as a medical device
    4. Post-Market Cybersecurity requirements for FDA-approved software as a medical device
    5. NIST Cybersecurity framework
    6. Risks, Legislation and Challenges in Cybersecurity for FDA-approved software as a medical device
    """
    return cyber_security_rag_chain.invoke({"question": query})

# Agent state definition
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

# Create the LLM model call function
def call_model(state):
    messages = state["messages"]
    response = model.invoke(messages)
    return {"messages": [response]}

# Initialize resources and create the graph
async def initialize_resources():
    global qdrant_vectorstore, cyber_security_qdrant_vectorstore, rag_chain, cyber_security_rag_chain, model, compiled_regulatory_framework_graph
    
    # Set up the embedding model
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")
    
    # Load FDA documents
    print("Loading FDA documents...")
    pdf_documents = read_pdf_documents()
    if pdf_documents:
        split_chunks = text_splitter.split_documents(pdf_documents)
        
        # Create FDA vectorstore
        qdrant_vectorstore = Qdrant.from_documents(
            split_chunks,
            embedding_model,
            location=":memory:",
            collection_name="fda_guidance_for_samd_and_aiml",
        )
        qdrant_retriever = qdrant_vectorstore.as_retriever()
        
        # Create FDA RAG prompt and chain
        RAG_PROMPT = """
        CONTEXT:
        {context}
    
        QUERY:
        {question}
    
        You are a helpful FDA Auditor. Use the available context to answer the question. If you can't answer the question, say you don't know.
        """
        rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT)
        openai_chat_model = ChatOpenAI(model="gpt-4o-mini")
        rag_chain = (
            {"context": itemgetter("question") | qdrant_retriever, "question": itemgetter("question")}
            | rag_prompt | openai_chat_model | StrOutputParser()
        )
    else:
        # If no documents are found, create a simple response chain
        rag_chain = lambda x: "I don't have information on that topic yet. Please add FDA regulatory documents to the docs folder."
    
    # Load cybersecurity documents
    print("Loading cybersecurity documents...")
    cybersecurity_docs_dir = "docs/cybersecurity"
    if os.path.exists(cybersecurity_docs_dir):
        pdf_documents = read_pdf_documents(docs_dir=cybersecurity_docs_dir)
        if pdf_documents:
            cyber_security_split_chunks = text_splitter.split_documents(pdf_documents)
            
            # Create cybersecurity vectorstore
            cyber_security_qdrant_vectorstore = Qdrant.from_documents(
                cyber_security_split_chunks,
                embedding_model,
                location=":memory:",
                collection_name="cybersecurity_requirements",
            )
            cyber_security_qdrant_retriever = cyber_security_qdrant_vectorstore.as_retriever()
            
            # Create cybersecurity RAG prompt and chain
            CYBER_RAG_PROMPT = """
            CONTEXT:
            {context}
    
            QUERY:
            {question}
    
            You are a helpful Cybersecurity Expert in the field of FDA Software as a Medical Device. Use the available context to answer the question. 
            If you can't answer the question, say you don't know.
            """
            cyber_security_rag_prompt = ChatPromptTemplate.from_template(CYBER_RAG_PROMPT)
            cyber_security_openai_chat_model = ChatOpenAI(model="gpt-4o-mini")
            cyber_security_rag_chain = (
                {"context": itemgetter("question") | cyber_security_qdrant_retriever, "question": itemgetter("question")}
                | cyber_security_rag_prompt | cyber_security_openai_chat_model | StrOutputParser()
            )
        else:
            # If no documents are found, create a simple response chain
            cyber_security_rag_chain = lambda x: "I don't have information on cybersecurity topics yet. Please add cybersecurity documents to the docs/cybersecurity folder."
    else:
        print(f"Directory {cybersecurity_docs_dir} does not exist.")
        # If no directory exists, create a simple response chain
        cyber_security_rag_chain = lambda x: "I don't have information on cybersecurity topics yet. Please create the docs/cybersecurity folder and add relevant documents."
    
    # Create the model and bind tools
    tool_belt = [
        retrieve_fda_information,
        retrieve_cybersecurity_information
    ]
    model = ChatOpenAI(model="gpt-4-turbo", temperature=0)
    model = model.bind_tools(tool_belt)
    
    # Create tool node for the graph
    tool_node = ToolNode(tool_belt)
    
    # Create the graph
    regulatory_framework_graph = StateGraph(AgentState)
    regulatory_framework_graph.add_node("agent", call_model)
    regulatory_framework_graph.add_node("action", tool_node)
    regulatory_framework_graph.set_entry_point("agent")
    
    def should_continue(state):
        last_message = state["messages"][-1]
        if last_message.tool_calls:
            return "action"
        return END
    
    regulatory_framework_graph.add_conditional_edges(
        "agent",
        should_continue
    )
    regulatory_framework_graph.add_edge("action", "agent")
    compiled_regulatory_framework_graph = regulatory_framework_graph.compile()
    print("Resources initialized successfully!")

@cl.on_chat_start
async def on_chat_start():
    # Show loading message
    loading_msg = cl.Message(content="Initializing FDA Regulatory Assistant... Loading documents and setting up knowledge base. This may take a minute.")
    await loading_msg.send()
    
    # Initialize resources
    await initialize_resources()
    
    # Update loading message with completion
    await loading_msg.remove()
    await cl.Message(content="FDA Regulatory Assistant initialized and ready to help!").send()
    
    # Send welcome message
    await cl.Message(content="I'm your FDA Regulatory Assistant. Ask me about Software as a Medical Device (SaMD) regulations, requirements, or cybersecurity considerations.").send()

@cl.on_message
async def on_message(message: cl.Message):
    query = message.content
    human_message = HumanMessage(content=query)
    
    # Initialize response message
    response_message = cl.Message(content="")
    await response_message.send()
    
    # Create inputs for the graph
    inputs = {"messages": [human_message]}
    
    # Stream the responses
    async for chunk in compiled_regulatory_framework_graph.astream(inputs, stream_mode="updates"):
        for node, values in chunk.items():
            if node == "agent":
                last_message = values["messages"][-1]
                # Skip tool calls
                if hasattr(last_message, "content") and last_message.content:
                    await response_message.stream_token(last_message.content)
            elif node == "action":
                # Show which tool is being used
                tool_message = values["messages"][0]
                # We don't display the tool content to avoid verbose UI
                tool_name = getattr(tool_message, "name", "unknown tool")
                await response_message.stream_token(f"\n\nResearching information from: {tool_name}...\n\n")
    
    # Finalize the message
    await response_message.update()
