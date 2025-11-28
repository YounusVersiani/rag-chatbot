import streamlit as st
import os
from dotenv import load_dotenv
import shutil

# --- ARCHITECTURE STACK ---
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain 
from langchain_core.prompts import ChatPromptTemplate


# Load environment variables
load_dotenv()

# Verify API key is loaded
if not os.getenv("OPENAI_API_KEY"):
    st.error("⚠️ OPENAI_API_KEY not found in environment variables. Please check your .env file.")
    st.stop()

# Set the API key for OpenAI
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# --- CONFIGURATION ---
st.set_page_config(
    page_title="Enterprise AI Engine",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for cleaner UI
st.markdown("""
<style>
    .reportview-container { background: #f0f2f6 }
    header {visibility: hidden;}
    .main {padding-top: 2rem;}
    div.stToast { background-color: #2e7bcf; color: white; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.title("🎛️ System Controls")
    
    # Connectivity check
    if os.getenv("OPENAI_API_KEY"):
        st.success("API Gateway: Online 🟢")
    else:
        st.error("API Gateway: Offline 🔴")
    
    st.divider()

    # Inference Model Selection
    st.subheader("Inference Architecture")
    model_choice = st.selectbox(
        "Select Model:",
        ["gpt-5-mini", "gpt-5-nano", "gpt-5", "gpt-5.1"],
        index=0,
        help="Select 'Nano' for fastest/cheapest, 'Mini' for speed, '5' for balanced, or '5.1' for complex reasoning."
    )
    
    st.info(f"Active Configuration: {model_choice}")
    st.divider()
    
    # Data Ingestion
    st.subheader("Knowledge Base")
    uploaded_file = st.file_uploader("Ingest Technical PDF", type="pdf")
    
    # Clear Context
    if st.button("Clear Session Context"):
        st.session_state.messages = []
        st.session_state.vector_db = None
        # Clean up temp files and chroma directory
        if os.path.exists("./temp_manual.pdf"):
            os.remove("./temp_manual.pdf")
        if os.path.exists("./chroma_db"):
            shutil.rmtree("./chroma_db")
        st.rerun()

# --- MAIN DASHBOARD ---
st.title("⚛️ Enterprise Neural Retrieval Engine")
st.caption(f"🚀 Architecture: RAG-v3 | Model: {model_choice} | Semantic Search Enabled")

# --- STATE MANAGEMENT ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

if "current_file" not in st.session_state:
    st.session_state.current_file = None

# --- ETL PIPELINE ---
if uploaded_file is not None:
    # Check if this is a new file
    if st.session_state.current_file != uploaded_file.name:
        with st.spinner("Processing Document Vectors & Building Index..."):
            try:
                # Save temp file
                temp_file = "./temp_manual.pdf"
                with open(temp_file, "wb") as f:
                    f.write(uploaded_file.getvalue())

                # Load and Chunk
                loader = PyPDFLoader(temp_file)
                docs = loader.load()
                
                # 1000 token chunks with 250 overlap for context continuity
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000, 
                    chunk_overlap=250
                )
                splits = text_splitter.split_documents(docs)
                
                # Generate Embeddings and Index
                embeddings = OpenAIEmbeddings()
                
                # Create persistent directory for vector database
                persist_directory = "./chroma_db"
                if os.path.exists(persist_directory):
                    shutil.rmtree(persist_directory)
                
                st.session_state.vector_db = Chroma.from_documents(
                    documents=splits, 
                    embedding=embeddings,
                    persist_directory=persist_directory
                )
                
                st.session_state.current_file = uploaded_file.name
                st.toast(f"✅ Ingestion Complete: {len(splits)} semantic chunks indexed.")
                
            except Exception as e:
                st.error(f"❌ Error processing document: {str(e)}")
                st.session_state.vector_db = None

# --- RENDER CHAT HISTORY ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- INFERENCE ENGINE ---
if prompt := st.chat_input("Query the knowledge base..."):
    # Append User Query
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Ensure DB is active
    if st.session_state.vector_db is not None:
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("🔄 Analyzing context...")
            
            try:
                # Initialize LLM
                # All GPT-5 models require default temperature=1 (cannot be customized)
                llm = ChatOpenAI(model=model_choice)
                
                retriever = st.session_state.vector_db.as_retriever(
                    search_kwargs={"k": 4}  # Retrieve top 4 most relevant chunks
                )
                
                # System Prompt for Technical Persona
                system_prompt = (
                    "You are an expert Technical Solutions Architect specializing in industrial systems. "
                    "Use the provided context to answer the user's technical query accurately. "
                    "Maintain a professional, concise, and engineering-focused tone. "
                    "If the answer is not in the context, state that information is missing from the provided document. "
                    "Context: {context}"
                )
                
                prompt_template = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    ("human", "{input}"),
                ])

                # Execute RAG Chain
                question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
                rag_chain = create_retrieval_chain(retriever, question_answer_chain)
                
                response = rag_chain.invoke({"input": prompt})
                
                # Render Response
                full_response = response['answer']
                message_placeholder.markdown(full_response)
                
                # Display Evidence (White-box AI)
                with st.expander("🔍 View Retrieval Context (Evidence)"):
                    for i, doc in enumerate(response["context"]):
                        st.caption(f"Source Chunk {i+1}:")
                        st.code(doc.page_content[:300] + "...", language="text")
                
                # Update History
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
            except Exception as e:
                error_message = f"⚠️ Error processing query: {str(e)}"
                message_placeholder.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})
            
    else:
        with st.chat_message("assistant"):
            st.error("⚠️ System Standby: Please ingest a document to begin analysis.")