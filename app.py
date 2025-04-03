import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader # Keep UnstructuredMarkdownLoader for loader_cls
from langchain_community.document_loaders import UnstructuredMarkdownLoader # Import separately to ensure it's available for loader_cls
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain import hub
from langchain.agents import create_openai_functions_agent, AgentExecutor
import logging
import shutil

# --- Configuration & Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("OpenAI API key not found. Please create a .env file in the app directory with OPENAI_API_KEY='your_key'.")
    st.stop()

DOCS_PATH = "docs"
VECTORSTORE_PATH = "vectorstore_faiss"

# --- Helper Functions ---

@st.cache_resource(show_spinner="Loading and processing documents...")
def load_and_process_docs(docs_path):
    """Loads markdown documents, splits them, and creates a FAISS vector store."""
    if not os.path.exists(docs_path) or not os.listdir(docs_path):
        logging.warning(f"'{docs_path}' directory not found or is empty. Local RAG will be unavailable.")
        return None # Indicate that local docs are unavailable

    try:
        # Use DirectoryLoader with UnstructuredMarkdownLoader for .md files
        # Ensure UnstructuredMarkdownLoader is imported before being used here
        loader = DirectoryLoader(docs_path, glob="**/*.md", loader_cls=UnstructuredMarkdownLoader, show_progress=True, use_multithreading=True)
        documents = loader.load()

        if not documents:
            logging.warning(f"No markdown documents found in '{docs_path}'.")
            return None

        logging.info(f"Loaded {len(documents)} documents.")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)
        logging.info(f"Split documents into {len(texts)} chunks.")

        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

        # Create and save FAISS index
        if os.path.exists(VECTORSTORE_PATH):
             logging.info(f"Removing existing vectorstore at {VECTORSTORE_PATH}")
             shutil.rmtree(VECTORSTORE_PATH) # Remove old index if exists

        logging.info(f"Creating new vectorstore at {VECTORSTORE_PATH}")
        vectorstore = FAISS.from_documents(texts, embeddings)
        vectorstore.save_local(VECTORSTORE_PATH)
        logging.info(f"FAISS vector store created and saved locally at {VECTORSTORE_PATH}.")
        return vectorstore

    except Exception as e:
        logging.error(f"Error loading/processing documents: {e}", exc_info=True)
        st.error(f"Error processing local documents: {e}")
        return None

@st.cache_resource(show_spinner="Loading vector store...")
def load_vector_store(path):
    """Loads the FAISS vector store from local disk."""
    if os.path.exists(path):
        try:
            embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
            vectorstore = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True) # Added allow_dangerous_deserialization
            logging.info("FAISS vector store loaded successfully.")
            return vectorstore
        except Exception as e:
            logging.error(f"Error loading vector store from {path}: {e}", exc_info=True)
            st.error(f"Could not load local document index: {e}")
            return None
    else:
        logging.warning(f"Vector store not found at {path}. Attempting to create.")
        return load_and_process_docs(DOCS_PATH) # Try creating it if not found


@st.cache_resource(show_spinner="Setting up Agent...")
def get_agent_executor(_vector_store): # Pass vector_store as argument
    """Creates and returns the LangChain agent executor."""
    tools = []

    # 1. Local RAG Tool (only if vector_store is available)
    if _vector_store:
        retriever = _vector_store.as_retriever(search_kwargs={"k": 5}) # Retrieve top 5 relevant chunks
        retriever_tool = create_retriever_tool(
            retriever,
            "local_markdown_retriever",
            "Searches and returns information from the local markdown documents. Use this for questions specifically about the content in the local 'docs' folder.",
        )
        tools.append(retriever_tool)
        logging.info("Local RAG tool created.")
    else:
        logging.warning("Local vector store not available. Skipping local RAG tool.")


    # 2. Web Search Tool
    web_search_tool = DuckDuckGoSearchRun()
    tools.append(web_search_tool)
    logging.info("Web search tool created.")

    # 3. Agent Setup
    llm = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=OPENAI_API_KEY, streaming=True)
    # Adjust prompt for function calling agent
    prompt = hub.pull("hwchase17/openai-functions-agent") # Standard prompt for this agent type

    agent = create_openai_functions_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True) # verbose=True for debugging
    logging.info("Agent Executor created.")
    return agent_executor

# --- Streamlit UI ---
st.set_page_config(page_title="Agentic RAG App", layout="wide")
st.title("📄 Agentic RAG with Local Docs & Web Search")
st.caption("Ask questions about your local markdown files or the web.")

# --- Initialization ---
# Attempt to load or create the vector store
vector_store = load_vector_store(VECTORSTORE_PATH)

# Display status of local docs
if vector_store:
    st.sidebar.success(f"✅ Local document index loaded from '{VECTORSTORE_PATH}'.")
    st.sidebar.info(f"Using documents from the '{DOCS_PATH}' folder.")
else:
    st.sidebar.warning(f"⚠️ Local document index ('{VECTORSTORE_PATH}') not found or couldn't be created. Local search unavailable.")
    st.sidebar.info(f"Ensure the '{DOCS_PATH}' folder exists and contains markdown (.md) files, then restart.")

# Create agent executor (pass the potentially None vector_store)
agent_executor = get_agent_executor(vector_store)

# --- Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("Ask a question:"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get agent response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        try:
            # Use invoke for direct response or stream for chunk-by-chunk
            response = agent_executor.invoke({"input": prompt})
            full_response = response.get("output", "Sorry, I encountered an issue.")

        except Exception as e:
            full_response = f"An error occurred: {e}"
            logging.error(f"Error during agent execution: {e}", exc_info=True)

        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})

# Add a button to clear chat history
if st.sidebar.button("Clear Chat History"):
    st.session_state.messages = []
    st.rerun()

# Add a button to rebuild the index
if st.sidebar.button("Rebuild Local Document Index"):
    if os.path.exists(DOCS_PATH):
        with st.spinner("Rebuilding index... Please wait."):
            # Clear cache for loading functions
            st.cache_resource.clear()
            # Attempt to rebuild
            new_vector_store = load_and_process_docs(DOCS_PATH)
            if new_vector_store:
                st.sidebar.success("Index rebuilt successfully!")
                # Need to update the agent with the new vector store
                # Clear agent cache and reload
                st.cache_resource.clear() # Clear all cache for simplicity
                st.rerun() # Rerun the app to reload everything
            else:
                st.sidebar.error("Failed to rebuild index. Check logs and 'docs' folder.")
    else:
        st.sidebar.error(f"'{DOCS_PATH}' folder not found. Cannot rebuild index.")

st.sidebar.markdown("---")
# st.sidebar.markdown("**Instructions:**")
# st.sidebar.markdown(f"1. Create a `.env` file in this directory (`G:\\My Drive\\Main\\Mac\\misc_proj\\agentic_rag_app`) with your `OPENAI_API_KEY`.")
# st.sidebar.markdown(f"2. Create a folder named `{DOCS_PATH}` in this directory.")
# st.sidebar.markdown(f"3. Place your markdown (`.md`) files inside the `{DOCS_PATH}` folder.")
# st.sidebar.markdown(f"4. Run the app using: `streamlit run app.py`")
# st.sidebar.markdown(f"5. Use the 'Rebuild Local Document Index' button if you add/change files in `{DOCS_PATH}`.")
