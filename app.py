import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain import hub
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_core.prompts import SystemMessagePromptTemplate
from langchain_core.documents import Document # Import Document type hint
from typing import List, Tuple, Any # For type hinting intermediate steps
import logging
import shutil
from pathlib import Path

# --- Configuration & Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
)
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("OpenAI API key not found. Please create a .env file with OPENAI_API_KEY='your_key'.")
    st.stop()

DOCS_PATH = "docs"
VECTORSTORE_PATH = "vectorstore_faiss"

# --- Define Helper Class Globally ---
class NoToolsExecutor:
    """A dummy executor to return when no tools/prompt are available for the agent."""
    def invoke(self, *args, **kwargs):
        msg = "I cannot function right now. No tools are available or the agent prompt failed to load. Please check logs, ensure local docs exist, or enable Web Search."
        logging.info(f"NoToolsExecutor invoked. Returning: {msg}")
        return {"output": msg, "intermediate_steps": []}

# --- Helper Functions ---

@st.cache_resource(show_spinner="Loading and processing documents...")
def load_and_process_docs(docs_path):
    """Loads markdown docs individually, logs progress, splits, creates FAISS store."""
    docs_path_obj = Path(docs_path)
    if not docs_path_obj.is_dir():
        logging.warning(f"'{docs_path}' directory not found.")
        return None

    markdown_files = list(docs_path_obj.rglob("*.md"))
    if not markdown_files:
        logging.warning(f"No markdown (.md) files found in '{docs_path}'.")
        return None

    logging.info(f"Found {len(markdown_files)} markdown files to process.")
    all_documents = []
    processed_files_count = 0
    failed_files_count = 0

    st.info(f"Found {len(markdown_files)} files. Starting processing...")
    progress_bar = st.progress(0)

    for i, file_path in enumerate(markdown_files):
        file_path_str = str(file_path)
        logging.info(f"Attempting to load file ({i+1}/{len(markdown_files)}): {file_path_str}")
        try:
            loader = UnstructuredMarkdownLoader(file_path_str)
            docs = loader.load()
            if docs:
                all_documents.extend(docs)
                processed_files_count += 1
                logging.info(f"Successfully loaded and added: {file_path.name}")
            else:
                logging.warning(f"No content loaded from: {file_path.name}")
                failed_files_count += 1
        except Exception as e:
            logging.error(f"Failed to load {file_path.name}: {e}", exc_info=False)
            failed_files_count += 1
        progress_bar.progress( (i + 1) / len(markdown_files) )

    progress_bar.empty()
    logging.info(f"Finished loading files. Success: {processed_files_count}, Failed/No Content: {failed_files_count}")
    st.info(f"File loading complete. Success: {processed_files_count}, Failed/No Content: {failed_files_count}")

    if not all_documents:
        logging.warning("No documents successfully loaded. Index not created.")
        st.warning("No documents successfully loaded. Index not created.")
        return None

    try:
        logging.info(f"Splitting {len(all_documents)} loaded documents...")
        st.info(f"Splitting {len(all_documents)} docs into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(all_documents)
        logging.info(f"Split into {len(texts)} chunks.")
        st.info(f"Split into {len(texts)} chunks. Starting embedding...")

        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

        if os.path.exists(VECTORSTORE_PATH):
            logging.info(f"Removing existing vectorstore at {VECTORSTORE_PATH}")
            shutil.rmtree(VECTORSTORE_PATH)

        logging.info(f"Creating new vectorstore at {VECTORSTORE_PATH}")
        vectorstore = FAISS.from_documents(texts, embeddings)
        vectorstore.save_local(VECTORSTORE_PATH)
        logging.info(f"FAISS vector store created and saved locally at {VECTORSTORE_PATH}.")
        st.success("Indexing complete!")
        return vectorstore
    except Exception as e:
        logging.error(f"Error during chunking/embedding/saving: {e}", exc_info=True)
        st.error(f"Error during indexing process: {e}")
        return None

@st.cache_resource(show_spinner="Loading vector store...")
def load_vector_store(path):
    """Loads FAISS vector store, rebuilds if necessary."""
    if os.path.exists(path):
        try:
            embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
            vectorstore = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
            logging.info("FAISS vector store loaded successfully.")
            return vectorstore
        except Exception as e:
            logging.error(f"Error loading vector store from {path}: {e}", exc_info=True)
            st.error(f"Could not load local index: {e}. Attempting to rebuild...")
            return load_and_process_docs(DOCS_PATH)
    else:
        logging.warning(f"Vector store not found at {path}. Attempting to create.")
        return load_and_process_docs(DOCS_PATH)


@st.cache_resource(show_spinner="Setting up Agent...")
def get_agent_executor(_vector_store, use_web_search: bool):
    """Creates/returns agent executor, conditionally including web search."""
    logging.info(f"--- Executing get_agent_executor --- use_web_search: {use_web_search}")
    tools = []
    tool_names = []

    # 1. Local RAG Tool
    if _vector_store:
        try:
            retriever = _vector_store.as_retriever(search_kwargs={"k": 5}) # Adjust 'k' as needed
            retriever_tool = create_retriever_tool(
                retriever,
                "local_markdown_retriever",
                "Searches local markdown docs. Use for questions about content in 'docs' folder.",
            )
            tools.append(retriever_tool)
            tool_names.append(retriever_tool.name)
            logging.info("Local RAG tool created.")
        except Exception as e:
            logging.error(f"Failed to create retriever tool: {e}", exc_info=True)
            st.error(f"Error creating local search tool: {e}")
    else:
        logging.warning("Local vector store not available. Skipping local RAG tool.")

    # 2. Web Search Tool (Conditional)
    if use_web_search:
        try:
            web_search_tool = DuckDuckGoSearchRun()
            web_search_tool.description = "Web search tool. Use for current events or info NOT in local docs."
            tools.append(web_search_tool)
            tool_names.append(web_search_tool.name)
            logging.info("Web search tool created.")
        except Exception as e:
            logging.error(f"Failed to create web search tool: {e}", exc_info=True)
            st.error(f"Could not initialize web search: {e}")
    else:
        logging.info("Web search is disabled by user.")

    # 3. Agent Setup
    llm = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=OPENAI_API_KEY, streaming=True)
    prompt = None
    try:
        prompt = hub.pull("hwchase17/openai-functions-agent")
    except Exception as e:
        logging.error(f"Failed to pull prompt 'hwchase17/openai-functions-agent': {e}", exc_info=True)
        st.error("Failed to load standard agent prompt.")

    # --- Prompt Modification (with stricter instruction) --- ## MODIFIED ##
    system_message_template = None
    if prompt:
        for msg_template in prompt.messages:
            if isinstance(msg_template, SystemMessagePromptTemplate):
                system_message_template = msg_template
                break
    has_local_retriever = any(t.name == "local_markdown_retriever" for t in tools)
    if system_message_template and not use_web_search and has_local_retriever:
        original_template = system_message_template.prompt.template
        # --- Make Instruction More Explicit --- ## MODIFIED ##
        strict_instruction = (
            "\n\nIMPORTANT: You MUST strictly use the 'local_markdown_retriever' tool to find information "
            "relevant to the user's query before formulating an answer. Do not use your internal knowledge. "
            "If the tool returns no relevant information or documents that do not contain the answer, "
            "you MUST respond ONLY by stating clearly that the answer cannot be found in the provided local documents. "
            "Do NOT provide any other information or definitions in that case."
        )
        # --- End Stricter Instruction ---
        if strict_instruction not in original_template:
             system_message_template.prompt.template = original_template + strict_instruction
             logging.info("Modified system prompt for strict local-only retrieval.")

    # Check if Agent can be created
    if not tools or not prompt:
        if not tools: logging.warning("No tools available for the agent!")
        if not prompt: logging.error("Prompt could not be loaded.")
        logging.info(f"--- Returning NoToolsExecutor instance ---")
        return NoToolsExecutor()

    # Create Agent Executor
    try:
        agent = create_openai_functions_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True,
            return_intermediate_steps=True
        )
        logging.info(f"--- Returning Agent Executor --- Tools included: {tool_names}")
        return agent_executor
    except Exception as e:
       logging.error(f"Failed to create agent executor: {e}", exc_info=True)
       st.error(f"Failed to create agent: {e}")
       logging.info(f"--- Returning None due to Agent/Executor Creation Error ---")
       return None


# --- Streamlit UI ---
st.set_page_config(page_title="Agentic RAG App + Web Search", layout="wide")
st.title("📄 Agentic RAG + Web Search")
st.caption("Ask questions about your local markdown files or the web.")

# --- Sidebar ---
st.sidebar.title("Controls")
web_search_enabled = st.sidebar.toggle("Web Search", value=False, help="Enable web search (Default: Off)")
st.sidebar.markdown("---")

# --- Initialization & Agent Setup ---
vector_store = load_vector_store(VECTORSTORE_PATH)
logging.info(f"--- Calling get_agent_executor --- web_search_enabled value: {web_search_enabled}")
agent_executor = get_agent_executor(vector_store, web_search_enabled)

# --- Sidebar Status Display ---
if vector_store: st.sidebar.success(f"✅ Local document index active.")
else: st.sidebar.warning(f"⚠️ Local document index inactive.")

# --- Display Sources in Sidebar ---
st.sidebar.markdown("---")
st.sidebar.subheader("Sources Used (Last Response)")
# Initialize in session state if not present
if 'last_sources_details' not in st.session_state:
    st.session_state['last_sources_details'] = []
# Check the list in session state
sources_to_display = st.session_state.get('last_sources_details', [])
if sources_to_display:
    with st.sidebar.expander("Show Sources", expanded=False): # Start collapsed
        displayed_combos = set()
        for idx, item in enumerate(sources_to_display):
            source = item.get('source', 'Unknown Source')
            page = item.get('page', None)
            content = item.get('content', '')
            try: filename = Path(source).name
            except Exception: filename = source
            display_key = f"{filename}_{page}_{idx}"
            if display_key not in displayed_combos:
                display_text_header = f"**{filename}**"
                if page is not None: display_text_header += f" (Page {page + 1})"
                st.markdown(display_text_header + ":", unsafe_allow_html=False)
                st.text_area(
                    label=f"content_{display_key}", value=content, height=200,
                    disabled=True, key=f"src_{display_key}"
                )
                st.markdown("---")
                displayed_combos.add(display_key)
else:
    st.sidebar.caption("No local sources identified.")
st.sidebar.markdown("---")


# --- Agent Status Check ---
chat_input_disabled = False
prompt_disabled_message = ""
if not agent_executor or isinstance(agent_executor, NoToolsExecutor):
    if not isinstance(agent_executor, NoToolsExecutor):
        chat_input_disabled = True
        prompt_disabled_message = "Agent initialization failed."
        st.error(prompt_disabled_message)
    else:
        prompt_disabled_message = "Agent non-functional (no tools/prompt)."
        st.warning(prompt_disabled_message)


# --- Sidebar Buttons ---
if st.sidebar.button("Clear Chat History"):
    st.session_state.messages = []
    if 'last_sources_details' in st.session_state:
        st.session_state['last_sources_details'] = []
    logging.info("Chat history and sources cleared.")
    st.rerun()

if st.sidebar.button("Rebuild Local Document Index"):
    with st.spinner("Rebuilding index... Please wait."):
        st.cache_resource.clear()
        logging.info("Cache cleared by rebuild button.")
        new_vector_store = load_and_process_docs(DOCS_PATH)
        if new_vector_store: st.sidebar.success("Index rebuilt successfully!")
        else: st.sidebar.error("Failed to rebuild index. Check logs and 'docs'.")
        if 'last_sources_details' in st.session_state:
            st.session_state['last_sources_details'] = []
        logging.info("Sources cleared after rebuild.")
        st.rerun()

# --- Chat Interface ---
# Initialize session state keys if they don't exist
if "messages" not in st.session_state:
    st.session_state.messages = []
if 'last_sources_details' not in st.session_state:
    st.session_state['last_sources_details'] = [] # Use a list for details

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]): st.markdown(message["content"])

# User input
if prompt := st.chat_input("Ask a question:", disabled=chat_input_disabled):
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Clear previous source details when a new prompt is entered
    st.session_state['last_sources_details'] = []
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        sources_details = []
        retriever_query_used = None

        try:
            if hasattr(agent_executor, 'invoke'):
                with st.spinner("Thinking..."):
                    response = agent_executor.invoke({"input": prompt})

                # --- Initialize variables ---
                full_response = response.get("output", "Sorry, I encountered an issue.")
                intermediate_steps = response.get("intermediate_steps", [])
                sources_details = [] # Re-initialize here too for safety
                retriever_query_used = None

                # --- Log and Find Retriever Query from Intermediate Steps ---
                logging.info(f"Agent Response Dictionary Keys: {response.keys()}")
                if intermediate_steps:
                    logging.info(f"Processing {len(intermediate_steps)} intermediate step(s)...")
                    for i, step in enumerate(intermediate_steps):
                        if isinstance(step, tuple) and len(step) == 2:
                            action, observation = step
                            action_tool = getattr(action, 'tool', None)
                            # logging.info(f"Step {i+1} used tool: {action_tool}") # Optional
                            if action_tool == "local_markdown_retriever":
                                action_input = getattr(action, 'tool_input', None)
                                if isinstance(action_input, dict) and 'query' in action_input:
                                    retriever_query_used = action_input['query']
                                    logging.info(f"  Retriever query identified: '{retriever_query_used}'")
                                    # break # Optional
                        # else: logging.warning(f"  Step {i+1} has unexpected format: {step}")
                else:
                    logging.info("Agent did not use any tools (intermediate_steps is empty or missing).")

                # --- Perform Re-retrieval if Retriever Query Found ---
                if retriever_query_used and vector_store:
                    logging.info(f"Performing re-retrieval using query: '{retriever_query_used}'")
                    try:
                        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
                        retrieved_docs: List[Document] = retriever.invoke(retriever_query_used)
                        logging.info(f"Re-retrieval found {len(retrieved_docs)} documents.")
                        for doc in retrieved_docs:
                            if isinstance(doc, Document):
                                source = doc.metadata.get('source', 'Unknown Source')
                                page_num = doc.metadata.get('page', None)
                                full_content = doc.page_content if doc.page_content else ""
                                sources_details.append({
                                    "source": source, "page": page_num, "content": full_content
                                })
                                logging.info(f"  Extracted details: Source='{source}', Page='{page_num}', Content Length='{len(full_content)}'")
                            # else: logging.warning(f"Item in retrieved_docs is not a Document: {type(doc)}")
                    except Exception as retrieve_err:
                        logging.error(f"Error during re-retrieval: {retrieve_err}", exc_info=True)
                        st.warning("Could not perform source re-retrieval.")
                # elif retriever_query_used: logging.warning("Retriever query found, but vector_store unavailable.")


                # --- !!! UPDATED POST-PROCESSING CHECK !!! --- ## MODIFIED ##
                # Define phrases indicating the AGENT ITSELF found nothing relevant locally
                not_found_phrases = [
                    "did not return relevant information",
                    "couldn't find information about",
                    "documents do not contain",
                    "information is not in the documents",
                    "search did not return relevant",
                    "not found in the provided documents",
                    "unable to find information",
                    "relevant information was not found",
                    # Add more phrases if you observe others
                ]
                # Check if web search is OFF AND the agent's response indicates failure to find in docs
                response_lower = full_response.lower()
                llm_indicated_failure = any(phrase in response_lower for phrase in not_found_phrases)

                if not web_search_enabled and llm_indicated_failure:
                    logging.warning("Agent response indicated retrieval failure/irrelevance while Web Search OFF. Overriding response.")
                    # Override with the standard message, preventing the LLM's extra info
                    full_response = "I couldn't find information about that in the local documents. You can enable 'Web Search' if you'd like me to look online."
                    # Clear sources as the retrieval wasn't useful according to the agent
                    sources_details = [] # Ensure no sources are displayed
                elif not web_search_enabled and not intermediate_steps:
                    # Also catch the original case where agent didn't use tools at all
                    logging.warning("Agent answered without using tools while Web Search was OFF. Overriding response.")
                    full_response = "I couldn't find information about that in the local documents. You can enable 'Web Search' if you'd like me to look online."
                    sources_details = []
                # --- End Updated Post-Processing Check ---


                # Store detailed sources in session state
                logging.info(f"Source details identified for response: {len(sources_details)} items")
                st.session_state['last_sources_details'] = sources_details.copy()

            else:
                full_response = prompt_disabled_message if prompt_disabled_message else "Agent is not available."

        except Exception as e:
            full_response = f"An error occurred: {e}"
            logging.error(f"Error during agent execution: {e}", exc_info=True)
            st.error(f"Agent execution error: {e}")

        message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        st.rerun() # Keep rerun uncommented to update sidebar

elif chat_input_disabled and prompt_disabled_message and not st.session_state.messages:
     st.warning(prompt_disabled_message)