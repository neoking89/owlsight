"""
Intelligent Document Search Application
------------------------------------
A Streamlit-based application build on Owlsight, that provides two main functionalities:
1. Web Search: Search and analyze online content using DuckDuckGo
2. Document Search: Upload and search through local documents, using Apache Tika.
The power of Apache Tika lies in its ability to extract text from a wide range of file formats, including PDF, DOCX, and more.

Features:
- Semantic search using sentence transformers
- Configurable chunk size for text processing
- GPU/CPU processing support
- Real-time search results with source links
- Export results to CSV
- Enhanced RAG integration for chat

run with:
```bash
streamlit run examples/streamlit_retrieval_app.py
```
"""

import sys
import streamlit as st
import hashlib
import time
from io import StringIO
from datetime import datetime
import re

sys.path.append("src")

from owlsight import OwlDefaultFunctions
# from owlsight.huggingface.leaderboards import get_mteb_leaderboard


def capture_console_output(func, *args, **kwargs):
    """
    Captures console output using contextlib.redirect_stdout/stderr.
    Works reliably across different platforms and with Streamlit.
    """
    from contextlib import redirect_stdout, redirect_stderr
    import logging

    # Set up string buffers
    stdout_buffer = StringIO()
    stderr_buffer = StringIO()

    # Set up logging capture
    log_buffer = StringIO()
    log_handler = logging.StreamHandler(log_buffer)
    log_handler.setLevel(logging.DEBUG)
    root_logger = logging.getLogger()
    old_handlers = root_logger.handlers
    root_logger.handlers = [log_handler]

    try:
        # Capture stdout and stderr
        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            result = func(*args, **kwargs)

        # Get outputs
        stdout_output = stdout_buffer.getvalue()
        stderr_output = stderr_buffer.getvalue()
        log_output = log_buffer.getvalue()

        # Combine all output
        full_output = stdout_output
        if stderr_output:
            full_output += f"\n=== Error Output ===\n{stderr_output}"
        if log_output:
            full_output += f"\n=== Log Output ===\n{log_output}"

    finally:
        # Clean up
        stdout_buffer.close()
        stderr_buffer.close()
        log_buffer.close()
        root_logger.handlers = old_handlers

    return result, full_output


def calculate_files_hash(uploaded_files):
    """
    Calculate a hash of the uploaded files to detect changes.
    """
    hasher = hashlib.sha256()
    for file in uploaded_files:
        content = file.getvalue()
        hasher.update(content)
        # Reset file pointer for subsequent reads
        file.seek(0)
    return hasher.hexdigest()


def run_web_search(
    query, max_results, transformer_model, device, chunk_length, top_k, query_prefix=None, document_prefix=None
):
    """
    Runs the document search via web scraping and captures console output.
    """
    owl_funcs = OwlDefaultFunctions({})

    try:
        # Capture console output while fetching documents
        documents, console_output_1 = capture_console_output(
            owl_funcs.owl_search_and_scrape, query, max_results=max_results
        )

        # Add prefix to documents if specified
        if document_prefix:
            documents = {k: f"{document_prefix} {v}" for k, v in documents.items()}

        # Capture console output while creating document searcher
        searcher, console_output_2 = capture_console_output(
            owl_funcs.owl_create_document_searcher,
            documents,
            sentence_transformer_model_name=transformer_model,
            device=device,
            target_chunk_length=chunk_length,
        )

        # Add prefix to query if specified
        search_query = f"{query_prefix} {query}" if query_prefix else query

        # Capture console output while performing search
        df, console_output_3 = capture_console_output(searcher.search, search_query, top_k=top_k)

        # Add source column and reorder columns
        df["source"] = df["document_name"].apply(lambda x: x.split("__split")[0])
        df = df[["source"] + [col for col in df.columns if col != "source"]]

        # Combine all console outputs
        full_console_output = console_output_1 + console_output_2 + console_output_3

        return df, full_console_output
    except Exception as e:
        return None, f"Error occurred: {str(e)}"


def process_uploaded_documents(uploaded_files, transformer_model, device, chunk_length, document_prefix=None):
    """
    Process uploaded documents and create a searcher.
    """
    owl_funcs = OwlDefaultFunctions({})
    documents = {}

    try:
        for uploaded_file in uploaded_files:
            document = owl_funcs.owl_read(uploaded_file.getvalue())
            if document_prefix:
                document = f"{document_prefix} {document}"
            documents[uploaded_file.name] = document

        # Create document searcher
        searcher, console_output = capture_console_output(
            owl_funcs.owl_create_document_searcher,
            documents,
            sentence_transformer_model_name=transformer_model,
            device=device,
            target_chunk_length=chunk_length,
        )

        return searcher, console_output
    except Exception as e:
        return None, f"Error occurred: {str(e)}"


def search_documents(searcher, query, top_k, query_prefix=None):
    """
    Search through processed documents with a query.
    """
    try:
        # Add prefix to query if specified
        search_query = f"{query_prefix} {query}" if query_prefix else query

        df, console_output = capture_console_output(searcher.search, search_query, top_k=top_k)

        # Add source column and reorder columns
        df["source"] = df["document_name"].apply(lambda x: x.split("__split")[0])
        df = df[["source"] + [col for col in df.columns if col != "source"]]

        return df, console_output
    except Exception as e:
        return None, f"Error occurred: {str(e)}"


# Enhanced RAG Integration Functions


def search_to_chat_context(df, search_type="Document"):
    """
    Convert search results DataFrame to a structured context string for chat.

    Args:
        df: DataFrame containing search results
        search_type: String indicating the type of search (Document/Web)

    Returns:
        str: Formatted context string
    """
    context_text = f"{search_type} Search Results:\n\n"

    # Add metadata about the search
    context_text += f"Search Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    context_text += f"Results Count: {len(df)}\n\n"

    # Add each result with its source and relevance score
    for i, row in df.iterrows():
        # Format the source information
        source = row["source"]
        source_name = source.split("__split")[0] if "__split" in source else source

        # Add the result with formatting
        context_text += f"[Result {i + 1}] Source: {source_name}\n"

        # Add relevance score if available
        if "similarity" in row:
            context_text += f"Relevance: {row['similarity']:.2f}\n"

        # Add the actual content
        context_text += f"Content: {row['chunk_text']}\n\n"

    return context_text


def add_results_to_chat_context(df, search_type, query):
    """
    Add search results to chat context and switch to chat mode.

    Args:
        df: DataFrame with search results
        search_type: String indicating search type (Document/Web)
        query: The original search query
    """
    # Convert results to context format
    context_text = search_to_chat_context(df, search_type)

    # Store in session state
    st.session_state.chat_context = context_text

    # Create a dictionary of documents for citation
    documents = {}
    for i, row in df.iterrows():
        source = row["source"]
        source_name = source.split("__split")[0] if "__split" in source else source
        if source_name not in documents:
            documents[source_name] = row["chunk_text"]
        else:
            documents[source_name] += " " + row["chunk_text"]

    st.session_state.chat_documents = documents

    # Add system message to chat
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    st.session_state.chat_messages.append(
        {"role": "system", "content": f"📄 {search_type} search results for '{query}' have been added as context."}
    )

    # Success message
    st.success(f"✅ {search_type} results added to chat! Switch to Chat mode to continue.")


def display_search_results_summary(df, query):
    """
    Display a visual summary of search results.

    Args:
        df: DataFrame with search results
        query: Original search query
    """
    num_results = len(df)
    sources = df["source"].nunique()

    # Create a summary card
    st.markdown(
        f"""
    <div style="background-color: #f8f9fa; padding: 15px; border-radius: 10px; margin-bottom: 20px; border-left: 5px solid #4c8bf5;">
        <h4 style="margin-top: 0;">Search Summary: "{query}"</h4>
        <p>Found <strong>{num_results}</strong> relevant passages from <strong>{sources}</strong> sources</p>
        <div style="display: flex; gap: 10px;">
            <div style="background-color: #e3f2fd; padding: 8px; border-radius: 5px; flex: 1;">
                <strong>Top Source:</strong> {df["source"].value_counts().index[0] if not df.empty else "None"}
            </div>
            <div style="background-color: #e3f2fd; padding: 8px; border-radius: 5px; flex: 1;">
                <strong>Avg. Relevance:</strong> {df["similarity"].mean():.2f if 'similarity' in df.columns else 'N/A'}
            </div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )


def chat_response(message, context=None, documents=None):
    """
    Generate a response based on user message and available context.

    Args:
        message (str): The user's message
        context (str, optional): Additional context from documents or web search
        documents (dict, optional): Dictionary of document name to content for citation

    Returns:
        str: The response message with citations if documents are provided
    """
    # In a real implementation, this would call your LLM with the context
    # For demonstration, we'll create a simulated response

    if not context:
        return (
            "I don't have any context to help answer your question. Try uploading documents or running a search first!"
        )

    # Simulate thinking time
    with st.spinner("Thinking..."):
        time.sleep(1.5)  # Simulates LLM processing time

    # Create a response that references the context
    response = f"Based on the provided information, I can answer your question about '{message}'.\n\n"

    # Add references to documents if available
    if documents:
        doc_names = list(documents.keys())[:3]  # Just use up to 3 docs for demonstration
        response += "I found information in the following sources:\n"
        for i, doc in enumerate(doc_names):
            response += f"[{i + 1}] {doc}\n"

        response += "\nThe answer to your query is: This is a simulated response that would normally come from an LLM using the context provided by your documents. "
        response += f"According to [1], the primary information related to your question is about document search functionality. "
        response += f"[2] mentions additional details about semantic search capabilities. "
        if len(doc_names) > 2:
            response += f"Furthermore, [3] provides insights about the user interface design."
    else:
        response += "The answer to your query is: This is a simulated response based on the context you provided."

    return response


# Process and Update Context for Document Management


def process_and_update_context(uploaded_files, existing_context=None):
    """
    Process newly uploaded files and update the existing context.

    Args:
        uploaded_files: List of Streamlit uploaded file objects
        existing_context: Existing context string if any

    Returns:
        tuple: (updated context string, documents dictionary)
    """
    owl_funcs = OwlDefaultFunctions({})
    documents = {}

    try:
        for uploaded_file in uploaded_files:
            document = owl_funcs.owl_read(uploaded_file.getvalue())
            documents[uploaded_file.name] = document

        # Create a formatted context string
        context_text = existing_context or "Document Context:\n\n"

        for doc_name, content in documents.items():
            # Add a snippet from each document (first 300 chars)
            snippet = content[:300] + "..." if len(content) > 300 else content
            context_text += f"Source: {doc_name}\nPreview: {snippet}\n\n"

        return context_text, documents

    except Exception as e:
        st.error(f"Error processing documents: {str(e)}")
        return existing_context, {}


# UI Enhancement Components


def display_document_cards():
    """
    Display uploaded documents as interactive cards with preview functionality.
    """
    if not st.session_state.chat_documents:
        st.info("No documents are currently loaded. Upload documents or add search results as context.")
        return

    st.markdown("### 📚 Active Documents")

    # Create columns for document cards
    cols = st.columns(2)

    for i, (doc_name, content) in enumerate(st.session_state.chat_documents.items()):
        with cols[i % 2]:
            with st.container():
                st.markdown(
                    f"""
                <div class="document-card">
                    <div class="document-title">{doc_name}</div>
                    <div class="document-info">
                        {len(content)} characters | Added: {datetime.now().strftime("%H:%M:%S")}
                    </div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

                with st.expander("Preview Content"):
                    st.text_area(
                        "Document Preview",
                        content[:500] + ("..." if len(content) > 500 else ""),
                        height=100,
                        disabled=True,
                    )

                    if st.button(f"Remove '{doc_name}'", key=f"remove_{i}"):
                        # Remove this document
                        del st.session_state.chat_documents[doc_name]

                        # Update context
                        if st.session_state.chat_documents:
                            context_text = "Document Context:\n\n"
                            for name, text in st.session_state.chat_documents.items():
                                snippet = text[:300] + "..." if len(text) > 300 else text
                                context_text += f"Source: {name}\nPreview: {snippet}\n\n"
                            st.session_state.chat_context = context_text
                        else:
                            st.session_state.chat_context = None

                        # Add system message
                        st.session_state.chat_messages.append(
                            {"role": "system", "content": f"🗑️ Removed document: {doc_name}"}
                        )

                        st.rerun()


def format_message_with_citations(content):
    """
    Format chat message content to highlight citations.

    Args:
        content: Message content string

    Returns:
        str: HTML formatted message with citation styling
    """
    # Find citation patterns like [1], [2], etc.
    citation_pattern = r"\[(\d+)\]"

    # Replace citations with styled spans
    formatted_content = re.sub(citation_pattern, r'<span class="message-citation">[&nbsp;\1&nbsp;]</span>', content)

    return formatted_content


def display_citation_info(citations, documents):
    """
    Display an expandable section showing which sources influenced the answer.

    Args:
        citations: List of citation keys/indices
        documents: Dictionary of document sources
    """
    if not citations or not documents:
        return

    with st.expander("📚 View Sources Used"):
        for i, cite_key in enumerate(citations):
            if cite_key in documents:
                st.markdown(f"**Source {i + 1}: {cite_key}**")

                # Show a preview of the document content
                preview = documents[cite_key][:300] + "..." if len(documents[cite_key]) > 300 else documents[cite_key]
                st.text_area(f"Content Preview", preview, height=100, disabled=True)
                st.markdown("---")


# Replace the display_chat_interface function with this updated version:


def display_chat_interface():
    """
    Display an enhanced chat interface with better context management and document upload.
    Uses a non-nested approach to avoid expander issues.
    """
    st.markdown("### 💬 Chat with Your Documents")

    # Initialize chat-related session state
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "chat_context" not in st.session_state:
        st.session_state.chat_context = None
    if "chat_documents" not in st.session_state:
        st.session_state.chat_documents = {}

    # Document management section (using a collapsible section with custom CSS)
    doc_mgmt_expanded = len(st.session_state.chat_documents) == 0
    doc_mgmt_expander = st.expander("📄 Document Management", expanded=doc_mgmt_expanded)

    with doc_mgmt_expander:
        st.markdown("#### Add Documents for Context")

        col1, col2 = st.columns([3, 1])

        with col1:
            # Direct file upload in chat
            chat_files = st.file_uploader(
                "Upload documents for chat context",
                accept_multiple_files=True,
                help="These documents will be used as context for the chat",
                key="chat_files",
            )

        with col2:
            st.markdown("<br>", unsafe_allow_html=True)  # Spacing
            if st.button("📥 Process Documents", use_container_width=True):
                if chat_files:
                    with st.spinner("Processing documents..."):
                        context, documents = process_and_update_context(chat_files, st.session_state.chat_context)

                        # Update the session state
                        st.session_state.chat_context = context
                        st.session_state.chat_documents.update(documents)

                        # Add a system message
                        st.session_state.chat_messages.append(
                            {"role": "system", "content": f"📚 Added {len(chat_files)} document(s) to chat context."}
                        )

                        st.success(f"✅ Added {len(chat_files)} document(s) to chat context")
                        st.rerun()
                else:
                    st.warning("Please upload documents first")

        # Clear all documents button
        if st.session_state.chat_documents:
            if st.button("🗑️ Clear All Documents", use_container_width=True):
                st.session_state.chat_documents = {}
                st.session_state.chat_context = None
                st.session_state.chat_messages.append(
                    {"role": "system", "content": "🗑️ All documents have been removed from chat context."}
                )
                st.success("All documents removed")
                st.rerun()

    # Display document cards OUTSIDE the expander to avoid nesting
    if st.session_state.chat_documents:
        display_document_cards()

    # Context indicator
    if st.session_state.chat_context:
        st.markdown(
            f"""
        <div style="background-color: #e8f4f8; padding: 10px; border-radius: 5px; margin-bottom: 15px; border-left: 5px solid #4c8bf5;">
            <p style="margin: 0; color: #1e3a8a;">
                <strong>🔍 Context Active:</strong> Your conversation is using 
                information from {len(st.session_state.chat_documents)} document(s). Ask questions about their content!
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Chat message display area with improved styling
    chat_container = st.container()
    with chat_container:
        for i, message in enumerate(st.session_state.chat_messages):
            role = message["role"]
            content = message["content"]

            if role == "user":
                st.markdown(
                    f"""
                <div class="chat-message user">
                    <div class="message-content">
                        <div class="message-avatar user-avatar">👤</div>
                        <div class="message-text">{content}</div>
                    </div>
                </div>
                """,
                    unsafe_allow_html=True,
                )
            elif role == "assistant":
                formatted_content = format_message_with_citations(content)
                st.markdown(
                    f"""
                <div class="chat-message assistant">
                    <div class="message-content">
                        <div class="message-avatar assistant-avatar">🤖</div>
                        <div class="message-text">{formatted_content}</div>
                    </div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

                # If the message contains citations, extract them and show source info
                citations = re.findall(r"\[(\d+)\]", content)
                if citations and st.session_state.chat_documents:
                    # Map citation numbers to document names
                    sources = list(st.session_state.chat_documents.keys())
                    cited_sources = [sources[int(c) - 1] for c in citations if 0 < int(c) <= len(sources)]

                    # Display citation info
                    display_citation_info(cited_sources, st.session_state.chat_documents)
            else:  # System message
                st.markdown(
                    f"""
                <div class="chat-message system">
                    <div class="message-content">
                        <div class="message-text">{content}</div>
                    </div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

    # Add a placeholder message if no context
    if not st.session_state.chat_context and len(st.session_state.chat_messages) == 0:
        st.markdown(
            """
        <div style="background-color: #fef9e7; padding: 15px; border-radius: 10px; margin-bottom: 20px; text-align: center;">
            <p style="margin: 0;">👋 <strong>Welcome to the Chat!</strong></p>
            <p style="margin-top: 10px;">Please upload documents or add search results as context before chatting.</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Chat input area
    user_input = st.text_input(
        "Ask about your documents...",
        key="chat_input",
        placeholder="What information are you looking for in your documents?",
        disabled=not st.session_state.chat_context,
    )

    col1, col2 = st.columns([4, 1])

    with col1:
        send_button = st.button(
            "Send Message", use_container_width=True, disabled=not st.session_state.chat_context, key="send_chat"
        )

    with col2:
        clear_button = st.button("Clear Chat", use_container_width=True)

    if send_button and user_input:
        # Add user message to chat history
        st.session_state.chat_messages.append({"role": "user", "content": user_input})

        # Get response using context and documents
        response = chat_response(user_input, st.session_state.chat_context, st.session_state.chat_documents)

        # Add assistant response to chat history
        st.session_state.chat_messages.append({"role": "assistant", "content": response})

        # Clear the input and refresh
        st.rerun()

    if clear_button:
        st.session_state.chat_messages = []
        st.rerun()


def main():
    # Set page configuration
    st.set_page_config(
        page_title="🦉 Intelligent Document Search",
        layout="wide",
        initial_sidebar_state="expanded",
        page_icon="🦉",
    )

    # Custom CSS for a more professional look
    st.markdown(
        """
        <style>
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .stButton button {
            width: 100%;
            border-radius: 4px;
            padding: 0.5rem;
        }
        .stTextInput div[data-baseweb="input"] {
            border-radius: 4px;
        }
        .stSelectbox div[data-baseweb="select"] {
            border-radius: 4px;
        }
        .stDownloadButton button {
            width: auto;
            margin-top: 1rem;
            margin-bottom: 1rem;
        }
        /* Chat interface styling */
        .chat-message {
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
            display: flex;
            flex-direction: column;
            box-shadow: 0 1px 2px rgba(0,0,0,0.1);
        }
        .chat-message.user {
            background-color: #f0f7ff;
            border-left: 5px solid #4c8bf5;
        }
        .chat-message.assistant {
            background-color: #f8fdf8;
            border-left: 5px solid #34a853;
        }
        .chat-message.system {
            background-color: #fffde7;
            border-left: 5px solid #fbbc05;
            opacity: 0.85;
        }
        .message-content {
            display: flex;
            margin-top: 0.5rem;
        }
        .message-avatar {
            width: 35px;
            height: 35px;
            border-radius: 50%;
            margin-right: 1rem;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.2rem;
        }
        .user-avatar {
            background-color: #e3f2fd;
            color: #1976d2;
        }
        .assistant-avatar {
            background-color: #e8f5e9;
            color: #2e7d32;
        }
        .message-text {
            flex-grow: 1;
        }
        .message-citation {
            background-color: #f5f5f5;
            border-radius: 4px;
            padding: 2px 6px;
            font-size: 0.85em;
            color: #666;
            cursor: pointer;
            display: inline-block;
            margin: 2px;
        }
        .context-badge {
            display: inline-block;
            padding: 3px 8px;
            border-radius: 12px;
            font-size: 0.8em;
            font-weight: bold;
            margin-right: 5px;
            background-color: #e3f2fd;
            color: #1976d2;
        }
        .source-info {
            padding: 10px;
            background-color: #f5f5f5;
            border-radius: 5px;
            margin-top: 10px;
            border-left: 3px solid #4c8bf5;
        }
        .upload-container {
            border: 2px dashed #ccc;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 20px;
            background-color: #fafafa;
        }
        .upload-container:hover {
            border-color: #4c8bf5;
            background-color: #f5f9ff;
        }
        .document-card {
            background-color: white;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 15px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border-left: 4px solid #4c8bf5;
        }
        .document-title {
            font-weight: bold;
            margin-bottom: 5px;
        }
        .document-info {
            color: #666;
            font-size: 0.9em;
        }
        .document-preview {
            margin-top: 10px;
            padding: 10px;
            background-color: #f9f9f9;
            border-radius: 4px;
            font-family: monospace;
            font-size: 0.9em;
            max-height: 100px;
            overflow-y: auto;
        }
        </style>
    """,
        unsafe_allow_html=True,
    )

    # Initialize session state for document processing
    if "processed_files_hash" not in st.session_state:
        st.session_state.processed_files_hash = None
    if "document_searcher" not in st.session_state:
        st.session_state.document_searcher = None
    if "processing_console_output" not in st.session_state:
        st.session_state.processing_console_output = ""
    if "query_prefix" not in st.session_state:
        st.session_state.query_prefix = ""
    if "document_prefix" not in st.session_state:
        st.session_state.document_prefix = ""
    if "top_k" not in st.session_state:
        st.session_state.top_k = 20
    # Chat-related session state
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "chat_context" not in st.session_state:
        st.session_state.chat_context = None
    if "chat_documents" not in st.session_state:
        st.session_state.chat_documents = {}
    if "search_mode" not in st.session_state:
        st.session_state.search_mode = "🌍 Online Search"

    # Dashboard Header with improved styling
    st.title("🦉 Intelligent Document Search")
    st.markdown(
        """
        <div style='background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; margin-bottom: 2rem;'>
        Upload your documents, search the web (using the DuckDuckGo search engine), or chat with an AI assistant for instant, relevant results.
        </div>
    """,
        unsafe_allow_html=True,
    )

    # Configuration Section
    st.sidebar.markdown("### 🛠️ Configuration")

    # Search Mode Selection
    search_mode = st.sidebar.radio(
        "Select Mode",
        ["🌍 Online Search", "📂 Document Search", "💬 Chat"],
        index=0
        if "search_mode" not in st.session_state
        else list(["🌍 Online Search", "📂 Document Search", "💬 Chat"]).index(st.session_state.search_mode),
        help="Choose between searching the web, your uploaded documents, or chatting with AI",
        key="mode_selection",
    )

    # Update session state when mode changes
    st.session_state.search_mode = search_mode

    # Advanced Settings in an expander
    with st.sidebar.expander("⚙️ Advanced Settings"):
        # Model Settings
        st.markdown("#### 🤖 Model Settings")
        transformer_model = st.selectbox(
            "Transformer Model",
            ["sentence-transformers/all-mpnet-base-v2", "sentence-transformers/all-MiniLM-L6-v2"],
            help="Choose the transformer model for semantic search",
        )

        devices = ["cuda", "cpu", "mps"]
        device = st.selectbox(
            "💻 Device",
            devices,
            help="Choose the device for processing",
        )

        st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)

        # Search Settings
        st.markdown("#### 🔍 Search Settings")
        col1, col2 = st.columns(2)

        with col1:
            st.session_state.top_k = st.number_input(
                "Top K Results",
                min_value=1,
                max_value=200,
                value=20,
                step=5,
                help="Number of top retrieval results to show",
            )

            max_results = st.number_input(
                "Web Search Results",
                min_value=1,
                max_value=200,
                value=10,
                step=5,
                help="Amount of results to use from web search",
            )

        with col2:
            chunk_length = st.number_input(
                "Chunk Length",
                min_value=100,
                max_value=2000,
                value=400,
                step=50,
                help="Length of text chunks (in characters) for text splitting",
            )

        st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)

        # Text Prompt Settings
        st.markdown("#### 📝 Text Prompt Settings")
        col3, col4 = st.columns(2)

        with col3:
            st.session_state.query_prefix = st.text_input(
                "Query Prefix",
                value="",
                help="Optional prefix to add before each search query (e.g., 'query:')",
                placeholder="e.g., query:",
            )

        with col4:
            st.session_state.document_prefix = st.text_input(
                "Document Prefix",
                value="",
                help="Optional prefix to add before each document text (e.g., 'passage:')",
                placeholder="e.g., passage:",
            )

    # Main content area
    if search_mode == "🌍 Online Search":
        st.markdown("### 🔎 Web Search")

        query = st.text_input(
            "Enter your search query", placeholder="Insert query here", help="Enter keywords or phrases to search for"
        )

        if st.button("🚀 Search Web", use_container_width=True):
            if query:
                with st.spinner("🔄 Searching and analyzing documents..."):
                    df, console_output = run_web_search(
                        query,
                        max_results,
                        transformer_model,
                        device,
                        chunk_length,
                        st.session_state.top_k,
                        st.session_state.query_prefix,
                        st.session_state.document_prefix,
                    )

                if df is not None:
                    st.success("✅ Search completed!")

                    # Display search summary
                    display_search_results_summary(df, query)

                    # Results in tabs
                    tab1, tab2, tab3 = st.tabs(["📊 Results", "📜 Logs", "🔄 Use in Chat"])
                    with tab1:
                        # Configure the columns for the data editor
                        column_config = {
                            "source": st.column_config.LinkColumn(
                                "Source",
                                help="Click to open source",
                                validate="^https?://",  # Validate URLs
                                max_chars=200,
                            )
                        }

                        # Display the DataFrame with clickable links
                        st.data_editor(
                            df,
                            column_config=column_config,
                            use_container_width=True,
                            disabled=True,  # Make it read-only
                            hide_index=True,
                        )

                        # Add CSV download button
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results as CSV",
                            data=csv,
                            file_name=f"web_search_results_{query[:30]}.csv",
                            mime="text/csv",
                            help="Download the search results as a CSV file",
                        )

                    with tab2:
                        st.text_area("Execution Logs", console_output, height=300)

                    with tab3:
                        st.markdown("#### Use Search Results in Chat")

                        col1, col2 = st.columns([3, 1])

                        with col1:
                            st.markdown(
                                "This will add the search results as context for chat. The assistant will be able "
                                "to answer questions based on these results."
                            )

                        with col2:
                            if st.button("📎 Add to Chat", use_container_width=True):
                                add_results_to_chat_context(df, "Web", query)

                                # Show the "switch to chat" button
                                st.markdown("---")
                                if st.button("🗣️ Switch to Chat Mode", use_container_width=True):
                                    # Update the search mode and rerun
                                    st.session_state.search_mode = "💬 Chat"
                                    st.rerun()
                else:
                    st.error(f"Search failed: {console_output}")
            else:
                st.warning("Please enter a search query")

    elif search_mode == "📂 Document Search":
        st.markdown("### 📂 Document Retrieval")

        # Better file upload area with drag and drop styling
        st.markdown(
            """
        <div class="upload-container">
            <h4>📄 Drag and Drop Documents Here</h4>
            <p>Upload PDF, DOCX, TXT, and other files to search through their content</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        uploaded_files = st.file_uploader(
            "Upload your documents",
            accept_multiple_files=True,
            help="Select one or more documents for retrieval",
            label_visibility="collapsed",
        )

        if uploaded_files:
            current_files_hash = calculate_files_hash(uploaded_files)

            # Process documents if needed
            if (
                st.session_state.processed_files_hash != current_files_hash
                or st.session_state.document_searcher is None
            ):
                with st.spinner("🔄 Processing documents..."):
                    searcher, console_output = process_uploaded_documents(
                        uploaded_files,
                        transformer_model,
                        device,
                        chunk_length,
                        st.session_state.document_prefix,
                    )

                    if searcher is not None:
                        st.session_state.document_searcher = searcher
                        st.session_state.processed_files_hash = current_files_hash
                        st.session_state.processing_console_output = console_output
                        st.success(f"✅ Processed {len(uploaded_files)} document(s)")
                    else:
                        st.error(f"Document processing failed: {console_output}")

            # Search interface
            if st.session_state.document_searcher is not None:
                query = st.text_input(
                    "Search within documents",
                    placeholder="Insert query here",
                    help="Enter keywords to search within your documents",
                )

                if st.button("🔍 Search Documents", use_container_width=True):
                    if query:
                        with st.spinner("🔄 Searching..."):
                            df, search_console_output = search_documents(
                                st.session_state.document_searcher,
                                query,
                                st.session_state.top_k,
                                st.session_state.query_prefix,
                            )

                        if df is not None:
                            st.success("✅ Search complete")

                            # Display search summary
                            display_search_results_summary(df, query)

                            # Results in tabs
                            tab1, tab2, tab3 = st.tabs(["📊 Results", "📜 Logs", "🔄 Use in Chat"])
                            with tab1:
                                # Display the DataFrame
                                st.dataframe(
                                    df,
                                    use_container_width=True,
                                    hide_index=True,
                                )

                                # Download button uses original DataFrame
                                csv = df.to_csv(index=False)
                                st.download_button(
                                    label="📥 Download Results as CSV",
                                    data=csv,
                                    file_name=f"document_search_results_{query[:30]}.csv",
                                    mime="text/csv",
                                    help="Download the search results as a CSV file",
                                )

                            with tab2:
                                full_console_output = (
                                    "Document Processing Output:\n"
                                    + st.session_state.processing_console_output
                                    + "\nSearch Output:\n"
                                    + search_console_output
                                )
                                st.text_area("Execution Logs", full_console_output, height=300)

                            with tab3:
                                st.markdown("#### Use Search Results in Chat")

                                col1, col2 = st.columns([3, 1])

                                with col1:
                                    st.markdown(
                                        "This will add the document search results as context for chat. The assistant "
                                        "will be able to answer questions based on these results."
                                    )

                                with col2:
                                    if st.button("📎 Add to Chat", use_container_width=True):
                                        add_results_to_chat_context(df, "Document", query)

                                        # Show the "switch to chat" button
                                        st.markdown("---")
                                        if st.button("🗣️ Switch to Chat Mode", use_container_width=True):
                                            # Update the search mode and rerun
                                            st.session_state.search_mode = "💬 Chat"
                                            st.rerun()
                        else:
                            st.error(f"Search failed: {search_console_output}")
                    else:
                        st.warning("Please enter a search query")
        else:
            st.info("👆 Start by uploading your documents")

    else:  # Chat mode
        # Use the enhanced chat interface
        display_chat_interface()


if __name__ == "__main__":
    main()
