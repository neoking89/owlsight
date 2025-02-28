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

run with:
```bash
streamlit run examples/streamlit_retrieval_app.py
```
"""

import sys
import streamlit as st
import hashlib
from io import StringIO

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

    # Dashboard Header with improved styling
    st.title("🦉 Intelligent Document Search")
    st.markdown(
        """
        <div style='background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; margin-bottom: 2rem;'>
        Upload your documents or search the web (using the DuckDuckGo search engine) for instant, relevant results.
        </div>
    """,
        unsafe_allow_html=True,
    )

    # Configuration Section
    st.sidebar.markdown("### 🛠️ Configuration")

    # Search Mode Selection
    search_mode = st.sidebar.radio(
        "Select Search Mode",
        ["🌍 Online Search", "📂 Document Search"],
        index=0,
        help="Choose between searching the web or your uploaded documents",
    )

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

                    # Results in tabs
                    tab1, tab2 = st.tabs(["📊 Results", "📜 Logs"])
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
                else:
                    st.error(f"Search failed: {console_output}")
            else:
                st.warning("Please enter a search query")

    else:  # Document Search mode
        st.markdown("### 📂 Document Retrieval")

        # File upload area with better styling
        uploaded_files = st.file_uploader(
            "Upload your documents", accept_multiple_files=True, help="Select one or more documents for retrieval"
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

                            # Results in tabs
                            tab1, tab2 = st.tabs(["📊 Results", "📜 Logs"])
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
                        else:
                            st.error(f"Search failed: {search_console_output}")
                    else:
                        st.warning("Please enter a search query")
        else:
            st.info("👆 Start by uploading your documents")


if __name__ == "__main__":
    main()
