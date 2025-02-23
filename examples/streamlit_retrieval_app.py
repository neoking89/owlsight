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
"""


import sys
import streamlit as st
import hashlib
from io import StringIO
import torch

sys.path.append("src")

from owlsight import OwlDefaultFunctions


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


def run_search(query, max_results, transformer_model, device, chunk_length):
    """
    Runs the document search via web scraping and captures console output.
    """
    owl_funcs = OwlDefaultFunctions({})

    try:
        # Capture console output while fetching documents
        documents, console_output_1 = capture_console_output(
            owl_funcs.owl_search_and_scrape, query, max_results=max_results
        )

        # Capture console output while creating document searcher
        searcher, console_output_2 = capture_console_output(
            owl_funcs.owl_create_document_searcher,
            documents,
            sentence_transformer_model_name=transformer_model,
            device=device,
            target_chunk_length=chunk_length,
        )

        # Capture console output while performing search
        df, console_output_3 = capture_console_output(searcher.search, query, top_k=50)

        # Add source column and reorder columns
        df["source"] = df["document_name"].apply(lambda x: x.split("__split")[0])
        df = df[["source"] + [col for col in df.columns if col != "source"]]

        # Combine all console outputs
        full_console_output = console_output_1 + console_output_2 + console_output_3

        return df, full_console_output
    except Exception as e:
        return None, f"Error occurred: {str(e)}"


def process_uploaded_documents(uploaded_files, transformer_model, device, chunk_length):
    """
    Process uploaded documents and create a searcher.
    """
    owl_funcs = OwlDefaultFunctions({})
    documents = {}

    try:
        for uploaded_file in uploaded_files:
            document = owl_funcs.owl_read(uploaded_file.getvalue())
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


def search_documents(searcher, query):
    """
    Search through processed documents with a query.
    """
    try:
        df, console_output = capture_console_output(searcher.search, query, top_k=50)

        # Add source column and reorder columns
        df["source"] = df["document_name"].apply(lambda x: x.split("__split")[0])
        df = df[["source"] + [col for col in df.columns if col != "source"]]

        return df, console_output
    except Exception as e:
        return None, f"Error occurred: {str(e)}"


def main():
    # Set page configuration
    st.set_page_config(page_title="🦉 Intelligent Document Search", layout="wide", initial_sidebar_state="expanded", page_icon="🦉",)

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
        transformer_model = st.text_input(
            "Sentence Transformer Model",
            value="all-MiniLM-L6-v2",
            help="Specify the embedding model (Sentence Transformer) for document retrieval",
        )

        device = st.selectbox(
            "Processing Unit",
            ["🔷 cuda", "💠 cpu"],
            index=0 if torch.cuda.is_available() else 1,
            format_func=lambda x: x.split()[-1],
            help="Select the processing unit for computations",
        )
        device = device.split()[-1].lower()

        chunk_length = st.slider(
            "Chunk Length",
            min_value=100,
            max_value=1000,
            value=400,
            step=50,
            help="Adjust the character chunk size for retrieval with a semantic-based approach. The size is a target value, but the actual chunk size may vary depending on the content.",
        )

    # Main content area
    if search_mode == "🌍 Online Search":
        st.markdown("### 🔎 Web Search")

        query = st.text_input(
            "Enter your search query", placeholder="Insert query here", help="Enter keywords or phrases to search for"
        )

        max_results = st.slider(
            "Number of results",
            min_value=5,
            max_value=50,
            value=20,
            step=5,
            help="Adjust the maximum number of search results",
        )

        if st.button("🚀 Search Web", use_container_width=True):
            if query:
                with st.spinner("🔄 Searching and analyzing documents..."):
                    df, console_output = run_search(query, max_results, transformer_model, device, chunk_length)

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
                        uploaded_files, transformer_model, device, chunk_length
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
                            df, search_console_output = search_documents(st.session_state.document_searcher, query)

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
