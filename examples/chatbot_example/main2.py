import sys
import streamlit as st
import hashlib
import time
import re
from io import StringIO
from datetime import datetime

sys.path.append("src")
from owlsight import OwlDefaultFunctions, select_processor_type
from styling import Styling

class DocumentSearchApp: 
    def __init__(self):
        # Alias Streamlit's session state to self.session for state management.
        self.session = st.session_state
        if "processed_files_hash" not in self.session:
            self.session.processed_files_hash = None
        if "document_searcher" not in self.session:
            self.session.document_searcher = None
        if "processing_console_output" not in self.session:
            self.session.processing_console_output = ""
        if "query_prefix" not in self.session:
            self.session.query_prefix = ""
        if "document_prefix" not in self.session:
            self.session.document_prefix = ""
        if "top_k" not in self.session:
            self.session.top_k = 20
        if "chat_messages" not in self.session:
            self.session.chat_messages = []
        if "chat_context" not in self.session:
            self.session.chat_context = None
        if "chat_documents" not in self.session:
            self.session.chat_documents = {}
        if "search_mode" not in self.session:
            self.session.search_mode = "🌍 Online Search"

        # Initialize OwlDefaultFunctions and Styling.
        self.owl_funcs = OwlDefaultFunctions({})
        self.styling = Styling

    def capture_console_output(self, func, *args, **kwargs):
        """
        Captures console output using contextlib.redirect_stdout/stderr.
        Works reliably across different platforms and with Streamlit.
        """
        from contextlib import redirect_stdout, redirect_stderr
        import logging

        stdout_buffer = StringIO()
        stderr_buffer = StringIO()
        log_buffer = StringIO()
        log_handler = logging.StreamHandler(log_buffer)
        log_handler.setLevel(logging.DEBUG)
        root_logger = logging.getLogger()
        old_handlers = root_logger.handlers
        root_logger.handlers = [log_handler]

        try:
            with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                result = func(*args, **kwargs)
            stdout_output = stdout_buffer.getvalue()
            stderr_output = stderr_buffer.getvalue()
            log_output = log_buffer.getvalue()
            full_output = stdout_output
            if stderr_output:
                full_output += f"\n=== Error Output ===\n{stderr_output}"
            if log_output:
                full_output += f"\n=== Log Output ===\n{log_output}"
        finally:
            stdout_buffer.close()
            stderr_buffer.close()
            log_buffer.close()
            root_logger.handlers = old_handlers

        return result, full_output

    def calculate_files_hash(self, uploaded_files):
        """
        Calculate a hash of the uploaded files to detect changes.
        """
        hasher = hashlib.sha256()
        for file in uploaded_files:
            content = file.getvalue()
            hasher.update(content)
            file.seek(0)
        return hasher.hexdigest()

    def run_web_search(self, query, max_results, transformer_model, device, chunk_length, top_k):
        """
        Runs the document search via web scraping and captures console output.
        """
        query_prefix = self.session.query_prefix
        document_prefix = self.session.document_prefix

        try:
            documents, console_output_1 = self.capture_console_output(
                self.owl_funcs.owl_search_and_scrape, query, max_results=max_results
            )

            if document_prefix:
                documents = {k: f"{document_prefix} {v}" for k, v in documents.items()}

            searcher, console_output_2 = self.capture_console_output(
                self.owl_funcs.owl_create_document_searcher,
                documents,
                sentence_transformer_model_name=transformer_model,
                device=device,
                target_chunk_length=chunk_length,
            )

            search_query = f"{query_prefix} {query}" if query_prefix else query
            df, console_output_3 = self.capture_console_output(searcher.search, search_query, top_k=top_k)

            df["source"] = df["document_name"].apply(lambda x: x.split("__split")[0])
            df = df[["source"] + [col for col in df.columns if col != "source"]]

            full_console_output = console_output_1 + console_output_2 + console_output_3

            return df, full_console_output
        except Exception as e:
            return None, f"Error occurred: {str(e)}"

    def process_uploaded_documents(self, uploaded_files, transformer_model, device, chunk_length):
        """
        Process uploaded documents and create a searcher.
        """
        document_prefix = self.session.document_prefix
        documents = {}

        try:
            for uploaded_file in uploaded_files:
                document = self.owl_funcs.owl_read(uploaded_file.getvalue())
                if document_prefix:
                    document = f"{document_prefix} {document}"
                documents[uploaded_file.name] = document

            searcher, console_output = self.capture_console_output(
                self.owl_funcs.owl_create_document_searcher,
                documents,
                sentence_transformer_model_name=transformer_model,
                device=device,
                target_chunk_length=chunk_length,
            )
            return searcher, console_output
        except Exception as e:
            return None, f"Error occurred: {str(e)}"

    def search_documents(self, searcher, query, top_k):
        """
        Search through processed documents with a query.
        """
        query_prefix = self.session.query_prefix

        try:
            search_query = f"{query_prefix} {query}" if query_prefix else query
            df, console_output = self.capture_console_output(searcher.search, search_query, top_k=top_k)

            df["source"] = df["document_name"].apply(lambda x: x.split("__split")[0])
            df = df[["source"] + [col for col in df.columns if col != "source"]]

            return df, console_output
        except Exception as e:
            return None, f"Error occurred: {str(e)}"

    def search_to_chat_context(self, df, search_type="Document"):
        """
        Convert search results DataFrame to a structured context string for chat.
        """
        context_text = f"{search_type} Search Results:\n\n"
        context_text += f"Search Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        context_text += f"Results Count: {len(df)}\n\n"

        for i, row in df.iterrows():
            source = row["source"]
            source_name = source.split("__split")[0] if "__split" in source else source
            context_text += f"[Result {i + 1}] Source: {source_name}\n"
            if "similarity" in row:
                context_text += f"Relevance: {row['similarity']:.2f}\n"
            context_text += f"Content: {row['chunk_text']}\n\n"

        return context_text

    def add_results_to_chat_context(self, df, search_type, query):
        """
        Add search results to chat context.
        """
        context_text = self.search_to_chat_context(df, search_type)
        self.session.chat_context = context_text

        documents = {}
        for i, row in df.iterrows():
            source = row["source"]
            source_name = source.split("__split")[0] if "__split" in source else source
            if source_name not in documents:
                documents[source_name] = row["chunk_text"]
            else:
                documents[source_name] += " " + row["chunk_text"]

        self.session.chat_documents = documents

        if "chat_messages" not in self.session:
            self.session.chat_messages = []
        self.session.chat_messages.append(
            {"role": "system", "content": f"📄 {search_type} search results for '{query}' have been added as context."}
        )

        st.success(f"✅ {search_type} results added to chat! Switch to Chat mode to continue.")

    def display_search_results_summary(self, df, query):
        """
        Display a visual summary of search results.
        """
        num_results = len(df)
        sources = df["source"].nunique()

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

    def chat_response(self, message):
        """
        Generate a response based on the user message and available context.
        """
        if not self.session.chat_context:
            return "I don't have any context to help answer your question. Try uploading documents or running a search first!"

        with st.spinner("Thinking..."):
            time.sleep(0.5)
            return "This is a simulated response."

    def process_and_update_context(self, uploaded_files):
        """
        Process newly uploaded files and update the existing context.
        """
        documents = {}

        try:
            for uploaded_file in uploaded_files:
                document = self.owl_funcs.owl_read(uploaded_file.getvalue())
                documents[uploaded_file.name] = document

            context_text = self.session.chat_context or "Document Context:\n\n"
            for doc_name, content in documents.items():
                snippet = content[:300] + "..." if len(content) > 300 else content
                context_text += f"Source: {doc_name}\nPreview: {snippet}\n\n"

            return context_text, documents

        except Exception as e:
            st.error(f"Error processing documents: {str(e)}")
            return self.session.chat_context, {}

    def display_document_cards(self):
        """
        Display uploaded documents as interactive cards with preview functionality.
        """
        if not self.session.chat_documents:
            st.info("No documents are currently loaded. Upload documents or add search results as context.")
            return

        st.markdown("### 📚 Active Documents")
        cols = st.columns(2)

        for i, (doc_name, content) in enumerate(self.session.chat_documents.items()):
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
                            del self.session.chat_documents[doc_name]
                            if self.session.chat_documents:
                                context_text = "Document Context:\n\n"
                                for name, text in self.session.chat_documents.items():
                                    snippet = text[:300] + "..." if len(text) > 300 else text
                                    context_text += f"Source: {name}\nPreview: {snippet}\n\n"
                                self.session.chat_context = context_text
                            else:
                                self.session.chat_context = None

                            self.session.chat_messages.append(
                                {"role": "system", "content": f"🗑️ Removed document: {doc_name}"}
                            )
                            st.rerun()

    def format_message_with_citations(self, content):
        """
        Format chat message content to highlight citations.
        """
        citation_pattern = r"\[(\d+)\]"
        formatted_content = re.sub(citation_pattern, r'<span class="message-citation">[&nbsp;\1&nbsp;]</span>', content)
        return formatted_content

    def display_citation_info(self, citations):
        """
        Display an expandable section showing which sources influenced the answer.
        """
        if not citations or not self.session.chat_documents:
            return

        with st.expander("📚 View Sources Used"):
            for i, cite_key in enumerate(citations):
                if cite_key in self.session.chat_documents:
                    st.markdown(f"**Source {i + 1}: {cite_key}**")
                    preview = self.session.chat_documents[cite_key]
                    preview = preview[:300] + "..." if len(preview) > 300 else preview
                    st.text_area("Content Preview", preview, height=100, disabled=True)
                    st.markdown("---")

    def display_chat_interface(self):
        """
        Display an enhanced chat interface with context management and document upload.
        """
        st.markdown("### 💬 Chat with Your Documents")

        if "chat_messages" not in self.session:
            self.session.chat_messages = []
        if "chat_context" not in self.session:
            self.session.chat_context = None
        if "chat_documents" not in self.session:
            self.session.chat_documents = {}

        doc_mgmt_expanded = len(self.session.chat_documents) == 0
        with st.expander("📄 Document Management", expanded=doc_mgmt_expanded):
            st.markdown("#### Add Documents for Context")
            col1, col2 = st.columns([3, 1])
            with col1:
                chat_files = st.file_uploader(
                    "Upload documents for chat context",
                    accept_multiple_files=True,
                    help="These documents will be used as context for the chat",
                    key="chat_files",
                )
            with col2:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("📥 Process Documents", use_container_width=True):
                    if chat_files:
                        with st.spinner("Processing documents..."):
                            context, documents = self.process_and_update_context(chat_files)
                            self.session.chat_context = context
                            self.session.chat_documents.update(documents)
                            self.session.chat_messages.append(
                                {"role": "system", "content": f"📚 Added {len(chat_files)} document(s) to chat context."}
                            )
                            st.success(f"✅ Added {len(chat_files)} document(s) to chat context")
                            st.rerun()
                    else:
                        st.warning("Please upload documents first")

            if self.session.chat_documents:
                if st.button("🗑️ Clear All Documents", use_container_width=True):
                    self.session.chat_documents = {}
                    self.session.chat_context = None
                    self.session.chat_messages.append(
                        {"role": "system", "content": "🗑️ All documents have been removed from chat context."}
                    )
                    st.success("All documents removed")
                    st.rerun()

        if self.session.chat_documents:
            self.display_document_cards()

        if self.session.chat_context:
            st.markdown(
                f"""
                <div style="background-color: #e8f4f8; padding: 10px; border-radius: 5px; margin-bottom: 15px; border-left: 5px solid #4c8bf5;">
                    <p style="margin: 0; color: #1e3a8a;">
                        <strong>🔍 Context Active:</strong> Your conversation is using 
                        information from {len(self.session.chat_documents)} document(s). Ask questions about their content!
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        chat_container = st.container()
        with chat_container:
            for i, message in enumerate(self.session.chat_messages):
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
                    formatted_content = self.format_message_with_citations(content)
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
                    citations = re.findall(r"\[(\d+)\]", content)
                    if citations and self.session.chat_documents:
                        sources = list(self.session.chat_documents.keys())
                        cited_sources = [sources[int(c) - 1] for c in citations if 0 < int(c) <= len(sources)]
                        self.display_citation_info(cited_sources)
                else:
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

        if not self.session.chat_context and len(self.session.chat_messages) == 0:
            st.markdown(
                """
                <div style="background-color: #fef9e7; padding: 15px; border-radius: 10px; margin-bottom: 20px; text-align: center;">
                    <p style="margin: 0;">👋 <strong>Welcome to the Chat!</strong></p>
                    <p style="margin-top: 10px;">Please upload documents or add search results as context before chatting.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        user_input = st.text_input(
            "Ask about your documents...",
            key="chat_input",
            placeholder="What information are you looking for in your documents?",
            disabled=not self.session.chat_context,
        )

        col1, col2 = st.columns([4, 1])
        with col1:
            send_button = st.button("Send Message", use_container_width=True, disabled=not self.session.chat_context, key="send_chat")
        with col2:
            clear_button = st.button("Clear Chat", use_container_width=True)

        if send_button and user_input:
            self.session.chat_messages.append({"role": "user", "content": user_input})
            response = self.chat_response(user_input)
            self.session.chat_messages.append({"role": "assistant", "content": response})
            st.rerun()

        if clear_button:
            self.session.chat_messages = []
            st.rerun()

    def run(self):
        # Set page configuration
        st.set_page_config(
            page_title="🦉 Intelligent Document Search",
            layout="wide",
            initial_sidebar_state="expanded",
            page_icon="🦉",
        )
        st.markdown(self.styling.main_page, unsafe_allow_html=True)

        # Sidebar Configuration
        st.sidebar.markdown("### 🛠️ Configuration")
        search_mode = st.sidebar.radio(
            "Select Mode",
            ["🌍 Online Search", "📂 Document Search", "💬 Chat"],
            index=0 if self.session.search_mode not in ["🌍 Online Search", "📂 Document Search", "💬 Chat"]
            else ["🌍 Online Search", "📂 Document Search", "💬 Chat"].index(self.session.search_mode),
            help="Choose between searching the web, your uploaded documents, or chatting with AI",
            key="mode_selection",
        )
        self.session.search_mode = search_mode

        with st.sidebar.expander("⚙️ Advanced Settings"):
            st.markdown("#### 🤖 Model Settings")
            transformer_model = st.selectbox(
                "Sentence Transformer Model",
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
            st.markdown("#### 🔍 Search Settings")
            col1, col2 = st.columns(2)
            with col1:
                self.session.top_k = st.number_input(
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
            st.markdown("#### 📝 Text Prompt Settings")
            col3, col4 = st.columns(2)
            with col3:
                self.session.query_prefix = st.text_input(
                    "Query Prefix",
                    value="",
                    help="Optional prefix to add before each search query (e.g., 'query:')",
                    placeholder="e.g., query:",
                )
            with col4:
                self.session.document_prefix = st.text_input(
                    "Document Prefix",
                    value="",
                    help="Optional prefix to add before each document text (e.g., 'passage:')",
                    placeholder="e.g., passage:",
                )

        st.title("🦉 Intelligent Document Search")
        st.markdown(
            """
            <div style='background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; margin-bottom: 2rem;'>
            Upload your documents, search the web (using the DuckDuckGo search engine), or chat with an AI assistant for instant, relevant results.
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Main Content Area based on Mode
        if self.session.search_mode == "🌍 Online Search":
            st.markdown("### 🔎 Web Search")
            query = st.text_input(
                "Enter your search query", placeholder="Insert query here", help="Enter keywords or phrases to search for"
            )
            if st.button("🚀 Search Web", use_container_width=True):
                if query:
                    with st.spinner("🔄 Searching and analyzing documents..."):
                        df, console_output = self.run_web_search(
                            query, max_results, transformer_model, device, chunk_length, self.session.top_k
                        )
                    if df is not None:
                        st.success("✅ Search completed!")
                        self.display_search_results_summary(df, query)
                        tab1, tab2, tab3 = st.tabs(["📊 Results", "📜 Logs", "🔄 Use in Chat"])
                        with tab1:
                            column_config = {
                                "source": st.column_config.LinkColumn(
                                    "Source",
                                    help="Click to open source",
                                    validate="^https?://",
                                    max_chars=200,
                                )
                            }
                            st.data_editor(
                                df,
                                column_config=column_config,
                                use_container_width=True,
                                disabled=True,
                                hide_index=True,
                            )
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
                                    self.add_results_to_chat_context(df, "Web", query)
                                    st.markdown("---")
                                    if st.button("🗣️ Switch to Chat Mode", use_container_width=True):
                                        self.session.search_mode = "💬 Chat"
                                        st.rerun()
                    else:
                        st.error(f"Search failed: {console_output}")
                else:
                    st.warning("Please enter a search query")
        elif self.session.search_mode == "📂 Document Search":
            st.markdown("### 📂 Document Retrieval")
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
                current_files_hash = self.calculate_files_hash(uploaded_files)
                if (
                    self.session.processed_files_hash != current_files_hash
                    or self.session.document_searcher is None
                ):
                    with st.spinner("🔄 Processing documents..."):
                        searcher, console_output = self.process_uploaded_documents(
                            uploaded_files, transformer_model, device, chunk_length
                        )
                        if searcher is not None:
                            self.session.document_searcher = searcher
                            self.session.processed_files_hash = current_files_hash
                            self.session.processing_console_output = console_output
                            st.success(f"✅ Processed {len(uploaded_files)} document(s)")
                        else:
                            st.error(f"Document processing failed: {console_output}")
                if self.session.document_searcher is not None:
                    query = st.text_input(
                        "Search within documents",
                        placeholder="Insert query here",
                        help="Enter keywords to search within your documents",
                    )
                    if st.button("🔍 Search Documents", use_container_width=True):
                        if query:
                            with st.spinner("🔄 Searching..."):
                                df, search_console_output = self.search_documents(
                                    self.session.document_searcher, query, self.session.top_k
                                )
                            if df is not None:
                                st.success("✅ Search complete")
                                self.display_search_results_summary(df, query)
                                tab1, tab2, tab3 = st.tabs(["📊 Results", "📜 Logs", "🔄 Use in Chat"])
                                with tab1:
                                    st.dataframe(
                                        df,
                                        use_container_width=True,
                                        hide_index=True,
                                    )
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
                                        + self.session.processing_console_output
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
                                            self.add_results_to_chat_context(df, "Document", query)
                                            st.markdown("---")
                                            if st.button("🗣️ Switch to Chat Mode", use_container_width=True):
                                                self.session.search_mode = "💬 Chat"
                                                st.rerun()
                            else:
                                st.error(f"Search failed: {search_console_output}")
                        else:
                            st.warning("Please enter a search query")
            else:
                st.info("👆 Start by uploading your documents")
        else:  # Chat mode
            self.display_chat_interface()

if __name__ == "__main__": 
    app = DocumentSearchApp()
    app.run()