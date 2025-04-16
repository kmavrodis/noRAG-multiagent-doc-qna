import streamlit as st
import PyPDF2
import os
from io import BytesIO
from openai import AzureOpenAI
import tiktoken
import json
import tempfile
from typing import List, Dict, Tuple
import logging
from configuration.config import ConfigLoader
from utils import count_tokens, split_text_into_chunks, extract_text_from_pdf_gpt, extract_text_from_pdf_pypdf2, get_summary, process_document_chunks, select_relevant_document, get_answer

# Page configuration
st.set_page_config(
    page_title="noRAG Multiagent Document QnA",
    page_icon="📚",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
        .main {
            padding: 2rem;
        }
        .stButton>button {
            width: 50%;
        }
        .upload-text {
            font-size: 1.2rem;
            font-weight: bold;
            margin-bottom: 1rem;
        }
        .status-box {
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 1rem 0;
        }
        .token-info {
            font-size: 0.9rem;
            color: #666;
            padding: 5px;
            border-radius: 5px;
            background-color: #f0f2f6;
        }
        .token-dashboard {
            background-color: #f8f9fa;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 20px;
            border: 1px solid #e9ecef;
        }
        .highlight-box {
            background-color: #e6f3ff;
            border-left: 4px solid #0d6efd;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
        }
        .answer-container {
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            border-left: 5px solid #198754;
            margin: 10px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .token-meter {
            height: 8px;
            border-radius: 4px;
            margin-top: 5px;
            background: linear-gradient(90deg, #4CAF50, #FFC107, #FF5722);
        }
        .info-tooltip {
            color: #6c757d;
            cursor: help;
            margin-left: 5px;
        }
        .extraction-badge {
            display: inline-block;
            padding: 3px 8px;
            border-radius: 12px;
            font-size: 0.8rem;
            margin-right: 10px;
        }
        .extraction-badge-pypdf {
            background-color: #d1e7dd;
            color: #0f5132;
        }
        .extraction-badge-gpt {
            background-color: #cfe2ff;
            color: #084298;
        }
        /* Add smaller title style */
        .small-title {
            font-size: 1.8rem;
            margin-bottom: 0.5rem;
        }
    </style>
""", unsafe_allow_html=True)

# Initialize configuration
if 'config' not in st.session_state:
    st.session_state.config = ConfigLoader()

# Configure OpenAI
azure_config = st.session_state.config.get_azure_config()
client = AzureOpenAI(
    api_key=azure_config['api_key'],
    api_version=azure_config['api_version'],
    azure_endpoint=azure_config['azure_endpoint']
)
deployment_name = azure_config['deployment_name']

# Initialize tokenizer
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

# Initialize session state for documents and UI control
if 'documents' not in st.session_state:
    st.session_state.documents = {}
if 'summaries' not in st.session_state:
    st.session_state.summaries = {}
if 'token_counts' not in st.session_state:
    st.session_state.token_counts = {}
if 'show_answer' not in st.session_state:
    st.session_state.show_answer = False
if 'extraction_method' not in st.session_state:
    st.session_state.extraction_method = 'PyPDF2'
if 'token_usage_history' not in st.session_state:
    st.session_state.token_usage_history = []
if 'total_tokens_processed' not in st.session_state:
    st.session_state.total_tokens_processed = 0
    
# App header with logo and title - updated with smaller header
st.markdown("""
    <h2 class="small-title">📚 noRAG Multiagent Document QnA</h2>
    <div class="highlight-box">
        This application uses multiple AI agents to analyze documents, extract relevant information,
        and answer your questions without using traditional RAG (Retrieval Augmented Generation).
    </div>
""", unsafe_allow_html=True)

# Create tabs
tab1, tab2, tab3 = st.tabs(["📝 Main", "📊 Token Analytics", "⚙️ Configuration"])

with tab1:
    # Main UI
    col1, col2 = st.columns([3, 2])

    with col1:
        # Information box about token usage
        with st.expander("ℹ️ Understanding Token Usage", expanded=False):
            st.markdown("""
                ### What are tokens?
                
                Tokens are pieces of text that the AI model processes. In OpenAI models, a token is approximately 4 characters or 3/4 of a word.
                
                ### Why do tokens matter?
                - **Processing limits**: Each AI model has a maximum context window (token limit)
                - **Processing time**: More tokens = longer processing time
                - **Document splitting**: Large documents are split into chunks based on token count
                - **Cost**: API usage is billed by token count
                
                ### How token usage affects this application:
                1. **Document size**: Larger documents consume more tokens
                2. **Chunking threshold**: Documents exceeding the max token limit are split into parts
                3. **Answer quality**: More relevant context tokens often lead to better answers
                4. **Processing speed**: Token count directly impacts how fast your document is processed
            """)

        st.markdown("### 📄 Upload Documents")
        
        # Extraction method with improved UI
        st.markdown("""
            <div style="display: flex; align-items: center;">
                <div style="flex-grow: 1;">
                    <p><strong>Select text extraction method:</strong></p>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        extraction_col1, extraction_col2 = st.columns(2)
        with extraction_col1:
            extraction_method = st.radio(
                "",
                options=['PyPDF2', 'GPT'],
                horizontal=True,
                key='extraction_method',
                label_visibility="collapsed"
            )
        
        with extraction_col2:
            if extraction_method == 'PyPDF2':
                st.markdown("""
                    <div class="extraction-badge extraction-badge-pypdf">
                        ⚡ Fast extraction (text only)
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div class="extraction-badge extraction-badge-gpt">
                        🔍 Advanced extraction (text, tables, images)
                    </div>
                """, unsafe_allow_html=True)
        
        st.markdown("<hr style='margin: 10px 0;'>", unsafe_allow_html=True)
        
        uploaded_files = st.file_uploader(
            "Upload PDF Documents",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload one or more PDF files to analyze",
            key="pdf_uploader",
            label_visibility="collapsed"
        )
        
        # Modified file processing section
        if uploaded_files:
            for file in uploaded_files:
                if file.name not in st.session_state.documents:
                    with st.spinner('🔄 Document Analysis Agent is processing document ' + file.name):
                        
                        try:
                            progress_bar = st.progress(0)
                            
                            progress_bar.progress(25)
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                                tmp_file.write(file.getvalue())
                                file_path = tmp_file.name
                            
                            # Use the selected extraction method
                            extraction_start_time = st.empty()
                            extraction_start_time.info(f"⏱️ Started extraction at {st.session_state.extraction_method} method...")
                            
                            if st.session_state.extraction_method == 'GPT':
                                chunks, chunk_tokens = extract_text_from_pdf_gpt(file_path)
                            else:
                                chunks, chunk_tokens = extract_text_from_pdf_pypdf2(BytesIO(file.read()))
                            
                            extraction_start_time.empty()
                            progress_bar.progress(50)
                            total_tokens = sum(chunk_tokens)
                            
                            # Update total token counter
                            st.session_state.total_tokens_processed += total_tokens
                            
                            # Update token usage history
                            st.session_state.token_usage_history.append({
                                'filename': file.name,
                                'tokens': total_tokens,
                                'extraction_method': st.session_state.extraction_method,
                                'chunks': len(chunks)
                            })
                            
                            if len(chunks) > 1:
                                st.info(f"""
                                    ℹ️ Document '{file.name}' is large ({total_tokens:,} tokens) and will be split into {len(chunks)} parts.
                                    Each part will be processed separately for better handling.
                                """)
                            
                            progress_bar.progress(75)
                            docs, sums, tokens = process_document_chunks(file.name, chunks, chunk_tokens)
                            
                            st.session_state.documents.update(docs)
                            st.session_state.summaries.update(sums)
                            st.session_state.token_counts.update(tokens)
                            
                            progress_bar.progress(100)
                            
                            # Add extraction method info to success message
                            extraction_info = "🔍 Extracted using: " + st.session_state.extraction_method
                            if len(chunks) > 1:
                                st.success(f"""
                                    ✅ Successfully processed {file.name}
                                    \n{extraction_info}
                                    \n📊 Total tokens: {total_tokens:,}
                                    \n📑 Split into {len(chunks)} parts of {', '.join(f"{tokens:,}" for tokens in chunk_tokens)} tokens each
                                """)
                            else:
                                st.success(f"""
                                    ✅ Successfully processed {file.name}
                                    \n{extraction_info}
                                    \n📊 Token count: {total_tokens:,} tokens
                                """)
                            
                            progress_bar.empty()
                            
                        except Exception as e:
                            st.error(f"""
                                ❌ Error processing {file.name}
                                \nError: {str(e)}
                                \nPlease try again with a different file or contact support if the issue persists.
                            """)
                            continue
        
        st.markdown("### ❓ Ask Your Question")
        question = st.text_input(
            "Enter your question",
            key="question_input",
            placeholder="Type your question about the uploaded documents here...",
            help="Ask a question about the uploaded documents",
            label_visibility="collapsed"
        )
        
        if st.button("🔍 Submit Question", type="primary", disabled=len(st.session_state.documents) == 0):
            st.session_state.show_answer = True
        else:
            st.session_state.show_answer = False

        if st.session_state.show_answer and question and st.session_state.documents:
            with st.spinner('🔍 Researcher Agent is analyzing document relevance...'):
                relevant_doc, relevance_scores = select_relevant_document(question, st.session_state.summaries)
                
                st.markdown("### 📊 Document Relevance")
                
                sorted_scores = dict(sorted(relevance_scores.items(), key=lambda x: x[1], reverse=True))
                
                with st.expander("View Relevance Scores"):
                    for doc, score in sorted_scores.items():
                        col_0, col_1, col_2 = st.columns([3, 2, 0.5])
                        with col_0:
                            st.markdown(f"{doc}")
                        with col_1:
                            st.progress(score / 100)
                        with col_2:
                            st.markdown(f"{score}%")

            with st.spinner('🔍 Reply Agent is generating an answer from the most relevant document...'):
                answer = get_answer(question, st.session_state.documents[relevant_doc])
                
                st.markdown("### 💡 Answer")
                st.markdown(f"""
                    <div class="highlight-box">
                        <strong>📄 Source:</strong> {relevant_doc}
                        <br><strong>📊 Document size:</strong> {st.session_state.token_counts[relevant_doc]:,} tokens
                        <br><strong>🎯 Relevance score:</strong> {relevance_scores[relevant_doc]}%
                    </div>
                """, unsafe_allow_html=True)
                
                st.markdown(
                    f"""
                    <div class="answer-container">
                        {answer}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    with col2:
        st.markdown("### 📑 Documents Dashboard")
        
        if st.session_state.summaries:
            # Token usage overview
            total_tokens = sum(st.session_state.token_counts.values())
            st.markdown(f"""
                <div class="token-dashboard">
                    <h4>📊 Token Usage Overview</h4>
                    <p><strong>Documents loaded:</strong> {len(st.session_state.documents)}</p>
                    <p><strong>Total tokens across documents:</strong> {total_tokens:,}</p>
                    <p><strong>Average tokens per document:</strong> {int(total_tokens / len(st.session_state.documents)):,}</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Document summaries
            for filename in st.session_state.summaries.keys():
                with st.expander(f"📄 {filename}"):
                    # Make summary editable with automatic saving
                    st.markdown("#### Document Summary")
                    st.markdown("<small>Edit this summary to refine document matching accuracy</small>", unsafe_allow_html=True)
                    
                    edited_summary = st.text_area(
                        "Document Summary (editable)",
                        value=st.session_state.summaries[filename],
                        height=250,
                        key=f"summary_{filename}",
                        label_visibility="collapsed"
                    )
                    
                    # Update the summary if changed
                    if edited_summary != st.session_state.summaries[filename]:
                        st.session_state.summaries[filename] = edited_summary
                    
                    # Token visualization
                    doc_tokens = st.session_state.token_counts[filename]
                    max_tokens = st.session_state.config.get_processing_config()['max_chunk_tokens']
                    token_percentage = min(100, (doc_tokens / max_tokens) * 100)
                    
                    st.markdown(
                        f"""
                        <div class="token-info">
                            <strong>Document Statistics:</strong>
                            <ul style="margin-top: 5px; margin-bottom: 5px;">
                                <li>Tokens: {doc_tokens:,} ({token_percentage:.1f}% of maximum chunk size)</li>
                                <li>Extraction Method: {st.session_state.extraction_method}</li>
                            </ul>
                            <div class="token-meter" style="width: {token_percentage}%"></div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
        else:
            st.info("📌 Upload documents to see their summaries and analytics here")

        st.markdown("### 🔧 System Status")
        with st.expander("View System Details", expanded=False):
            st.markdown(f"**Documents Loaded:** {len(st.session_state.documents)}")
            st.markdown(f"**Model:** {deployment_name}")
            st.markdown(f"**Total Processed Tokens:** {st.session_state.total_tokens_processed:,}")
            st.markdown(f"**System Status:** 🟢 System Ready")
            st.markdown(f"**Extraction Method:** {st.session_state.extraction_method}")

with tab2:
    # Token Analytics Tab
    st.markdown("### 📊 Token Usage Analytics")
    
    # Token usage explanation
    st.markdown("""
        <div class="highlight-box">
            <h4>Why Token Analytics Matter</h4>
            <p>Tokens are the basic units that AI models process. Understanding token usage helps you:</p>
            <ul>
                <li><strong>Optimize costs:</strong> OpenAI API usage is billed based on token count</li>
                <li><strong>Improve performance:</strong> Large token counts may slow down processing</li>
                <li><strong>Ensure quality:</strong> Documents split into many chunks may lose context</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
    
    # Token usage metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Total Tokens Processed", 
            f"{st.session_state.total_tokens_processed:,}",
            help="Total number of tokens processed across all documents"
        )
    
    with col2:
        if st.session_state.token_counts:
            st.metric(
                "Average Tokens per Document", 
                f"{int(sum(st.session_state.token_counts.values()) / len(st.session_state.token_counts)):,}",
                help="Average number of tokens per document"
            )
        else:
            st.metric("Average Tokens per Document", "0")
    
    with col3:
        if st.session_state.token_usage_history:
            st.metric(
                "Documents Processed", 
                len(st.session_state.token_usage_history),
                help="Total number of documents processed"
            )
        else:
            st.metric("Documents Processed", "0")
    
    # Token visualization
    if st.session_state.token_counts:
        st.markdown("### Document Token Distribution")
        
        # Create simple bar chart of token counts
        token_data = {"Document": [], "Tokens": []}
        for doc, tokens in st.session_state.token_counts.items():
            token_data["Document"].append(doc)
            token_data["Tokens"].append(tokens)
        
        st.bar_chart(token_data, x="Document", y="Tokens", use_container_width=True)
        
        # Token usage history
        if st.session_state.token_usage_history:
            st.markdown("### Document Processing History")
            history_data = st.session_state.token_usage_history
            
            for i, entry in enumerate(history_data):
                st.markdown(f"""
                    <div style="padding: 10px; background-color: {'#f8f9fa' if i % 2 == 0 else '#ffffff'}; border-radius: 5px; margin-bottom: 5px;">
                        <strong>{entry['filename']}</strong>: {entry['tokens']:,} tokens 
                        <span class="extraction-badge {'extraction-badge-pypdf' if entry['extraction_method'] == 'PyPDF2' else 'extraction-badge-gpt'}">
                            {entry['extraction_method']}
                        </span>
                        {f"<span style='color: #dc3545;'>(Split into {entry['chunks']} chunks)</span>" if entry['chunks'] > 1 else ""}
                    </div>
                """, unsafe_allow_html=True)

with tab3:
    st.markdown("### ⚙️ System Configuration")

    # Document Processing Settings
    with st.expander("📄 Document Processing Settings"):
        processing_config = st.session_state.config.get_processing_config()
        new_max_tokens = st.number_input(
            "Maximum Tokens per Chunk",
            min_value=1000,
            max_value=1000000,
            value=processing_config['max_chunk_tokens'],
            help="Maximum number of tokens per document chunk. When a document exceeds this size, it will be split into multiple chunks.",
            key="doc_proc_max_tokens"
        )
        if new_max_tokens != processing_config['max_chunk_tokens']:
            st.session_state.config.update_config('document_processing', 'max_chunk_tokens', new_max_tokens)
        
        st.info("""
            **Token Chunking Explained:**
            Setting a higher maximum tokens per chunk allows for processing larger document sections together, which can improve answer quality but may slow down processing.
            Setting a lower value will split documents into more manageable pieces but may sometimes lose cross-section context.
        """)

    # Document Analysis Agent Configuration
    with st.expander("📊 Document Analysis Agent"):
        st.markdown("""
            <div class="highlight-box">
                The Document Analysis Agent extracts key information from documents, creates summaries, and prepares content for analysis.
                Adjust these settings to control how documents are summarized and processed.
            </div>
        """, unsafe_allow_html=True)
        
        doc_analysis_config = st.session_state.config.get_agent_config('document_analysis_agent')

        new_system_prompt = st.text_area(
            "System Prompt",
            value=doc_analysis_config['system_prompt'],
            height=100,
            help="System prompt that defines the agent's role and behavior",
            key="doc_analysis_system_prompt"
        )
        if new_system_prompt != doc_analysis_config['system_prompt']:
            st.session_state.config.update_config('document_analysis_agent', 'system_prompt', new_system_prompt)

        new_model_prompt = st.text_area(
            "Model Prompt Template",
            value=doc_analysis_config['model_prompt'],
            height=100,
            help="Template for the model prompt (actual document text will be appended)",
            key="doc_analysis_model_prompt"
        )
        if new_model_prompt != doc_analysis_config['model_prompt']:
            st.session_state.config.update_config('document_analysis_agent', 'model_prompt', new_model_prompt)
        
        col1, col2 = st.columns(2)
        with col1:
            new_max_tokens = st.number_input(
                "Maximum Tokens",
                min_value=100,
                max_value=2000,
                value=doc_analysis_config['max_tokens'],
                help="Maximum number of tokens for response. Higher values allow for more detailed summaries.",
                key="doc_analysis_max_tokens"
            )
            if new_max_tokens != doc_analysis_config['max_tokens']:
                st.session_state.config.update_config('document_analysis_agent', 'max_tokens', new_max_tokens)
        
        with col2:
            new_temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=doc_analysis_config['temperature'],
                step=0.1,
                help="Controls randomness in generation (0 = deterministic, 1 = creative). Lower values are better for factual summaries.",
                key="doc_analysis_temperature"
            )
            if new_temperature != doc_analysis_config['temperature']:
                st.session_state.config.update_config('document_analysis_agent', 'temperature', new_temperature)

    # Researcher Agent Configuration
    with st.expander("🔍 Researcher Agent"):
        st.markdown("""
            <div class="highlight-box">
                The Researcher Agent determines which document is most relevant to your question.
                It analyzes document summaries and assigns relevance scores to each document.
            </div>
        """, unsafe_allow_html=True)
        
        researcher_config = st.session_state.config.get_agent_config('researcher_agent')
        
        new_system_prompt = st.text_area(
            "System Prompt",
            value=researcher_config['system_prompt'],
            height=100,
            help="System prompt that defines the agent's role and behavior",
            key="researcher_system_prompt"
        )
        if new_system_prompt != researcher_config['system_prompt']:
            st.session_state.config.update_config('researcher_agent', 'system_prompt', new_system_prompt)
        
        new_model_prompt = st.text_area(
            "Model Prompt Template",
            value=researcher_config['model_prompt'],
            height=100,
            help="Template for the model prompt (document details will be appended)",
            key="researcher_model_prompt"
        )
        if new_model_prompt != researcher_config['model_prompt']:
            st.session_state.config.update_config('researcher_agent', 'model_prompt', new_model_prompt)
        
        col1, col2 = st.columns(2)
        with col1:
            new_max_tokens = st.number_input(
                "Maximum Tokens",
                min_value=100,
                max_value=2000,
                value=researcher_config['max_tokens'],
                help="Maximum number of tokens for response",
                key="researcher_max_tokens"
            )
            if new_max_tokens != researcher_config['max_tokens']:
                st.session_state.config.update_config('researcher_agent', 'max_tokens', new_max_tokens)
        
        with col2:
            new_temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=researcher_config['temperature'],
                step=0.1,
                help="Controls randomness in generation (0 = deterministic, 1 = creative). Lower values are better for consistent document selection.",
                key="researcher_temperature"
            )
            if new_temperature != researcher_config['temperature']:
                st.session_state.config.update_config('researcher_agent', 'temperature', new_temperature)

    # Reply Agent Configuration
    with st.expander("💡 Reply Agent"):
        st.markdown("""
            <div class="highlight-box">
                The Reply Agent generates the final answer to your question based on the selected document.
                It analyzes the document content and provides a comprehensive response.
            </div>
        """, unsafe_allow_html=True)
        
        reply_config = st.session_state.config.get_agent_config('reply_agent')
        
        new_system_prompt = st.text_area(
            "System Prompt",
            value=reply_config['system_prompt'],
            height=100,
            help="System prompt that defines the agent's role and behavior",
            key="reply_system_prompt"
        )
        if new_system_prompt != reply_config['system_prompt']:
            st.session_state.config.update_config('reply_agent', 'system_prompt', new_system_prompt)
        
        new_model_prompt = st.text_area(
            "Model Prompt Template",
            value=reply_config['model_prompt'],
            height=100,
            help="Template for the model prompt (document context will be appended)",
            key="reply_model_prompt"
        )
        if new_model_prompt != reply_config['model_prompt']:
            st.session_state.config.update_config('reply_agent', 'model_prompt', new_model_prompt)
        
        col1, col2 = st.columns(2)
        with col1:
            new_max_tokens = st.number_input(
                "Maximum Tokens",
                min_value=100,
                max_value=4000,
                value=reply_config['max_tokens'],
                help="Maximum number of tokens for response. Higher values allow for more detailed answers.",
                key="reply_max_tokens"
            )
            if new_max_tokens != reply_config['max_tokens']:
                st.session_state.config.update_config('reply_agent', 'max_tokens', new_max_tokens)
        
        with col2:
            new_temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=reply_config['temperature'],
                step=0.1,
                help="Controls randomness in generation (0 = deterministic, 1 = creative)",
                key="reply_temperature"
            )
            if new_temperature != reply_config['temperature']:
                st.session_state.config.update_config('reply_agent', 'temperature', new_temperature)

    # Model Information
    st.markdown("### 🤖 Model Information")
    azure_config = st.session_state.config.get_azure_config()
    st.info(f"""
        **Current Model:** {azure_config['deployment_name']}
        \n**API Version:** {azure_config['api_version']}
        \n**Endpoint:** {azure_config['azure_endpoint']}
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #666;">
        <p>📚 noRAG Multiagent Document QnA | Made using Streamlit and Azure OpenAI</p>
        <p style="font-size: 0.8rem;">Process documents efficiently with AI agents that extract information and answer questions</p>
    </div>
    """,
    unsafe_allow_html=True
)
