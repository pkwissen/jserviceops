import os
import sys

# Prevent __pycache__ creation
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

import re
import streamlit as st
from intelligent_agent_assist_code.ingestion.ingest_pipeline import run_ingestion
from intelligent_agent_assist_code.search.opensearch_retriever import OpenSearchRetriever
from intelligent_agent_assist_code.chat.prompt_builder import build_prompt, enhance_question_with_context
from intelligent_agent_assist_code.chat.chat_model import ask_llm
from intelligent_agent_assist_code.chat.chat_history_ui import render_chat_history_sidebar, save_current_chat
from intelligent_agent_assist_code.ingestion.sharepoint_list_client import list_kb_files
import pandas as pd


# ========== PAGE CONFIG ==========
st.set_page_config(
    page_title="Intelligent Agent Assist",
    layout="centered",
    initial_sidebar_state="collapsed"
)
def main():
    if st.button("Back to Homepage", key="heat_back_homepage"):
        st.session_state["current_app"] = "Homepage"
        st.rerun()
    # ========== INITIALIZE SESSION STATE ==========
    if "page" not in st.session_state:
        st.session_state.page = "home"

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "current_chat_id" not in st.session_state:
        st.session_state.current_chat_id = None

    # ========== CUSTOM STYLING ==========
    st.markdown("""
    <style>
        .title {
            font-size: 36px;
            font-weight: 800;
            color: #e74266;
            text-align: center;
            margin-bottom: 5px;
        }
        .subtitle {
            font-size: 18px;
            font-weight: 600;
            color: #101330;
            text-align: center;
            margin-bottom: 40px;
        }
        .card {
            background-color: #ffffff;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            margin-bottom: 30px;
            text-align: center;
        }
        .card h3 {
            color: #101330;
            margin-bottom: 15px;
        }
        .card p {
            color: #666;
            margin-bottom: 20px;
            font-size: 14px;
        }
        .nav-btn {
            display: inline-block;
            padding: 12px 24px;
            background-color: #e74266;
            color: #ffffff !important;
            font-size: 15px;
            font-weight: 700;
            border-radius: 8px;
            text-decoration: none;
            border: none;
            cursor: pointer;
            margin: 5px;
        }
        .nav-btn:hover {
            background-color: #cf3c5c;
        }
        .back-btn {
            display: inline-block;
            padding: 8px 16px;
            background-color: #e0e0e0;
            color: #333 !important;
            font-size: 14px;
            font-weight: 600;
            border-radius: 6px;
            text-decoration: none;
            border: none;
            cursor: pointer;
            margin-bottom: 20px;
        }
        .back-btn:hover {
            background-color: #d0d0d0;
        }
        /* Chat Dark Theme */
        .chat-header {
            background: linear-gradient(135deg, #1a1f3a 0%, #2d3e50 100%);
            color: white;
            padding: 40px 20px;
            border-radius: 12px 12px 0 0;
            margin-bottom: 30px;
            text-align: left;
        }
        .chat-header h1 {
            font-size: 32px;
            font-weight: 800;
            margin: 0;
            margin-bottom: 8px;
        }
        .chat-header p {
            font-size: 16px;
            color: #b0b8c8;
            margin: 0;
        }
        /* Chat message styles */
        .chat-msg {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            font-size: 16px; /* same size for questions and answers */
            line-height: 1.5;
            color: #163238;
        }
        .chat-user-box {
            background-color: #efefef;
            padding: 12px 16px;
            border-radius: 10px;
            margin: 8px 0;
            color: #233;
        }
        .chat-assistant-box {
            background-color: #eaf7f1; /* faint teal */
            padding: 12px 16px;
            border-radius: 10px;
            margin: 8px 0;
            color: #0b3d2e; /* dark text for contrast */
        }
        .chat-reference {
            font-size: 12px; /* smaller reference font */
            color: #6b6f76;
            margin-top: 4px;
        }
        .chat-bullets {
            margin: 6px 0 0 18px;
        }
    </style>
    """, unsafe_allow_html=True)

    # ========== HOME PAGE ==========
    def page_home():
        st.markdown("<div class='title'>Intelligent Agent Assist</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='subtitle'>Real-time KB upload and AI-powered guidance for GSD agents</div>",
            unsafe_allow_html=True
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="card">
                <h3>📄 Upload Knowledge Base Article</h3>
                <p>Upload approved KB documents for indexing and retrieval.</p>
            </div>
            """, unsafe_allow_html=True)
            if st.button("📤 Go to Upload", key="btn_upload", use_container_width=True):
                st.session_state.page = "upload"
                st.rerun()
        
        with col2:
            st.markdown("""
            <div class="card">
                <h3>💬 Chat with Intelligent Agent</h3>
                <p>Ask questions and get answers from indexed KB articles and SOPs.</p>
            </div>
            """, unsafe_allow_html=True)
            if st.button("💬 Go to Chat", key="btn_chat", use_container_width=True):
                st.session_state.page = "chat"
                st.rerun()

    # ========== UPLOAD PAGE ==========
    def page_upload():
        col_back, col_spacer = st.columns([1, 3])
        with col_back:
            if st.button("← Back", key="back_upload"):
                st.session_state.page = "home"
                st.rerun()
        
        st.markdown("<div class='title'>📄 Upload Knowledge Base</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='subtitle'>Upload KB documents for indexing and retrieval</div>",
            unsafe_allow_html=True
        )
        
        st.markdown("""
        <div class="card">
            <p style="text-align:left; color:#333;">Upload your KB documents below. Supported formats: DOCX</p>
        </div>
        """, unsafe_allow_html=True)
        
        file = st.file_uploader("Choose a KB document", type=["pdf", "docx"], key="upload_file")
        
        if file:
            try:
                with st.spinner("📚 Ingesting document..."):
                    run_ingestion(file)
                st.success(f"✅ **{file.name}** ingested successfully!")
                st.info("Your document has been indexed and is ready for search.")
            except Exception as e:
                st.error(f"❌ Failed to ingest document: {str(e)}")

        st.markdown("### 📂 Existing Files")

        try:
            files = list_kb_files()

            if files:

                df = pd.DataFrame(files)

                st.dataframe(df, use_container_width=True)

                existing_files = [f["name"] for f in files]

            else:
                st.info("No files found.")
                existing_files = []

        except Exception as e:
            st.error(str(e))
            existing_files = []


    # ========== CHAT PAGE ==========
    def page_chat():
        render_chat_history_sidebar()
        
        col_back, col_spacer = st.columns([1, 3])
        with col_back:
            if st.button("← Back", key="back_chat"):
                st.session_state.page = "home"
                st.rerun()
        
        # Dark header
        st.markdown("""
        <div class="chat-header">
            <h1>Intelligent Agent Assist</h1>
            <p>Instant answers from approved KB articles and SOPs</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Prepare placeholders for chat and status so we can update them in-place
        chat_placeholder = st.empty()
        status_placeholder = st.empty()

        def render_history(container):
            """Render the full chat history into given container."""
            container.empty()
            with container.container():
                for message in st.session_state.chat_history:
                    if message["role"] == "user":
                        col_left, col_right = st.columns([0.6, 0.4])
                        with col_left:
                            st.markdown(
                                f"""
                                <div class="chat-user-box chat-msg">
                                    <p style="margin:0;"><strong>👤 You:</strong> {message['content']}</p>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                    else:
                        col_left, col_right = st.columns([0.4, 0.6])
                        with col_right:
                            content_html = str(message.get('content', '')).replace('\n', '<br>')
                            # include reference HTML inside the assistant box so it always appears directly under the answer
                            reference_html = ''
                            if message.get('reference'):
                                reference_html = f"<div class='chat-reference'>📚 Reference: {message['reference']}</div>"

                            st.markdown(
                                f"""
                                <div class="chat-assistant-box chat-msg">
                                    <p style="margin: 0;"><strong>🤖 Answer:</strong></p>
                                    <div style="margin-top:6px;">{content_html}</div>
                                    {reference_html}
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

        # initial render of existing history
        render_history(chat_placeholder)
        
        st.markdown("")
        
        # Chat input at bottom
        question = st.chat_input(
            "How may I help you?",
            key="chat_input"
        )
        
        if question:
            # Add user message to history IMMEDIATELY
            st.session_state.chat_history.append({
                "role": "user",
                "content": question
            })
            
            # IMMEDIATELY save after adding user message to preserve continuity
            save_current_chat()
            
            try:
                # Render history so the user's question appears before we start searching
                render_history(chat_placeholder)

                # Show single spinner in the status placeholder below the chat
                with status_placeholder.container():
                    with st.spinner("🔍 Searching knowledge base and generating answer..."):
                        try:
                            # 🎯 CHAT CONTINUITY: Enhance question with chat history context
                            # This keeps search within the same topic (e.g., Citrix stays Citrix)
                            enhanced_question = enhance_question_with_context(question, st.session_state.chat_history)
                            
                            # 🚀 OPTIMIZED: 45s timeout for faster retrieval
                            retriever = OpenSearchRetriever()
                            results = retriever.search(query_text=enhanced_question, timeout=45)
                        except Exception as e:
                            print(f"[WARNING] Search error: {str(e)}")
                            results = []

                        if not results:
                            # Add assistant message indicating no results
                            bot_response = "⚠️ No relevant documents found for your question. Please try different keywords or rephrase your question."
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": bot_response,
                                "reference": None
                            })
                        else:
                            # Generate answer from LLM (within same spinner)
                            context = "\n\n".join(r["text"] for r in results)
                            prompt = build_prompt(context, question, search_results=results)
                            answer = ask_llm(prompt, timeout=90)  # ⚡ Optimized: 90s timeout

                            # We'll decide on the reference after we inspect the LLM answer
                            reference_text = None

                            # Strip any LLM-provided Reference section from the answer
                            def strip_reference_section(text: str):
                                m = re.search(r"(?mi)\n[_\- ]*reference[s]?[:_\- ]*\n?", text)
                                if m:
                                    main = text[:m.start()].strip()
                                    ref = text[m.end():].strip()
                                    return main, ref
                                m2 = re.search(r"(?mi)Reference[:]\s*", text)
                                if m2:
                                    main = text[:m2.start()].strip()
                                    ref = text[m2.end():].strip()
                                    return main, ref
                                return text, None

                            main_answer_text, llm_ref_text = strip_reference_section(answer)

                            # NOTE: Do NOT fall back to extracting KB ids from the LLM-generated reference text.
                            # That can introduce unrelated or spurious KB references. We'll only use index metadata.

                            # Build a cleaned reference string from the first search result metadata (if present)
                            if results and len(results) > 0:
                                first_result = results[0]
                                md = first_result.get("metadata", {})
                                kb_number = md.get("kb_number") or None
                                kb_title_raw = md.get("kb_title") or md.get("document_title") or md.get("source_file")

                                def _clean_title(t: str) -> str:
                                    if not t:
                                        return ""
                                    s = t
                                    # remove common file extensions
                                    s = re.sub(r"\.(pdf|docx|doc|txt)$", "", s, flags=re.IGNORECASE)
                                    # remove repeated KB numbers (e.g. "KB0015259" inside title)
                                    if kb_number:
                                        s = re.sub(re.escape(kb_number), "", s, flags=re.IGNORECASE)
                                    # remove duplicate patterns like 'KB0015259_GSD' if kb_number already present
                                    s = re.sub(r"KB\s?\d{4,8}_?", "", s, flags=re.IGNORECASE)
                                    # replace multiple separators with single ' - '
                                    s = re.sub(r"[\-_–]{1,}\s*", " - ", s)
                                    # collapse whitespace
                                    s = re.sub(r"\s+", " ", s).strip()
                                    # trim leading/trailing separators
                                    s = s.strip(" -_")
                                    return s

                                cleaned_title = _clean_title(kb_title_raw) if kb_title_raw else ""

                                if kb_number and cleaned_title:
                                    reference_text = f"{kb_number}_{cleaned_title}"
                                elif kb_number:
                                    reference_text = kb_number
                                elif cleaned_title:
                                    reference_text = cleaned_title

                            # If the LLM explicitly says the KB context does NOT contain the requested information,
                            # suppress the displayed reference to avoid implying the KB answered the question.
                            refusal_patterns = [
                                r"does not include",
                                r"do not include",
                                r"doesn't include",
                                r"don't include",
                                r"can'?t answer",
                                r"cannot answer",
                                r"no relevant",
                                r"no information",
                                r"not include information",
                                r"not contained",
                                r"not present",
                            ]

                            answer_text_for_check = (main_answer_text or "").lower()
                            if any(re.search(p, answer_text_for_check, re.IGNORECASE) for p in refusal_patterns):
                                reference_text = None

                            # Append assistant message to history (plain text)
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": main_answer_text,
                                "reference": reference_text
                            })

                # clear status placeholder after work completes
                status_placeholder.empty()

                # Re-render chat so assistant reply (or no-results) appears immediately
                render_history(chat_placeholder)

            except Exception as e:
                error_msg = f"❌ Error during search: {str(e)}"
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": error_msg,
                    "reference": None
                })
                # Re-render chat with the error message
                status_placeholder.empty()
                render_history(chat_placeholder)
        
            # ALWAYS save current chat after processing (idempotent)
            if st.session_state.chat_history:
                save_current_chat()

        # ========== PAGE ROUTING ==========
    if st.session_state.page == "home":
        page_home()
    elif st.session_state.page == "upload":
        page_upload()
    elif st.session_state.page == "chat":
        page_chat()

if __name__ == "__main__":
    main()