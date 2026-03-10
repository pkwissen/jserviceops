import os
import sys
import re
import pandas as pd
import streamlit as st

# Prevent __pycache__ creation
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True

# Import Backend Logic
from intelligent_agent_assist_code.utils.document_loader import save_upload_to_temp, load_documents_from_file
from intelligent_agent_assist_code.ingestion.ingest_pipeline import run_ingestion
from intelligent_agent_assist_code.retrieval.chat_lang_osearch_rank import chat, get_session, create_session
from intelligent_agent_assist_code.sharepoint.sp_uploader import upload_file_to_sharepoint
from intelligent_agent_assist_code.sharepoint.sp_list_client import list_kb_files  # Your MSAL listing logic
from intelligent_agent_assist_code.chat.chat_history_ui import render_chat_history_sidebar, save_current_chat

# ========== PAGE CONFIG ==========
# st.set_page_config(
#     page_title="Intelligent Agent Assist",
#     layout="centered",
#     initial_sidebar_state="collapsed"
# )

# We will define main() at the bottom.

# ========== HELPER FUNCTIONS ==========
def go_home():
    st.session_state.page = "home"
    st.rerun()

# ========== PAGE: HOME ==========
def page_home():
    st.markdown("<div class='title'>Intelligent Agent Assist</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Real-time KB upload and AI-powered guidance for GSD agents</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="card"><h3>📄 Upload & Manage KB</h3><p>Upload new articles and view existing files in SharePoint.</p></div>', unsafe_allow_html=True)
        if st.button("📤 Go to Upload", use_container_width=True):
            st.session_state.page = "upload"
            st.rerun()
    with col2:
        st.markdown('<div class="card"><h3>💬 Chat with AI</h3><p>Ask questions and get answers from indexed KB articles.</p></div>', unsafe_allow_html=True)
        if st.button("💬 Go to Chat", use_container_width=True):
            st.session_state.page = "chat"
            st.rerun()

# ========== PAGE: UPLOAD & LIST ==========
def page_upload():
    if st.button("← Back to Home"): go_home()
    
    st.markdown("<div class='title'>📄 Knowledge Management</div>", unsafe_allow_html=True)
    
    # --- Upload Section ---
    st.markdown("### 📤 Upload New Document")
    file = st.file_uploader("Choose a DOCX or PDF file", type=["pdf", "docx"])
    
    if file and st.button("Start Ingestion Process"):
        try:
            with st.status("Processing Document...", expanded=True) as status:
                st.write("Uploading to SharePoint...")
                upload_file_to_sharepoint(file.getvalue(), file.name)
                
                st.write("Indexing content for AI search...")
                path, _ = save_upload_to_temp(file)
                docs = load_documents_from_file(path)
                run_ingestion(docs)
                
                status.update(label="✅ Ingestion Complete!", state="complete", expanded=False)
            st.success(f"'{file.name}' is now active in the knowledge base.")
            st.rerun() 
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

    st.markdown("---")

    # --- List Section ---
    st.markdown("### 📂 Existing Articles (SharePoint)")
    with st.spinner("Fetching latest file list from SharePoint..."):
        files = list_kb_files()
    
    if files:
        df = pd.DataFrame(files)
        st.dataframe(
            df, 
            use_container_width=True, 
            hide_index=True,
            column_config={
                "File Name": st.column_config.TextColumn("Document Name", width="large"),
                "Size (KB)": "Size (KB)",
                "Last Modified": "Modified Date"
            }
        )
    else:
        st.info("No documents found in the SharePoint target folder.")

# ========== PAGE: CHAT ==========
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
                        content_html = str(message.get('content', '')).replace('\\n', '<br>')
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

    # User Input at bottom
    if question := st.chat_input("Ask about a GSD process..."):
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

            # Show spinner in the status placeholder
            with status_placeholder.container():
                with st.spinner("🔍 Analyzing Knowledge Base..."):
                    # Use existing session if it exists, otherwise create a new one matching the frontend chat ID
                    session_id = st.session_state.current_chat_id
                    session = get_session(session_id) if session_id else None
                    if session is None:
                        session = create_session(session_id)
                        
                    # Execute the existing engine logic
                    result = chat(question, session)
                    
                    answer_text = result.get("answer", "")
                    
                    # Extract references conceptually (if your pipeline returned docs)
                    source_docs = result.get("source_documents", [])
                    reference_text = ""
                    if source_docs:
                        titles = set()
                        for doc in source_docs:
                            metadata = doc.get("metadata", {})
                            # The file_name or source might be in metadata depending on how it was indexed
                            title = metadata.get("file_name") or metadata.get("source") or metadata.get("title") or "Unknown Document"
                            titles.add(title)
                        
                        if titles:
                            reference_text = ", ".join(sorted(list(titles)))
                        else:
                            reference_text = "Retrieved from Knowledge Base"
                    else:
                        reference_text = None
                    
                    # Store assistant message
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer_text,
                        "reference": reference_text
                    })
                    
            status_placeholder.empty()
            render_history(chat_placeholder)
            
        except Exception as e:
            error_msg = f"❌ Error during search: {str(e)}"
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": error_msg,
                "reference": None
            })
            status_placeholder.empty()
            render_history(chat_placeholder)
            
        # ALWAYS save current chat after processing
        if st.session_state.chat_history:
            save_current_chat()
    


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

    # ========== CUSTOM STYLING (GSD Branded) ==========
    st.markdown("""
<style>
    .title { font-size: 36px; font-weight: 800; color: #e74266; text-align: center; margin-bottom: 5px; }
    .subtitle { font-size: 18px; font-weight: 600; color: #101330; text-align: center; margin-bottom: 40px; }
    .card { background-color: #ffffff; padding: 30px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); margin-bottom: 30px; text-align: center; }
    .card h3 { color: #101330; margin-bottom: 15px; }
    .chat-header { background: linear-gradient(135deg, #1a1f3a 0%, #2d3e50 100%); color: white; padding: 40px 20px; border-radius: 12px 12px 0 0; margin-bottom: 30px; }
    .chat-user-box { background-color: #efefef; padding: 12px 16px; border-radius: 10px; margin: 8px 0; color: #233; }
    .chat-assistant-box { background-color: #eaf7f1; padding: 12px 16px; border-radius: 10px; margin: 8px 0; color: #0b3d2e; }
    .chat-msg { font-family: 'Segoe UI'; font-size: 16px; line-height: 1.5; }
    .chat-reference { font-size: 12px; color: #6b6f76; margin-top: 4px; }
</style>
""", unsafe_allow_html=True)

    # ========== ROUTING ==========
    if st.session_state.page == "home":
        page_home()
    elif st.session_state.page == "upload":
        page_upload()
    elif st.session_state.page == "chat":
        page_chat()

if __name__ == "__main__":
    main()