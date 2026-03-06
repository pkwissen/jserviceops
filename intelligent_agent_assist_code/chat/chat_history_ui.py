"""
Chat History UI - Displays chat history in sidebar without disrupting existing chat logic.
"""
import streamlit as st
from datetime import datetime
from .chat_store import ChatStore

def format_date(iso_string: str) -> str:
    """Format ISO datetime to readable format."""
    try:
        dt = datetime.fromisoformat(iso_string)
        today = datetime.now().date()
        if dt.date() == today:
            return dt.strftime("%I:%M %p")
        else:
            return dt.strftime("%b %d")
    except (ValueError, AttributeError):
        return iso_string

def render_chat_history_sidebar():
    """
    Render chat history in sidebar.
    Call this ONCE at the top of your chat page (page_chat function).
    """
    store = ChatStore()
    
    with st.sidebar:
        # New Chat Button
        if st.button("➕ New Chat", use_container_width=True, key="btn_new_chat"):
            # Clear the current chat history and reset
            st.session_state.chat_history = []
            st.session_state.current_chat_id = None
            st.rerun()
        
        st.divider()
        
        # Chat History List
        st.markdown("### 📋 Your Chats")
        
        chats = store.list_chats()
        
        if not chats:
            st.info("No saved chats yet. Start a new conversation!", icon="💭")
        else:
            for chat in chats:
                col1, col2 = st.columns([0.85, 0.15])
                
                with col1:
                    # Load this chat when clicked
                    if st.button(
                        chat["title"],
                        key=f"load_chat_{chat['id']}",
                        use_container_width=True,
                        help=f"Created: {chat['created_at']}"
                    ):
                        loaded_chat = store.load_chat(chat["id"])
                        if loaded_chat:
                            st.session_state.chat_history = loaded_chat["messages"]
                            st.session_state.current_chat_id = chat["id"]
                            st.rerun()
                
                with col2:
                    # Delete button for this chat
                    if st.button(
                        "🗑️",
                        key=f"delete_chat_{chat['id']}",
                        help="Delete this chat"
                    ):
                        store.delete_chat(chat["id"])
                        # If we deleted the current chat, clear it
                        if st.session_state.get("current_chat_id") == chat["id"]:
                            st.session_state.chat_history = []
                            st.session_state.current_chat_id = None
                        st.rerun()
                
                # Show metadata
                st.caption(
                    f"📅 {format_date(chat['created_at'])} • "
                    f"{chat['message_count']} messages"
                )
        
        st.divider()
        
        # Settings Section
        if st.checkbox("⚙️ Settings", key="show_settings"):
            col_clear, col_info = st.columns([0.6, 0.4])
            
            with col_clear:
                if st.button("🗑️ Clear All Chats", key="clear_all_chats", use_container_width=True):
                    store.clear_all_chats()
                    st.session_state.chat_history = []
                    st.session_state.current_chat_id = None
                    st.success("All chats cleared!")
                    st.rerun()
            
            with col_info:
                total_chats = len(chats)
                st.metric("Total Chats", total_chats)


def save_current_chat(force_save=False):
    """
    Save the current chat from session state with proper error handling.
    Call this in your app after the user sends a message.
    
    Args:
        force_save: If True, always save even if chat_history is empty
    
    Returns:
        Chat ID if saved, None otherwise
    """
    if not st.session_state.chat_history and not force_save:
        return None
    
    if len(st.session_state.chat_history) == 0 and not force_save:
        return None
    
    try:
        store = ChatStore()
        
        # If we're updating an existing chat, update it
        if "current_chat_id" in st.session_state and st.session_state.current_chat_id:
            chat_id = st.session_state.current_chat_id
            store.update_chat(chat_id, st.session_state.chat_history)
            return chat_id
        
        # Otherwise, save as a new chat
        else:
            chat_id = store.save_chat(st.session_state.chat_history)
            st.session_state.current_chat_id = chat_id
            return chat_id
    
    except Exception as e:
        print(f"[WARNING] Failed to save chat: {str(e)}")
        return None
