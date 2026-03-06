"""
Chat Store Module - Persists chat history to disk without disrupting existing code.
Handles saving, loading, and managing chat sessions.
"""
import json
import os
from pathlib import Path
from datetime import datetime, timezone, timedelta
import hashlib
from typing import List, Dict, Optional

# Indian Standard Time (IST) offset
IST = timezone(timedelta(hours=5, minutes=30))


class ChatStore:
    """Manages persistent chat storage in JSON format."""
    
    def __init__(self, storage_dir: str = ".chats"):
        """Initialize chat store with a given storage directory."""
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        self.metadata_file = self.storage_dir / "metadata.json"
        self._ensure_metadata_file()
    
    def _ensure_metadata_file(self):
        """Create metadata file if it doesn't exist."""
        if not self.metadata_file.exists():
            self._save_metadata({})
    
    def _load_metadata(self) -> Dict:
        """Load metadata file."""
        try:
            with open(self.metadata_file, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {}
    
    def _save_metadata(self, metadata: Dict):
        """Save metadata file."""
        with open(self.metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
    
    def _generate_chat_id(self) -> str:
        """Generate a unique chat ID."""
        timestamp = datetime.now(tz=IST).isoformat()
        hash_input = f"{timestamp}-{os.urandom(8).hex()}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:12]
    
    def _extract_title_from_messages(self, messages: List[Dict]) -> str:
        """Extract a title from the first user message."""
        for msg in messages:
            if msg.get("role") == "user":
                text = msg.get("content", "").strip()
                # Take first 50 chars and clean up
                title = text[:50]
                if len(text) > 50:
                    title += "..."
                return title or "New Chat"
        return "New Chat"
    
    def save_chat(self, messages: List[Dict]) -> str:
        """
        Save a chat session and return its ID.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
        
        Returns:
            str: Chat ID
        """
        chat_id = self._generate_chat_id()
        chat_file = self.storage_dir / f"{chat_id}.json"
        
        # Save chat messages
        chat_data = {
            "id": chat_id,
            "created_at": datetime.now(tz=IST).isoformat(),
            "title": self._extract_title_from_messages(messages),
            "messages": messages
        }
        
        with open(chat_file, "w") as f:
            json.dump(chat_data, f, indent=2)
        
        # Update metadata
        metadata = self._load_metadata()
        metadata[chat_id] = {
            "title": chat_data["title"],
            "created_at": chat_data["created_at"],
            "message_count": len(messages)
        }
        self._save_metadata(metadata)
        
        return chat_id
    
    def update_chat(self, chat_id: str, messages: List[Dict]):
        """Update an existing chat with new messages."""
        chat_file = self.storage_dir / f"{chat_id}.json"
        
        chat_data = {
            "id": chat_id,
            "created_at": datetime.now(tz=IST).isoformat(),
            "title": self._extract_title_from_messages(messages),
            "messages": messages
        }
        
        with open(chat_file, "w") as f:
            json.dump(chat_data, f, indent=2)
        
        # Update metadata
        metadata = self._load_metadata()
        metadata[chat_id] = {
            "title": chat_data["title"],
            "created_at": chat_data["created_at"],
            "message_count": len(messages)
        }
        self._save_metadata(metadata)
    
    def load_chat(self, chat_id: str) -> Optional[Dict]:
        """Load a saved chat by ID."""
        chat_file = self.storage_dir / f"{chat_id}.json"
        
        if not chat_file.exists():
            return None
        
        try:
            with open(chat_file, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError, IOError):
            return None
    
    def list_chats(self, limit: int = 50) -> List[Dict]:
        """
        List all saved chats sorted by creation date (newest first).
        
        Args:
            limit: Max number of chats to return
        
        Returns:
            List of chat metadata dicts with keys: id, title, created_at, message_count
        """
        metadata = self._load_metadata()
        
        chats = []
        for chat_id, info in metadata.items():
            chats.append({
                "id": chat_id,
                "title": info.get("title", "Untitled Chat"),
                "created_at": info.get("created_at"),
                "message_count": info.get("message_count", 0)
            })
        
        # Sort by created_at (newest first)
        chats.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        
        return chats[:limit]
    
    def delete_chat(self, chat_id: str) -> bool:
        """Delete a saved chat."""
        chat_file = self.storage_dir / f"{chat_id}.json"
        
        if chat_file.exists():
            try:
                chat_file.unlink()
                metadata = self._load_metadata()
                metadata.pop(chat_id, None)
                self._save_metadata(metadata)
                return True
            except (OSError, IOError):
                return False
        
        return False
    
    def clear_all_chats(self):
        """Delete all saved chats."""
        for chat_file in self.storage_dir.glob("*.json"):
            if chat_file.name != "metadata.json":
                try:
                    chat_file.unlink()
                except (OSError, IOError):
                    pass
        self._save_metadata({})
