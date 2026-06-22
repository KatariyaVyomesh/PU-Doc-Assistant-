from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import os
import requests
import re
import json
import uuid
from datetime import datetime
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from supabase import create_client, Client

import sys
# Ensure current directory is in sys.path for Vercel deployment
sys.path.append(os.path.dirname(__file__))

import processor
processor = processor.processor

load_dotenv()

# Vercel-specific: Use /tmp for any file operations
TMP_DIR = "/tmp"
UPLOADS_DIR = os.path.join(TMP_DIR, "uploads")
CONVERSATIONS_FILE = os.path.join(TMP_DIR, "conversations.json")
LINKS_FILE = os.path.join(TMP_DIR, "links.json")
RESOURCES_FILE = os.path.join(TMP_DIR, "resources.json")

# Initialize /tmp storage if not exists
if not os.path.exists(UPLOADS_DIR):
    os.makedirs(UPLOADS_DIR, exist_ok=True)

# Supabase Client for index.py (separate from processor)
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
supabase: Client = None
if SUPABASE_URL and SUPABASE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Update processor to use /tmp locations
processor.links_file = LINKS_FILE
processor.resources_file = RESOURCES_FILE

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Automatically re-index files and links from /tmp
    print("Application starting... Re-indexing resources from /tmp.")
    processor.reindex_all(UPLOADS_DIR)
    yield

def ensure_synced():
    """Ensures that the in-memory index matches whatever is in /tmp."""
    # Only re-index if memory is currently empty (typical code start or reset)
    if not processor.chunks or processor.index is None:
        print("Instance reset detected. Syncing from /tmp...")
        processor._load_resources()
        processor.reindex_all(UPLOADS_DIR)

app = FastAPI(lifespan=lifespan)

# Enable CORS (Vercel handles this too, but keeping for safety)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY = os.getenv("SAMBANOVA_API_KEY", "")

def clean_llm_response(text: str) -> str:
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'Thought:.*?\n', '', text, flags=re.IGNORECASE)
    text = text.replace('<think>', '').replace('</think>', '')
    return text.strip()

@app.get("/api/files")
async def get_files(conversation_id: str):
    # Always ensure we have the latest list from the DB for the sidebar
    if supabase:
        try:
            response = supabase.table("resources").select("*").eq("conv_id", conversation_id).execute()
            return {"resources": response.data}
        except Exception as e:
            print(f"Error fetching files from Supabase: {e}")
    
    # Fallback to in-memory if DB fails
    conv_resources = [r for r in processor.indexed_resources if r.get("conv_id") == conversation_id]
    return {"resources": conv_resources}

@app.post("/api/upload")
async def upload_file(conversation_id: str = Form(...), file: UploadFile = File(...)):
    try:
        save_dir = os.path.join(UPLOADS_DIR, conversation_id)
        os.makedirs(save_dir, exist_ok=True)
        
        file_path = os.path.join(save_dir, file.filename)
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())
        
        text = processor.load_document(file_path)
        num_chunks = processor.process_content(file.filename, text, conv_id=conversation_id, is_link=False)
        
        # PERSIST: Upload to Supabase for stateless persistence
        supabase_success = processor.upload_file_to_supabase(file_path, file.filename, conversation_id)
        
        # Cleanup local /tmp file
        if os.path.exists(file_path): os.remove(file_path)
        
        msg = f"Success! I've processed **{file.filename}**."
        if not supabase_success:
            msg += " (Warning: Persistent storage failed, may not survive reset)"
            
        add_message_to_history(conversation_id, "ai", msg)
        
        return {"filename": file.filename, "chunks": num_chunks, "status": "success"}
    except Exception as e:
        import traceback
        error_msg = f"UPLOAD ERROR: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload-link")
async def upload_link(request: dict):
    url = request.get("url")
    conv_id = request.get("conversation_id")
    if not url or not conv_id:
        raise HTTPException(status_code=400, detail="URL and conversation_id are required")
    
    try:
        text = processor.scrape_url(url)
        if text.startswith("Error"):
          raise HTTPException(status_code=400, detail=text)
          
        num_chunks = processor.process_content(url, text, conv_id=conv_id, is_link=True)
        add_message_to_history(conv_id, "ai", f"Success! I've indexed the link: **{url}**.")
        
        return {"url": url, "chunks": num_chunks, "status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/files/{name:path}")
async def delete_file(name: str, conversation_id: str):
    try:
        processor.remove_resource(name, conv_id=conversation_id, upload_dir=UPLOADS_DIR)
        add_message_to_history(conversation_id, "ai", f"Removed **{name}** from this chat's memory.")
        return {"status": "success", "message": f"Resource {name} deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/conversations")
async def list_conversations():
    if not supabase: return []
    try:
        response = supabase.table("conversations").select("id, title, date").order("created_at", desc=True).execute()
        return response.data
    except Exception as e:
        print(f"Error listing conversations: {e}")
        return []

@app.get("/api/conversations/{id}")
async def get_conversation(id: str):
    if not supabase: raise HTTPException(status_code=500, detail="Supabase not configured")
    try:
        response = supabase.table("conversations").select("*").eq("id", id).execute()
        if not response.data:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return response.data[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/conversations")
async def create_conversation(request: dict):
    if not supabase: raise HTTPException(status_code=500, detail="Supabase not configured")
    title = request.get("title", "New Chat")
    new_id = str(uuid.uuid4())
    new_conv = {
        "id": new_id,
        "title": title,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "messages": [
            {"role": "ai", "content": "This is a fresh workspace. Upload documents to this chat to get started!"}
        ]
    }
    try:
        supabase.table("conversations").insert(new_conv).execute()
        return new_conv
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/conversations/{id}")
async def delete_conversation(id: str):
    if not supabase: raise HTTPException(status_code=500, detail="Supabase not configured")
    try:
        supabase.table("conversations").delete().eq("id", id).execute()
        return {"status": "success", "message": "Conversation deleted"}
    except Exception as e:
        import traceback
        error_msg = f"CREATE CONVERSATION ERROR: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        raise HTTPException(status_code=500, detail=str(e))

def add_message_to_history(conv_id, role, content):
    if not conv_id or not supabase: return
    try:
        # Fetch current messages
        response = supabase.table("conversations").select("messages").eq("id", conv_id).execute()
        if response.data:
            messages = response.data[0]["messages"]
            messages.append({"role": role, "content": content})
            # Update in Supabase
            supabase.table("conversations").update({"messages": messages}).eq("id", conv_id).execute()
    except Exception as e:
        print(f"Error adding to history: {e}")

def update_conv_title(conv_id, title):
    if not conv_id or not supabase: return
    try:
        supabase.table("conversations").update({"title": title}).eq("id", conv_id).execute()
    except Exception as e:
        print(f"Error updating title: {e}")

@app.post("/api/chat")
async def chat(request: dict):
    ensure_synced()
    query = request.get("query")
    conv_id = request.get("conversation_id")
    if not query:
        raise HTTPException(status_code=400, detail="Query is required")
    
    system_prompt = """
    You are a professional Document Analysis Assistant.
    Analyze the provided content and answer the question accurately.
    1. NO reasoning/<think> tags.
    2. Clean, direct answer.
    3. IF NO content: "Please upload a document or provide a link."
    Context: {context_text}
    """
    
    try:
        messages = [{"role": "system", "content": system_prompt}]
        if conv_id:
            # Fetch history from Supabase
            res = supabase.table("conversations").select("messages, title").eq("id", conv_id).execute()
            if res.data:
                conv_data = res.data[0]
                history = []
                for msg in conv_data["messages"][-5:]:
                    role = "assistant" if msg["role"] == "ai" else msg["role"]
                    history.append({"role": role, "content": msg["content"]})
                messages.extend(history)
                current_title = conv_data["title"]
        
        messages.append({"role": "user", "content": query})
        context_chunks = processor.search(query, conv_id=conv_id)
        context_text = "\n---\n".join(context_chunks)
        messages[0]["content"] = messages[0]["content"].replace("{context_text}", context_text)

        response = requests.post(
            "https://api.sambanova.ai/v1/chat/completions",
            headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
            json={"model": "Qwen3-32B", "messages": messages, "temperature": 0.1, "max_tokens": 1024}
        )
        
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"SambaNova error: {response.text}")
        
        clean_response = clean_llm_response(response.json()["choices"][0]["message"]["content"])
        
        conv_resources = [r for r in processor.indexed_resources if r["conv_id"] == conv_id]
        if not conv_resources and "Please upload a document" not in clean_response:
            clean_response += "\n\nPlease upload a document or provide a link to proceed."
            
        if conv_id:
            # Automatic title update
            if current_title == "New Chat":
                new_title = query[:30] + "..." if len(query) > 30 else query
                update_conv_title(conv_id, new_title)
            
            # Persist message history to Supabase
            add_message_to_history(conv_id, "user", query)
            add_message_to_history(conv_id, "ai", clean_response)

        return {"response": clean_response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
