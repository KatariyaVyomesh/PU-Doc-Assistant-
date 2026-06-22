from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import os
# Fix Hugging Face warnings and improve connection stability
os.environ["USER_AGENT"] = "AI-Document-Assistant/1.0"
os.environ["HF_HUB_READ_TIMEOUT"] = "60" # Increase timeout for model loading

import requests
import re
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from supabase import create_client, Client
from processor import processor

load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Automatically re-index files and links
    print("Application starting... Re-indexing resources.")
    processor.reindex_all("uploads")
    yield
    # Shutdown logic (if any) here

app = FastAPI(lifespan=lifespan)

# Enable CORS for frontend interaction
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Supabase for main.py
print(f"DEBUG: Loading .env from: {os.getcwd()}")
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
supabase: Client = None

if SUPABASE_URL and SUPABASE_KEY and "your-" not in SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("DEBUG: Supabase Status -> CONNECTED")
    except Exception as se:
        print(f"DEBUG: Supabase Connection Error -> {se}")
else:
    print("DEBUG: Supabase Status -> NOT CONFIGURED (Check your .env keys)")

API_KEY = os.getenv("SAMBANOVA_API_KEY", "")

@app.get("/api/status")
async def get_status():
    """Simple status check for Supabase connectivity."""
    if supabase:
        return {"status": "connected", "url": SUPABASE_URL}
    else:
        return {"status": "disconnected", "error": "Supabase client not initialized"}

def clean_llm_response(text: str) -> str:
    """Removes <think> tags and any internal reasoning blocks."""
    # Remove everything between <think> and </think> tags
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    # Also remove common markers if they appear
    text = re.sub(r'Thought:.*?\n', '', text, flags=re.IGNORECASE)
    # Final check for any hanging tags
    text = text.replace('<think>', '').replace('</think>', '')
    return text.strip()

@app.get("/files")
async def get_files(conversation_id: str):
    """Returns the list of indexed resources from Supabase for parity."""
    if supabase:
        try:
            response = supabase.table("resources").select("*").eq("conv_id", conversation_id).execute()
            return {"resources": response.data}
        except Exception as e:
            print(f"Error fetching files from Supabase: {e}")
    
    # Fallback
    conv_resources = [r for r in processor.indexed_resources if r.get("conv_id") == conversation_id]
    return {"resources": conv_resources}

@app.post("/upload")
async def upload_file(conversation_id: str = Form(...), file: UploadFile = File(...)):
    """Uploads and processes a file for a specific conversation."""
    try:
        save_dir = f"uploads/{conversation_id}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        file_path = f"{save_dir}/{file.filename}"
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())
        
        text = processor.load_document(file_path)
        num_chunks = processor.process_content(file.filename, text, conv_id=conversation_id, is_link=False)
        
        # PERSIST: Upload to Supabase for parity
        processor.upload_file_to_supabase(file_path, file.filename, conversation_id)
        
        # PERSIST: Success message in history
        success_msg = f"Success! I've processed **{file.filename}** for this conversation."
        add_message_to_history(conversation_id, "ai", success_msg)
        
        return {"filename": file.filename, "chunks": num_chunks, "status": "success"}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-link")
async def upload_link(request: dict):
    """Scrapes and processes content from a URL for a specific conversation."""
    url = request.get("url")
    conv_id = request.get("conversation_id")
    if not url or not conv_id:
        raise HTTPException(status_code=400, detail="URL and conversation_id are required")
    
    try:
        text = processor.scrape_url(url)
        if text.startswith("Error"):
          raise HTTPException(status_code=400, detail=text)
          
        num_chunks = processor.process_content(url, text, conv_id=conv_id, is_link=True)
        
        # PERSIST: Success message in history
        success_msg = f"Success! I've indexed the link for this workspace: **{url}**."
        add_message_to_history(conv_id, "ai", success_msg)
        
        return {"url": url, "chunks": num_chunks, "status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/files/{name:path}")
async def delete_file(name: str, conversation_id: str):
    """Deletes a file or link for a specific conversation and rebuilds the index."""
    try:
        processor.remove_resource(name, conv_id=conversation_id)
        
        # PERSIST: Removal message in history
        remove_msg = f"Removed **{name}** from this chat's memory."
        add_message_to_history(conversation_id, "ai", remove_msg)
        
        return {"status": "success", "message": f"Resource {name} deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

import json
import uuid
from datetime import datetime

@app.get("/conversations")
async def list_conversations():
    if not supabase: return []
    try:
        response = supabase.table("conversations").select("id, title, date").order("created_at", desc=True).execute()
        return response.data
    except Exception as e:
        print(f"Error listing conversations: {e}")
        return []

@app.get("/conversations/{id}")
async def get_conversation(id: str):
    if not supabase: raise HTTPException(status_code=500, detail="Supabase not configured")
    try:
        response = supabase.table("conversations").select("*").eq("id", id).execute()
        if not response.data:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return response.data[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/conversations")
async def create_conversation(request: dict):
    if not supabase: 
        print("CREATE ERROR: Supabase client is None (check config)")
        raise HTTPException(status_code=500, detail="Supabase not configured")
    
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
    print(f"DEBUG: Creating new conversation in Supabase: {new_id}")
    try:
        response = supabase.table("conversations").insert(new_conv).execute()
        print(f"DEBUG: Supabase Insert Response -> {response}")
        return new_conv
    except Exception as e:
        import traceback
        error_msg = f"CREATE ERROR: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/conversations/{id}")
async def delete_conversation(id: str):
    if not supabase: raise HTTPException(status_code=500, detail="Supabase not configured")
    try:
        supabase.table("conversations").delete().eq("id", id).execute()
        return {"status": "success", "message": "Conversation deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def add_message_to_history(conv_id, role, content):
    if not conv_id or not supabase: return
    try:
        response = supabase.table("conversations").select("messages").eq("id", conv_id).execute()
        if response.data:
            messages = response.data[0]["messages"]
            messages.append({"role": role, "content": content})
            supabase.table("conversations").update({"messages": messages}).eq("id", conv_id).execute()
    except Exception as e:
        print(f"Error adding to history: {e}")

def update_conv_title(conv_id, title):
    if not conv_id or not supabase: return
    try:
        supabase.table("conversations").update({"title": title}).eq("id", conv_id).execute()
    except Exception as e:
        print(f"Error updating title: {e}")

@app.post("/chat")
async def chat(request: dict):
    """Answers questions and persists history."""
    query = request.get("query")
    conv_id = request.get("conversation_id")
    if not query:
        raise HTTPException(status_code=400, detail="Query is required")
    
    system_prompt = """
    You are a highly professional Document Analysis Assistant.
    Your job is to analyze the provided document content and answer the user's question accurately.

    -------------------------------------
    RESPONSE RULES (STRICT)
    -------------------------------------
    1. DO NOT include internal reasoning, chain-of-thought, or <think> tags.
    2. ALWAYS provide a clean, direct, and final answer with clear formatting.
    3. IF counting items/skills (e.g., tools, technologies), provide the TOTAL count and a categorized breakdown.
    4. IF the question is general (hi/role), respond politely.
    5. IF NO document AND NO link is uploaded to your system, you MUST append this exact sentence at the end: 
       "Please upload a document or provide a link to proceed."
    6. IF any document or link IS uploaded (even if it doesn't contain the specific answer), do NOT include the above message.
    7. DO NOT hallucinate. Only use provided content. If missing, say: "The document does not provide this information."

    Context (Strictly for this conversation):
    {context_text}
    """
    
    try:
        # Get history if available
        messages = [{"role": "system", "content": system_prompt}]
        if conv_id:
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

        # RAG: Search for relevant context (FILTERED BY CONV_ID)
        context_chunks = processor.search(query, conv_id=conv_id)
        context_text = "\n---\n".join(context_chunks)
        
        # Update system prompt with fresh context
        messages[0]["content"] = messages[0]["content"].replace("{context_text}", context_text)

        response = requests.post(
            "https://api.sambanova.ai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "Qwen3-32B",
                "messages": messages,
                "temperature": 0.1,
                "max_tokens": 1024
            }
        )
        
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"SambaNova API error: {response.text}")
        
        result = response.json()
        raw_response = result["choices"][0]["message"]["content"]
        clean_response = clean_llm_response(raw_response)
        
        # New Rule Persistence: Append reminder if no content in index for THIS CONV
        conv_resources = [r for r in processor.indexed_resources if r["conv_id"] == conv_id]
        if not conv_resources:
            if "Please upload a document or provide a link to proceed." not in clean_response:
                clean_response += "\n\nPlease upload a document or provide a link to proceed."
            
        # PERSIST: Update conversations.json
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
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
