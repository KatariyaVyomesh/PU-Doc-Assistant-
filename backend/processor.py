import os
import numpy as np
import faiss
import json
from huggingface_hub import InferenceClient
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

# LangChain Imports
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader, Docx2txtLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

class DocumentProcessor:
    def __init__(self):
        # HF API Settings
        self.hf_api_key = os.getenv("HUGGINGFACE_API_KEY", "")
        self.model_id = "sentence-transformers/all-MiniLM-L6-v2"
        self.client = InferenceClient(api_key=self.hf_api_key)
        
        # Supabase Settings
        self.supabase_url = os.getenv("SUPABASE_URL", "")
        self.supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
        self.supabase: Client = None
        if self.supabase_url and self.supabase_key:
            self.supabase = create_client(self.supabase_url, self.supabase_key)

        self.index = None
        self.chunks = [] # Stores: {"text": str, "conv_id": str}
        self.indexed_resources = [] # Stores: {"name": str, "type": str, "conv_id": str}
        self.links_file = "links.json"
        self.resources_file = "resources.json"
        
        # Smarter text splitting that respects sentence/paragraph boundaries
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len,
        )

    def load_document(self, file_path):
        """Loads text from various file types using LangChain loaders."""
        ext = os.path.splitext(file_path)[1].lower()
        loader = None
        
        if ext == '.pdf':
            loader = PyPDFLoader(file_path)
        elif ext == '.txt':
            loader = TextLoader(file_path, encoding='utf-8')
        elif ext == '.docx':
            loader = Docx2txtLoader(file_path)
        elif ext == '.csv':
            loader = CSVLoader(file_path)
        else:
            return ""

        if loader:
            docs = loader.load()
            return "\n\n".join([doc.page_content for doc in docs])
        return ""

    def scrape_url(self, url):
        """Scrapes text from a given URL using LangChain's WebBaseLoader."""
        try:
            loader = WebBaseLoader(url)
            docs = loader.load()
            return "\n\n".join([doc.page_content for doc in docs])
        except Exception as e:
            return f"Error scraping URL: {str(e)}"

    def _get_embeddings(self, texts):
        """Calls Hugging Face Inference API to get embeddings using InferenceClient."""
        if not self.hf_api_key:
            print("WARNING: HUGGINGFACE_API_KEY is missing!")
            return np.zeros((len(texts), 384))

        try:
            # InferenceClient handles batching and task-specific URLs
            response = self.client.feature_extraction(
                model=self.model_id,
                text=texts
            )
            # Response is a numpy-like list of embeddings
            return np.array(response)
        except Exception as e:
            print(f"Error calling HF API via InferenceClient: {e}")
            return np.zeros((len(texts), 384))

    def process_content(self, content_id, content_text, conv_id, is_link=False, skip_db=False):
        """Processes raw text from any source and associates it with a specific conversation."""
        if not content_text.strip() or not conv_id:
            return 0
            
        new_text_chunks = self.text_splitter.split_text(content_text)
        
        # Track chunks with their conversation ID
        for text in new_text_chunks:
            self.chunks.append({"text": text, "conv_id": conv_id})
        
        # Track resource with its conversation ID
        if not any(r["name"] == content_id and r["conv_id"] == conv_id for r in self.indexed_resources):
            self.indexed_resources.append({
                "name": content_id,
                "type": "link" if is_link else "file",
                "conv_id": conv_id
            })
        
        # Persistence: Save link
        if is_link and not skip_db:
            self._save_link(content_id, conv_id)
        
        # Update Vector Index via HF API
        embeddings = self._get_embeddings(new_text_chunks)
        dimension = embeddings.shape[1]
        
        if self.index is None:
            self.index = faiss.IndexFlatL2(dimension)
            
        self.index.add(np.array(embeddings).astype('float32'))
        return len(new_text_chunks)

    def _save_link(self, url, conv_id):
        """Saves a URL to links.json for persistence with conv_id."""
        if self.supabase:
            try:
                self.supabase.table("resources").insert({
                    "name": url,
                    "type": "link",
                    "conv_id": conv_id
                }).execute()
            except Exception as e:
                print(f"Error saving link to Supabase: {e}")
        
        # Local fallback
        links = []
        if os.path.exists(self.links_file):
            try:
                with open(self.links_file, 'r') as f:
                    links = json.load(f)
            except:
                links = []
        
        exists = any(isinstance(l, dict) and l.get('url') == url and l.get('conv_id') == conv_id for l in links)
        if not exists:
            links.append({"url": url, "conv_id": conv_id})
            with open(self.links_file, 'w') as f:
                json.dump(links, f)

    def upload_file_to_supabase(self, file_path, filename, conv_id):
        """Uploads file to Supabase Storage and records metadata in Database."""
        if not self.supabase:
            return False
            
        try:
            # 1. Upload to Storage
            with open(file_path, 'rb') as f:
                storage_path = f"{conv_id}/{filename}"
                self.supabase.storage.from_("document-assistant").upload(
                    path=storage_path,
                    file=f,
                    file_options={"upsert": "true"}
                )
            
            # 2. Add to Database Table
            self.supabase.table("resources").insert({
                "name": filename,
                "type": "file",
                "conv_id": conv_id
            }).execute()
            
            return True
        except Exception as e:
            print(f"Supabase Upload Error: {e}")
            return False

    def reindex_all(self, upload_dir="uploads"):
        """Re-indexes resources from Supabase for local parity."""
        if not self.supabase:
            print("Supabase not set up, using local folders instead.")
            # FALLBACK: Keep original local folder logic for offline dev
            self.chunks = []
            self.index = None
            self.indexed_resources = []
            if os.path.exists(upload_dir):
                for root, dirs, files in os.walk(upload_dir):
                    for filename in files:
                        file_path = os.path.join(root, filename)
                        rel_path = os.path.relpath(root, upload_dir)
                        cid = "global" if rel_path == "." else rel_path
                        text = self.load_document(file_path)
                        if text:
                            self.process_content(filename, text, cid, is_link=False, skip_db=True)
            return

        print("Re-indexing resources from Supabase...")
        self.chunks = []
        self.index = None
        self.indexed_resources = []
        
        try:
            response = self.supabase.table("resources").select("*").execute()
            resources = response.data
            
            for res in resources:
                name = res["name"]
                type = res["type"]
                cid = res["conv_id"]
                
                if type == "link":
                    text = self.scrape_url(name)
                    if not text.startswith("Error"):
                        self.process_content(name, text, cid, is_link=True, skip_db=True)
                else:
                    storage_path = f"{cid}/{name}"
                    tmp_path = os.path.join("uploads", name)
                    os.makedirs("uploads", exist_ok=True)
                    try:
                        data = self.supabase.storage.from_("document-assistant").download(storage_path)
                        with open(tmp_path, "wb") as f:
                            f.write(data)
                        
                        text = self.load_document(tmp_path)
                        if text:
                            self.process_content(name, text, cid, is_link=False, skip_db=True)
                        
                        if os.path.exists(tmp_path): os.remove(tmp_path)
                    except Exception as fe:
                        print(f"Error processing storage file {name}: {fe}")
        except Exception as e:
            print(f"Supabase Sync Error: {e}")

    def remove_resource(self, name, conv_id, upload_dir="uploads"):
        """Removes resource from Supabase and local folder."""
        # 1. Local cleanup
        file_path = os.path.join(upload_dir, conv_id, name)
        if os.path.exists(file_path): os.remove(file_path)

        # 2. Supabase cleanup
        if self.supabase:
            try:
                self.supabase.table("resources").delete().match({"name": name, "conv_id": conv_id}).execute()
                storage_path = f"{conv_id}/{name}"
                try:
                    self.supabase.storage.from_("document-assistant").remove([storage_path])
                except:
                    pass
            except Exception as e:
                print(f"Supabase Deletion Error: {e}")
        
        self.reindex_all(upload_dir)
        return True

    def search(self, query, conv_id, top_k=5):
        """Finds the most relevant chunks specifically for the given conversation."""
        if self.index is None or not self.chunks or not conv_id:
            return []
            
        query_embedding = self._get_embeddings([query])
        # We search more than top_k because we'll filter by conv_id
        distances, indices = self.index.search(np.array(query_embedding).astype('float32'), 50)
        
        results = []
        for i in indices[0]:
            if i != -1 and i < len(self.chunks):
                chunk = self.chunks[i]
                # FILTER: Only include if it belongs to this conversation OR is global
                if chunk["conv_id"] == conv_id:
                    results.append(chunk["text"])
                    if len(results) >= top_k:
                        break
        return results

# Shared instance
processor = DocumentProcessor()
