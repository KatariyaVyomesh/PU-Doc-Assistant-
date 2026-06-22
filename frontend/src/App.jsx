import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import ReactMarkdown from 'react-markdown';
import { 
  Plus, 
  Trash2, 
  Send, 
  Paperclip, 
  CloudUpload, 
  FileText, 
  Link as LinkIcon, 
  History, 
  Settings,
  MoreVertical,
  ChevronDown
} from 'lucide-react';

function App() {
  const [conversations, setConversations] = useState([]);
  const [activeConversationId, setActiveConversationId] = useState(null);
  const [messages, setMessages] = useState([
    { role: 'ai', content: 'Hello! I am your AI Document Assistant. Upload PDF, DOCX, CSV, or a URL to get started!' }
  ]);
  const [input, setInput] = useState('');
  const [linkInput, setLinkInput] = useState('');
  const [files, setFiles] = useState([]);
  const [isUploading, setIsUploading] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [isTyping, setIsTyping] = useState(false);
  const [chatCache, setChatCache] = useState({}); // { [id]: { messages: [], files: [] } }
  
  // Responsive sidebar states
  const [leftSidebarOpen, setLeftSidebarOpen] = useState(false);
  const [rightSidebarOpen, setRightSidebarOpen] = useState(false);

  const chatEndRef = useRef(null);
  const fileInputRef = useRef(null);

  const scrollToBottom = () => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  // Sync resources and conversations on startup
  useEffect(() => {
    const fetchData = async () => {
      try {
        const convRes = await axios.get('/api/conversations');
        setConversations(convRes.data);
        if (convRes.data.length > 0) {
          handleSelectChat(convRes.data[0].id);
        } else {
          handleNewChat();
        }
      } catch (err) {
        console.error("Could not fetch data:", err);
      }
    };
    fetchData();
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, isTyping]);

  const fetchFiles = async (id) => {
    try {
      const res = await axios.get(`/api/files?conversation_id=${id}`);
      const resources = res.data.resources || [];
      setFiles(resources);
      setChatCache(prev => ({
        ...prev,
        [id]: { ...prev[id], files: resources }
      }));
    } catch (err) {
      console.error("Error fetching files:", err);
    }
  };

  const handleNewChat = async () => {
    try {
      const response = await axios.post('/api/conversations', { title: "New Chat" });
      const newChat = response.data;
      setConversations(prev => [newChat, ...prev]);
      setActiveConversationId(newChat.id);
      const initialMessages = [{ role: 'ai', content: 'This is a fresh workspace. Upload documents to this chat to get started!' }];
      setMessages(initialMessages);
      setFiles([]);
      setChatCache(prev => ({
        ...prev,
        [newChat.id]: { messages: initialMessages, files: [] }
      }));
      setLeftSidebarOpen(false);
    } catch (err) {
      console.error(err);
    }
  };

  const handleSelectChat = async (id) => {
    if (activeConversationId === id) return;
    
    setActiveConversationId(id);
    setLeftSidebarOpen(false);

    // OPTIMISTIC RENDER: Load from cache instantly if available
    if (chatCache[id]) {
      setMessages(chatCache[id].messages || []);
      setFiles(chatCache[id].files || []);
    }

    try {
      // PARALLEL FETCH: Load both history and files at once
      const [convRes, filesRes] = await Promise.all([
        axios.get(`/api/conversations/${id}`),
        axios.get(`/api/files?conversation_id=${id}`)
      ]);

      const updatedMessages = convRes.data?.messages?.length > 0 
        ? convRes.data.messages 
        : [{ role: 'ai', content: 'This is a fresh workspace. Upload documents to this chat to get started!' }];
      
      const updatedFiles = filesRes.data.resources || [];

      // Update State
      setMessages(updatedMessages);
      setFiles(updatedFiles);

      // Update Cache
      setChatCache(prev => ({
        ...prev,
        [id]: { messages: updatedMessages, files: updatedFiles }
      }));
    } catch (err) {
      console.error("Error selecting chat:", err);
    }
  };

  const handleDeleteConversation = async (e, id) => {
    e.stopPropagation();
    try {
      const response = await axios.delete(`/api/conversations/${id}`);
      if (response.data.status === 'success') {
        setConversations(prev => prev.filter(c => c.id !== id));
        setChatCache(prev => {
          const updated = { ...prev };
          delete updated[id];
          return updated;
        });
        if (activeConversationId === id) handleNewChat();
      }
    } catch (err) {
      console.error(err);
    }
  };

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file || !activeConversationId) return;

    setIsUploading(true);
    const formData = new FormData();
    formData.append('file', file);
    formData.append('conversation_id', activeConversationId);

    try {
      const response = await axios.post('/api/upload', formData, {
        onUploadProgress: (progressEvent) => {
          const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          setUploadProgress(percentCompleted);
        }
      });
      if (response.data.status === 'success') {
        fetchFiles(activeConversationId);
        const successMsg = { role: 'ai', content: `Success! I've processed **${response.data.filename}** for this conversation.` };
        setMessages(prev => {
          const updated = [...prev, successMsg];
          setChatCache(cache => ({
            ...cache,
            [activeConversationId]: { ...cache[activeConversationId], messages: updated }
          }));
          return updated;
        });
      }
    } catch (err) {
      console.error(err);
    }
    setIsUploading(false);
    setUploadProgress(0);
  };

  const handleLinkUpload = async () => {
    if (!linkInput.trim() || isUploading || !activeConversationId) return;
    setIsUploading(true);
    try {
      const response = await axios.post('/api/upload-link', { 
        url: linkInput,
        conversation_id: activeConversationId 
      });
      if (response.data.status === 'success') {
        const successMsg = { role: 'ai', content: `Success! I've indexed the link for this workspace: **${linkInput}**.` };
        
        // Update Local State via refresh or manual append
        fetchFiles(activeConversationId);
        setMessages(prev => {
          const updated = [...prev, successMsg];
          setChatCache(cache => ({
            ...cache,
            [activeConversationId]: { ...cache[activeConversationId], messages: updated }
          }));
          return updated;
        });
        setLinkInput('');
      }
    } catch (err) {
      console.error(err);
    }
    setIsUploading(false);
  };

  const handleDeleteResource = async (name) => {
    if (!activeConversationId) return;
    setIsDeleting(true);
    setUploadProgress(100);
    try {
      const encodedName = encodeURIComponent(name);
      const response = await axios.delete(`/api/files/${encodedName}?conversation_id=${activeConversationId}`);
      if (response.data.status === 'success') {
        const removeMsg = { role: 'ai', content: `Removed **${name}** from this chat's memory.` };
        setFiles(prev => {
          const updated = prev.filter(f => f.name !== name);
          setChatCache(cache => ({
            ...cache,
            [activeConversationId]: { ...cache[activeConversationId], files: updated }
          }));
          return updated;
        });
        setMessages(prev => {
          const updated = [...prev, removeMsg];
          setChatCache(cache => ({
            ...cache,
            [activeConversationId]: { ...cache[activeConversationId], messages: updated }
          }));
          return updated;
        });
      }
    } catch (err) {
      console.error(err);
    }
    setIsDeleting(false);
    setUploadProgress(0);
  };

  const handleSendMessage = async () => {
    if (!input.trim() || isTyping) return;
    const userMsg = { role: 'user', content: input };
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setIsTyping(true);

    try {
      const response = await axios.post('/api/chat', { 
        query: input,
        conversation_id: activeConversationId
      });
      const aiMsg = { role: 'ai', content: response.data.response };
      setMessages(prev => {
        const updated = [...prev, aiMsg];
        setChatCache(cache => ({
          ...cache,
          [activeConversationId]: { ...cache[activeConversationId], messages: updated }
        }));
        return updated;
      });
      const convRes = await axios.get('/api/conversations');
      setConversations(convRes.data);
    } catch (err) {
      console.error(err);
      setMessages(prev => [...prev, { role: 'ai', content: 'Oops! Something went wrong.' }]);
    }
    setIsTyping(false);
  };

  return (
    <div className="app-container">
      {/* Sidebar: History */}
      <aside className={`sidebar left-sidebar ${leftSidebarOpen ? 'open' : ''}`}>
        <div className="branding">
          <span>History</span>
          <ChevronDown size={18} strokeWidth={2.5} style={{ color: 'var(--text-muted)' }} />
        </div>
        
        <button className="gradient-btn" onClick={handleNewChat}>
          <Plus size={18} /> New Chat
        </button>

        <div className="item-container">
          {conversations.map((c) => (
            <div 
              key={c.id} 
              onClick={() => handleSelectChat(c.id)}
              className={`clickable-item ${activeConversationId === c.id ? 'active' : ''}`}
            >
              <div className="content">
                <div className="title">{c.title}</div>
                <div className="subtitle">{c.date} <History size={10} style={{display:'inline', marginLeft: '4px'}}/></div>
              </div>
              <button 
                className="icon-btn-small"
                onClick={(e) => handleDeleteConversation(e, c.id)}
              >
                <Trash2 size={14} />
              </button>
            </div>
          ))}
        </div>
      </aside>

      {/* Main Content Area */}
      <main className="main-chat">
        <header className="chat-header">
          Doc-Intel Hub
        </header>

        <section className="chat-messages">
          {messages.map((m, i) => (
            <div key={i} className={`message ${m.role}`}>
              <ReactMarkdown>{m.content}</ReactMarkdown>
            </div>
          ))}
          {isTyping && (
            <div className="message ai" style={{fontStyle: 'italic', opacity: 0.7}}>
              AI is thinking...
            </div>
          )}
          <div ref={chatEndRef} />
        </section>

        <div className="chat-controls">
          <div className="input-pill">
            <input 
              type="text" 
              placeholder="Ask anything..." 
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleSendMessage()}
            />
            <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
              <button className="icon-btn-small" onClick={() => fileInputRef.current.click()}>
                <Paperclip size={20} />
              </button>
              <button className="icon-btn-small" onClick={handleSendMessage} style={{ color: 'white' }}>
                <Send size={20} fill="currentColor" />
              </button>
            </div>
          </div>
        </div>
      </main>

      {/* Sidebar: Resources */}
      <aside className={`sidebar right-sidebar ${rightSidebarOpen ? 'open' : ''}`}>
        <h2>Resources</h2>
        
        <div className="item-container">
          {files.map((f, i) => (
            <div key={i} className="clickable-item">
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', overflow: 'hidden' }}>
                {f.type === 'file' ? <FileText size={18} /> : <LinkIcon size={18} />}
                <div className="title" style={{ fontSize: '0.85rem' }}>{f.name}</div>
              </div>
              <button className="icon-btn-small" onClick={() => handleDeleteResource(f.name)}>
                <Trash2 size={14} />
              </button>
            </div>
          ))}
        </div>

        {(isUploading || isDeleting) && (
          <div className="status-progress-container">
            <div className="status-label">
              {isDeleting ? 'Deleting...' : 'Processing...'} {uploadProgress}%
            </div>
            <div className="progress-track">
              <div className="progress-fill" style={{ width: `${uploadProgress}%` }}></div>
            </div>
          </div>
        )}

        <div className="url-input-section">
          <input 
            type="text" 
            placeholder="Add URL" 
            className="modern-url-bar"
            value={linkInput}
            onChange={(e) => setLinkInput(e.target.value)}
          />
          <button className="square-gradient-btn" onClick={handleLinkUpload}>
            <Plus size={20} strokeWidth={3} />
          </button>
        </div>

        <div className="modern-upload-zone" onClick={() => fileInputRef.current.click()}>
          <CloudUpload size={32} />
          <div style={{ textAlign: 'center' }}>
            <div className="upload-title">Upload Files</div>
            <div className="upload-subtitle">drag & drop or click</div>
          </div>
          <input 
            type="file" 
            ref={fileInputRef} 
            onChange={handleFileUpload} 
            style={{ display: 'none' }} 
            accept=".pdf,.txt,.docx,.csv" 
          />
        </div>
      </aside>
    </div>
  );
}

export default App;
