import { useState, useEffect } from 'react';
import { Conversation } from '../types';
import { 
  getAllConversations, 
  deleteConversation,
} from '../services/conversationStorage';
import './Sidebar.css';

interface SidebarProps {
  currentConversationId: string | null;
  onConversationSelect: (conversationId: string) => void;
  onNewConversation: () => void;
  isOpen: boolean;
  onToggle: () => void;
}

export default function Sidebar({ 
  currentConversationId, 
  onConversationSelect,
  onNewConversation,
  isOpen,
  onToggle,
}: SidebarProps) {
  const [conversations, setConversations] = useState<Conversation[]>([]);

  // Load conversations from storage
  useEffect(() => {
    const loadConversations = () => {
      const loaded = getAllConversations();
      setConversations(loaded);
    };

    loadConversations();
    
    // Listen for storage changes (in case of updates from other tabs)
    const handleStorageChange = () => {
      loadConversations();
    };
    
    window.addEventListener('storage', handleStorageChange);
    return () => window.removeEventListener('storage', handleStorageChange);
  }, []);

  // Refresh conversations when current conversation changes
  useEffect(() => {
    const interval = setInterval(() => {
      setConversations(getAllConversations());
    }, 1000);
    return () => clearInterval(interval);
  }, [currentConversationId]);

  const handleNewChat = () => {
    // Let App handle the conversation creation to avoid duplicates
    onNewConversation();
  };

  const handleDeleteConversation = (e: React.MouseEvent, conversationId: string) => {
    e.stopPropagation();
    deleteConversation(conversationId);
    setConversations(getAllConversations());
    
    if (currentConversationId === conversationId) {
      const remaining = getAllConversations();
      if (remaining.length > 0) {
        onConversationSelect(remaining[0].id);
      } else {
        handleNewChat();
      }
    }
  };

  const formatDate = (date: Date) => {
    const now = new Date();
    const diff = now.getTime() - date.getTime();
    const days = Math.floor(diff / (1000 * 60 * 60 * 24));
    
    if (days === 0) return 'Today';
    if (days === 1) return 'Yesterday';
    if (days < 7) return `${days} days ago`;
    return date.toLocaleDateString();
  };

  return (
    <>
      <div className={`sidebar ${isOpen ? 'sidebar-open' : 'sidebar-closed'}`}>
        <div className="sidebar-header">
          <button 
            className="new-chat-button"
            onClick={handleNewChat}
            title="New Chat"
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <line x1="12" y1="5" x2="12" y2="19"></line>
              <line x1="5" y1="12" x2="19" y2="12"></line>
            </svg>
            <span>New Chat</span>
          </button>
        </div>

        <div className="sidebar-content">
          <div className="conversations-list">
            {conversations.length === 0 ? (
              <div className="empty-conversations">
                <p>No conversations yet</p>
                <p className="empty-hint">Start a new chat to begin</p>
              </div>
            ) : (
              conversations.map((conv) => (
                <div
                  key={conv.id}
                  className={`conversation-item ${
                    currentConversationId === conv.id ? 'active' : ''
                  }`}
                  onClick={() => onConversationSelect(conv.id)}
                  title={conv.title}
                >
                  <div className="conversation-content">
                    <div className="conversation-info">
                      <div className="conversation-title">{conv.title}</div>
                      <div className="conversation-meta">
                        {formatDate(conv.updatedAt)}
                      </div>
                    </div>
                  </div>
                  <button
                    className="delete-conversation-button"
                    onClick={(e) => handleDeleteConversation(e, conv.id)}
                    title="Delete conversation"
                  >
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <polyline points="3 6 5 6 21 6"></polyline>
                      <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
                    </svg>
                  </button>
                </div>
              ))
            )}
          </div>
        </div>
      </div>
      
      <button
        className={`sidebar-toggle ${!isOpen ? 'sidebar-closed-toggle' : ''}`}
        onClick={onToggle}
        title={isOpen ? 'Close sidebar' : 'Open sidebar'}
      >
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          {isOpen ? (
            <polyline points="9 18 15 12 9 6"></polyline>
          ) : (
            <polyline points="15 18 9 12 15 6"></polyline>
          )}
        </svg>
      </button>
    </>
  );
}

