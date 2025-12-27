import { useState, useCallback, useRef, useEffect } from 'react';
import { Message, Conversation } from '../types';
import { sendMessage } from '../services/api';
import { 
  getAllConversations, 
  saveConversation,
  updateConversationTitle,
} from '../services/conversationStorage';
import MessageList from './MessageList';
import MessageInput from './MessageInput';
import TypingIndicator from './TypingIndicator';
import './ChatContainer.css';

interface ChatContainerProps {
  conversationId: string | null;
  onConversationUpdate: () => void;
  sidebarOpen?: boolean;
}

export default function ChatContainer({ conversationId, onConversationUpdate, sidebarOpen = true }: ChatContainerProps) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const isFirstMessageRef = useRef(true);

  // Load conversation when conversationId changes
  useEffect(() => {
    if (!conversationId) {
      setMessages([]);
      return;
    }

    const conversations = getAllConversations();
    const conversation = conversations.find(c => c.id === conversationId);
    
    if (conversation) {
      setMessages(conversation.messages);
      isFirstMessageRef.current = conversation.messages.length === 0;
    } else {
      setMessages([]);
      isFirstMessageRef.current = true;
    }
  }, [conversationId]);

  // Auto-scroll to bottom when new messages arrive
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSendMessage = useCallback(async (content: string) => {
    if (!content.trim() || isLoading || !conversationId) return;

    // Add user message
    const userMessage: Message = {
      id: `user-${Date.now()}`,
      content: content.trim(),
      role: 'user',
      timestamp: new Date(),
    };

    const updatedMessages = [...messages, userMessage];
    setMessages(updatedMessages);
    setIsLoading(true);
    setError(null);

    // Update conversation title if this is the first message
    if (isFirstMessageRef.current) {
      updateConversationTitle(conversationId, content.trim());
      isFirstMessageRef.current = false;
      onConversationUpdate();
    }

    try {
      // Call the real ML pipeline API
      const response = await sendMessage(content, updatedMessages);

      // Add assistant response
      const assistantMessage: Message = {
        id: `assistant-${Date.now()}`,
        content: response.message,
        role: 'assistant',
        timestamp: new Date(response.timestamp),
      };

      const finalMessages = [...updatedMessages, assistantMessage];
      setMessages(finalMessages);

      // Save conversation to storage
      const conversations = getAllConversations();
      const conversation = conversations.find(c => c.id === conversationId);
      if (conversation) {
        conversation.messages = finalMessages;
        conversation.updatedAt = new Date();
        saveConversation(conversation);
        onConversationUpdate();
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to get response';
      setError(errorMessage);
      
      // Add error message to chat
      const errorMsg: Message = {
        id: `error-${Date.now()}`,
        content: `Sorry, I encountered an error: ${errorMessage}`,
        role: 'assistant',
        timestamp: new Date(),
      };
      
      const finalMessages = [...updatedMessages, errorMsg];
      setMessages(finalMessages);

      // Save conversation even with error
      const conversations = getAllConversations();
      const conversation = conversations.find(c => c.id === conversationId);
      if (conversation) {
        conversation.messages = finalMessages;
        conversation.updatedAt = new Date();
        saveConversation(conversation);
        onConversationUpdate();
      }
    } finally {
      setIsLoading(false);
    }
  }, [isLoading, messages, conversationId, onConversationUpdate]);

  const handleClearChat = useCallback(() => {
    if (!conversationId) return;
    
    setMessages([]);
    setError(null);
    isFirstMessageRef.current = true;

    // Update conversation in storage
    const conversations = getAllConversations();
    const conversation = conversations.find(c => c.id === conversationId);
    if (conversation) {
      conversation.messages = [];
      conversation.title = 'New Chat';
      conversation.updatedAt = new Date();
      saveConversation(conversation);
      onConversationUpdate();
    }
  }, [conversationId, onConversationUpdate]);

  return (
    <div className={`chat-container ${!sidebarOpen ? 'sidebar-closed' : ''}`}>
      <div className="chat-content">
        <div className="chat-header">
          <h1>KNIT-LLM</h1>
          <button 
            type="button"
            className="clear-button" 
            onClick={handleClearChat}
            disabled={messages.length === 0 || !conversationId}
          >
            Clear Chat
          </button>
        </div>
        
        <div className="chat-messages-wrapper">
          <MessageList messages={messages} />
          {isLoading && <TypingIndicator />}
          <div ref={messagesEndRef} />
        </div>

        {error && (
          <div className="error-banner">
            {error}
          </div>
        )}

        <MessageInput 
          onSendMessage={handleSendMessage} 
          disabled={isLoading}
        />
      </div>
    </div>
  );
}

