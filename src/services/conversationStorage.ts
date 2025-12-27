import { Conversation, Message } from '../types';

const STORAGE_KEY = 'lknit-conversations';
const CURRENT_CONVERSATION_KEY = 'lknit-current-conversation-id';

/**
 * Conversation storage service using localStorage
 * 
 * TODO: Replace with backend API when ML pipeline is integrated
 * This service manages conversation persistence on the client side
 */

/**
 * Get all conversations from localStorage
 */
export function getAllConversations(): Conversation[] {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (!stored) return [];
    
    const conversations = JSON.parse(stored);
    // Convert date strings back to Date objects
    return conversations.map((conv: any) => ({
      ...conv,
      createdAt: new Date(conv.createdAt),
      updatedAt: new Date(conv.updatedAt),
      messages: conv.messages.map((msg: any) => ({
        ...msg,
        timestamp: new Date(msg.timestamp),
      })),
    }));
  } catch (error) {
    console.error('Error loading conversations:', error);
    return [];
  }
}

/**
 * Save a conversation to localStorage
 */
export function saveConversation(conversation: Conversation): void {
  try {
    const conversations = getAllConversations();
    const index = conversations.findIndex(c => c.id === conversation.id);
    
    if (index >= 0) {
      conversations[index] = conversation;
    } else {
      conversations.push(conversation);
    }
    
    // Sort by updatedAt (most recent first)
    conversations.sort((a, b) => b.updatedAt.getTime() - a.updatedAt.getTime());
    
    localStorage.setItem(STORAGE_KEY, JSON.stringify(conversations));
  } catch (error) {
    console.error('Error saving conversation:', error);
  }
}

/**
 * Delete a conversation
 */
export function deleteConversation(conversationId: string): void {
  try {
    const conversations = getAllConversations();
    const filtered = conversations.filter(c => c.id !== conversationId);
    localStorage.setItem(STORAGE_KEY, JSON.stringify(filtered));
  } catch (error) {
    console.error('Error deleting conversation:', error);
  }
}

/**
 * Create a new conversation
 */
export function createConversation(title?: string): Conversation {
  const now = new Date();
  const conversation: Conversation = {
    id: `conv-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
    title: title || 'New Chat',
    messages: [],
    createdAt: now,
    updatedAt: now,
  };
  
  saveConversation(conversation);
  return conversation;
}

/**
 * Update conversation title based on first user message
 */
export function updateConversationTitle(conversationId: string, firstMessage: string): void {
  const conversations = getAllConversations();
  const conversation = conversations.find(c => c.id === conversationId);
  
  if (conversation && conversation.title === 'New Chat') {
    // Generate title from first message (first 50 chars)
    const title = firstMessage.length > 50 
      ? firstMessage.substring(0, 50) + '...'
      : firstMessage;
    
    conversation.title = title;
    conversation.updatedAt = new Date();
    saveConversation(conversation);
  }
}

/**
 * Get current conversation ID from localStorage
 */
export function getCurrentConversationId(): string | null {
  return localStorage.getItem(CURRENT_CONVERSATION_KEY);
}

/**
 * Set current conversation ID
 */
export function setCurrentConversationId(conversationId: string | null): void {
  if (conversationId) {
    localStorage.setItem(CURRENT_CONVERSATION_KEY, conversationId);
  } else {
    localStorage.removeItem(CURRENT_CONVERSATION_KEY);
  }
}

