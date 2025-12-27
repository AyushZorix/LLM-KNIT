import { Message } from '../types';
import MessageBubble from './MessageBubble';
import './MessageList.css';

interface MessageListProps {
  messages: Message[];
}

export default function MessageList({ messages }: MessageListProps) {
  if (messages.length === 0) {
    return (
      <div className="empty-state">
        <h2>Start a conversation</h2>
        <p>Send a message to begin chatting with the assistant.</p>
      </div>
    );
  }

  return (
    <div className="message-list">
      {messages.map((message) => (
        <MessageBubble key={message.id} message={message} />
      ))}
    </div>
  );
}

