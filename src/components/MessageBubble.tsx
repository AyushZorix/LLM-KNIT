import { Message } from '../types';
import './MessageBubble.css';

interface MessageBubbleProps {
  message: Message;
}

export default function MessageBubble({ message }: MessageBubbleProps) {
  const isUser = message.role === 'user';
  const timeString = message.timestamp.toLocaleTimeString([], { 
    hour: '2-digit', 
    minute: '2-digit' 
  });

  return (
    <div className={`message-bubble ${isUser ? 'message-user' : 'message-assistant'}`}>
      <div className="message-content-wrapper">
        <div className="message-content">
          {message.content}
        </div>
        <div className="message-timestamp">
          {timeString}
        </div>
      </div>
    </div>
  );
}

