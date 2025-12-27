import './TypingIndicator.css';

export default function TypingIndicator() {
  return (
    <div className="typing-indicator">
      <div className="typing-content">
        <div className="typing-label">Thinking…</div>
        <div className="typing-dots">
          <span></span>
          <span></span>
          <span></span>
        </div>
      </div>
    </div>
  );
}

