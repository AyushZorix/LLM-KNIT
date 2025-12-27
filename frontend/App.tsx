import { useState, useEffect } from 'react';
import ChatContainer from './components/ChatContainer';
import Sidebar from './components/Sidebar';
import { 
  getAllConversations, 
  createConversation,
  getCurrentConversationId,
  setCurrentConversationId,
} from './services/conversationStorage';
import './App.css';

function App() {
  const [currentConversationId, setCurrentConversationIdState] = useState<string | null>(null);
  const [conversations, setConversations] = useState(getAllConversations());
  const [sidebarOpen, setSidebarOpen] = useState(true);

  // Initialize with current conversation or create new one
  useEffect(() => {
    const storedId = getCurrentConversationId();
    const allConversations = getAllConversations();
    
    if (storedId && allConversations.find(c => c.id === storedId)) {
      setCurrentConversationIdState(storedId);
    } else if (allConversations.length > 0) {
      setCurrentConversationIdState(allConversations[0].id);
      setCurrentConversationId(allConversations[0].id);
    } else {
      // Create first conversation
      const newConv = createConversation();
      setCurrentConversationIdState(newConv.id);
      setCurrentConversationId(newConv.id);
      setConversations([newConv]);
    }
  }, []);

  const handleConversationSelect = (conversationId: string) => {
    setCurrentConversationIdState(conversationId);
    setCurrentConversationId(conversationId);
    setConversations(getAllConversations());
  };

  const handleNewConversation = () => {
    const newConv = createConversation();
    setCurrentConversationIdState(newConv.id);
    setCurrentConversationId(newConv.id);
    setConversations(getAllConversations());
  };

  return (
    <div className="app">
      <Sidebar
        currentConversationId={currentConversationId}
        onConversationSelect={handleConversationSelect}
        onNewConversation={handleNewConversation}
        isOpen={sidebarOpen}
        onToggle={() => setSidebarOpen(!sidebarOpen)}
      />
      <ChatContainer
        conversationId={currentConversationId}
        onConversationUpdate={() => setConversations(getAllConversations())}
        sidebarOpen={sidebarOpen}
      />
    </div>
  );
}

export default App;
