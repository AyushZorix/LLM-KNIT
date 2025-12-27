import { MockApiResponse } from '../types';

/**
 * Mock API service for chatbot responses
 * 
 * TODO: Replace this with actual LLM/ML API integration when "ADD ML PIPELINE" is triggered
 * This service simulates API calls with mock responses and delays
 */

// Mock responses pool for variety
const MOCK_RESPONSES = [
  "That's an interesting question! Let me think about that...",
  "I understand what you're asking. Here's my perspective on that topic.",
  "Great question! Based on what you've shared, I'd say that's a valid point.",
  "Thanks for asking! I'd be happy to help you think through this.",
  "That's a thoughtful observation. Let me provide some context on that.",
  "I see where you're coming from. Here's what I think might be helpful:",
  "Interesting point! There are a few ways to approach this...",
  "I appreciate you sharing that. My response would be...",
];

/**
 * Simulates an API call to get a chatbot response
 * @param userMessage - The user's input message
 * @returns Promise with a mock response
 * 
 * TODO: Replace with actual API endpoint when ML pipeline is integrated
 * Expected endpoint: POST /api/chat
 * Expected request: { message: string, history?: Message[] }
 * Expected response: { message: string, timestamp: string }
 */
export async function sendMessage(userMessage: string): Promise<MockApiResponse> {
  // Simulate network delay (500-1500ms)
  const delay = Math.random() * 1000 + 500;
  await new Promise(resolve => setTimeout(resolve, delay));

  // Generate a mock response
  const randomResponse = MOCK_RESPONSES[Math.floor(Math.random() * MOCK_RESPONSES.length)];
  
  // Sometimes include the user's message in the response for more realistic feel
  const response = userMessage.length > 20 
    ? `${randomResponse} You mentioned "${userMessage.substring(0, 30)}...". That's worth exploring further.`
    : randomResponse;

  return {
    message: response,
    timestamp: new Date().toISOString(),
  };
}

/**
 * Simulates checking if the API is available
 * @returns Promise<boolean>
 * 
 * TODO: Replace with actual health check endpoint when ML pipeline is integrated
 */
export async function checkApiHealth(): Promise<boolean> {
  // Simulate a quick health check
  await new Promise(resolve => setTimeout(resolve, 100));
  return true; // Always return true for mock
}

