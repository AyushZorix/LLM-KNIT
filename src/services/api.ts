import { Message } from '../types';

/**
 * Real API service for KNIT-LLM backend
 * Connects to the FastAPI backend running the dual-pipeline ML system
 */

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export interface ApiResponse {
  message: string;
  timestamp: string;
  metadata?: {
    selected_id?: number;
    confidence?: number;
    domain?: string;
    hyde_response?: string;
    num_candidates?: number;
    selection_metadata?: any;
  };
}

export interface ApiRequest {
  message: string;
  history?: Array<{
    role: string;
    content: string;
  }>;
}

/**
 * Send a message to the KNIT-LLM API
 * @param userMessage - The user's input message
 * @param history - Optional conversation history
 * @returns Promise with API response
 */
export async function sendMessage(
  userMessage: string,
  history?: Message[]
): Promise<ApiResponse> {
  try {
    // Convert Message[] to API format if provided
    const apiHistory = history?.map(msg => ({
      role: msg.role,
      content: msg.content,
    }));

    const requestBody: ApiRequest = {
      message: userMessage,
      ...(apiHistory && { history: apiHistory }),
    };

    const response = await fetch(`${API_BASE_URL}/api/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(requestBody),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
      throw new Error(errorData.detail || `API error: ${response.statusText}`);
    }

    const data: ApiResponse = await response.json();
    return data;
  } catch (error) {
    console.error('Error calling API:', error);
    throw error;
  }
}

/**
 * Check if the API is available
 * @returns Promise<boolean>
 */
export async function checkApiHealth(): Promise<boolean> {
  try {
    const response = await fetch(`${API_BASE_URL}/health`, {
      method: 'GET',
    });
    return response.ok;
  } catch (error) {
    console.error('Health check failed:', error);
    return false;
  }
}

