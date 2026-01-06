import { ChatRequest, ChatResponse, SessionDataResponse } from "@/types";

const API_BASE_URL = "http://localhost:8000";

export async function fileToBase64(file: File): Promise<string> {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result as string);
        reader.onerror = reject;
        reader.readAsDataURL(file);
    });
}

export async function filesToBase64(files: File[]): Promise<string[]> {
    return Promise.all(files.map(fileToBase64));
}

export async function sendMessage(request: ChatRequest): Promise<ChatResponse> {
    const response = await fetch(`${API_BASE_URL}/chat`, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify(request),
    });

    if (!response.ok) {
        throw new Error(`API error: ${response.status}`)
    }

    return response.json();
}

export async function getSession(sessionId: string): Promise<SessionDataResponse | null> {
    const response = await fetch(`${API_BASE_URL}/session/${sessionId}`, {
        method: "GET",
        headers: {
            "Content-Type": "application/json",
        },
    });

    if (response.status === 404) {
        return null;
    }
    if (!response.ok) {
        throw new Error(`API error: ${response.status}`)
    }

    return response.json();
}   