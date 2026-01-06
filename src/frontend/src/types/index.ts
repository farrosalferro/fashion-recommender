export interface ChatRequest {
    query: string;
    session_id: string | null;
    images: string[] | null;
    model_image: string | null;
}

export interface ImageResult {
    image_id: string;
    url: string;
    bbox: [number, number, number, number] | null;
    type: "user_provided" | "retrieved" | "virtual_try_on";
}

export interface ChatResponse {
    answer: string;
    session_id: string;
    images: ImageResult[] | null;
}

export interface Message {
    id: string;
    role: "user" | "assistant";
    content: string;
    images?: ImageResult[];
    timestamp: Date;
}

export interface MessageHistory {
    role: "user" | "assistant";
    content: string;
    images: ImageResult[] | null;
}

export interface SessionDataResponse {
    session_id: string;
    messages: MessageHistory[];
    has_model_image: boolean;
}