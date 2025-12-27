"use client"

import { useState, useRef, useEffect } from "react";
import ChatMessage from "@/components/ChatMessage";
import ChatInput from "@/components/ChatInput";
import { Message, ChatResponse } from "@/types";
import { filesToBase64, fileToBase64, sendMessage } from "@/lib/api";


export default function Home() {
  const [modelImage, setModelImage] = useState<File | null>(null);
  const [modelImagePreview, setModelImagePreview] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const handleSetModelImage = (file: File | null) => {
    setModelImage(file);
    if (file) {
      setModelImagePreview(URL.createObjectURL(file));
    } else {
      setModelImagePreview(null);
    }
  }

  const handleSendMessage = async (content: string, images: File[]) => {
    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content: content,
      images: images.length > 0
        ? images.map((file, idx) => ({
          image_id: `local-${idx}`,
          url: URL.createObjectURL(file),
          bbox: null,
          type: "user_provided" as const,
        }))
        : undefined,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setIsLoading(true);

    try {
      const imageUrls = images.length > 0 ? await filesToBase64(images) : null;
      const modelImageBase64 = modelImage ? await fileToBase64(modelImage) : null;

      const response: ChatResponse = await sendMessage({
        query: content,
        session_id: sessionId,
        images: imageUrls,
        model_image: modelImageBase64,
      });

      if (response.session_id) {
        setSessionId(response.session_id);
      }

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: response.answer,
        images: response.images || undefined,
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (error) {
      console.error("Error sending message:", error);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading])

  return (
    <div className="flex h-screen flex-col bg-white dark:bg-gray-900">
      <header className="border-b border-gray-200 bg-white px-6 py-4 dark:border-gray-700 dark:bg-gray-900">
        <div className="flex items-center justify-between">
          <h1 className="text-xl font-semibold text-gray-900 dark:text-white">
            Fashion Recommender Agent
          </h1>

          <div className="flex items-center gap-3">
            <input
              type="file"
              id="model-image-input"
              accept="image/*"
              className="hidden"
              onChange={(e) => {
                const file = e.target.files?.[0] || null;
                handleSetModelImage(file);
              }}
            />

            {modelImagePreview ? (
              <div className="relative">
                <img
                  src={modelImagePreview}
                  alt="Your photo"
                  className="h-10 w-10 rounded-full object-cover border-2 border-blue-500"
                />
                <button
                  onClick={() => handleSetModelImage(null)}
                  className="absolute -top-1 -right-1 h-4 w-4 rounded-full bg-red-500 text-white text-xs"
                >
                  x
                </button>
              </div>
            ) : (
              <button
                onClick={() => document.getElementById('model-image-input')?.click()}
                className="flex items-center gap-2 px-3 py-2 rounded-lg bg-gray-100 hover:bg-gray-200 dark:bg-gray-800 dark:hover:bg-gray-700 text-sm"
              >
                🙋 Set My Photo
              </button>
            )}

          </div>
        </div>
      </header>
      <main className="flex-1 overflow-y-auto p-6">
        {messages.length === 0 ? (
          <div className="flex h-full items-center justify-center text-gray-500">
            <p>Start a conversation about fashion!</p>
          </div>
        ) : (
          <div className="mx-auto max-w-3xl">
            {messages.map((message) => (
              <ChatMessage key={message.id} message={message} />
            ))}
            {isLoading && (
              <div className="flex justify-start mb-4">
                <div className="bg-gray-100 rounded-2xl px-4 py-3 dark:bg-gray-800">
                  <p className="text-gray-500">Thinking...</p>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
        )}
      </main>
      <ChatInput onSendMessage={handleSendMessage} isLoading={isLoading} />
    </div>
  );
}