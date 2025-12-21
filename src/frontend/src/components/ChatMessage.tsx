import { Message } from "@/types";

interface ChatMessageProps {
    message: Message;
}

export default function ChatMessage({ message }: ChatMessageProps) {
    const isUser = message.role === "user";

    return (
        <div className={`flex ${isUser ? "justify-end" : "justify-start"} mb-4`}>
            <div
                className={`max-w-[80%] rounded-2xl px-4 py-3 ${isUser
                    ? "bg-blue-600 text-white"
                    : "bg-gray-100 text-gray-900 dark:bg-gray-800 dark:text-gray-100"
                    }`}
            >
                <p className="whitespace-pre-wrap">{message.content}</p>
                {message.images && message.images.length > 0 && (
                    <div className="mt-3 flex flex-wrap gap-2">
                        {message.images.map((img) => (
                            <img
                                key={img.image_id}
                                src={img.url}
                                alt={`${img.type} image`}
                                className="h-24 w-24 rounded-lg object-cover"
                            />
                        ))}
                    </div>
                )}
            </div>
        </div>
    );
}