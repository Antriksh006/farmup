"use client"

import { useState, useRef, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Send, Mic, MicOff, Bot, User, Leaf } from "lucide-react"

interface Message {
    id: string
    type: "user" | "assistant"
    content: string
    timestamp: Date
}

export default function AssistantPage() {
    const [messages, setMessages] = useState<Message[]>([
        {
            id: "1",
            type: "assistant",
            content:
                "Hello! I'm your AI farming assistant. I can help you with crop recommendations, pest management, weather insights, and general farming questions. What would you like to know?",
            timestamp: new Date(),
        },
    ])
    const [inputMessage, setInputMessage] = useState("")
    const [isLoading, setIsLoading] = useState(false)
    const [isListening, setIsListening] = useState(false)
    const messagesEndRef = useRef<HTMLDivElement>(null)

    const quickPrompts = [
        "What crops should I plant this season?",
        "How do I treat aphid infestation?",
        "When should I irrigate my wheat crop?",
        "What's the best fertilizer for mustard?",
        "How to improve soil health?",
        "Market prices for chickpea",
    ]

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
    }

    useEffect(() => {
        scrollToBottom()
    }, [messages])

    const handleSendMessage = async (message?: string) => {
        const messageToSend = message || inputMessage.trim()
        if (!messageToSend) return

        const userMessage: Message = {
            id: Date.now().toString(),
            type: "user",
            content: messageToSend,
            timestamp: new Date(),
        }

        setMessages((prev) => [...prev, userMessage])
        setInputMessage("")
        setIsLoading(true)

        // Simulate AI response
        setTimeout(() => {
            const aiResponse = generateAIResponse(messageToSend)
            const assistantMessage: Message = {
                id: (Date.now() + 1).toString(),
                type: "assistant",
                content: aiResponse,
                timestamp: new Date(),
            }
            setMessages((prev) => [...prev, assistantMessage])
            setIsLoading(false)
        }, 1500)
    }

    const generateAIResponse = (userMessage: string): string => {
        const lowerMessage = userMessage.toLowerCase()

        if (lowerMessage.includes("crop") || lowerMessage.includes("plant")) {
            return "Based on your farm's soil type and current season, I recommend considering wheat, mustard, or chickpea. Wheat (HD-2967 variety) has a 92% suitability score for your region and high market demand. Would you like detailed planting guidelines for any specific crop?"
        }

        if (lowerMessage.includes("aphid") || lowerMessage.includes("pest")) {
            return "For aphid control, I recommend an integrated approach: 1) Spray neem oil solution (2-3ml per liter) early morning or evening, 2) Introduce beneficial insects like ladybugs, 3) Use yellow sticky traps. Monitor your crops daily and repeat neem treatment every 7-10 days if needed."
        }

        if (lowerMessage.includes("irrigat") || lowerMessage.includes("water")) {
            return "For wheat crops, irrigate when soil moisture drops below 50%. Best times are early morning (6-8 AM) or evening (6-8 PM). Your current soil moisture in Zone A is at 45%, so irrigation is recommended today. Apply 25-30mm of water per irrigation cycle."
        }

        if (lowerMessage.includes("fertilizer") || lowerMessage.includes("mustard")) {
            return "For mustard crops, use NPK fertilizer in 12:32:16 ratio. Apply 100kg/hectare as basal dose during sowing, followed by 50kg/hectare of urea after 30 days. Mustard also benefits from sulfur - apply 20kg/hectare of elemental sulfur for better oil content."
        }

        if (lowerMessage.includes("soil") || lowerMessage.includes("health")) {
            return "To improve soil health: 1) Add organic matter like compost or farmyard manure (5-10 tons/hectare), 2) Practice crop rotation with legumes, 3) Use cover crops during fallow periods, 4) Minimize tillage, 5) Test soil pH regularly and maintain 6.0-7.5 range."
        }

        if (lowerMessage.includes("price") || lowerMessage.includes("market")) {
            return "Current market prices in your region: Chickpea ₹5,200/quintal (+8% from last week), Wheat ₹2,150/quintal (stable), Mustard ₹5,800/quintal (+12%). Chickpea prices are trending upward due to export demand. Consider timing your sales accordingly."
        }

        return "I understand you're asking about farming practices. Could you be more specific about what aspect you'd like help with? I can assist with crop selection, pest management, irrigation scheduling, fertilizer recommendations, soil health, or market information."
    }

    const handleVoiceInput = () => {
        if (!isListening) {
            // Start voice recognition
            setIsListening(true)
            // Simulate voice input (in real app, use Web Speech API)
            setTimeout(() => {
                setInputMessage("What crops should I plant this season?")
                setIsListening(false)
            }, 2000)
        } else {
            setIsListening(false)
        }
    }

    return (
        <div className="min-h-screen bg-background flex flex-col">
            {/* Header */}
            <div className="bg-card border-b border-border p-4">
                <div className="max-w-4xl mx-auto flex items-center gap-4">
                    <div className="bg-primary rounded-full p-2">
                        <Bot className="h-6 w-6 text-primary-foreground" />
                    </div>
                    <div>
                        <h1 className="text-xl font-semibold">AI Farming Assistant</h1>
                        <p className="text-sm text-muted-foreground">24/7 expert guidance for your farm</p>
                    </div>
                    <Badge variant="secondary" className="ml-auto bg-green-100 text-green-800">
                        Online
                    </Badge>
                </div>
            </div>

            {/* Messages */}
            <div className="flex-1 overflow-y-auto">
                <div className="max-w-4xl mx-auto p-4 space-y-4">
                    {messages.map((message) => (
                        <div key={message.id} className={`flex gap-3 ${message.type === "user" ? "justify-end" : "justify-start"}`}>
                            {message.type === "assistant" && (
                                <div className="bg-primary rounded-full p-2 h-8 w-8 flex items-center justify-center">
                                    <Leaf className="h-4 w-4 text-primary-foreground" />
                                </div>
                            )}

                            <Card
                                className={`max-w-[80%] ${message.type === "user" ? "bg-primary text-primary-foreground" : "bg-card"}`}
                            >
                                <CardContent className="p-3">
                                    <p className="text-sm">{message.content}</p>
                                    <p
                                        className={`text-xs mt-2 ${
                                            message.type === "user" ? "text-primary-foreground/70" : "text-muted-foreground"
                                        }`}
                                    >
                                        {message.timestamp.toLocaleTimeString([], {
                                            hour: "2-digit",
                                            minute: "2-digit",
                                        })}
                                    </p>
                                </CardContent>
                            </Card>

                            {message.type === "user" && (
                                <div className="bg-muted rounded-full p-2 h-8 w-8 flex items-center justify-center">
                                    <User className="h-4 w-4" />
                                </div>
                            )}
                        </div>
                    ))}

                    {isLoading && (
                        <div className="flex gap-3 justify-start">
                            <div className="bg-primary rounded-full p-2 h-8 w-8 flex items-center justify-center">
                                <Leaf className="h-4 w-4 text-primary-foreground" />
                            </div>
                            <Card className="bg-card">
                                <CardContent className="p-3">
                                    <div className="flex items-center gap-2">
                                        <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-primary"></div>
                                        <span className="text-sm text-muted-foreground">AI is thinking...</span>
                                    </div>
                                </CardContent>
                            </Card>
                        </div>
                    )}

                    <div ref={messagesEndRef} />
                </div>
            </div>

            {/* Quick Prompts */}
            <div className="border-t border-border bg-muted/30 p-4">
                <div className="max-w-4xl mx-auto">
                    <p className="text-sm text-muted-foreground mb-3">Quick questions:</p>
                    <div className="flex flex-wrap gap-2">
                        {quickPrompts.map((prompt, index) => (
                            <Button
                                key={index}
                                variant="outline"
                                size="sm"
                                onClick={() => handleSendMessage(prompt)}
                                className="text-xs bg-background"
                            >
                                {prompt}
                            </Button>
                        ))}
                    </div>
                </div>
            </div>

            {/* Input */}
            <div className="border-t border-border bg-card p-4">
                <div className="max-w-4xl mx-auto flex gap-2">
                    <div className="flex-1 relative">
                        <Input
                            placeholder="Ask me anything about farming..."
                            value={inputMessage}
                            onChange={(e) => setInputMessage(e.target.value)}
                            onKeyPress={(e) => e.key === "Enter" && handleSendMessage()}
                            className="pr-12"
                        />
                        <Button
                            variant="ghost"
                            size="sm"
                            className={`absolute right-1 top-1/2 transform -translate-y-1/2 h-8 w-8 p-0 ${
                                isListening ? "text-red-500" : "text-muted-foreground"
                            }`}
                            onClick={handleVoiceInput}
                        >
                            {isListening ? <MicOff className="h-4 w-4" /> : <Mic className="h-4 w-4" />}
                        </Button>
                    </div>
                    <Button onClick={() => handleSendMessage()} disabled={!inputMessage.trim() || isLoading}>
                        <Send className="h-4 w-4" />
                    </Button>
                </div>
            </div>
        </div>
    )
}