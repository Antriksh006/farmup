"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { Card, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Plus, Heart, MessageCircle, Share2, Camera, MapPin, Clock } from "lucide-react"

interface Post {
    id: string
    author: {
        name: string
        avatar: string
        location: string
        verified: boolean
    }
    content: string
    image?: string
    timestamp: string
    likes: number
    comments: number
    isLiked: boolean
    category: "question" | "tip" | "success" | "problem"
}

interface Comment {
    id: string
    author: string
    content: string
    timestamp: string
    helpful: boolean
}

export default function CommunityPage() {
    const [activeTab, setActiveTab] = useState("feed")
    const [newPostContent, setNewPostContent] = useState("")
    const [newPostCategory, setNewPostCategory] = useState<Post["category"]>("question")
    const [showCreatePost, setShowCreatePost] = useState(false)

    const [posts, setPosts] = useState<Post[]>([
        {
            id: "1",
            author: {
                name: "Rajesh Kumar",
                avatar: "/farmer-avatar-1.jpg",
                location: "Ludhiana, Punjab",
                verified: true,
            },
            content:
                "Just harvested my wheat crop and got 45 quintals per hectare! Used the NPK fertilizer recommended by the AI assistant. Very happy with the results. What yields are you getting this season?",
            image: "/wheat-harvest-field.jpg",
            timestamp: "2 hours ago",
            likes: 24,
            comments: 8,
            isLiked: false,
            category: "success",
        },
        {
            id: "2",
            author: {
                name: "Priya Singh",
                avatar: "/farmer-avatar-2.jpg",
                location: "Karnal, Haryana",
                verified: false,
            },
            content:
                "Has anyone dealt with aphid infestation on mustard crops? I'm seeing small green insects on the leaves. Need urgent advice on organic treatment methods.",
            timestamp: "4 hours ago",
            likes: 12,
            comments: 15,
            isLiked: true,
            category: "problem",
        },
        {
            id: "3",
            author: {
                name: "Suresh Patel",
                avatar: "/farmer-avatar-3.jpg",
                location: "Patiala, Punjab",
                verified: true,
            },
            content:
                "Pro tip: Always test your soil pH before applying fertilizers. I learned this the hard way after poor yields last season. Now I test every 6 months and adjust accordingly. pH should be between 6.0-7.5 for most crops.",
            timestamp: "1 day ago",
            likes: 45,
            comments: 12,
            isLiked: false,
            category: "tip",
        },
        {
            id: "4",
            author: {
                name: "Meera Devi",
                avatar: "/farmer-avatar-4.jpg",
                location: "Amritsar, Punjab",
                verified: false,
            },
            content:
                "What's the best time to plant chickpea in our region? I'm planning to start next week but want to make sure about the timing. Also, which variety gives better yield - Kabuli or Desi?",
            timestamp: "2 days ago",
            likes: 8,
            comments: 6,
            isLiked: false,
            category: "question",
        },
    ])

    const handleLikePost = (postId: string) => {
        setPosts((prev) =>
            prev.map((post) =>
                post.id === postId
                    ? { ...post, isLiked: !post.isLiked, likes: post.isLiked ? post.likes - 1 : post.likes + 1 }
                    : post,
            ),
        )
    }

    const handleCreatePost = () => {
        if (!newPostContent.trim()) return

        const newPost: Post = {
            id: Date.now().toString(),
            author: {
                name: "You",
                avatar: "/your-avatar.jpg",
                location: "Your Location",
                verified: false,
            },
            content: newPostContent,
            timestamp: "Just now",
            likes: 0,
            comments: 0,
            isLiked: false,
            category: newPostCategory,
        }

        setPosts((prev) => [newPost, ...prev])
        setNewPostContent("")
        setShowCreatePost(false)
    }

    const getCategoryColor = (category: Post["category"]) => {
        switch (category) {
            case "question":
                return "bg-blue-100 text-blue-800"
            case "tip":
                return "bg-green-100 text-green-800"
            case "success":
                return "bg-yellow-100 text-yellow-800"
            case "problem":
                return "bg-red-100 text-red-800"
            default:
                return "bg-gray-100 text-gray-800"
        }
    }

    const getCategoryIcon = (category: Post["category"]) => {
        switch (category) {
            case "question":
                return "❓"
            case "tip":
                return "💡"
            case "success":
                return "🎉"
            case "problem":
                return "⚠️"
            default:
                return "📝"
        }
    }

    return (
        <div className="min-h-screen bg-background">
            {/* Header */}
            <div className="bg-card border-b border-border p-4">
                <div className="max-w-4xl mx-auto flex items-center justify-between">
                    <div>
                        <h1 className="text-2xl font-bold">Farming Community</h1>
                        <p className="text-muted-foreground">Connect, share, and learn from fellow farmers</p>
                    </div>

                    <Dialog open={showCreatePost} onOpenChange={setShowCreatePost}>
                        <DialogTrigger asChild>
                            <Button>
                                <Plus className="h-4 w-4 mr-2" />
                                Create Post
                            </Button>
                        </DialogTrigger>
                        <DialogContent className="max-w-2xl">
                            <DialogHeader>
                                <DialogTitle>Share with the Community</DialogTitle>
                            </DialogHeader>
                            <div className="space-y-4">
                                <div className="flex gap-2">
                                    {(["question", "tip", "success", "problem"] as const).map((category) => (
                                        <Button
                                            key={category}
                                            variant={newPostCategory === category ? "default" : "outline"}
                                            size="sm"
                                            onClick={() => setNewPostCategory(category)}
                                        >
                                            {getCategoryIcon(category)} {category.charAt(0).toUpperCase() + category.slice(1)}
                                        </Button>
                                    ))}
                                </div>

                                <Textarea
                                    placeholder="What's on your mind? Share your farming experience, ask questions, or give advice..."
                                    value={newPostContent}
                                    onChange={(e) => setNewPostContent(e.target.value)}
                                    rows={4}
                                />

                                <div className="flex items-center gap-2">
                                    <Button variant="outline" size="sm">
                                        <Camera className="h-4 w-4 mr-2" />
                                        Add Photo
                                    </Button>
                                    <Button variant="outline" size="sm">
                                        <MapPin className="h-4 w-4 mr-2" />
                                        Add Location
                                    </Button>
                                </div>

                                <div className="flex gap-2">
                                    <Button variant="outline" onClick={() => setShowCreatePost(false)} className="flex-1">
                                        Cancel
                                    </Button>
                                    <Button onClick={handleCreatePost} disabled={!newPostContent.trim()} className="flex-1">
                                        Share Post
                                    </Button>
                                </div>
                            </div>
                        </DialogContent>
                    </Dialog>
                </div>
            </div>

            <div className="max-w-4xl mx-auto p-6">
                <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
                    <TabsList className="grid w-full grid-cols-4">
                        <TabsTrigger value="feed">All Posts</TabsTrigger>
                        <TabsTrigger value="questions">Questions</TabsTrigger>
                        <TabsTrigger value="tips">Tips</TabsTrigger>
                        <TabsTrigger value="success">Success Stories</TabsTrigger>
                    </TabsList>

                    <TabsContent value="feed" className="space-y-6">
                        {posts.map((post) => (
                            <Card key={post.id} className="overflow-hidden">
                                <CardContent className="p-6">
                                    {/* Post Header */}
                                    <div className="flex items-start gap-4 mb-4">
                                        <Avatar>
                                            <AvatarImage src={post.author.avatar || "/placeholder.svg"} />
                                            <AvatarFallback>{post.author.name.charAt(0)}</AvatarFallback>
                                        </Avatar>

                                        <div className="flex-1">
                                            <div className="flex items-center gap-2">
                                                <h3 className="font-semibold">{post.author.name}</h3>
                                                {post.author.verified && (
                                                    <Badge variant="secondary" className="bg-green-100 text-green-800 text-xs">
                                                        Verified
                                                    </Badge>
                                                )}
                                                <Badge className={`text-xs ${getCategoryColor(post.category)}`}>
                                                    {getCategoryIcon(post.category)} {post.category}
                                                </Badge>
                                            </div>
                                            <div className="flex items-center gap-2 text-sm text-muted-foreground">
                                                <MapPin className="h-3 w-3" />
                                                <span>{post.author.location}</span>
                                                <span>•</span>
                                                <Clock className="h-3 w-3" />
                                                <span>{post.timestamp}</span>
                                            </div>
                                        </div>
                                    </div>

                                    {/* Post Content */}
                                    <div className="mb-4">
                                        <p className="text-sm leading-relaxed">{post.content}</p>
                                        {post.image && (
                                            <img
                                                src={post.image || "/placeholder.svg"}
                                                alt="Post image"
                                                className="mt-3 rounded-lg w-full max-h-64 object-cover"
                                            />
                                        )}
                                    </div>

                                    {/* Post Actions */}
                                    <div className="flex items-center gap-6 pt-3 border-t border-border">
                                        <Button
                                            variant="ghost"
                                            size="sm"
                                            onClick={() => handleLikePost(post.id)}
                                            className={`gap-2 ${post.isLiked ? "text-red-500" : "text-muted-foreground"}`}
                                        >
                                            <Heart className={`h-4 w-4 ${post.isLiked ? "fill-current" : ""}`} />
                                            {post.likes} Helpful
                                        </Button>

                                        <Button variant="ghost" size="sm" className="gap-2 text-muted-foreground">
                                            <MessageCircle className="h-4 w-4" />
                                            {post.comments} Comments
                                        </Button>

                                        <Button variant="ghost" size="sm" className="gap-2 text-muted-foreground">
                                            <Share2 className="h-4 w-4" />
                                            Share
                                        </Button>
                                    </div>
                                </CardContent>
                            </Card>
                        ))}
                    </TabsContent>

                    <TabsContent value="questions">
                        <div className="space-y-6">
                            {posts
                                .filter((post) => post.category === "question")
                                .map((post) => (
                                    <Card key={post.id} className="overflow-hidden">
                                        <CardContent className="p-6">
                                            {/* Same post structure as above */}
                                            <div className="flex items-start gap-4 mb-4">
                                                <Avatar>
                                                    <AvatarImage src={post.author.avatar || "/placeholder.svg"} />
                                                    <AvatarFallback>{post.author.name.charAt(0)}</AvatarFallback>
                                                </Avatar>
                                                <div className="flex-1">
                                                    <div className="flex items-center gap-2">
                                                        <h3 className="font-semibold">{post.author.name}</h3>
                                                        {post.author.verified && (
                                                            <Badge variant="secondary" className="bg-green-100 text-green-800 text-xs">
                                                                Verified
                                                            </Badge>
                                                        )}
                                                    </div>
                                                    <div className="flex items-center gap-2 text-sm text-muted-foreground">
                                                        <MapPin className="h-3 w-3" />
                                                        <span>{post.author.location}</span>
                                                        <span>•</span>
                                                        <Clock className="h-3 w-3" />
                                                        <span>{post.timestamp}</span>
                                                    </div>
                                                </div>
                                            </div>
                                            <p className="text-sm leading-relaxed mb-4">{post.content}</p>
                                            <div className="flex items-center gap-6 pt-3 border-t border-border">
                                                <Button
                                                    variant="ghost"
                                                    size="sm"
                                                    onClick={() => handleLikePost(post.id)}
                                                    className={`gap-2 ${post.isLiked ? "text-red-500" : "text-muted-foreground"}`}
                                                >
                                                    <Heart className={`h-4 w-4 ${post.isLiked ? "fill-current" : ""}`} />
                                                    {post.likes} Helpful
                                                </Button>
                                                <Button variant="ghost" size="sm" className="gap-2 text-muted-foreground">
                                                    <MessageCircle className="h-4 w-4" />
                                                    {post.comments} Answers
                                                </Button>
                                            </div>
                                        </CardContent>
                                    </Card>
                                ))}
                        </div>
                    </TabsContent>

                    <TabsContent value="tips">
                        <div className="space-y-6">
                            {posts
                                .filter((post) => post.category === "tip")
                                .map((post) => (
                                    <Card key={post.id} className="overflow-hidden border-l-4 border-l-green-500">
                                        <CardContent className="p-6">
                                            <div className="flex items-start gap-4 mb-4">
                                                <Avatar>
                                                    <AvatarImage src={post.author.avatar || "/placeholder.svg"} />
                                                    <AvatarFallback>{post.author.name.charAt(0)}</AvatarFallback>
                                                </Avatar>
                                                <div className="flex-1">
                                                    <div className="flex items-center gap-2">
                                                        <h3 className="font-semibold">{post.author.name}</h3>
                                                        {post.author.verified && (
                                                            <Badge variant="secondary" className="bg-green-100 text-green-800 text-xs">
                                                                Verified
                                                            </Badge>
                                                        )}
                                                        <Badge className="bg-green-100 text-green-800 text-xs">💡 Pro Tip</Badge>
                                                    </div>
                                                    <div className="flex items-center gap-2 text-sm text-muted-foreground">
                                                        <MapPin className="h-3 w-3" />
                                                        <span>{post.author.location}</span>
                                                        <span>•</span>
                                                        <Clock className="h-3 w-3" />
                                                        <span>{post.timestamp}</span>
                                                    </div>
                                                </div>
                                            </div>
                                            <p className="text-sm leading-relaxed mb-4">{post.content}</p>
                                            <div className="flex items-center gap-6 pt-3 border-t border-border">
                                                <Button
                                                    variant="ghost"
                                                    size="sm"
                                                    onClick={() => handleLikePost(post.id)}
                                                    className={`gap-2 ${post.isLiked ? "text-red-500" : "text-muted-foreground"}`}
                                                >
                                                    <Heart className={`h-4 w-4 ${post.isLiked ? "fill-current" : ""}`} />
                                                    {post.likes} Helpful
                                                </Button>
                                                <Button variant="ghost" size="sm" className="gap-2 text-muted-foreground">
                                                    <MessageCircle className="h-4 w-4" />
                                                    {post.comments} Comments
                                                </Button>
                                            </div>
                                        </CardContent>
                                    </Card>
                                ))}
                        </div>
                    </TabsContent>

                    <TabsContent value="success">
                        <div className="space-y-6">
                            {posts
                                .filter((post) => post.category === "success")
                                .map((post) => (
                                    <Card key={post.id} className="overflow-hidden border-l-4 border-l-yellow-500">
                                        <CardContent className="p-6">
                                            <div className="flex items-start gap-4 mb-4">
                                                <Avatar>
                                                    <AvatarImage src={post.author.avatar || "/placeholder.svg"} />
                                                    <AvatarFallback>{post.author.name.charAt(0)}</AvatarFallback>
                                                </Avatar>
                                                <div className="flex-1">
                                                    <div className="flex items-center gap-2">
                                                        <h3 className="font-semibold">{post.author.name}</h3>
                                                        {post.author.verified && (
                                                            <Badge variant="secondary" className="bg-green-100 text-green-800 text-xs">
                                                                Verified
                                                            </Badge>
                                                        )}
                                                        <Badge className="bg-yellow-100 text-yellow-800 text-xs">🎉 Success Story</Badge>
                                                    </div>
                                                    <div className="flex items-center gap-2 text-sm text-muted-foreground">
                                                        <MapPin className="h-3 w-3" />
                                                        <span>{post.author.location}</span>
                                                        <span>•</span>
                                                        <Clock className="h-3 w-3" />
                                                        <span>{post.timestamp}</span>
                                                    </div>
                                                </div>
                                            </div>
                                            <p className="text-sm leading-relaxed mb-4">{post.content}</p>
                                            {post.image && (
                                                <img
                                                    src={post.image || "/placeholder.svg"}
                                                    alt="Success story image"
                                                    className="mb-4 rounded-lg w-full max-h-64 object-cover"
                                                />
                                            )}
                                            <div className="flex items-center gap-6 pt-3 border-t border-border">
                                                <Button
                                                    variant="ghost"
                                                    size="sm"
                                                    onClick={() => handleLikePost(post.id)}
                                                    className={`gap-2 ${post.isLiked ? "text-red-500" : "text-muted-foreground"}`}
                                                >
                                                    <Heart className={`h-4 w-4 ${post.isLiked ? "fill-current" : ""}`} />
                                                    {post.likes} Congratulations
                                                </Button>
                                                <Button variant="ghost" size="sm" className="gap-2 text-muted-foreground">
                                                    <MessageCircle className="h-4 w-4" />
                                                    {post.comments} Comments
                                                </Button>
                                            </div>
                                        </CardContent>
                                    </Card>
                                ))}
                        </div>
                    </TabsContent>
                </Tabs>
            </div>
        </div>
    )
}