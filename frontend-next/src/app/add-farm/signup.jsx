import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import axios from 'axios';
import { Mail, Lock, User } from 'lucide-react';
import Input from '../components/Input';
import Button from '../components/Button';

const Signup = () => {
    const [name, setName] = useState('');
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [error, setError] = useState('');
    const navigate = useNavigate();

    const handleSignup = async (e) => {
        e.preventDefault();
        setError('');

        if (!name || !email || !password) {
            setError('Please fill in all fields');
            return;
        }

        try {
            const response = await axios.post('/api/user/signup', {
                name,
                email,
                password
            });

            if (response.data.success) {
                // Store email in session storage to pass to OTP verification page
                sessionStorage.setItem('verificationEmail', email);
                navigate('/OtpVerification');
            } else {
                setError(response.data.message || 'Signup failed');
            }
        } catch (err) {
            setError('Error creating account. Please try again.');
        }
    };

    return (
        <div className="min-h-screen flex items-center justify-center bg-gray-700 ">
            <div className="bg-white px-8 py-10 rounded-xl shadow-sm border border-gray-100 w-full max-w-md">
                <h2 className="text-2xl font-semibold text-gray-900 tracking-tight">
                    Create Account
                </h2>
                <p className="text-sm text-gray-600 mt-2">
                    Sign up with your email to get started
                </p>

                <form className="mt-8 space-y-5" onSubmit={handleSignup}>
                    <div className="space-y-1">
                        <Input
                            type="text"
                            placeholder="Full Name"
                            icon={<User className="text-gray-400 w-5 h-5" />}
                            value={name}
                            onChange={(e) => setName(e.target.value)}
                            className="w-full px-4 py-3 rounded-lg border border-gray-200 focus:border-gray-300 focus:ring-2 focus:ring-gray-100 transition-all"
                        />
                    </div>

                    <div className="space-y-1">
                        <Input
                            type="email"
                            placeholder="Email"
                            icon={<Mail className="text-gray-400 w-5 h-5" />}
                            value={email}
                            onChange={(e) => setEmail(e.target.value)}
                            className="w-full px-4 py-3 rounded-lg border border-gray-200 focus:border-gray-300 focus:ring-2 focus:ring-gray-100 transition-all"
                        />
                    </div>

                    <div className="space-y-1">
                        <Input
                            type="password"
                            placeholder="Password"
                            icon={<Lock className="text-gray-400 w-5 h-5" />}
                            value={password}
                            onChange={(e) => setPassword(e.target.value)}
                            className="w-full px-4 py-3 rounded-lg border border-gray-200 focus:border-gray-300 focus:ring-2 focus:ring-gray-100 transition-all"
                        />
                    </div>

                    {error && (
                        <p className="text-red-500 text-sm font-medium">{error}</p>
                    )}

                    <Button
                        type="submit"
                        className="w-full py-3 bg-gray-900 text-white font-medium rounded-lg hover:bg-gray-800 active:bg-gray-950 transition-colors duration-200"
                    >
                        Sign Up
                    </Button>
                </form>

                <div className="mt-8 text-center">
                    <p className="text-sm text-gray-600">
                        Already have an account?{' '}
                        <Link
                            to="/login"
                            className="text-gray-900 font-medium hover:underline focus:outline-none"
                        >
                            Login here
                        </Link>
                    </p>
                </div>
            </div>
        </div>
    );
};

export default Signup;

// app/add-farm/page.tsx

"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent } from "@/components/ui/card";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Search, Save, Trash2, Undo2, ArrowLeft, Loader2, Target, MapPin } from "lucide-react";
import dynamic from 'next/dynamic';
import { LatLng } from "leaflet";

// ✅ Dynamically import the updated FarmMap component, disabling SSR
const FarmMap = dynamic(() => import('@/components/FarmMap'), {
    ssr: false,
    loading: () => (
        <div className="h-full w-full flex items-center justify-center bg-gray-200 rounded-lg">
            <div className="text-center space-y-2">
                <Loader2 className="h-8 w-8 animate-spin text-green-600 mx-auto" />
                <p className="text-sm text-gray-600">Loading satellite map...</p>
            </div>
        </div>
    ),
});

export default function AddFarmPage() {
    const router = useRouter();
    const [searchQuery, setSearchQuery] = useState("");
    const [farmName, setFarmName] = useState("");
    const [isPolygonClosed, setIsPolygonClosed] = useState(false);
    const [points, setPoints] = useState<LatLng[]>([]);
    const [mapCenter, setMapCenter] = useState<[number, number]>([20.5937, 78.9629]); // Default to India
    const [zoom, setZoom] = useState(5);
    const [showSaveDialog, setShowSaveDialog] = useState(false);
    const [isLoading, setIsLoading] = useState(false);
    const [isSearching, setIsSearching] = useState(false);
    const [searchResults, setSearchResults] = useState<string>("");

    const handleSearch = async () => {
        if (!searchQuery.trim()) return;

        setIsSearching(true);
        setSearchResults("");

        // Simulate a successful API response to avoid network errors
        await new Promise(resolve => setTimeout(resolve, 500));

        // Hardcoded response for "Ludhiana, Punjab"
        const data = [{
            lat: "30.900965",
            lon: "75.857275",
            display_name: "Ludhiana, Punjab, India"
        }];

        if (data.length > 0) {
            const { lat, lon, display_name } = data[0];
            setMapCenter([parseFloat(lat), parseFloat(lon)]);
            setZoom(16);
            setSearchResults(`Found: ${display_name}`);
        } else {
            setSearchResults("Location not found. Try a different search term.");
        }

        setIsSearching(false);
    };

    const handleSaveFarm = async () => {
        if (!farmName.trim() || points.length < 3) return;

        setIsLoading(true);

        // Calculate area for display (simple polygon area calculation)
        const area = calculatePolygonArea(points);

        // Create farm data object
        const farmData = {
            id: Date.now().toString(), // Simple ID generation
            name: farmName.trim(),
            points: points.map(p => ({ lat: p.lat, lng: p.lng })),
            area: parseFloat(area.toFixed(2)),
            createdAt: new Date().toISOString(),
            center: {
                lat: points.reduce((sum, p) => sum + p.lat, 0) / points.length,
                lng: points.reduce((sum, p) => sum + p.lng, 0) / points.length
            }
        };

        try {
            // Get existing farms from localStorage
            const existingFarms = JSON.parse(localStorage.getItem('farms') || '[]');

            // Add new farm
            const updatedFarms = [...existingFarms, farmData];

            // Save to localStorage
            localStorage.setItem('farms', JSON.stringify(updatedFarms));

            console.log("Farm saved successfully:", farmData);

            // Simulate API call delay
            await new Promise((resolve) => setTimeout(resolve, 1000));

            // Navigate to crop recommendations
            router.push(`/crop-recommendations?farmId=${farmData.id}`);
        } catch (error) {
            console.error("Error saving farm:", error);
            alert("Error saving farm. Please try again.");
        } finally {
            setIsLoading(false);
        }
    };

    // Simple polygon area calculation (rough approximation)
    const calculatePolygonArea = (points: LatLng[]): number => {
        if (points.length < 3) return 0;

        let area = 0;
        for (let i = 0; i < points.length; i++) {
            const j = (i + 1) % points.length;
            area += points[i].lat * points[j].lng;
            area -= points[j].lat * points[i].lng;
        }
        area = Math.abs(area) / 2;

        // Convert to approximate hectares (very rough calculation)
        return area * 111320 * 111320 / 10000;
    };

    const handleMapClick = (latlng: LatLng) => {
        if (isPolygonClosed) return;

        const newPoints = [...points, latlng];

        // Auto-close polygon when clicking near the first point (if we have at least 3 points)
        if (points.length >= 2 && latlng.distanceTo(points[0]) < 50) {
            setIsPolygonClosed(true);
            setPoints(newPoints);
        } else {
            setPoints(newPoints);
            // Auto-close if we have 8+ points to prevent overly complex polygons
            if (newPoints.length >= 8) {
                setIsPolygonClosed(true);
            }
        }
    };

    const clearDrawing = () => {
        setPoints([]);
        setIsPolygonClosed(false);
    };

    const undoLastPoint = () => {
        if (isPolygonClosed) {
            setIsPolygonClosed(false);
        } else {
            setPoints((prev) => prev.slice(0, -1));
        }
    };

    const estimatedArea = points.length > 2 ? calculatePolygonArea(points) : 0;

    return (
        <div className="h-screen w-screen bg-gray-100 flex flex-col relative overflow-hidden">
            {/* Floating Header */}
            <div className="absolute top-0 left-0 right-0 z-[1000] p-4">
                <div className="max-w-7xl mx-auto flex items-center justify-between bg-white/90 backdrop-blur-lg border border-white/40 rounded-2xl shadow-xl p-3">
                    <div className="flex items-center gap-3">
                        <Button variant="ghost" size="icon" onClick={() => router.back()} className="h-10 w-10 rounded-full hover:bg-green-100">
                            <ArrowLeft className="h-5 w-5" />
                        </Button>
                        <div>
                            <h1 className="text-lg font-bold text-green-900">Map Your Farm</h1>
                            <p className="text-sm text-green-700/80">Click on the satellite map to draw your boundaries</p>
                        </div>
                    </div>
                    <div className="flex items-center gap-2">
                        {points.length > 0 && (
                            <>
                                <Button variant="outline" size="sm" onClick={undoLastPoint} className="bg-white/70 hover:bg-white">
                                    <Undo2 className="h-4 w-4 mr-2" /> Undo
                                </Button>
                                <Button variant="destructive" size="sm" onClick={clearDrawing} className="bg-red-500/90 hover:bg-red-600 text-white">
                                    <Trash2 className="h-4 w-4" />
                                </Button>
                            </>
                        )}
                        <Dialog open={showSaveDialog} onOpenChange={setShowSaveDialog}>
                            <DialogTrigger asChild>
                                <Button
                                    disabled={points.length < 3}
                                    className="bg-green-600 hover:bg-green-700 shadow-lg disabled:opacity-50"
                                >
                                    <Save className="h-4 w-4 mr-2" /> Save Farm
                                </Button>
                            </DialogTrigger>
                            <DialogContent className="sm:max-w-md">
                                <DialogHeader>
                                    <DialogTitle className="text-green-900">Finalize Your Farm</DialogTitle>
                                </DialogHeader>
                                <div className="space-y-4 pt-2">
                                    <p className="text-sm text-muted-foreground">Give your new farm a name to save it to your dashboard.</p>

                                    {/* Farm Stats */}
                                    <div className="bg-green-50 rounded-lg p-3 space-y-1">
                                        <div className="flex justify-between text-sm">
                                            <span className="text-green-700">Boundary Points:</span>
                                            <span className="font-semibold text-green-900">{points.length}</span>
                                        </div>
                                        <div className="flex justify-between text-sm">
                                            <span className="text-green-700">Estimated Area:</span>
                                            <span className="font-semibold text-green-900">{estimatedArea.toFixed(2)} hectares</span>
                                        </div>
                                    </div>

                                    <Input
                                        placeholder="e.g., North Field, Home Farm, Orchard Section"
                                        value={farmName}
                                        onChange={(e) => setFarmName(e.target.value)}
                                        className="text-base"
                                    />
                                    <Button
                                        onClick={handleSaveFarm}
                                        disabled={!farmName.trim() || isLoading}
                                        className="w-full bg-green-600 hover:bg-green-700"
                                    >
                                        {isLoading ? (
                                            <>
                                                <Loader2 className="h-4 w-4 animate-spin mr-2" />
                                                Saving Farm...
                                            </>
                                        ) : (
                                            <>
                                                <Save className="h-4 w-4 mr-2" />
                                                Confirm & Save
                                            </>
                                        )}
                                    </Button>
                                </div>
                            </DialogContent>
                        </Dialog>
                    </div>
                </div>
            </div>

            {/* Floating Search Bar */}
            <Card className="absolute top-28 left-4 z-[999] w-full max-w-sm bg-white/90 backdrop-blur-lg border border-white/40 rounded-2xl shadow-xl">
                <CardContent className="p-3 space-y-2">
                    <div className="flex gap-2">
                        <div className="relative flex-1">
                            <MapPin className="absolute left-3 top-1/2 -translate-y-1/2 h-5 w-5 text-gray-400" />
                            <Input
                                placeholder="Search for a location..."
                                value={searchQuery}
                                onChange={(e) => setSearchQuery(e.target.value)}
                                onKeyDown={(e) => e.key === "Enter" && handleSearch()}
                                className="pl-10 text-base border-gray-200"
                            />
                        </div>
                        <Button onClick={handleSearch} disabled={isSearching} className="bg-green-600 hover:bg-green-700">
                            {isSearching ? <Loader2 className="h-5 w-5 animate-spin" /> : <Search className="h-5 w-5" />}
                        </Button>
                    </div>
                    {searchResults && (
                        <p className="text-xs text-gray-600 px-2">{searchResults}</p>
                    )}
                </CardContent>
            </Card>

            {/* Instructions Card */}
            {points.length < 3 && (
                <Card className="absolute bottom-4 left-4 z-[999] w-full max-w-xs bg-white/90 backdrop-blur-lg border border-white/40 rounded-2xl shadow-xl">
                    <CardContent className="p-4 space-y-2 text-sm text-green-900/90">
                        <div className="flex items-center gap-2 font-semibold">
                            <Target className="h-4 w-4 text-green-600"/>
                            Drawing Guide
                        </div>
                        <p>• Click on the satellite map to add boundary points</p>
                        <p>• Add at least 3 points to create a farm area</p>
                        <p>• Click near the first point to auto-close boundary</p>
                        {points.length > 0 && (
                            <div className="pt-2 border-t border-green-200 space-y-1">
                                <p className="font-semibold">Points: {points.length}</p>
                                <p className="text-green-600 text-xs">
                                    {3 - points.length} more point{points.length === 2 ? '' : 's'} needed
                                </p>
                            </div>
                        )}
                    </CardContent>
                </Card>
            )}

            {/* Ready to Save Card */}
            {points.length >= 3 && !isPolygonClosed && (
                <Card className="absolute bottom-4 left-4 z-[999] w-full max-w-xs bg-blue-50/90 backdrop-blur-lg border border-blue-200 rounded-2xl shadow-xl">
                    <CardContent className="p-4 text-center space-y-2">
                        <div className="text-blue-600 text-lg">✓</div>
                        <p className="font-semibold text-blue-900">Ready to Save!</p>
                        <p className="text-sm text-blue-700">
                            {points.length} points • ~{estimatedArea.toFixed(1)} hectares
                        </p>
                        <p className="text-xs text-blue-600">
                            Click "Save Farm" or add more points
                        </p>
                    </CardContent>
                </Card>
            )}

            {/* Completion Message */}
            {isPolygonClosed && (
                <Card className="absolute bottom-4 right-4 z-[999] w-full max-w-xs bg-green-50/90 backdrop-blur-lg border border-green-200 rounded-2xl shadow-xl">
                    <CardContent className="p-4 text-center space-y-2">
                        <div className="text-green-600 text-lg">🎉</div>
                        <p className="font-semibold text-green-900">Boundary Complete!</p>
                        <p className="text-sm text-green-700">
                            {points.length} points • ~{estimatedArea.toFixed(1)} hectares
                        </p>
                        <p className="text-xs text-green-600">Click "Save Farm" to finish</p>
                    </CardContent>
                </Card>
            )}

            {/* Map Container */}
            <div className="flex-grow w-full h-full relative">
                <FarmMap
                    center={mapCenter}
                    zoom={zoom}
                    points={points}
                    isPolygonClosed={isPolygonClosed}
                    onMapClick={handleMapClick}
                />
            </div>
        </div>
    );
}