"use client";

import { useState, useEffect, Suspense } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
    Sprout,
    Loader2,
    CheckCircle2,
    MapPin,
    Wheat,
    Bean,
    Apple,
    PlusCircle,
    Search
} from "lucide-react";

// --- Type Definitions ---
interface Farm {
    id: string;
    name: string;
    area: number;
    points: { lat: number; lng: number }[];
    center: { lat: number; lng: number };
    createdAt: string;
}

interface CropRecommendation {
    id: string;
    name: string;
    icon: React.ReactNode;
    description: string;
    idealArea: number;
    maxArea: number;
    growthPeriod: string;
    waterRequirement: string;
    profitability: 'High' | 'Medium' | 'Low';
    difficulty: 'Easy' | 'Medium' | 'Hard';
    marketPrice: string;
    benefits: string[];
    color: string;
}

interface SelectedCrop {
    cropId: string;
    allocatedArea: number;
}

interface CropData {
    crop: string;
    category: string;
    season: string;
    typical_sowing_months: string;
    typical_harvest_months: string;
    major_states: string;
    notes: string;
}

// ✅ Hardcoded the crop data to eliminate fetching errors
const allCropsData: CropData[] = [
    {"crop":"Rice","category":"Cereal","season":"Kharif/Rabi","typical_sowing_months":"Jun-Jul / Sep-Oct","typical_harvest_months":"Sep-Dec / Mar-May","major_states":"West Bengal; Uttar Pradesh; Punjab","notes":"Main staple; irrigated and rainfed ecologies"},
    {"crop":"Wheat","category":"Cereal","season":"Rabi","typical_sowing_months":"Oct-Dec","typical_harvest_months":"Mar-May","major_states":"Punjab; Haryana; Uttar Pradesh","notes":"Temperate/rainfed-irrigated; major rabi cereal"},
    {"crop":"Maize","category":"Cereal","season":"Kharif/Rabi/Zaid","typical_sowing_months":"Jun-Jul / Oct-Nov / Mar-Apr","typical_harvest_months":"Sep-Dec / Jan-Mar / Apr-Jun","major_states":"Madhya Pradesh; Karnataka; Maharashtra","notes":"Used as food, feed, and industrial"},
    {"crop":"Sorghum (Jowar)","category":"Cereal","season":"Kharif/Rabi","typical_sowing_months":"Jun-Jul / Oct-Nov","typical_harvest_months":"Sep-Dec / Mar-May","major_states":"Maharashtra; Karnataka; Telangana","notes":"Drought tolerant; used as staple and fodder"},
    {"crop":"Pearl Millet (Bajra)","category":"Cereal","season":"Kharif","typical_sowing_months":"Jun-Jul","typical_harvest_months":"Sep-Nov","major_states":"Rajasthan; Gujarat; Haryana","notes":"Rainfed, arid zones"},
    {"crop":"Chickpea (Gram)","category":"Pulse","season":"Rabi","typical_sowing_months":"Oct-Dec","typical_harvest_months":"Mar-May","major_states":"Madhya Pradesh; Rajasthan; Maharashtra","notes":"Major rabi pulse"},
    {"crop":"Pigeon Pea (Arhar/Toor)","category":"Pulse","season":"Kharif/Rabi","typical_sowing_months":"Jun-Jul / Nov-Dec","typical_harvest_months":"Sep-Dec / Feb-Apr","major_states":"Madhya Pradesh; Maharashtra; Karnataka","notes":"Important kharif pulse"},
    {"crop":"Groundnut (Peanut)","category":"Oilseed","season":"Kharif/Rabi","typical_sowing_months":"Jun-Jul / Oct-Nov","typical_harvest_months":"Sep-Dec / Feb-Apr","major_states":"Gujarat; Andhra Pradesh; Tamil Nadu","notes":"Major oilseed; rainfed and irrigated"},
    {"crop":"Soybean","category":"Oilseed","season":"Kharif","typical_sowing_months":"Jun-Jul","typical_harvest_months":"Sep-Oct","major_states":"Madhya Pradesh; Maharashtra; Rajasthan","notes":"Key oilseed and protein source"},
    {"crop":"Rapeseed & Mustard","category":"Oilseed","season":"Rabi","typical_sowing_months":"Oct-Nov","typical_harvest_months":"Mar-Apr","major_states":"Rajasthan; Uttar Pradesh; Haryana","notes":"Rabi oilseed"},
    {"crop":"Cotton","category":"Fiber","season":"Kharif","typical_sowing_months":"Jun-Jul","typical_harvest_months":"Oct-Dec","major_states":"Maharashtra; Gujarat; Telangana","notes":"Major fiber crop; Bt cotton widely adopted"},
    {"crop":"Sugarcane","category":"Cash","season":"Year-round","typical_sowing_months":"Oct-Mar","typical_harvest_months":"Oct-Dec","major_states":"Uttar Pradesh; Maharashtra; Karnataka","notes":"Long-duration ratoonable crop"},
    {"crop":"Potato","category":"Vegetable","season":"Rabi/Kharif/Zaid","typical_sowing_months":"Oct-Dec / Feb-Mar / Mar-Apr","typical_harvest_months":"Feb-Apr / May-Jun / Jun-Jul","major_states":"West Bengal; Uttar Pradesh; Bihar","notes":"Staple tuber; multiple planting windows"},
    {"crop":"Onion","category":"Vegetable","season":"Rabi/Kharif/Zaid","typical_sowing_months":"Sep-Nov / Feb-Mar / Mar-Apr","typical_harvest_months":"Feb-May / Jun-Jul / Jun-Aug","major_states":"Maharashtra; Karnataka; Gujarat","notes":"Bulb crop with storability"},
    {"crop":"Tomato","category":"Vegetable","season":"Kharif/Rabi/Zaid","typical_sowing_months":"Oct-Dec / Jun-Jul / Feb-Mar","typical_harvest_months":"Jan-Mar / Sep-Nov / Apr-May","major_states":"Andhra Pradesh; Karnataka; Maharashtra","notes":"Perishable, warm-season varieties"},
];

const getInitialRecommendations = (farmArea: number): CropRecommendation[] => [
    {id:'wheat',name:'Winter Wheat',icon:<Wheat className="h-8 w-8"/>,description:'High-yield grain crop suitable for your soil type.',idealArea:Math.round(farmArea * 0.4),maxArea:Math.round(farmArea * 0.6),growthPeriod:'8-10 months',waterRequirement:'Medium',profitability:'High',difficulty:'Easy',marketPrice:'₹2,500/quintal',benefits:['Stable market demand','Good storage life'],color:'#F59E0B'},
    {id:'corn',name:'Sweet Corn',icon:<Bean className="h-8 w-8"/>,description:'Fast-growing crop with excellent market demand.',idealArea:Math.round(farmArea * 0.3),maxArea:Math.round(farmArea * 0.5),growthPeriod:'3-4 months',waterRequirement:'High',profitability:'High',difficulty:'Medium',marketPrice:'₹3,200/quintal',benefits:['Quick returns','High nutrition value'],color:'#10B981'},
    {id:'vegetables',name:'Mixed Vegetables',icon:<Apple className="h-8 w-8"/>,description:'Diverse vegetable crops for local market supply.',idealArea:Math.round(farmArea * 0.2),maxArea:Math.round(farmArea * 0.4),growthPeriod:'2-6 months',waterRequirement:'Medium',profitability:'Medium',difficulty:'Medium',marketPrice:'₹1,800/quintal',benefits:['Diverse income','Local market'],color:'#8B5CF6'},
];

function RecommendationsContent() {
    const router = useRouter();
    const searchParams = useSearchParams();
    const farmId = searchParams.get('farmId');

    const [farm, setFarm] = useState<Farm | null>(null);
    const [isLoading, setIsLoading] = useState(true);
    const [isAnalyzing, setIsAnalyzing] = useState(false);
    const [selectedCrops, setSelectedCrops] = useState<SelectedCrop[]>([]);
    const [analysisProgress, setAnalysisProgress] = useState(0);
    const [cropRecommendations, setCropRecommendations] = useState<CropRecommendation[]>([]);
    const [searchTerm, setSearchTerm] = useState("");
    const [isModalOpen, setIsModalOpen] = useState(false);

    useEffect(() => {
        if (!farmId) return;

        const loadData = () => {
            try {
                const savedFarms = JSON.parse(localStorage.getItem('farms') || '[]');
                const currentFarm = savedFarms.find((f: Farm) => f.id === farmId);

                if (!currentFarm) {
                    console.error("Farm ID not found:", farmId);
                    router.push('/dashboard');
                    return;
                }

                setFarm(currentFarm);
                setCropRecommendations(getInitialRecommendations(currentFarm.area));

                setIsLoading(false);
                setIsAnalyzing(true);

                const progressInterval = setInterval(() => {
                    setAnalysisProgress(prev => {
                        const newProgress = prev + Math.random() * 20;
                        if (newProgress >= 100) {
                            clearInterval(progressInterval);
                            setTimeout(() => setIsAnalyzing(false), 500);
                            return 100;
                        }
                        return newProgress;
                    });
                }, 300);
            } catch (error) {
                console.error('Error loading data from localStorage:', error);
                alert("Error reading farm data. Your saved data might be corrupted. Please try clearing your browser's local storage for this site and create the farm again.");
                router.push('/dashboard');
            }
        };

        loadData();
    }, [farmId, router]);

    const handleAddCustomCrop = (cropData: CropData) => {
        if (cropRecommendations.some(c => c.name.toLowerCase() === cropData.crop.toLowerCase())) return;

        const newRec: CropRecommendation = {
            id: cropData.crop.toLowerCase().replace(/\s/g, '-'),
            name: cropData.crop,
            icon: <Sprout className="h-8 w-8" />,
            description: cropData.notes || `A ${cropData.category} for the ${cropData.season} season.`,
            idealArea: Math.round((farm?.area || 0) * 0.1),
            maxArea: Math.round((farm?.area || 0) * 0.8),
            growthPeriod: `${cropData.typical_sowing_months} to ${cropData.typical_harvest_months}`,
            waterRequirement: 'Medium',
            profitability: 'Medium',
            difficulty: 'Medium',
            marketPrice: 'Varies',
            benefits: [cropData.category, cropData.season],
            color: `#${Math.floor(Math.random()*16777215).toString(16).padStart(6, '0')}`
        };
        setCropRecommendations(prev => [...prev, newRec]);
        setIsModalOpen(false);
    };

    const handleCropSelection = (cropId: string, area: number) => {
        setSelectedCrops(prev => {
            const existing = prev.find(c => c.cropId === cropId);
            if (existing) {
                return prev.map(c => (c.cropId === cropId ? { ...c, allocatedArea: area } : c));
            }
            return [...prev, { cropId, allocatedArea: area }];
        });
    };

    const removeCropSelection = (cropId: string) => {
        setSelectedCrops(prev => prev.filter(c => c.cropId !== cropId));
    };

    const totalAllocatedArea = selectedCrops.reduce((sum, crop) => sum + crop.allocatedArea, 0);
    const remainingArea = (farm?.area || 0) - totalAllocatedArea;

    const handleFinalizeCropPlan = async () => {
        setIsLoading(true);
        const cropPlan = {
            farmId: farm?.id,
            crops: selectedCrops.map(sc => {
                const crop = cropRecommendations.find(c => c.id === sc.cropId);
                return {...sc, cropName:crop?.name, color:crop?.color, percentage: Math.round((sc.allocatedArea / (farm?.area || 1)) * 100)};
            }),
            totalArea: farm?.area || 0,
            allocatedArea: totalAllocatedArea,
            remainingArea,
            createdAt: new Date().toISOString()
        };
        localStorage.setItem(`cropPlan_${farm?.id}`, JSON.stringify(cropPlan));
        await new Promise(resolve => setTimeout(resolve, 1000));
        router.push('/dashboard');
    };

    const filteredCrops = allCropsData.filter(crop =>
        crop.crop.toLowerCase().includes(searchTerm.toLowerCase()) ||
        crop.category.toLowerCase().includes(searchTerm.toLowerCase())
    );

    if (isLoading) {
        return (
            <div className="flex h-screen items-center justify-center"><Loader2 className="h-16 w-16 animate-spin text-green-600"/></div>
        );
    }

    if (isAnalyzing) {
        return (
            <div className="min-h-screen flex items-center justify-center p-6 bg-gray-50">
                <Card className="w-full max-w-2xl shadow-2xl"><CardContent className="p-8 text-center space-y-6">
                    <Sprout className="h-20 w-20 text-green-600 mx-auto animate-pulse"/>
                    <h1 className="text-3xl font-bold text-gray-800">Analyzing Your Farm</h1>
                    <p className="text-gray-600 text-lg">Our AI is studying your {farm?.area} hectare farm...</p>
                    <div className="space-y-3">
                        <Progress value={analysisProgress} className="h-3"/>
                        <p className="text-sm text-green-600">
                            {analysisProgress < 50 ? "Analyzing soil and climate..." : "Calculating recommendations..."}
                        </p>
                    </div>
                </CardContent></Card>
            </div>
        );
    }

    return (
        <div className="min-h-screen bg-gray-50 p-6">
            <div className="max-w-7xl mx-auto space-y-8">
                <div className="text-center">
                    <h1 className="text-4xl font-bold text-gray-800">Crop Recommendations</h1>
                    <p className="text-xl text-gray-600 mt-2">For <strong>{farm?.name}</strong> ({farm?.area} hectares)</p>
                </div>

                <Card className="sticky top-4 z-10 backdrop-blur-lg bg-white/80 shadow-lg">
                    <CardHeader><CardTitle className="flex items-center gap-2"><MapPin className="h-5 w-5 text-green-600"/>Area Allocation</CardTitle></CardHeader>
                    <CardContent>
                        <div className="grid grid-cols-3 gap-4 text-center">
                            <div><p className="text-3xl font-bold text-green-600">{farm?.area?.toFixed(1)}</p><p className="text-sm text-gray-600">Total Hectares</p></div>
                            <div><p className="text-3xl font-bold text-blue-600">{totalAllocatedArea.toFixed(1)}</p><p className="text-sm text-gray-600">Allocated</p></div>
                            <div><p className="text-3xl font-bold text-orange-600">{remainingArea.toFixed(1)}</p><p className="text-sm text-gray-600">Remaining</p></div>
                        </div>
                        {totalAllocatedArea > 0 && (<div className="mt-4"><Progress value={(totalAllocatedArea / (farm?.area || 1)) * 100} className="h-2"/></div>)}
                    </CardContent>
                </Card>

                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                    {cropRecommendations.map(crop => {
                        const isSelected = selectedCrops.some(sc => sc.cropId === crop.id);
                        const selectedCrop = selectedCrops.find(sc => sc.cropId === crop.id);
                        return (
                            <Card key={crop.id} className={`transition-all hover:shadow-xl ${isSelected ? 'ring-2 ring-green-500' : 'shadow-md'}`}>
                                <CardHeader><div className="flex items-center justify-between">
                                    <div className="flex items-center gap-3">
                                        <div className="p-2 rounded-lg" style={{backgroundColor:crop.color+'20'}}><div style={{color:crop.color}}>{crop.icon}</div></div>
                                        <div><CardTitle className="text-lg">{crop.name}</CardTitle><p className="text-sm text-gray-500">{crop.growthPeriod}</p></div>
                                    </div>
                                    <Badge variant={crop.profitability === 'High' ? 'default':'secondary'}>{crop.profitability}</Badge>
                                </div></CardHeader>
                                <CardContent className="space-y-4">
                                    <p className="text-sm text-gray-600 h-10">{crop.description}</p>
                                    <div className="space-y-2">
                                        <p className="text-sm font-medium">Recommended: {crop.idealArea} ha (Max: {crop.maxArea} ha)</p>
                                        {isSelected ? (
                                            <div className="flex items-center gap-2">
                                                <Input type="number" min="0" max={Math.min(crop.maxArea, remainingArea + (selectedCrop?.allocatedArea || 0))} value={selectedCrop?.allocatedArea || ''} onChange={e => handleCropSelection(crop.id, parseFloat(e.target.value) || 0)} placeholder="Hectares" className="text-sm"/>
                                                <Button variant="outline" size="sm" onClick={() => removeCropSelection(crop.id)}>Remove</Button>
                                            </div>
                                        ) : (
                                            <Button onClick={() => handleCropSelection(crop.id, crop.idealArea)} className="w-full" disabled={remainingArea < 0.1} style={{backgroundColor:crop.color,color:'white'}}>Select This Crop</Button>
                                        )}
                                    </div>
                                </CardContent>
                            </Card>
                        );
                    })}

                    <Dialog open={isModalOpen} onOpenChange={setIsModalOpen}>
                        <DialogTrigger asChild><Card className="border-2 border-dashed flex items-center justify-center min-h-[350px] cursor-pointer hover:border-green-500"><div className="text-center text-gray-500"><PlusCircle className="h-12 w-12 mx-auto"/><h3 className="font-semibold mt-2">Add Custom Crop</h3></div></Card></DialogTrigger>
                        <DialogContent className="sm:max-w-lg"><DialogHeader><DialogTitle>Select a Custom Crop</DialogTitle><CardDescription>Add a crop from the master list to your plan.</CardDescription></DialogHeader>
                            <div className="relative my-4"><Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground"/><Input placeholder="Search crops..." value={searchTerm} onChange={e=>setSearchTerm(e.target.value)} className="pl-10"/></div>
                            <ScrollArea className="h-[50vh]"><div className="space-y-2 pr-4">
                                {filteredCrops.map(crop => (
                                    <div key={crop.crop} className="flex items-center justify-between p-2 border rounded-lg hover:bg-muted">
                                        <div><p className="font-semibold">{crop.crop}</p><p className="text-sm text-muted-foreground">{crop.category}</p></div>
                                        <Button size="sm" variant="outline" onClick={() => handleAddCustomCrop(crop)}><PlusCircle className="h-4 w-4 mr-2"/>Add</Button>
                                    </div>
                                ))}
                            </div></ScrollArea>
                        </DialogContent>
                    </Dialog>
                </div>

                <div className="flex justify-center gap-4 pt-8">
                    <Button variant="outline" onClick={() => router.push('/dashboard')}>Back to Dashboard</Button>
                    <Button onClick={handleFinalizeCropPlan} disabled={selectedCrops.length === 0} className="bg-green-600 hover:bg-green-700" size="lg">
                        <CheckCircle2 className="h-4 w-4 mr-2"/>Finalize Crop Plan
                    </Button>
                </div>
            </div>
        </div>
    );
}


export default function CropRecommendationsPage() {
    return (
        <Suspense fallback={<div className="flex h-screen items-center justify-center"><Loader2 className="h-16 w-16 animate-spin text-green-600"/></div>}>
            <RecommendationsContent />
        </Suspense>
    );
}