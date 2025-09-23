// Frontend type definitions

export interface UploadedImage {
  file?: File;
  previewUrl: string;
  base64: string;
  mimeType: string;
}

// API Types
export interface ApiFile {
  base64: string;
  mimeType: string;
}

export interface ClothingItems {
  top: ApiFile | null;
  pants: ApiFile | null;
  shoes: ApiFile | null;
  outer: ApiFile | null;
}

export interface VirtualTryOnRequest {
  // Person image is now optional to allow outfit-only composition
  person: ApiFile | null;
  clothingItems: ClothingItems;
}

export interface VirtualTryOnResponse {
  generatedImage: string;
  error?: string;
}

export interface RecommendationRequest {
  person: ApiFile | null;
  clothingItems: ClothingItems | null;
  generatedImage: string | null;
  options: RecommendationOptions | null;
  selectedProductIds: {
    top: string | null;
    pants: string | null;
    shoes: string | null;
    outer: string | null;
  } | null;
}

export interface RecommendationItem {
  id: string;
  pos?: number; // embedding index when known
  title: string;
  price: number;
  imageUrl?: string;
  productUrl?: string;
  score?: number;
  tags: string[];
  category: string;
  discountRate?: number; // 할인율 (0.0 ~ 1.0)
  discountPercentage?: number; // 할인 퍼센트 (0 ~ 100)
  brandName?: string; // 브랜드명
}

export interface CategoryRecommendations {
  top: RecommendationItem[];
  pants: RecommendationItem[];
  shoes: RecommendationItem[];
  outer: RecommendationItem[];
  accessories: RecommendationItem[];
}

export interface RecommendationResponse {
  recommendations: RecommendationItem[] | CategoryRecommendations;
  error?: string;
}

// Recommendation options (aligns with backend API)
export interface RecommendationOptions {
  maxPerCategory?: number;
  minPrice?: number;
  maxPrice?: number;
  excludeTags?: string[];
}

export interface VideoGenerationRequest {
  prompt: string;
  imageData: string;
  mimeType?: string;
  parameters?: Record<string, unknown>;
}

export interface VideoGenerationStartResponse {
  operationName: string;
  raw?: unknown;
}

export interface VideoGenerationStatusResponse {
  done: boolean;
  videoUris: string[];
  operation: unknown;
  progressPercent?: number | null;
}

// Style tips
export interface StyleTipsResponse {
  tips: string[];
  tone?: string;
  occasion?: string;
  source: "ai" | "fallback";
  requestId?: string;
  timestamp?: string;
  score?: number; // 0..100
}

// Frontend State Types
export interface VirtualTryOnState {
  personImage: UploadedImage | null;
  clothingItems: {
    top: UploadedImage | null;
    pants: UploadedImage | null;
    shoes: UploadedImage | null;
    outer: UploadedImage | null;
  };
  generatedImage: string | null;
  recommendations: CategoryRecommendations | null;
  isLoading: boolean;
  error: string | null;
}
