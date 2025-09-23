import {
  RecommendationItem,
  RecommendationRequest,
  RecommendationResponse,
  VideoGenerationRequest,
  VideoGenerationStartResponse,
  VideoGenerationStatusResponse,
  VirtualTryOnRequest,
  VirtualTryOnResponse,
} from "../types";
import { apiClient, ApiError } from "./api.service";

/**
 * Virtual Try-On API Service
 * Provides typed methods for virtual try-on and recommendation API calls
 */
export class VirtualTryOnService {
  /**
   * Generate virtual try-on image by combining person and clothing items
   * @param request - Virtual try-on request with person image and clothing items
   * @returns Promise resolving to generated image data
   */
  async combineImages(
    request: VirtualTryOnRequest
  ): Promise<VirtualTryOnResponse> {
    try {
      const response = await apiClient.post<VirtualTryOnResponse>(
        "/api/generate",
        request,
        {
          timeout: 60000, // Extended timeout for AI processing
        }
      );

      if (response.error) {
        throw new ApiError(response.error, 400, "GENERATION_ERROR");
      }

      return response;
    } catch (error) {
      if (error instanceof ApiError) {
        throw error;
      }
      throw new ApiError(
        "Failed to generate virtual try-on image",
        500,
        "GENERATION_FAILED"
      );
    }
  }

  /**
   * Get product recommendations based on uploaded images
   * @param request - Recommendation request with person and/or clothing items
   * @returns Promise resolving to product recommendations
   */
  async getRecommendations(
    request: RecommendationRequest
  ): Promise<RecommendationResponse> {
    try {
      const response = await apiClient.post<RecommendationResponse>(
        "/api/recommend",
        request,
        {
          timeout: 45000, // Extended timeout for AI processing
        }
      );

      if (response.error) {
        throw new ApiError(response.error, 400, "RECOMMENDATION_ERROR");
      }

      return response;
    } catch (error) {
      if (error instanceof ApiError) {
        throw error;
      }
      throw new ApiError(
        "Failed to get recommendations",
        500,
        "RECOMMENDATION_FAILED"
      );
    }
  }

  // getRecommendationsFromFitting 제거됨 - getRecommendations로 통합

  /**
   * Get concise style tips based on generated image or history images
   * @param payload - { generatedImage?: string; historyImages?: string[]; options?: { tone?: 'warm'|'cool'|'neutral'; occasion?: string; maxTips?: number } }
   */
  async getStyleTips(payload: {
    generatedImage?: string;
    historyImages?: string[];
    person?: any;
    clothingItems?: any;
    options?: { tone?: string; occasion?: string; maxTips?: number };
  }): Promise<import("../types").StyleTipsResponse> {
    const response = await apiClient.post<import("../types").StyleTipsResponse>(
      "/api/tips",
      payload,
      { timeout: 20000 }
    );
    if ((response as any).error) {
      return { tips: [], source: "fallback" } as any;
    }
    return response as any;
  }

  /**
   * Recommend by selected positions with optional full metadata
   * @param payload { positions:number[]; items?: any[]; final_k?: number; categories?: string[]; use_llm_rerank?: boolean }
   */
  async getRecommendationsByPositions(payload: {
    positions: number[];
    items?: Array<{
      pos: number;
      category?: string;
      title?: string;
      tags?: string[];
      price?: number;
      brand?: string;
      gender?: string;
      productUrl?: string;
      imageUrl?: string;
      description?: string;
    }>;
    top_k?: number;
    final_k?: number;
    categories?: string[];
    use_llm_rerank?: boolean;
  }): Promise<RecommendationItem[]> {
    const res = await apiClient.post<RecommendationItem[]>(
      "/api/recommend/by-positions",
      payload,
      { timeout: 20000 }
    );
    return res;
  }

  /**
   * Check if virtual try-on generation is currently loading
   */
  isGenerating(): boolean {
    return apiClient.isLoading("POST", "/api/generate");
  }

  /**
   * Check if recommendations are currently loading
   */
  isLoadingRecommendations(): boolean {
    return apiClient.isLoading("POST", "/api/recommend");
  }

  /**
   * Check if any API call is currently loading
   */
  isLoading(): boolean {
    return this.isGenerating() || this.isLoadingRecommendations();
  }

  /**
   * Start Vertex AI video generation based on the current fitting image
   */
  async startVideoGeneration(
    request: VideoGenerationRequest
  ): Promise<VideoGenerationStartResponse> {
    try {
      const response = await apiClient.post<VideoGenerationStartResponse>(
        "/api/try-on/video",
        request,
        { timeout: 90000 }
      );
      if (
        !response ||
        typeof response.operationName !== "string" ||
        !response.operationName.trim()
      ) {
        throw new ApiError(
          "Video generation failed: invalid response from server",
          502,
          "VIDEO_GENERATION_INVALID",
          response
        );
      }
      return response;
    } catch (error) {
      if (error instanceof ApiError) {
        throw error;
      }
      const message =
        error instanceof Error
          ? error.message
          : "Failed to start video generation";
      throw new ApiError(
        message || "Failed to start video generation",
        500,
        "VIDEO_GENERATION_FAILED"
      );
    }
  }

  /**
   * Fetch status for video generation job
   */
  async fetchVideoStatus(
    operationName: string
  ): Promise<VideoGenerationStatusResponse> {
    try {
      const response = await apiClient.post<VideoGenerationStatusResponse>(
        "/api/try-on/video/status",
        { operationName },
        { timeout: 20000 }
      );
      if (
        !response ||
        typeof response.done !== "boolean" ||
        !Array.isArray(response.videoUris)
      ) {
        throw new ApiError(
          "Video status response malformed",
          502,
          "VIDEO_STATUS_INVALID",
          response
        );
      }
      return response;
    } catch (error) {
      if (error instanceof ApiError) {
        throw error;
      }
      const message =
        error instanceof Error
          ? error.message
          : "Failed to fetch video generation status";
      throw new ApiError(
        message || "Failed to fetch video generation status",
        500,
        "VIDEO_STATUS_FAILED"
      );
    }
  }

  /** Evaluate outfits (result images) with LLM */
  async evaluateOutfits(payload: {
    images: string[];
    options?: { occasion?: string; tone?: string; style?: string };
  }): Promise<{
    results: { index: number; score: number; reasoning?: string }[];
    source: "ai" | "fallback";
  }> {
    const res = await apiClient.post<any>("/api/evaluate", payload, {
      timeout: 30000,
    });
    return res as any;
  }
}

// Create and export singleton instance
export const virtualTryOnService = new VirtualTryOnService();
