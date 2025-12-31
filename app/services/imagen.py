"""
Google Gemini Imagen API 客户端
支持文生图和图生图功能
"""
import json
import time
import base64
from typing import Optional, Dict, Any, List
import httpx

from app.models.schemas import (
    ImageGenerationRequest,
    ImageGenerationResponse,
    ImageData,
)
from app.utils.logging import log


class ImagenClient:
    """Google Gemini Imagen API 客户端"""
    
    # 可用的 Imagen 模型列表
    AVAILABLE_MODELS = [
        "imagen-3.0-generate-002",
        "imagen-3.0-generate-001",
        "imagen-3.0-fast-generate-001",
    ]
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
    
    def _size_to_aspect_ratio(self, size: Optional[str]) -> str:
        """将 OpenAI 的 size 参数转换为 Gemini 的 aspect_ratio"""
        size_mapping = {
            "1024x1024": "1:1",
            "1792x1024": "16:9",
            "1024x1792": "9:16",
            "512x512": "1:1",
            "256x256": "1:1",
        }
        return size_mapping.get(size or "1024x1024", "1:1")
    
    def convert_openai_to_gemini_request(
        self, request: ImageGenerationRequest
    ) -> Dict[str, Any]:
        """
        将 OpenAI 格式的图像请求转换为 Gemini Imagen 格式
        
        OpenAI 格式:
        {
            "model": "dall-e-3",
            "prompt": "a white cat",
            "n": 1,
            "size": "1024x1024"
        }
        
        Gemini 格式:
        {
            "instances": [{"prompt": "a white cat"}],
            "parameters": {
                "sampleCount": 1,
                "aspectRatio": "1:1"
            }
        }
        """
        # 确定宽高比
        aspect_ratio = request.aspect_ratio or self._size_to_aspect_ratio(request.size)
        
        # 构建 Gemini 请求
        gemini_request: Dict[str, Any] = {
            "instances": [{"prompt": request.prompt}],
            "parameters": {
                "sampleCount": min(request.n, 4),  # Gemini 最多支持 4 张
                "aspectRatio": aspect_ratio,
            }
        }
        
        # 添加负面提示词
        if request.negative_prompt:
            gemini_request["parameters"]["negativePrompt"] = request.negative_prompt
        
        # 图生图：添加参考图像
        if request.image:
            gemini_request["instances"][0]["image"] = {
                "bytesBase64Encoded": request.image
            }
        
        return gemini_request
    
    def convert_gemini_to_openai_response(
        self,
        gemini_response: Dict[str, Any],
        response_format: str = "b64_json"
    ) -> ImageGenerationResponse:
        """
        将 Gemini Imagen 响应转换为 OpenAI 格式
        
        Gemini 响应格式:
        {
            "predictions": [
                {"bytesBase64Encoded": "..."},
                {"bytesBase64Encoded": "..."}
            ]
        }
        
        OpenAI 响应格式:
        {
            "created": 1234567890,
            "data": [
                {"b64_json": "..."},
                {"b64_json": "..."}
            ]
        }
        """
        images: List[ImageData] = []
        predictions = gemini_response.get("predictions", [])
        
        for prediction in predictions:
            if response_format == "b64_json":
                image_data = ImageData(
                    b64_json=prediction.get("bytesBase64Encoded"),
                    revised_prompt=prediction.get("prompt")
                )
            else:
                # URL 格式 - Gemini 原生不支持，返回 base64
                image_data = ImageData(
                    b64_json=prediction.get("bytesBase64Encoded"),
                    revised_prompt=prediction.get("prompt")
                )
            images.append(image_data)
        
        return ImageGenerationResponse(
            created=int(time.time()),
            data=images
        )

    async def generate_image(
        self,
        request: ImageGenerationRequest
    ) -> ImageGenerationResponse:
        """
        生成图像 (文生图)
        
        Args:
            request: OpenAI 格式的图像生成请求
        
        Returns:
            ImageGenerationResponse: OpenAI 格式的响应
        """
        model = request.model
        if model not in self.AVAILABLE_MODELS:
            model = "imagen-3.0-generate-002"
        
        # 转换请求格式
        gemini_request = self.convert_openai_to_gemini_request(request)
        
        url = f"{self.base_url}/models/{model}:predict?key={self.api_key}"
        headers = {"Content-Type": "application/json"}
        
        extra_log = {
            "key": self.api_key[:8],
            "request_type": "image-generation",
            "model": model,
        }
        log("INFO", "图像生成请求开始", extra=extra_log)
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    url,
                    headers=headers,
                    json=gemini_request,
                    timeout=120  # 图像生成可能需要较长时间
                )
                response.raise_for_status()
                
            gemini_response = response.json()
            log("INFO", "图像生成请求成功", extra=extra_log)
            
            return self.convert_gemini_to_openai_response(
                gemini_response,
                request.response_format or "b64_json"
            )
            
        except httpx.HTTPStatusError as e:
            log("ERROR", f"图像生成请求失败: {e.response.text}", extra=extra_log)
            raise
        except Exception as e:
            log("ERROR", f"图像生成请求异常: {str(e)}", extra=extra_log)
            raise
    
    async def edit_image(
        self,
        image: str,
        prompt: str,
        mask: Optional[str] = None,
        model: str = "imagen-3.0-generate-002",
        n: int = 1,
        size: Optional[str] = None,
        response_format: str = "b64_json"
    ) -> ImageGenerationResponse:
        """
        图生图编辑
        
        Args:
            image: base64 编码的原始图像
            prompt: 编辑提示词
            mask: base64 编码的遮罩图像 (可选)
            model: 使用的模型
            n: 生成数量
            size: 图像尺寸
            response_format: 响应格式
        
        Returns:
            ImageGenerationResponse: OpenAI 格式的响应
        """
        # 构建图生图请求
        request = ImageGenerationRequest(
            model=model,
            prompt=prompt,
            n=n,
            size=size,
            response_format=response_format,
            image=image
        )
        
        return await self.generate_image(request)
    
    @staticmethod
    def is_imagen_model(model: str) -> bool:
        """检查是否为 Imagen 模型"""
        return model in ImagenClient.AVAILABLE_MODELS or model.startswith("imagen")
