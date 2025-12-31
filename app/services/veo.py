"""
Google Gemini Veo API 客户端
支持文生视频和图生视频功能
"""
import json
import time
from typing import Optional, Dict, Any, List
import httpx

from app.models.schemas import (
    VideoGenerationRequest,
    VideoGenerationOperation,
    VideoOperationStatus,
    VideoGenerationResponse,
    VideoData,
)
from app.utils.logging import log


class VeoClient:
    """Google Gemini Veo API 客户端"""
    
    # 可用的 Veo 模型列表
    AVAILABLE_MODELS = [
        "veo-2.0-generate-001",
    ]
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
    
    def convert_request_to_gemini(
        self, request: VideoGenerationRequest
    ) -> Dict[str, Any]:
        """
        将视频生成请求转换为 Gemini Veo 格式
        
        Gemini Veo 请求格式:
        {
            "instances": [{"prompt": "..."}],
            "parameters": {
                "aspectRatio": "16:9",
                "durationSeconds": 5
            }
        }
        """
        gemini_request: Dict[str, Any] = {
            "instances": [{"prompt": request.prompt}],
            "parameters": {
                "aspectRatio": request.aspect_ratio or "16:9",
                "durationSeconds": request.duration,
            }
        }
        
        # 图生视频：添加参考图像
        if request.image:
            gemini_request["instances"][0]["image"] = {
                "bytesBase64Encoded": request.image
            }
        
        return gemini_request
    
    async def generate_video(
        self,
        request: VideoGenerationRequest
    ) -> VideoGenerationOperation:
        """
        发起视频生成请求 (异步操作)
        
        Args:
            request: 视频生成请求
        
        Returns:
            VideoGenerationOperation: 包含操作 ID 的响应
        """
        model = request.model
        if model not in self.AVAILABLE_MODELS:
            model = "veo-2.0-generate-001"
        
        # 转换请求格式
        gemini_request = self.convert_request_to_gemini(request)
        
        url = f"{self.base_url}/models/{model}:predictLongRunning?key={self.api_key}"
        headers = {"Content-Type": "application/json"}
        
        extra_log = {
            "key": self.api_key[:8],
            "request_type": "video-generation",
            "model": model,
        }
        log("INFO", "视频生成请求开始", extra=extra_log)
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    url,
                    headers=headers,
                    json=gemini_request,
                    timeout=60
                )
                response.raise_for_status()
            
            gemini_response = response.json()
            operation_name = gemini_response.get("name", "")
            
            log("INFO", f"视频生成任务已创建: {operation_name}", extra=extra_log)
            
            return VideoGenerationOperation(
                id=operation_name,
                status="pending",
                created=int(time.time())
            )
            
        except httpx.HTTPStatusError as e:
            log("ERROR", f"视频生成请求失败: {e.response.text}", extra=extra_log)
            raise
        except Exception as e:
            log("ERROR", f"视频生成请求异常: {str(e)}", extra=extra_log)
            raise

    async def get_operation_status(
        self,
        operation_id: str
    ) -> VideoOperationStatus:
        """
        查询视频生成操作状态
        
        Args:
            operation_id: 操作 ID
        
        Returns:
            VideoOperationStatus: 操作状态
        """
        url = f"{self.base_url}/{operation_id}?key={self.api_key}"
        headers = {"Content-Type": "application/json"}
        
        extra_log = {
            "key": self.api_key[:8],
            "request_type": "video-status",
            "operation_id": operation_id[:20] if operation_id else "",
        }
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, headers=headers, timeout=30)
                response.raise_for_status()
            
            gemini_response = response.json()
            
            # 解析状态
            done = gemini_response.get("done", False)
            error = gemini_response.get("error")
            metadata = gemini_response.get("metadata", {})
            
            if error:
                status = "failed"
                error_message = error.get("message", "Unknown error")
            elif done:
                status = "completed"
                error_message = None
            else:
                status = "processing"
                error_message = None
            
            # 计算进度
            progress = metadata.get("progress", 0.0) if not done else 1.0
            
            return VideoOperationStatus(
                id=operation_id,
                status=status,
                progress=progress,
                created=int(time.time()),
                completed=int(time.time()) if done else None,
                error=error_message
            )
            
        except httpx.HTTPStatusError as e:
            log("ERROR", f"查询视频状态失败: {e.response.text}", extra=extra_log)
            raise
        except Exception as e:
            log("ERROR", f"查询视频状态异常: {str(e)}", extra=extra_log)
            raise
    
    async def get_video_result(
        self,
        operation_id: str
    ) -> VideoGenerationResponse:
        """
        获取生成的视频结果
        
        Args:
            operation_id: 操作 ID
        
        Returns:
            VideoGenerationResponse: 视频生成结果
        """
        url = f"{self.base_url}/{operation_id}?key={self.api_key}"
        headers = {"Content-Type": "application/json"}
        
        extra_log = {
            "key": self.api_key[:8],
            "request_type": "video-result",
            "operation_id": operation_id[:20] if operation_id else "",
        }
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, headers=headers, timeout=30)
                response.raise_for_status()
            
            gemini_response = response.json()
            
            # 检查是否完成
            if not gemini_response.get("done", False):
                raise ValueError("Video generation not completed yet")
            
            # 检查错误
            error = gemini_response.get("error")
            if error:
                raise ValueError(f"Video generation failed: {error.get('message', 'Unknown error')}")
            
            # 提取视频数据
            result = gemini_response.get("response", {})
            predictions = result.get("predictions", [])
            
            videos: List[VideoData] = []
            for prediction in predictions:
                video_data = VideoData(
                    b64_json=prediction.get("bytesBase64Encoded"),
                    url=prediction.get("uri")
                )
                videos.append(video_data)
            
            log("INFO", f"视频生成完成，共 {len(videos)} 个视频", extra=extra_log)
            
            return VideoGenerationResponse(
                id=operation_id,
                status="completed",
                created=int(time.time()),
                completed=int(time.time()),
                data=videos
            )
            
        except httpx.HTTPStatusError as e:
            log("ERROR", f"获取视频结果失败: {e.response.text}", extra=extra_log)
            raise
        except Exception as e:
            log("ERROR", f"获取视频结果异常: {str(e)}", extra=extra_log)
            raise
    
    @staticmethod
    def is_veo_model(model: str) -> bool:
        """检查是否为 Veo 模型"""
        return model in VeoClient.AVAILABLE_MODELS or model.startswith("veo")
