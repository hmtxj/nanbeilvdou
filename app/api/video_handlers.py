"""
视频生成请求处理器
处理 /v1/videos/generations 端点的请求
"""
import asyncio
from typing import Optional

from app.models.schemas import (
    VideoGenerationRequest,
    VideoGenerationOperation,
    VideoOperationStatus,
    VideoGenerationResponse,
)
from app.services.veo import VeoClient
from app.utils import update_api_call_stats
from app.utils.error_handling import handle_gemini_error
from app.utils.logging import log
from app.utils.stats import get_api_key_usage
import app.config.settings as settings


async def process_video_generation(
    request: VideoGenerationRequest,
    key_manager,
) -> VideoGenerationOperation:
    """
    处理视频生成请求 (发起异步任务)
    
    Args:
        request: 视频生成请求
        key_manager: API 密钥管理器
    
    Returns:
        VideoGenerationOperation: 包含操作 ID 的响应
    """
    max_retry_num = settings.MAX_RETRY_NUM
    current_try_num = 0
    last_error: Optional[Exception] = None
    
    # 尝试使用不同 API 密钥
    while current_try_num < max_retry_num:
        # 获取可用密钥
        api_key = await key_manager.get_available_key()
        if not api_key:
            break
        
        # 检查密钥使用限制
        usage = await get_api_key_usage(settings.api_call_stats, api_key)
        if usage >= settings.API_KEY_DAILY_LIMIT:
            log(
                "warning",
                f"API密钥 {api_key[:8]}... 已达到每日调用限制",
                extra={
                    "key": api_key[:8],
                    "request_type": "video-generation",
                    "model": request.model,
                },
            )
            current_try_num += 1
            continue
        
        current_try_num += 1
        
        try:
            log(
                "info",
                f"视频生成请求开始，使用密钥: {api_key[:8]}...",
                extra={
                    "key": api_key[:8],
                    "request_type": "video-generation",
                    "model": request.model,
                },
            )
            
            # 创建客户端并发起视频生成
            client = VeoClient(api_key)
            response = await client.generate_video(request)
            
            # 更新 API 调用统计
            await update_api_call_stats(
                settings.api_call_stats,
                endpoint=api_key,
                model=request.model,
                token=0,
            )
            
            log(
                "info",
                f"视频生成任务已创建: {response.id}",
                extra={
                    "key": api_key[:8],
                    "request_type": "video-generation",
                    "model": request.model,
                },
            )
            
            return response
            
        except Exception as e:
            last_error = e
            handle_gemini_error(e, api_key)
            log(
                "error",
                f"视频生成失败: {str(e)}",
                extra={
                    "key": api_key[:8],
                    "request_type": "video-generation",
                    "model": request.model,
                },
            )
    
    # 所有密钥都失败
    log(
        "error",
        "所有 API 密钥均请求失败",
        extra={"request_type": "video-generation", "model": request.model},
    )
    
    if last_error:
        raise last_error
    else:
        raise ValueError("No available API keys for video generation")


async def process_video_status(
    operation_id: str,
    key_manager,
) -> VideoOperationStatus:
    """
    查询视频生成状态
    
    Args:
        operation_id: 操作 ID
        key_manager: API 密钥管理器
    
    Returns:
        VideoOperationStatus: 操作状态
    """
    # 获取可用密钥
    api_key = await key_manager.get_available_key()
    if not api_key:
        raise ValueError("No available API keys")
    
    try:
        client = VeoClient(api_key)
        return await client.get_operation_status(operation_id)
    except Exception as e:
        handle_gemini_error(e, api_key)
        raise


async def process_video_result(
    operation_id: str,
    key_manager,
) -> VideoGenerationResponse:
    """
    获取视频生成结果
    
    Args:
        operation_id: 操作 ID
        key_manager: API 密钥管理器
    
    Returns:
        VideoGenerationResponse: 视频生成结果
    """
    # 获取可用密钥
    api_key = await key_manager.get_available_key()
    if not api_key:
        raise ValueError("No available API keys")
    
    try:
        client = VeoClient(api_key)
        return await client.get_video_result(operation_id)
    except Exception as e:
        handle_gemini_error(e, api_key)
        raise
