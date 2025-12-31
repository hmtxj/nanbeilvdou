"""
图像生成请求处理器
处理 /v1/images/generations 端点的请求
"""
import asyncio
from typing import Optional

from app.models.schemas import (
    ImageGenerationRequest,
    ImageGenerationResponse,
    ImageData,
)
from app.services.imagen import ImagenClient
from app.utils import update_api_call_stats
from app.utils.error_handling import handle_gemini_error
from app.utils.logging import log
from app.utils.stats import get_api_key_usage
import app.config.settings as settings


async def process_image_generation(
    request: ImageGenerationRequest,
    key_manager,
) -> ImageGenerationResponse:
    """
    处理图像生成请求
    
    Args:
        request: 图像生成请求
        key_manager: API 密钥管理器
    
    Returns:
        ImageGenerationResponse: 图像生成响应
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
                    "request_type": "image-generation",
                    "model": request.model,
                },
            )
            current_try_num += 1
            continue
        
        current_try_num += 1
        
        try:
            log(
                "info",
                f"图像生成请求开始，使用密钥: {api_key[:8]}...",
                extra={
                    "key": api_key[:8],
                    "request_type": "image-generation",
                    "model": request.model,
                },
            )
            
            # 创建客户端并生成图像
            client = ImagenClient(api_key)
            response = await client.generate_image(request)
            
            # 更新 API 调用统计
            await update_api_call_stats(
                settings.api_call_stats,
                endpoint=api_key,
                model=request.model,
                token=0,  # 图像生成不计算 token
            )
            
            log(
                "info",
                f"图像生成成功，生成 {len(response.data)} 张图像",
                extra={
                    "key": api_key[:8],
                    "request_type": "image-generation",
                    "model": request.model,
                },
            )
            
            return response
            
        except Exception as e:
            last_error = e
            handle_gemini_error(e, api_key)
            log(
                "error",
                f"图像生成失败: {str(e)}",
                extra={
                    "key": api_key[:8],
                    "request_type": "image-generation",
                    "model": request.model,
                },
            )
    
    # 所有密钥都失败
    log(
        "error",
        "所有 API 密钥均请求失败",
        extra={"request_type": "image-generation", "model": request.model},
    )
    
    if last_error:
        raise last_error
    else:
        raise ValueError("No available API keys for image generation")
