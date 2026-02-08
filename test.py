import os
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage, ModelClientStreamingChunkEvent
from autogen_core import CancellationToken
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv
from autogen_ext.models.openai._model_info import ModelInfo

"""
[流式输出]核心变化
pyautogen 0.10.0 基于 v0.4 新架构
流式输出不再是旧版的 stream=True 参数，而是通过 model_client_stream=True 和 run_stream() 方法实现。
"""

load_dotenv()
async def main():
    # qwen-plus 配置
    model_info = ModelInfo(
        vision=False,           # qwen-plus 不支持视觉（qwen-vl 才支持）
        function_calling=True,  # 支持工具/函数调用
        json_output=True,       # 支持 JSON 输出
        family="qwen"           # 模型家族标识
    )
    # 1. 创建模型客户端（替代你的 config_list）
    model_client = OpenAIChatCompletionClient(
        model=os.getenv("OPENAI_MODEL", "gpt-4o"),
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        model_info=model_info,  # 手动提供模型信息
    )
    
    # 2. 创建 Agent
    agent = AssistantAgent(
        name="assistant",
        model_client=model_client,
        system_message="你是助手。",
        model_client_stream=True,  # 启用流式
    )
    
    # 3. 流式运行
    async for message in agent.on_messages_stream(
        [TextMessage(content="你好，请介绍自己", source="user")],
        cancellation_token=CancellationToken(),
    ):
        if isinstance(message, ModelClientStreamingChunkEvent):
            print(message.content, end="", flush=True)
        else:
            print(f"\n\n[{type(message).__name__}]")
    
    # 4. 关闭客户端
    await model_client.close()

if __name__ == "__main__":
    asyncio.run(main())