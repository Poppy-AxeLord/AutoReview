# 定义流式输出回调函数
def stream_callback(
    chunk: Dict[str, Any],
    sender: autogen.Agent,
    recipient: autogen.Agent,
    request_id: Optional[str] = None,
) -> None:
    """
    流式输出回调函数：实时打印AI生成的内容
    - chunk: 流式返回的单个数据块
    - sender: 发送消息的Agent
    - recipient: 接收消息的Agent
    """
    # 提取chunk中的文本内容
    if "choices" in chunk and len(chunk["choices"]) > 0:
        delta = chunk["choices"][0].get("delta", {})
        content = delta.get("content", "")
        # 实时打印（end='' 不换行，flush=True 强制刷新输出）
        if content:
            print(content, end="", flush=True)