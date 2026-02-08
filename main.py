import os
import asyncio
from typing import Dict, List, Any
from dotenv import load_dotenv

# 新版 autogen 导入
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.messages import TextMessage, ModelClientStreamingChunkEvent, ToolCallExecutionEvent
from autogen_core import CancellationToken
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.models.openai._model_info import ModelInfo

# 加载外部 PR Diff 函数
from PR_diff import get_gitee_pr_with_diff

# -------------------------- 1. 配置基础环境 --------------------------
load_dotenv()

# qwen-plus 模型信息
model_info = ModelInfo(
    vision=False,
    function_calling=True,
    json_output=True,
    family="qwen"
)

# 创建模型客户端
model_client = OpenAIChatCompletionClient(
    model=os.getenv("OPENAI_MODEL", "qwen-plus"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    model_info=model_info,
)

def get_pr_diff():
    GITEE_OWNER = "poppyaxelord"
    GITEE_REPO = "Langgraph-task-split"
    PR_NUMBER = 1
    diff_text = get_gitee_pr_with_diff(GITEE_OWNER, GITEE_REPO, PR_NUMBER)
    print(f"✅ 成功获取 PR Diff 数据:\n{diff_text}")
    return diff_text

# -------------------------- 3. 定义 4 个审核智能体 --------------------------

# Agent1: 语法规范审核员
syntax_reviewer_agent = AssistantAgent(
    name="SyntaxReviewer",
    system_message="""你是专业的代码语法规范审核员，你的审核范围严格限定为：
    1. 代码格式：缩进、空格、换行、注释规范
    2. 命名规范：变量、函数、类、常量的命名是否符合行业标准
    3. 语法错误：语法层面的错误、未定义变量、类型错误等
    4. 语言特性：是否合理使用对应编程语言的特性，避免反模式
    
    审核要求：
    - 针对每个问题标注严重程度（致命/高危/中危/低危）
    - 给出具体的修改建议和示例
    - 只关注语法规范，不涉及逻辑、安全、性能问题
    - 基于提供的 diff 代码进行审核，不要假设其他内容
    
    输出格式：使用 Markdown，结构清晰。""",
    model_client=model_client,
    model_client_stream=True,
)

# Agent2: 逻辑安全审核员
security_reviewer_agent = AssistantAgent(
    name="SecurityReviewer",
    system_message="""你是专业的代码逻辑安全审核员，你的审核范围严格限定为：
    1. 业务逻辑漏洞：边界条件处理、异常场景覆盖、逻辑判断错误
    2. 安全风险：SQL 注入、XSS、越权访问、敏感信息泄露、密码明文存储
    3. 资源安全：内存泄漏、文件句柄未释放、连接池未关闭
    4. 权限控制：访问控制是否严格、最小权限原则是否遵守
    
    审核要求：
    - 针对每个问题标注严重程度（致命/高危/中危/低危）
    - 给出具体的修复方案和安全建议
    - 只关注逻辑和安全，不涉及语法、性能问题
    - 基于提供的 diff 代码进行审核，不要假设其他内容
    
    输出格式：使用 Markdown，结构清晰。""",
    model_client=model_client,
    model_client_stream=True,
)

# Agent3: 性能优化审核员
performance_reviewer_agent = AssistantAgent(
    name="PerformanceReviewer",
    system_message="""你是专业的代码性能优化审核员，你的审核范围严格限定为：
    1. 循环效率：不必要的循环、嵌套过深、循环内的高开销操作
    2. 资源占用：内存使用、CPU 占用、网络请求频次
    3. 算法复杂度：时间复杂度、空间复杂度是否最优
    4. 缓存策略：是否合理使用缓存、避免重复计算
    
    审核要求：
    - 针对每个问题标注严重程度（致命/高危/中危/低危）
    - 给出具体的性能优化建议和代码示例
    - 只关注性能，不涉及语法、安全问题
    - 基于提供的 diff 代码进行审核，不要假设其他内容
    
    输出格式：使用 Markdown，结构清晰。""",
    model_client=model_client,
    model_client_stream=True,
)

# Agent4: 汇总报告生成员
report_summarizer_agent = AssistantAgent(
    name="ReportSummarizer",
    system_message="""你是代码审核报告汇总专家，你的唯一职责是：
    1. 收集语法规范、逻辑安全、性能优化三位审核员的所有意见
    2. 按严重程度（致命/高危/中危/低危）分类整理所有问题
    3. 为每个问题保留审核员的原始建议，并补充可落地的执行步骤
    4. 生成结构化、清晰易读的最终审核报告，包含：
       - 审核概要（问题总数、各严重程度分布）
       - 分严重程度的问题列表
       - 优先级修复建议
       - 整体代码质量评分（1-10分）
    
    输出格式要求：
    - 使用 Markdown 格式
    - 结构清晰，分章节展示
    - 语言简洁，重点突出""",
    model_client=model_client,
    model_client_stream=True,
)

# -------------------------- 3. 流式输出处理函数 --------------------------
async def stream_agent_response(agent, message: str, prefix: str = ""):
    """
    统一的流式输出处理函数
    """
    if prefix:
        print(f"\n{'='*50}")
        print(prefix)
        print(f"{'='*50}")
    
    full_content = ""
    
    async for msg in agent.run_stream(task=message):
        # 处理流式文本片段
        if isinstance(msg, ModelClientStreamingChunkEvent):
            print(msg.content, end="", flush=True)
            full_content += msg.content
        # 处理工具调用执行结果
        elif isinstance(msg, ToolCallExecutionEvent):
            print(f"\n[工具执行: {msg.name}]")
    
    print()  # 换行
    return full_content

# -------------------------- 4. 并行审核逻辑实现 --------------------------
async def run_parallel_review():
    """
    执行并行代码审核流程
    """
    # Step 1: 获取 PR Diff 数据
    print("=== 步骤1：获取 PR Diff 数据 ===")
    diff_text = get_pr_diff()
    
    # Step 2: 并行执行三个审核员的审核
    print("\n=== 步骤2：并行执行代码审核 ===")
    # 创建三个并行审核任务
    syntax_task = stream_agent_response(
        syntax_reviewer_agent,
        message=f"""请严格按照你的职责审核以下代码 diff 的语法规范问题：
        ```diff
        {diff_text}
        ```
        审核要求：
        逐行检查 diff 中的新增 / 修改代码
        对每个问题标注：
        严重程度（致命 / 高危 / 中危 / 低危）
        问题所在代码行
        具体修改建议（附代码示例）
        只关注语法规范相关问题，不要超出职责范围 """,
        prefix="🔍 语法规范审核（SyntaxReviewer）",
    )

    security_task = stream_agent_response(
        security_reviewer_agent,
        message=f"""
        请审核以下代码 diff 的逻辑安全问题：
        ```diff
        {diff_text}
        ```
        审核要求：
        重点检查 diff 中的新增 / 修改代码的安全风险
        对每个问题标注：
        严重程度（致命 / 高危 / 中危 / 低危）
        风险类型（如 SQL 注入、越权访问等）
        具体修复方案（附代码示例）
        只关注逻辑和安全相关问题，不要超出职责范围 """,
        prefix="🔒 逻辑安全审核（SecurityReviewer）"
    )
    performance_task = stream_agent_response (
        performance_reviewer_agent,
        message=f"""
        请审核以下代码 diff 的性能优化问题：
        ```diff
        {diff_text}
        ```
        审核要求：
        分析 diff 中代码的性能瓶颈
        对每个问题标注：
        严重程度（致命 / 高危 / 中危 / 低危）
        性能影响（如时间复杂度、资源占用）
        具体优化建议（附代码示例和性能提升预期）
        只关注性能相关问题，不要超出职责范围 """,
        prefix="🚀 性能优化审核（PerformanceReviewer）"
    )
    # 等待所有并行审核任务完成
    syntax_result, security_result, performance_result = await asyncio.gather(syntax_task, security_task, performance_task)

    # 提取各审核员的审核意见
    syntax_comments = syntax_result
    security_comments = security_result
    performance_comments = performance_result
    print ("✅ 所有审核任务完成")

    #Step 3: 生成最终汇总报告
    print ("\n=== 步骤 3：生成最终审核报告 ===")
    report_result = await stream_agent_response (
        report_summarizer_agent,
        message=f""" 请汇总以下三位审核员的意见，生成最终的代码审核报告：
        1. 语法规范审核意见：
        {syntax_comments}
        2. 逻辑安全审核意见：
        {security_comments}
        3. 性能优化审核意见：
        {performance_comments}
        最终报告要求：
        按严重程度（致命 > 高危 > 中危 > 低危）分类所有问题
        每个问题包含：问题描述、严重程度、影响范围、修复建议
        增加审核总结和修复优先级建议
        给出整体代码质量评分（1-10 分）
        使用 Markdown 格式，结构清晰易读 """,
        prefix="最终审核报告（ReportSummarizer）"
    )

    #提取并输出最终报告
    final_report = report_result
    print ("\n" + "="*50)
    print ("🎉 最终代码审核报告")
    print ("="*50)
    print (final_report)
    #保存报告到文件
    report_filename = f"review_results/pr_review_report.md"
    os.makedirs ("review_results", exist_ok=True)
    with open (report_filename, "w", encoding="utf-8") as f:f.write (final_report)
    print (f"\n✅ 报告已保存到：{report_filename}")

    return final_report

# -------------------------- 5. 主函数 --------------------------
if __name__ == "__main__":
    # 运行并行审核流程
    try:
        asyncio.run (run_parallel_review ())
        print ("\n✅ 代码审核流程全部完成！")
    except Exception as e:
        print (f"\n❌ 审核流程执行失败：{str (e)}")
        raise
