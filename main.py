import os
from typing import TypedDict

from dotenv import load_dotenv
from deepagents import create_deep_agent
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph


# 1. 加载环境变量
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL")
model_name = os.getenv("OPENAI_MODEL", "deepseek-chat")

if not api_key:
    raise RuntimeError(
        "没有配置 OPENAI_API_KEY，请检查项目根目录下的 .env 文件。"
    )


# 2. 创建模型实例
# LangChain 与 DeepAgents 都使用这个模型配置。
model = ChatOpenAI(
    model=model_name,
    api_key=api_key,
    base_url=base_url,
    temperature=0.2,
)


# 3. 创建业务分析 Agent：使用 LangChain 官方 API
business_agent = create_agent(
    model=model,
    tools=[],
    system_prompt="""
你是一名企业产品与商业分析专家。

你的职责：
1. 分析目标客户和核心业务问题。
2. 分析产品价值和潜在应用场景。
3. 判断需求是否明确，指出关键假设。
4. 分析商业风险、推广难点和收益衡量方式。

输出要求：
- 使用中文。
- 结论必须有清晰的推理依据。
- 不要编造市场规模、客户数据或财务数字。
- 缺少数据时明确标注“需要进一步验证”。
- 只负责业务与商业分析，不负责设计具体技术架构。
""",
)


# 4. 创建技术分析 Agent：使用 DeepAgents 官方 API
technical_agent = create_deep_agent(
    model=model,
    tools=[],
    system_prompt="""
你是一名企业 AI 应用架构师。

你的职责：
1. 分析系统实现所需的主要模块。
2. 评估大模型、RAG、工具调用、MCP、数据库等技术
   是否有必要使用。
3. 分析系统集成、权限、安全、可观测性和部署方面的风险。
4. 给出分阶段实施建议。
5. 指出技术方案依赖哪些尚未确认的业务条件。

输出要求：
- 使用中文。
- 只提出与需求相关的技术方案，不要为了使用新技术而堆砌技术。
- 不要虚构已经完成的系统或测试结果。
- 对没有充分信息的结论明确说明假设。
- 只负责技术分析，不负责最终商业立项决策。
""",
)


# 5. 定义 LangGraph 的共享状态
class ProjectState(TypedDict, total=False):
    project_requirement: str
    business_analysis: str
    technical_analysis: str
    final_report: str


# 6. 定义业务分析节点
def analyze_business(state: ProjectState) -> dict:
    """调用业务分析 Agent。"""

    result = business_agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "请对下面的产品需求进行业务分析。\n\n"
                        f"{state['project_requirement']}"
                    ),
                }
            ]
        }
    )

    answer = result["messages"][-1].content

    return {
        "business_analysis": answer,
    }


# 7. 定义技术分析节点
def analyze_technology(state: ProjectState) -> dict:
    """调用技术分析 Agent。"""

    result = technical_agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "请对下面的产品需求进行技术可行性分析。\n\n"
                        f"{state['project_requirement']}"
                    ),
                }
            ]
        }
    )

    answer = result["messages"][-1].content

    return {
        "technical_analysis": answer,
    }


# 8. 定义汇总节点
def generate_report(state: ProjectState) -> dict:
    """整合两个专业 Agent 的分析结果。"""

    report_prompt = f"""
你是一名企业项目评审负责人。

请基于下面两份分析结果，输出一份新产品可行性评估报告。

【原始需求】
{state["project_requirement"]}

【业务分析结果】
{state["business_analysis"]}

【技术分析结果】
{state["technical_analysis"]}

请按以下结构输出：

一、项目背景与目标
二、业务价值分析
三、技术实现建议
四、主要风险与待验证假设
五、建议的实施阶段
六、是否建议进入下一阶段评审

要求：
    1. 区分已知信息、分析判断和待验证假设。
2. 不得编造业务数据或测试结果。
3. 不要将技术可行性直接等同于商业可行性。
4. 对信息不足的地方，给出需要补充的数据。
5. 最终结论应谨慎，并说明判断依据。
"""

    result = model.invoke(report_prompt)

    return {
        "final_report": str(result.content),
    }


# 9. 使用 LangGraph 官方 StateGraph 编排
builder = StateGraph(ProjectState)

builder.add_node("business_analysis", analyze_business)
builder.add_node("technical_analysis", analyze_technology)
builder.add_node("generate_report", generate_report)

# 本章采用固定流程：
# 先做业务分析，再做技术分析，最后汇总。
builder.add_edge(START, "business_analysis")
builder.add_edge("business_analysis", "technical_analysis")
builder.add_edge("technical_analysis", "generate_report")
builder.add_edge("generate_report", END)

# 编译成可以执行的图
project_graph = builder.compile()


# 10. 运行项目
def main() -> None:
    requirement = """
我们计划为中小型电商企业开发一款 AI 客服产品。

产品希望帮助商家处理常见商品咨询、订单状态查询、
退换货政策查询和售后问题。

初期先服务少量试点商家，希望控制开发成本，
并且需要考虑后续接入商家的订单系统、商品系统和知识库。

目前还没有准确的客户数量、预算和业务效果数据。
请评估这个产品是否值得进入下一阶段的立项评审。
"""

    result = project_graph.invoke(
        {
            "project_requirement": requirement,
        }
    )

    print("\n" + "=" * 60)
    print("企业新产品可行性分析报告")
    print("=" * 60)
    print(result["final_report"])


if __name__ == "__main__":
    main()


