import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL")
model_name = os.getenv("OPENAI_MODEL", "deepseek-chat")

if not api_key:
    raise RuntimeError("请先在 .env 中配置 OPENAI_API_KEY")

model = ChatOpenAI(
    model=model_name,
    api_key=api_key,
    base_url=base_url,
    temperature=0.2,
)


# 1. 创建业务分析 Agent
business_agent = create_agent(
    model=model,
    tools=[],
    system_prompt="""
你是企业产品商业分析专家。

重点分析：
- 目标客户和用户痛点
- 产品价值与业务场景
- 商业风险和关键假设
- 需要验证的数据

只负责业务分析，不负责技术架构设计。
没有可靠数据时，不要编造市场规模、营收或客户数量。
请用中文回答，输出清晰、结构化的结论。
""",
)


# 2. 创建技术分析 Agent
technical_agent = create_agent(
    model=model,
    tools=[],
    system_prompt="""
你是企业 AI 应用架构师。

重点分析：
- 系统模块和技术架构
- 大模型、RAG、工具调用、MCP 的使用必要性
- 数据库、权限、安全和部署
- 实施难点、技术风险和阶段计划

只负责技术分析，不负责商业价值判断。
不要为了使用新技术而堆砌技术。
请用中文回答。
""",
)


# 3. 将业务 Agent 封装成 Supervisor 可以调用的工具
@tool
def analyze_business(requirement: str) -> str:
    """分析产品的商业价值、目标客户、业务需求和商业风险。"""

    result = business_agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": requirement,
                }
            ]
        }
    )

    return str(result["messages"][-1].content)


# 4. 将技术 Agent 封装成工具
@tool
def analyze_technology(requirement: str) -> str:
    """分析产品的技术可行性、系统架构、技术风险和实施计划。"""

    result = technical_agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": requirement,
                }
            ]
        }
    )

    return str(result["messages"][-1].content)


# 5. 创建 Supervisor Agent
supervisor = create_agent(
    model=model,
    tools=[
        analyze_business,
        analyze_technology,
    ],
    system_prompt="""
你是企业项目评审 Supervisor，负责协调专业分析 Agent。

你可以使用以下工具：
1. analyze_business：业务价值、客户需求、商业风险分析。
2. analyze_technology：技术架构、技术可行性、实施风险分析。

工作原则：

- 商业价值问题优先使用业务分析工具。
- 技术选型、架构和实施问题优先使用技术分析工具。
- 需要综合评估立项时，应结合业务和技术两方面的分析。
- 不相关的专业 Agent 不要无故调用。
- 对缺少证据的数据，明确标注需要验证。
- 工具返回的结果是分析依据，不是绝对事实。
- 不要声称已经执行了实际测试或查询了不存在的业务系统。
- 最终用中文输出结论、依据、风险和后续建议。
""",
)


def main() -> None:
    print("企业项目评审 Supervisor Agent")
    print("输入 exit 退出。\n")

    while True:
        question = input("请输入项目问题：").strip()

        if question.lower() in {"exit", "quit"}:
            break

        if not question:
            continue

        result = supervisor.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ]
            }
        )

        print("\n===== Supervisor 最终答复 =====")
        print(result["messages"][-1].content)
        print()


if __name__ == "__main__":
    main()