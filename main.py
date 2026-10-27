import os
from typing import Literal, TypedDict

from dotenv import load_dotenv
from langchain.agents import create_agent
from deepagents import create_deep_agent
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL")
model_name = os.getenv("OPENAI_MODEL", "deepseek-chat")

if not api_key:
    raise RuntimeError("请先配置 OPENAI_API_KEY")

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
你是企业产品商业分析师。
分析目标客户、业务痛点、商业价值、关键假设和业务风险。
不要编造市场规模、客户数量或收入数据。
请使用中文结构化输出。
""",
)


# 2. 创建技术分析 Agent
technical_agent = create_agent(
    model=model,
    tools=[],
    system_prompt="""
你是企业 AI 应用架构师。
分析系统模块、技术选型、集成方案、权限、安全、部署和实施风险。
不要为了使用新技术而堆砌技术。
请使用中文结构化输出。
""",
)


# 3. 定义 LangGraph 共享状态
class CollaborationState(TypedDict, total=False):
    question: str
    mode: Literal["sequential", "parallel", "conditional"]
    route: Literal["business", "technical", "both"]
    business_result: str
    technical_result: str
    final_result: str


# 4. 专业 Agent 节点
def run_business(state: CollaborationState) -> dict:
    result = business_agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": state["question"],
                }
            ]
        }
    )
    return {"business_result": str(result["messages"][-1].content)}


def run_technical(state: CollaborationState) -> dict:
    result = technical_agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": state["question"],
                }
            ]
        }
    )
    return {"technical_result": str(result["messages"][-1].content)}


# 5. 汇总节点
def summarize(state: CollaborationState) -> dict:
    business_result = state.get(
        "business_result",
        "本次没有执行业务分析。",
    )
    technical_result = state.get(
        "technical_result",
        "本次没有执行技术分析。",
    )

    prompt = f"""
请根据下面的结果生成最终答复。

用户问题：
{state["question"]}

业务分析：
{business_result}

技术分析：
{technical_result}

要求：
1. 不要将缺失结果编造成已完成的分析。
2. 区分事实、判断和待验证假设。
3. 根据用户原始问题组织答案。
4. 使用中文。
"""

    response = model.invoke(prompt)
    return {"final_result": str(response.content)}


# 6. 构建串行 Workflow
sequential_builder = StateGraph(CollaborationState)
sequential_builder.add_node("business", run_business)
sequential_builder.add_node("technical", run_technical)
sequential_builder.add_node("summary", summarize)

sequential_builder.add_edge(START, "business")
sequential_builder.add_edge("business", "technical")
sequential_builder.add_edge("technical", "summary")
sequential_builder.add_edge("summary", END)

sequential_graph = sequential_builder.compile()


# 7. 构建并行 Workflow
parallel_builder = StateGraph(CollaborationState)
parallel_builder.add_node("business", run_business)
parallel_builder.add_node("technical", run_technical)
parallel_builder.add_node("summary", summarize)

parallel_builder.add_edge(START, "business")
parallel_builder.add_edge(START, "technical")

parallel_builder.add_edge("business", "summary")
parallel_builder.add_edge("technical", "summary")
parallel_builder.add_edge("summary", END)

parallel_graph = parallel_builder.compile()


# 8. 条件路由：只决定执行哪个专业节点
def choose_route(
    state: CollaborationState,
) -> Literal["business", "technical", "both"]:
    prompt = f"""
判断用户问题需要哪些分析能力。

只能返回以下一个标签：
business
technical
both

business：主要分析商业价值、客户、市场、业务需求。
technical：主要分析架构、技术选型、开发、部署、安全。
both：需要综合分析商业价值和技术可行性。

用户问题：
{state["question"]}
"""

    response = model.invoke(prompt)
    value = str(response.content).strip().lower()

    if value == "business":
        return "business"

    if value == "technical":
        return "technical"

    return "both"


def route_after_classification(
    state: CollaborationState,
) -> Literal["business", "technical", "both"]:
    return state["route"]


def classify_question(state: CollaborationState) -> dict:
    route = choose_route(state)
    return {"route": route}


def build_conditional_graph():
    builder = StateGraph(CollaborationState)

    builder.add_node("classify", classify_question)
    builder.add_node("business", run_business)
    builder.add_node("technical", run_technical)
    builder.add_node("summary", summarize)

    builder.add_edge(START, "classify")

    builder.add_conditional_edges(
        "classify",
        route_after_classification,
        {
            "business": "business",
            "technical": "technical",
            "both": "business",
        },
    )

    # 为保持示例简洁，both 路由先执行业务分析，
    # 再通过条件边进入技术分析。
    # 其他路由在各自分析后进入汇总。
    def next_after_business(
        state: CollaborationState,
    ) -> Literal["technical", "summary"]:
        if state["route"] == "both":
            return "technical"
        return "summary"

    builder.add_conditional_edges(
        "business",
        next_after_business,
        {
            "technical": "technical",
            "summary": "summary",
        },
    )

    builder.add_edge("technical", "summary")
    builder.add_edge("summary", END)

    return builder.compile()


conditional_graph = build_conditional_graph()


def main() -> None:
    question = """
我们计划开发一款面向中小商家的 AI 客服产品，
希望帮助商家处理商品咨询、订单查询和售后问题。
目前缺少准确的预算、客户数量和业务效果数据。
请评估项目的商业价值和技术可行性。
"""

    print("\n===== 串行协作 =====")
    sequential_result = sequential_graph.invoke(
        {
            "question": question,
            "mode": "sequential",
        }
    )
    print(sequential_result["final_result"])

    print("\n===== 并行协作 =====")
    parallel_result = parallel_graph.invoke(
        {
            "question": question,
            "mode": "parallel",
        }
    )
    print(parallel_result["final_result"])

    print("\n===== 条件路由 =====")
    conditional_result = conditional_graph.invoke(
        {
            "question": question,
            "mode": "conditional",
        }
    )
    print("路由结果：", conditional_result["route"])
    print(conditional_result["final_result"])


if __name__ == "__main__":
    main()