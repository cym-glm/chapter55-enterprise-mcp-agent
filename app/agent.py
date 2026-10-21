import json
import os

from langchain_core.messages import (
HumanMessage,
SystemMessage,
ToolMessage,
)
from langchain_openai import ChatOpenAI

from app.gateway import gateway


def to_llm_tool(tool: dict) -> dict:
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool["description"],
            "parameters": tool["input_schema"],
        },
    }


async def run_agent(
    question: str,
    role: str,
) -> dict:
    if not os.getenv("LLM_API_KEY"):
        raise RuntimeError(
            "请先在 .env 中配置 LLM_API_KEY"
        )

    model = ChatOpenAI(
        model=os.getenv("LLM_MODEL", "deepseek-chat"),
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL") or None,
        temperature=0,
    )

    # 第一步：发现当前角色有权限使用的工具
    tools = await gateway.discover_tools(role)

    if not tools:
        return {
            "success": False,
            "message": "当前角色没有可用的 MCP 工具",
        }

    llm = model.bind_tools([
        to_llm_tool(tool)
        for tool in tools
    ])

    messages = [
        SystemMessage(
            content=(
                "你是企业业务 Agent。"
                "必须根据 MCP 工具的实际结果回答问题，"
                "不能编造订单、客户或知识库数据。"
                "涉及退款时，只能提交退款申请，"
                "不得声称退款已经实际完成。"
            )
        ),
        HumanMessage(content=question),
    ]

    # 第二步：执行大模型工具调用循环
    max_rounds = 6

    for _ in range(max_rounds):
        response = await llm.ainvoke(messages)
        messages.append(response)

        if not response.tool_calls:
            return {
                "success": True,
                "role": role,
                "answer": response.content,
                "traces": gateway.traces[-10:],
            }

        for call in response.tool_calls:
            tool_name = call["name"]
            arguments = call["args"]

            # 第三步：Gateway 重新校验权限并调用 MCP Server
            result = await gateway.call_tool(
                role=role,
                tool_name=tool_name,
                arguments=arguments,
            )

            # 第四步：把真实工具结果返回给大模型
            messages.append(
                ToolMessage(
                    content=json.dumps(
                        result,
                        ensure_ascii=False,
                        default=str,
                    ),
                    tool_call_id=call["id"],
                )
            )

    return {
        "success": False,
        "message": "Agent 已达到最大工具调用轮数",
        "traces": gateway.traces[-10:],
    }