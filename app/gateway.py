import logging
import time
import uuid

from mcp import Client, StdioServerParameters

logger = logging.getLogger(__name__)

SERVER_CONFIG = {
    "order": {
        "command": "uv",
        "args": [
            "run",
            "--project",
            ".",
            "python",
            "-m",
            "app.servers.order_server",
        ],
        "tools": {"get_order", "refund_order"},
    },
    "crm": {
        "command": "uv",
        "args": [
            "run",
            "--project",
            ".",
            "python",
            "-m",
            "app.servers.crm_server",
        ],
        "tools": {"get_customer"},
    },
    "knowledge": {
        "command": "uv",
        "args": [
            "run",
            "--project",
            ".",
            "python",
            "-m",
            "app.servers.kb_server",
        ],
        "tools": {"search_knowledge"},
    },
}

ROLE_TOOLS = {
    "sales": {
        "get_order",
        "get_customer",
        "search_knowledge",
    },
    "support": {
        "get_order",
        "get_customer",
        "search_knowledge",
        "refund_order",
    },
    "finance": {
        "get_order",
        "get_customer",
        "search_knowledge",
        "refund_order",
    },
}


class MCPGateway:
    def __init__(self):
        self.traces: list[dict] = []
        self.approvals: list[dict] = []

    def get_allowed_tools(self, role: str) -> set[str]:
        return ROLE_TOOLS.get(role, set())

    async def discover_tools(self, role: str) -> list[dict]:
        """动态发现工具，并根据角色过滤。"""

        allowed = self.get_allowed_tools(role)
        discovered = []

        for server_name, config in SERVER_CONFIG.items():
            params = StdioServerParameters(
                command=config["command"],
                args=config["args"],
            )

            async with Client(params) as client:
                result = await client.list_tools()

                for tool in result.tools:
                    if tool.name not in config["tools"]:
                        continue

                    if tool.name not in allowed:
                        continue

                    discovered.append({
                        "server": server_name,
                        "name": tool.name,
                        "description": tool.description or "",
                        "input_schema": tool.input_schema,
                    })

        return discovered

    async def call_tool(
        self,
        role: str,
        tool_name: str,
        arguments: dict,
        approved: bool = False,
    ) -> dict:
        """执行权限校验、审批检查、MCP 工具调用和 Trace。"""

        trace_id = str(uuid.uuid4())
        started_at = time.perf_counter()

        allowed = self.get_allowed_tools(role)

        if tool_name not in allowed:
            return self._record_trace({
                "trace_id": trace_id,
                "tool": tool_name,
                "role": role,
                "status": "permission_denied",
                "duration_ms": 0,
                "result": {
                    "success": False,
                    "message": "当前用户没有权限调用该工具",
                },
            })

        if tool_name == "refund_order":
            approval = {
                "approval_id": str(uuid.uuid4()),
                "trace_id": trace_id,
                "tool": tool_name,
                "role": role,
                "arguments": arguments,
                "status": "approved" if approved else "pending",
            }

            self.approvals.append(approval)

            if not approved:
                return self._record_trace({
                    "trace_id": trace_id,
                    "tool": tool_name,
                    "role": role,
                    "status": "pending_approval",
                    "duration_ms": 0,
                    "result": {
                        "success": False,
                        "approval_id": approval["approval_id"],
                        "message": "退款申请需要人工审批，尚未调用业务工具",
                    },
                })

        server_name = next(
            (
                name
                for name, config in SERVER_CONFIG.items()
                if tool_name in config["tools"]
            ),
            None,
        )

        if server_name is None:
            return self._record_trace({
                "trace_id": trace_id,
                "tool": tool_name,
                "role": role,
                "status": "tool_not_found",
                "duration_ms": 0,
                "result": {
                    "success": False,
                    "message": "没有找到对应的 MCP Server",
                },
            })

        config = SERVER_CONFIG[server_name]

        params = StdioServerParameters(
            command=config["command"],
            args=config["args"],
        )

        try:
            async with Client(params) as client:
                result = await client.call_tool(
                    tool_name,
                    arguments,
                )

                payload = {
                    "structured_content": result.structured_content,
                    "content": [
                        getattr(item, "text", "")
                        for item in result.content
                    ],
                    "is_error": result.is_error,
                }

                status = "tool_error" if result.is_error else "success"

        except Exception as exc:
            logger.exception(
                "MCP tool call failed: %s",
                tool_name,
            )

            payload = {
                "success": False,
                "message": f"{type(exc).__name__}: {exc}",
            }
            status = "exception"

        duration_ms = round(
            (time.perf_counter() - started_at) * 1000,
            2,
        )

        return self._record_trace({
            "trace_id": trace_id,
            "tool": tool_name,
            "server": server_name,
            "role": role,
            "status": status,
            "duration_ms": duration_ms,
            "result": payload,
        })

    def _record_trace(self, trace: dict) -> dict:
        self.traces.append(trace)

        logger.info(
            "mcp_trace=%s",
            trace,
        )

        return trace


gateway = MCPGateway()