
from mcp.server import MCPServer
from app.mock_systems import query_order, request_refund



mcp = MCPServer(
    "order_server",
    instructions="提供企业订单查询和退款申请工具。"
)

@mcp.tool()
def get_order(order_id: str) -> dict:
    """ 查询订单详情。"""
    return query_order(order_id)


@mcp.tool()
def refund_order(order_id: str, reason: str) -> dict:
    """提交退款申请。此工具会创建待审批申请，不会直接退款。"""

    return request_refund(order_id, reason)

if __name__ == "__main__":
    mcp.run()