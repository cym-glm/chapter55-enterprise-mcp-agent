from mcp.server import MCPServer

from app.mock_systems import query_customer

mcp = MCPServer(
    "crm-server",
    instructions="提供企业客户信息查询能力。",
)


@mcp.tool()
def get_customer(customer_id: str) -> dict:
    """根据客户编号查询客户信息。"""

    return query_customer(customer_id)


if __name__ == "__main__":
    mcp.run()