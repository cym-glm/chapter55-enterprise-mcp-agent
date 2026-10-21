from mcp.server import MCPServer

from app.mock_systems import query_knowledge

mcp = MCPServer(
    "knowledge-server",
    instructions="提供企业制度和业务规则查询能力。",
)


@mcp.tool()
def search_knowledge(topic: str) -> dict:
    """根据主题查询企业知识库内容。"""

    return query_knowledge(topic)


if __name__ == "__main__":
    mcp.run()