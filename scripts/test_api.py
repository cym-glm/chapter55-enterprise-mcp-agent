import asyncio

from app.gateway import gateway


async def main():
    # 销售角色没有退款申请权限
    denied = await gateway.call_tool(
        role="sales",
        tool_name="refund_order",
        arguments={
            "order_id": "O1001",
            "reason": "客户申请退款",
        },
    )

    print("销售角色：", denied)

    # 支持角色有提交申请权限，但仍需审批
    pending = await gateway.call_tool(
        role="support",
        tool_name="refund_order",
        arguments={
            "order_id": "O1001",
            "reason": "客户申请退款",
        },
    )

    print("支持角色：", pending)


if __name__ == "__main__":
    asyncio.run(main())