ORDERS = {
    "O1001": {
        "order_id": "O1001",
        "customer_id": "C1001",
        "amount": 26800,
        "status": "shipped",
        "status_text": "已发货",
    },
    "O1002": {
        "order_id": "O1002",
        "customer_id": "C1002",
        "amount": 6800,
        "status": "pending",
        "status_text": "待发货",
    },
}

CUSTOMERS = {
    "C1001": {
        "customer_id": "C1001",
        "name": "星河科技",
        "level": "VIP",
        "owner": "张三",
        "last_contact": "2026-10-18",
    },
    "C1002": {
        "customer_id": "C1002",
        "name": "云帆科技",
        "level": "普通客户",
        "owner": "李四",
        "last_contact": "2026-10-16",
    },
}

KNOWLEDGE = {
    "refund_policy": (
        "退款申请需要验证订单状态、退款金额和申请原因。"
        "高风险退款必须由有权限的人员审批。"
    ),
    "shipping_policy": (
        "待发货订单由仓储系统处理。"
        "已经发货的订单需要先查询物流状态。"
    ),
}


def query_order(order_id: str) -> dict:
    order = ORDERS.get(order_id)

    if order is None:
        return {
            "success": False,
            "message": f"订单 {order_id} 不存在",
        }

    return {
        "success": True,
        "data": order,
    }


def query_customer(customer_id: str) -> dict:
    customer = CUSTOMERS.get(customer_id)

    if customer is None:
        return {
            "success": False,
            "message": f"客户 {customer_id} 不存在",
        }

    return {
        "success": True,
        "data": customer,
    }


def query_knowledge(topic: str) -> dict:
    content = KNOWLEDGE.get(topic)

    if content is None:
        return {
            "success": False,
            "message": f"知识主题 {topic} 不存在",
        }

    return {
        "success": True,
        "data": {
            "topic": topic,
            "content": content,
        },
    }


def request_refund(
    order_id: str,
    reason: str,
) -> dict:
    """教学用退款操作模拟，不会真正执行退款。"""

    order = ORDERS.get(order_id)

    if order is None:
        return {
            "success": False,
            "message": f"订单 {order_id} 不存在",
        }

    return {
        "success": True,
        "message": "退款申请已提交至模拟审批流程",
        "order_id": order_id,
        "reason": reason,
        "status": "pending_approval",
    }