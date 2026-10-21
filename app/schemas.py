

from pydantic import BaseModel, Field

class AgentRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000, description="用户的问题")
    role: str = Field(default="sales", description="用户的角色，默认为'sales'")
