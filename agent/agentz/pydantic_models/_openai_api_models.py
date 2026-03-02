
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field, ConfigDict, field_serializer
from datetime import datetime, timezone
from typing import Optional

class GPTClientRequest(BaseModel):
    prompt: str = Field(..., description="Prompt or instruction for the tool.")
    tool_schema: type[BaseModel] = Field(..., description="Pydantic model class for the tool schema.")
    overrides: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Overrides for tool execution.")
    cid: Optional[str] = Field(None, description="Conversation id for the job.")

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    @field_serializer("tool_schema")
    def _serialize_tool_schema(self, v: type[BaseModel]) -> str:
        """
        Serialize a class object into 'module.ClassName' for JSON compatibility.
        """
        try:
            return f"{v.__module__}.{v.__name__}"
        except Exception:
            return str(v)
class GPTClientConversationMessage(BaseModel):
    agent : Any = Field(..., description = "")
    llm   : Any = Field(..., description = "")
    timestamp : datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="")

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

class GPTClientConversation(BaseModel):
    cid         : str = Field(..., description="Conversation unique identificator")
    description : Optional[str] = Field("", description="Conversation details")
    created_at  : datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="")
    chat        : List[GPTClientConversationMessage] = Field(default_factory=list, description="")

    model_config = ConfigDict(extra="forbid", validate_assignment=True)