from __future__ import annotations

import asyncio
from typing import Any, AsyncGenerator, Dict, List, Optional

from pydantic import BaseModel

from agentz.pydantic_models import (
    ExperimentConfiguration,
    GPTClientConversation,
    GPTClientConversationMessage,
    GPTClientRequest,
)

from ._gpt_client import GPTClient


class GPTClientManager(GPTClient):
    def __init__(self, settings: ExperimentConfiguration):
        """
        Initialize `GPTClientManager` dependencies and runtime state.
        
        Parameters
        ----------
        settings : ExperimentConfiguration
            Runtime settings for this component.
        
        Returns
        -------
        None
            No return value.
        """
        super().__init__(settings)
        self.settings = settings.gpt_client_settings
        self.conversations: Dict[str, GPTClientConversation] = {}

    def create_conversation(self, cid: str) -> None:
        """
        Process create conversation.
        
        Parameters
        ----------
        cid : str
            Function argument.
        
        Returns
        -------
        None
            No return value.
        
        """
        description = "Chat messages are not part of a conversation" if cid == "no-conversation" else ""
        self.conversations[cid] = GPTClientConversation(
            cid=cid,
            description=description,
            chat=[],
        )

    def reset(self, settings: Optional[ExperimentConfiguration] = None) -> None:
        """
        Run reset for the current workflow step.
        
        Parameters
        ----------
        settings : Optional[ExperimentConfiguration]
            Runtime settings for this component.
        
        Returns
        -------
        None
            No return value.
        """
        if settings is not None:
            super().__init__(settings)
            self.settings = settings.gpt_client_settings
        self.conversations = {}

    def _append_to_conversation(self, request: GPTClientRequest, response: Optional[BaseModel]) -> None:
        """
        Process append to conversation.
        
        Parameters
        ----------
        request : GPTClientRequest
            Request payload.
        response : Optional[BaseModel]
            Response payload.
        
        Returns
        -------
        None
            No return value.
        
        """
        if request.cid is None:
            request.cid = "no-conversation"

        if self.conversations.get(request.cid) is None:
            self.create_conversation(request.cid)

        self.conversations[request.cid].chat.append(
            GPTClientConversationMessage(
                agent=request,
                llm=response,
            )
        )

    def chat_with_tool(self, request: GPTClientRequest) -> BaseModel:
        """
        Process chat with tool.
        
        Parameters
        ----------
        request : GPTClientRequest
            Request payload.
        
        Returns
        -------
        BaseModel
            Function result.
        
        """
        if request.cid is None:
            request.cid = "no-conversation"

        response = super().chat_with_tool(request)
        self._append_to_conversation(request, response)
        return response

    def chat_with_tools_batch(
        self,
        jobs: List[GPTClientRequest],
        parallelism: Optional[int] = None,
        allow_partial: bool = False,
    ) -> List[Optional[BaseModel]]:
        """
        Process chat with tools batch.
        
        Parameters
        ----------
        jobs : List[GPTClientRequest]
            Function argument.
        parallelism : Optional[int]
            Function argument.
        allow_partial : Optional[bool]
            Function argument.
        
        Returns
        -------
        List[Optional[BaseModel]]
            List with computed values.
        
        """
        del parallelism  # Current GPTClient has no native batch API.

        responses: List[Optional[BaseModel]] = []
        for request in jobs:
            try:
                responses.append(self.chat_with_tool(request))
            except Exception:
                if allow_partial:
                    self._append_to_conversation(request, None)
                    responses.append(None)
                else:
                    raise
        return responses

    async def achat_with_tool(self, request: GPTClientRequest) -> BaseModel:
        """
        Process achat with tool.
        
        Parameters
        ----------
        request : GPTClientRequest
            Request payload.
        
        Returns
        -------
        BaseModel
            Function result.
        
        """
        return await asyncio.to_thread(self.chat_with_tool, request)

    async def achat_with_tools_batch(
        self,
        jobs: List[GPTClientRequest],
        parallelism: Optional[int] = None,
        allow_partial: bool = False,
    ) -> List[Optional[BaseModel]]:
        """
        Process achat with tools batch.
        
        Parameters
        ----------
        jobs : List[GPTClientRequest]
            Function argument.
        parallelism : Optional[int]
            Function argument.
        allow_partial : Optional[bool]
            Function argument.
        
        Returns
        -------
        List[Optional[BaseModel]]
            List with computed values.
        
        """
        return await asyncio.to_thread(self.chat_with_tools_batch, jobs, parallelism, allow_partial)

    async def achat_with_tool_stream(
        self,
        request: GPTClientRequest,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Process achat with tool stream.
        
        Parameters
        ----------
        request : GPTClientRequest
            Request payload.
        
        Returns
        -------
        AsyncGenerator[Dict[str, Any], None]
            Dictionary with computed fields.
        
        """
        response = await self.achat_with_tool(request)
        args = response.model_dump() if hasattr(response, "model_dump") else {}
        yield {
            "finished": True,
            "final_tool_call": {"args": args},
        }

    def conversations_model_dump(self) -> List[Dict[str, Any]]:
        """
        Run conversations model dump for the current workflow step.
        
        Returns
        -------
        List[Dict[str, Any]]
            List with computed output entries.
        """
        items: List[Dict[str, Any]] = []
        for conv in self.conversations.values():
            items.append(conv.model_dump(mode="json", exclude_none=True))
        return items
