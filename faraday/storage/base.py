from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional


class StorageBackend(ABC):
    """Backend interface for relational persistence operations."""

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Human-readable backend identifier."""

    @abstractmethod
    def initialize(self) -> None:
        """Initialize backend resources and schema."""

    @abstractmethod
    def healthcheck(self) -> bool:
        """Return True when backend is healthy."""

    # Usage records
    @abstractmethod
    def create_usage_record(self, data: dict[str, Any]) -> Optional[dict[str, Any]]:
        pass

    @abstractmethod
    def update_usage_record_by_query_id(
        self,
        query_id: str,
        status: str,
        additional_data: Optional[dict[str, Any]] = None,
        mode: str = "dev",
    ) -> bool:
        pass

    @abstractmethod
    def complete_usage_record_by_query_id(
        self,
        query_id: str,
        final_status: str = "completed",
        final_data: Optional[dict[str, Any]] = None,
        mode: str = "dev",
    ) -> bool:
        pass

    @abstractmethod
    def get_conversation_title(
        self,
        user_id: str,
        chat_id: str,
        mode: str = "dev",
    ) -> Optional[str]:
        pass

    # User balance / cost tracking
    @abstractmethod
    def get_user_balance(self, user_id: str, mode: str = "dev") -> Optional[float]:
        pass

    @abstractmethod
    def set_user_balance(self, user_id: str, balance: float, mode: str = "dev") -> bool:
        pass

    @abstractmethod
    def get_usage_cost_cumulative(self, user_id: str, mode: str = "dev") -> float:
        pass

    @abstractmethod
    def set_usage_cost_cumulative(self, user_id: str, value: float, mode: str = "dev") -> bool:
        pass

    # Tool jobs
    @abstractmethod
    def create_tool_job(self, job_data: dict[str, Any]) -> dict[str, Any]:
        pass

    @abstractmethod
    def update_tool_job(self, tool_run_id: str, updates: dict[str, Any]) -> Optional[dict[str, Any]]:
        pass

    @abstractmethod
    def get_tool_job(self, tool_run_id: str) -> Optional[dict[str, Any]]:
        pass
