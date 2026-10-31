"""REST + WebSocket API of the dashboard (Task 17)."""

from idsai.product.api.app import create_app
from idsai.product.api.state import AppState

__all__ = ["AppState", "create_app"]
