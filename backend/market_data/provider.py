"""Contrato para integrar fuentes oficiales mediante adaptadores explícitos."""
from collections.abc import AsyncIterator, Sequence
from typing import Protocol

from market_data.models import MarketDataCandle


class MarketDataProvider(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def spread_pct(self) -> float | None: ...

    async def connect(self) -> None: ...

    def stream(self) -> AsyncIterator[Sequence[MarketDataCandle]]: ...

    async def disconnect(self) -> None: ...
