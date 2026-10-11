"""
Interfaz abstracta de ejecución. Ver README.md de este módulo: no hay
implementación de ejecución real contra ningún bróker en este repositorio.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class OrderRequest:
    asset: str
    direction: str  # "CALL" | "PUT"
    amount: float
    expiration_seconds: int
    signal_confidence: float
    reasons: list[str]
    timeframe: str = "1min"
    indicator_snapshot: dict = field(default_factory=dict)
    market_regime: str | None = None
    strategy: str = "signal-engine"
    model_version: str | None = None


@dataclass
class OrderResult:
    order_id: str
    status: str  # "SIMULATED" | "REJECTED_BY_RISK_MANAGER" | ...
    detail: str | None = None


class ExecutionManagerBase(ABC):
    """Cualquier motor de ejecución (real o simulado) implementa esto.
    El RiskManager se consulta SIEMPRE antes de aceptar una orden, y su
    veto no puede ser sorteado por ninguna implementación concreta."""

    @abstractmethod
    def submit_order(self, order: OrderRequest) -> OrderResult: ...

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool: ...

    @abstractmethod
    def get_open_positions(self) -> list[dict]: ...

    @abstractmethod
    def kill_switch(self) -> None:
        """Detiene inmediatamente toda ejecución y cancela lo cancelable."""
        ...
