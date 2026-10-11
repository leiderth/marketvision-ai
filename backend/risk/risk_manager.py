"""
RiskManager — módulo independiente e inmutable desde el punto de vista del
modelo de IA. Ningún LearningManager ni modelo de ML puede modificar,
ignorar o sortear estos límites (regla invariable #7 del proyecto).

Fase 1: solo expone los límites configurados y un estado en memoria.
La lógica de bloqueo automático al alcanzar límites y las pausas
obligatorias se implementan en la Fase 10.
"""
import json
import math
import tempfile
from dataclasses import asdict, dataclass, field, replace
from datetime import date, datetime, timezone
from numbers import Real
from pathlib import Path

from config import settings


@dataclass(frozen=True)
class RiskLimits:
    max_loss_per_trade_pct: float
    max_daily_loss_pct: float
    max_consecutive_losses: int
    max_trades_per_day: int
    paper_account_balance: float = 10_000.0
    max_trades_per_hour: int = 5
    max_daily_profit_pct: float = 5.0

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.max_loss_per_trade_pct)
            or not 0 < self.max_loss_per_trade_pct <= 100
            or not math.isfinite(self.max_daily_loss_pct)
            or not 0 < self.max_daily_loss_pct <= 100
            or not math.isfinite(self.max_daily_profit_pct)
            or not 0 <= self.max_daily_profit_pct <= 100
            or self.max_consecutive_losses <= 0
            or self.max_trades_per_day <= 0
            or self.max_trades_per_hour <= 0
            or not math.isfinite(self.paper_account_balance)
            or self.paper_account_balance <= 0
        ):
            raise ValueError("Los límites de RiskManager deben ser positivos y finitos.")


@dataclass
class RiskState:
    trading_day: date = field(default_factory=lambda: datetime.now(timezone.utc).date())
    trades_today: int = 0
    consecutive_losses: int = 0
    daily_pnl_pct: float = 0.0
    trade_timestamps: list[str] = field(default_factory=list)
    trading_locked: bool = False
    lock_reason: str | None = None


class RiskManager:
    """
    Los límites (RiskLimits) se cargan una sola vez desde configuración y
    son de solo lectura para el resto del sistema. El estado (RiskState)
    se actualiza operación por operación en fases posteriores.
    """

    def __init__(
        self,
        *,
        limits: RiskLimits | None = None,
        state_path: str | Path | None = None,
    ) -> None:
        self._limits = limits or RiskLimits(
            max_loss_per_trade_pct=settings.risk_max_loss_per_trade_pct,
            max_daily_loss_pct=settings.risk_max_daily_loss_pct,
            max_consecutive_losses=settings.risk_max_consecutive_losses,
            max_trades_per_day=settings.risk_max_trades_per_day,
            paper_account_balance=settings.risk_paper_account_balance,
            max_trades_per_hour=settings.risk_max_trades_per_hour,
            max_daily_profit_pct=settings.risk_max_daily_profit_pct,
        )
        self._state = RiskState()
        self._state_path = Path(state_path) if state_path is not None else None
        self._load_state()

    def get_limits(self) -> dict:
        return asdict(self._limits)

    def get_state(self, *, now: datetime | None = None) -> dict:
        current_time = now or datetime.now(timezone.utc)
        self._roll_day(current_time)
        self._roll_hour(current_time)
        state = asdict(self._state)
        state["trading_day"] = self._state.trading_day.isoformat()
        return state

    def is_trading_allowed(self, *, now: datetime | None = None) -> bool:
        current_time = now or datetime.now(timezone.utc)
        self._roll_day(current_time)
        self._roll_hour(current_time)
        if self._state.trading_locked:
            return False
        if len(self._state.trade_timestamps) >= self._limits.max_trades_per_hour:
            self.lock_trading("Límite de operaciones por hora alcanzado.")
            return False
        if self._state.trades_today >= self._limits.max_trades_per_day:
            self.lock_trading("Límite de operaciones diarias alcanzado.")
            return False
        return True

    def authorize_trade(self, amount: float, *, now: datetime | None = None) -> tuple[bool, str | None]:
        current_time = now or datetime.now(timezone.utc)
        self._roll_day(current_time)
        self._roll_hour(current_time)
        if not self.is_trading_allowed(now=current_time):
            return False, self._state.lock_reason or "Operativa bloqueada por RiskManager."
        if (
            isinstance(amount, bool)
            or not isinstance(amount, Real)
            or not math.isfinite(amount)
            or amount <= 0
        ):
            return False, "Importe de operación inválido."
        maximum_amount = (
            self._limits.paper_account_balance
            * self._limits.max_loss_per_trade_pct
            / 100
        )
        if amount > maximum_amount:
            return False, (
                "Importe supera el límite por operación del "
                f"{self._limits.max_loss_per_trade_pct:.2f}% del balance paper."
            )
        return True, None

    def register_trade(self, amount: float, *, now: datetime | None = None) -> tuple[bool, str | None]:
        current_time = now or datetime.now(timezone.utc)
        allowed, reason = self.authorize_trade(amount, now=now)
        if not allowed:
            return False, reason
        previous_state = replace(
            self._state,
            trade_timestamps=list(self._state.trade_timestamps),
        )
        self._state.trades_today += 1
        self._state.trade_timestamps.append(current_time.isoformat())
        if self._state.trades_today >= self._limits.max_trades_per_day:
            self._state.trading_locked = True
            self._state.lock_reason = "Límite de operaciones diarias alcanzado."
        try:
            self._save_state()
        except OSError:
            self._state = previous_state
            raise
        return True, None

    def record_trade_result(
        self,
        pnl_pct: float,
        *,
        now: datetime | None = None,
    ) -> None:
        current_time = now or datetime.now(timezone.utc)
        self._roll_day(current_time)
        if (
            isinstance(pnl_pct, bool)
            or not isinstance(pnl_pct, Real)
            or not math.isfinite(pnl_pct)
        ):
            raise ValueError("El resultado de operación debe ser un porcentaje finito.")
        if pnl_pct < -self._limits.max_loss_per_trade_pct:
            raise ValueError("El resultado excede la pérdida máxima configurada por operación.")
        previous_state = replace(
            self._state,
            trade_timestamps=list(self._state.trade_timestamps),
        )
        self._state.daily_pnl_pct += float(pnl_pct)
        if pnl_pct < 0:
            self._state.consecutive_losses += 1
        elif pnl_pct > 0:
            self._state.consecutive_losses = 0

        if self._state.daily_pnl_pct <= -self._limits.max_daily_loss_pct:
            self.lock_trading("Límite de pérdida diaria alcanzado.")
        elif (
            self._limits.max_daily_profit_pct > 0
            and self._state.daily_pnl_pct >= self._limits.max_daily_profit_pct
        ):
            self.lock_trading("Objetivo de beneficio diario alcanzado.")
        elif self._state.consecutive_losses >= self._limits.max_consecutive_losses:
            self.lock_trading("Límite de pérdidas consecutivas alcanzado.")
        elif self._state.trades_today >= self._limits.max_trades_per_day:
            self.lock_trading("Límite de operaciones diarias alcanzado.")
        try:
            self._save_state()
        except OSError:
            self._state = previous_state
            raise

    def lock_trading(self, reason: str) -> None:
        if not reason.strip():
            raise ValueError("El bloqueo requiere un motivo explícito.")
        previous_state = replace(self._state)
        self._state.trading_locked = True
        self._state.lock_reason = reason
        try:
            self._save_state()
        except OSError:
            self._state = previous_state
            raise

    def unlock_trading(self) -> None:
        """Requiere acción explícita del usuario (regla invariable #9).
        En fases posteriores esto se expone solo vía un endpoint que
        exige confirmación explícita en la UI, no automáticamente."""
        previous_state = replace(self._state)
        self._state.trading_locked = False
        self._state.lock_reason = None
        try:
            self._save_state()
        except OSError:
            self._state = previous_state
            raise

    def _roll_day(self, now: datetime) -> None:
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("El reloj de RiskManager requiere zona horaria.")
        current_day = now.astimezone(timezone.utc).date()
        if current_day == self._state.trading_day:
            return
        previous_state = replace(self._state)
        self._state.trading_day = current_day
        self._state.trades_today = 0
        self._state.consecutive_losses = 0
        self._state.daily_pnl_pct = 0.0
        try:
            self._save_state()
        except OSError:
            self._state = previous_state
            raise

    def _roll_hour(self, now: datetime) -> None:
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("El reloj de RiskManager requiere zona horaria.")
        cutoff = now.astimezone(timezone.utc).timestamp() - 3600
        retained = [
            timestamp
            for timestamp in self._state.trade_timestamps
            if datetime.fromisoformat(timestamp).astimezone(timezone.utc).timestamp() > cutoff
        ]
        if retained == self._state.trade_timestamps:
            return
        previous_state = replace(self._state, trade_timestamps=list(self._state.trade_timestamps))
        self._state.trade_timestamps = retained
        try:
            self._save_state()
        except OSError:
            self._state = previous_state
            raise

    def _load_state(self) -> None:
        if self._state_path is None or not self._state_path.exists():
            return
        try:
            stored = json.loads(self._state_path.read_text(encoding="utf-8"))
            if (
                not isinstance(stored, dict)
                or not isinstance(stored.get("trades_today"), int)
                or isinstance(stored.get("trades_today"), bool)
                or not isinstance(stored.get("consecutive_losses"), int)
                or isinstance(stored.get("consecutive_losses"), bool)
                or not isinstance(stored.get("trading_locked"), bool)
                or (
                    "trade_timestamps" in stored
                    and (
                        not isinstance(stored["trade_timestamps"], list)
                        or not all(isinstance(value, str) for value in stored["trade_timestamps"])
                    )
                )
                or (
                    stored.get("lock_reason") is not None
                    and not isinstance(stored.get("lock_reason"), str)
                )
            ):
                raise ValueError("Tipos inválidos en el estado de riesgo.")
            trading_day = date.fromisoformat(stored["trading_day"])
            candidate = RiskState(
                trading_day=trading_day,
                trades_today=stored["trades_today"],
                consecutive_losses=stored["consecutive_losses"],
                daily_pnl_pct=float(stored["daily_pnl_pct"]),
                trade_timestamps=stored.get("trade_timestamps", []),
                trading_locked=bool(stored["trading_locked"]),
                lock_reason=stored.get("lock_reason"),
            )
            if any(
                datetime.fromisoformat(timestamp).tzinfo is None
                or datetime.fromisoformat(timestamp).utcoffset() is None
                for timestamp in candidate.trade_timestamps
            ):
                raise ValueError("Timestamps inválidos en el estado de riesgo.")
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            self._state.trading_locked = True
            self._state.lock_reason = f"Estado de riesgo persistido inválido ({type(exc).__name__})."
            return
        if (
            candidate.trades_today < 0
            or candidate.consecutive_losses < 0
            or not math.isfinite(candidate.daily_pnl_pct)
            or candidate.trading_locked and not candidate.lock_reason
        ):
            self._state.trading_locked = True
            self._state.lock_reason = "Estado de riesgo persistido inválido."
            return
        self._state = candidate
        self._roll_day(datetime.now(timezone.utc))

    def _save_state(self) -> None:
        if self._state_path is None:
            return
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self._state_path.parent,
                prefix=f".{self._state_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                state = asdict(self._state)
                state["trading_day"] = self._state.trading_day.isoformat()
                json.dump(state, temporary, sort_keys=True, allow_nan=False)
                temporary.flush()
                temporary_path = Path(temporary.name)
            temporary_path.replace(self._state_path)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()
