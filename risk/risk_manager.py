from datetime import date
import math
import config
from database.db_manager import get_recent_losses


class RiskManager:
    def __init__(self):
        for name in ("MAX_RISK_PER_TRADE", "MAX_DAILY_DRAWDOWN", "MAX_WEEKLY_DRAWDOWN"):
            value = getattr(config, name)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= 100:
                raise ValueError(f"{name} must be positive and at most 100 percent")
        if isinstance(config.MAX_CONSECUTIVE_LOSSES, bool) or not isinstance(config.MAX_CONSECUTIVE_LOSSES, int) or config.MAX_CONSECUTIVE_LOSSES <= 0:
            raise ValueError("MAX_CONSECUTIVE_LOSSES must be a positive integer")
        self._daily_start = None
        self._weekly_start = None
        self._today = date.today()
        self._week = self._today.isocalendar()[:2]

    def _reset_if_new_day(self, balance):
        today = date.today()
        week = today.isocalendar()[:2]
        if self._week != week:
            self._weekly_start = balance
            self._week = week
        if self._today != today:
            self._daily_start = balance
            self._today = today

    def seed_balance(self, balance):
        if self._daily_start is None:
            self._daily_start = balance
        if self._weekly_start is None:
            self._weekly_start = balance

    def check_risk(self, balance, equity):
        if any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in (balance, equity)) or balance <= 0:
            return {"allowed": False, "violations": ["Invalid account balance or equity"],
                    "warnings": [], "daily_drawdown": 100.0, "weekly_drawdown": 100.0,
                    "consecutive_losses": 0, "daily_start": self._daily_start,
                    "weekly_start": self._weekly_start}
        self.seed_balance(balance)
        self._reset_if_new_day(balance)

        violations = []
        warnings = []

        daily_dd = (self._daily_start - equity) / self._daily_start * 100
        weekly_dd = (self._weekly_start - equity) / self._weekly_start * 100

        if daily_dd >= config.MAX_DAILY_DRAWDOWN:
            violations.append(
                f"Daily drawdown {daily_dd:.2f}% exceeds limit ({config.MAX_DAILY_DRAWDOWN}%)"
            )
        elif daily_dd >= config.MAX_DAILY_DRAWDOWN * 0.75:
            warnings.append(f"Daily drawdown at {daily_dd:.2f}% — approaching limit")

        if weekly_dd >= config.MAX_WEEKLY_DRAWDOWN:
            violations.append(
                f"Weekly drawdown {weekly_dd:.2f}% exceeds limit ({config.MAX_WEEKLY_DRAWDOWN}%)"
            )

        consec_losses = len(get_recent_losses(config.MAX_CONSECUTIVE_LOSSES))
        if consec_losses >= config.MAX_CONSECUTIVE_LOSSES:
            violations.append(
                f"{consec_losses} consecutive losses — trading suspended per constitution"
            )

        return {
            "allowed": len(violations) == 0,
            "violations": violations,
            "warnings": warnings,
            "daily_drawdown": round(daily_dd, 2),
            "weekly_drawdown": round(weekly_dd, 2),
            "consecutive_losses": consec_losses,
            "daily_start": self._daily_start,
            "weekly_start": self._weekly_start,
        }

    def calculate_position_size(self, account_balance, entry, sl, pip_value=10.0, digits=5):
        """Return lot size based on MAX_RISK_PER_TRADE%."""
        if any(not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0
               for v in (account_balance, entry, sl, pip_value)) or entry == sl:
            return 0.0
        risk_amount = account_balance * (config.MAX_RISK_PER_TRADE / 100)
        if isinstance(digits, bool) or digits not in (2, 3, 4, 5):
            return 0.0
        point = 10 ** -digits
        pip_size = point * 10 if digits in (3, 5) else point
        sl_pips = abs(entry - sl) / pip_size
        lot_size = risk_amount / (sl_pips * pip_value)
        return max(0.0, math.floor(min(lot_size, 100.0) * 100 + 1e-12) / 100)

    def is_elite_trade_allowed(self, scoring, risk_status):
        """Elite trades require ELITE grade + no violations."""
        return (
            scoring.get("grade") == "ELITE"
            and risk_status.get("allowed") is True
            and risk_status.get("daily_drawdown", 99) < config.MAX_DAILY_DRAWDOWN * 0.5
        )
