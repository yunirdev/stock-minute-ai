"""
watchdog.py
健康看门狗 + 急停开关。

HeartbeatWatchdog：检查 DuckDB heartbeat 新鲜度 + 数据缺口 → Alert。
FileKillSwitch：用状态文件（conf/kill_switch.json）实现急停；runtime 每轮先查。
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List

from .models import Alert, utc_now

logger = logging.getLogger(__name__)

_KILL_SWITCH_PATH = Path("conf/kill_switch.json")
_HEARTBEAT_MAX_STALE_SECS = 120   # 2 分钟无心跳则告警


class HeartbeatWatchdog:
    """实现 Watchdog Protocol —— 检查 heartbeat 新鲜度与数据缺口。"""

    def __init__(
        self,
        db_path: str = "trade.duckdb",
        max_stale_secs: int = _HEARTBEAT_MAX_STALE_SECS,
    ) -> None:
        self._db_path = db_path
        self._max_stale = max_stale_secs

    def check(self) -> List[Alert]:
        alerts: List[Alert] = []
        try:
            import duckdb
            conn = duckdb.connect(self._db_path, read_only=True)
            row = conn.execute(
                "SELECT MAX(ts) FROM audit_heartbeats"
            ).fetchone()
            conn.close()
            if row and row[0]:
                last_hb = row[0]
                if isinstance(last_hb, str):
                    last_hb = datetime.fromisoformat(last_hb)
                if last_hb.tzinfo is None:
                    last_hb = last_hb.replace(tzinfo=timezone.utc)
                age = (utc_now() - last_hb).total_seconds()
                if age > self._max_stale:
                    alerts.append(Alert(
                        level="critical",
                        source="watchdog",
                        message=f"心跳超时 {age:.0f}s（阈值 {self._max_stale}s）",
                    ))
                    logger.warning("⚠️ watchdog: 心跳超时 %.0fs", age)
            else:
                alerts.append(Alert(
                    level="warn",
                    source="watchdog",
                    message="audit_heartbeats 表为空或不存在",
                ))
        except Exception as exc:
            logger.warning("watchdog 检查失败: %s", exc)
            alerts.append(Alert(level="warn", source="watchdog",
                                message=f"watchdog 检查异常: {exc}"))
        return alerts


class FileKillSwitch:
    """实现 KillSwitch Protocol —— 用 JSON 状态文件控制急停。"""

    def __init__(self, path: Path | str = _KILL_SWITCH_PATH) -> None:
        self._path = Path(path)

    def engaged(self) -> bool:
        if not self._path.exists():
            return False
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            return bool(data.get("engaged", False))
        except Exception:
            return False

    def engage(self, reason: str = "手动急停") -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps({"engaged": True, "reason": reason, "ts": utc_now().isoformat()},
                       ensure_ascii=False),
            encoding="utf-8",
        )
        logger.warning("🛑 KillSwitch 已激活: %s", reason)

    def disengage(self) -> None:
        if self._path.exists():
            self._path.write_text(
                json.dumps({"engaged": False, "reason": "", "ts": utc_now().isoformat()},
                           ensure_ascii=False),
                encoding="utf-8",
            )
        logger.info("✅ KillSwitch 已解除")
