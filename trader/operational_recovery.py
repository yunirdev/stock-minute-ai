"""Non-destructive database backup/restore and fault-drill evidence."""
from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import duckdb

_FAULT_KINDS = {"API", "DATABASE", "RUNTIME"}


def classify_fault(exc: BaseException) -> str:
    name = type(exc).__name__.upper()
    message = str(exc).upper()
    if "DUCKDB" in name or "DATABASE" in message or "CATALOG" in message:
        return "DATABASE"
    if "API" in name or "CONNECTION" in name or "HTTP" in message:
        return "API"
    return "RUNTIME"


class RecoveryManager:
    def create_backup(
        self,
        sources: list[str | Path],
        *,
        destination: str | Path,
        created_at: datetime,
    ) -> dict[str, Any]:
        if created_at.tzinfo is None or created_at.utcoffset() is None:
            raise ValueError("BACKUP_TIME_TZ_REQUIRED")
        target = Path(destination).resolve()
        target.mkdir(parents=True, exist_ok=True)
        records = []
        for source_value in sources:
            source = Path(source_value).resolve()
            if not source.is_file() or source.suffix.lower() != ".duckdb":
                raise ValueError("BACKUP_SOURCE_INVALID")
            backup = target / source.name
            if backup.exists():
                raise FileExistsError(backup)
            shutil.copy2(source, backup)
            self._verify_duckdb(backup)
            records.append(
                {
                    "source": str(source),
                    "backup": str(backup),
                    "size": backup.stat().st_size,
                    "sha256": self._sha256(backup),
                }
            )
        manifest_payload = {
            "created_at": created_at.isoformat(),
            "files": records,
        }
        manifest_id = "backup-" + hashlib.sha256(
            json.dumps(manifest_payload, sort_keys=True).encode()
        ).hexdigest()[:24]
        manifest_payload["manifest_id"] = manifest_id
        manifest_path = target / f"{manifest_id}.json"
        manifest_path.write_text(
            json.dumps(manifest_payload, indent=2),
            encoding="utf-8",
        )
        manifest_payload["manifest_path"] = str(manifest_path)
        return manifest_payload

    def restore_to_new_directory(
        self,
        manifest_path: str | Path,
        *,
        destination: str | Path,
    ) -> list[Path]:
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        target = Path(destination).resolve()
        target.mkdir(parents=True, exist_ok=True)
        restored = []
        for record in manifest["files"]:
            backup = Path(record["backup"]).resolve()
            if self._sha256(backup) != record["sha256"]:
                raise ValueError("BACKUP_CHECKSUM_MISMATCH")
            output = target / backup.name
            if output.exists():
                raise FileExistsError(output)
            shutil.copy2(backup, output)
            self._verify_duckdb(output)
            restored.append(output)
        return restored

    @staticmethod
    def _verify_duckdb(path: Path) -> None:
        connection = duckdb.connect(str(path), read_only=True)
        try:
            connection.execute("SELECT 1").fetchone()
        finally:
            connection.close()

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()


def run_fault_drill(
    kind: str,
    operation: Callable[[], Any],
    *,
    expected_exception: type[BaseException],
) -> dict[str, Any]:
    normalized = kind.strip().upper()
    if normalized not in _FAULT_KINDS:
        raise ValueError("FAULT_DRILL_KIND_INVALID")
    try:
        operation()
    except expected_exception as exc:
        classified = classify_fault(exc)
        return {
            "kind": normalized,
            "passed": classified == normalized,
            "classified_as": classified,
            "error_type": type(exc).__name__,
            "error_code": str(exc)[:200],
            "recovered": True,
        }
    return {
        "kind": normalized,
        "passed": False,
        "classified_as": "",
        "error_type": "",
        "error_code": "EXPECTED_FAILURE_NOT_RAISED",
        "recovered": False,
    }
