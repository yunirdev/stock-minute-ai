"""Browser-side UI fault and layout reporting primitives."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .bug_reporting import BugReporter


class UIHealthReport(BaseModel):
    model_config = ConfigDict(extra="ignore")

    kind: str = Field(max_length=80)
    message: str = Field(max_length=500)
    path: str = Field(default="", max_length=500)
    width: int = Field(default=0, ge=0, le=20000)
    height: int = Field(default=0, ge=0, le=20000)
    details: dict[str, Any] = Field(default_factory=dict)


def format_health_age(seconds: float) -> str:
    seconds = max(0.0, seconds)
    if seconds < 120:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.0f}m"
    if seconds < 86400:
        return f"{seconds / 3600:.0f}h"
    return f"{seconds / 86400:.0f}d"

def record_ui_health(report: UIHealthReport, db_path: str) -> str:
    """Persist a bounded, sanitized browser health report."""
    return BugReporter(db_path, "ui").capture_message(
        report.message,
        error_type=report.kind,
        operation="ui.browser_health",
        context={
            "path": report.path,
            "viewport": {"width": report.width, "height": report.height},
            "details": report.details,
        },
    )


UI_HEALTH_SCRIPT = r"""
(() => {
  const endpoint = '/api/ui-health/report';
  const sent = new Set();
  const report = (kind, message, details = {}) => {
    const key = kind + '|' + message;
    if (sent.has(key)) return;
    sent.add(key);
    fetch(endpoint, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        kind, message, path: location.pathname,
        width: window.innerWidth, height: window.innerHeight, details,
      }),
    }).catch(() => {});
  };

  window.addEventListener('error', event => {
    report('javascript_error', String(event.message || 'JavaScript error'), {
      source: String(event.filename || ''), line: Number(event.lineno || 0),
    });
  });
  window.addEventListener('unhandledrejection', event => {
    report('unhandled_rejection', String(event.reason || 'Unhandled promise rejection'));
  });

  const inspectLayout = () => {
    const root = document.querySelector('.q-layout, .nicegui-content');
    if (!root || root.getBoundingClientRect().height < 20) {
      report('ui_root_missing', 'UI root is missing or empty');
      return;
    }
    const overflow = document.documentElement.scrollWidth - document.documentElement.clientWidth;
    if (overflow > 8) {
      report('horizontal_overflow', 'Page content exceeds viewport width', {overflow});
    }
    const visible = [...document.querySelectorAll('body *')].filter(element => {
      const style = getComputedStyle(element);
      const rect = element.getBoundingClientRect();
      return style.position !== 'fixed' && style.display !== 'none' &&
             style.visibility !== 'hidden' && rect.width > 20 && rect.height > 20;
    });
    const clipped = visible.filter(element => {
      const rect = element.getBoundingClientRect();
      return rect.left < -8 || rect.right > window.innerWidth + 8;
    }).length;
    if (clipped > 2) {
      report('elements_clipped_by_viewport', 'Multiple visible elements are clipped by the viewport', {
        clipped, visible: visible.length,
      });
    }
  };
  window.addEventListener('load', () => setTimeout(inspectLayout, 2500));
  window.addEventListener('resize', () => setTimeout(inspectLayout, 250));
})();
"""