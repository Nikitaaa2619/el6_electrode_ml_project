from __future__ import annotations

import json
import logging
import os
from urllib.request import Request, urlopen

LOGGER = logging.getLogger(__name__)


class AlertNotifier:
    """Send a vendor-neutral JSON webhook for a new operator alert."""

    def __init__(self) -> None:
        self.url = os.getenv("ALERT_WEBHOOK_URL")
        self.token = os.getenv("ALERT_WEBHOOK_TOKEN")

    @property
    def configured(self) -> bool:
        return bool(self.url)

    def send(self, payload: dict) -> None:
        if not self.url:
            return
        headers = {"Content-Type": "application/json", "User-Agent": "el6-monitor/1.0"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        request = Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urlopen(request, timeout=5) as response:
                if response.status >= 300:
                    raise RuntimeError(f"Webhook returned HTTP {response.status}")
        except Exception:
            LOGGER.exception("Failed to deliver alert webhook")
