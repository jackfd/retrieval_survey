from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@dataclass
class FailureRecord:
    record_type: str
    record_id: str
    error_type: str
    error_message: str
    stage: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "record_type": self.record_type,
            "record_id": self.record_id,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "timestamp_utc": utc_now_iso(),
            "stage": self.stage,
        }
