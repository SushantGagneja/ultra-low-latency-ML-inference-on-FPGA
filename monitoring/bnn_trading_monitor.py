#!/usr/bin/env python3
"""BNN inference audit and latency monitor."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class InferenceRecord:
    timestamp: float
    spike_vector: str
    decision: int
    latency_ns: int
    status: str = "SUCCESS"


class AuditLogger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log_inference(self, record: InferenceRecord) -> None:
        row = {**asdict(record), "log_timestamp": time.time()}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")


class BNNTradingMonitor:
    def __init__(self, latency_sla_ns: int = 300, audit_path: Path = Path("monitoring/bnn_audit.jsonl")):
        self.latency_sla_ns = latency_sla_ns
        self.history: deque[InferenceRecord] = deque(maxlen=100000)
        self.audit = AuditLogger(audit_path)

    def log_inference(self, record: InferenceRecord) -> bool:
        self.history.append(record)
        self.audit.log_inference(record)
        return record.status == "SUCCESS" and record.latency_ns <= self.latency_sla_ns

    def get_bnn_metrics(self) -> dict:
        if not self.history:
            return {}

        latencies = [r.latency_ns for r in self.history]
        counts = Counter(r.decision for r in self.history)
        sorted_latencies = sorted(latencies)
        p99_idx = int(0.99 * (len(sorted_latencies) - 1))
        return {
            "total_inferences": len(self.history),
            "decision_distribution": {
                "BUY": counts.get(0, 0),
                "HOLD": counts.get(1, 0),
                "SELL": counts.get(2, 0),
                "INVALID": counts.get(3, 0),
            },
            "latency_ns": {
                "min": min(latencies),
                "mean": statistics.mean(latencies),
                "p99": sorted_latencies[p99_idx],
                "max": max(latencies),
                "sla": self.latency_sla_ns,
            },
            "sla_breach_count": sum(1 for r in self.history if r.latency_ns > self.latency_sla_ns),
            "error_count": sum(1 for r in self.history if r.status != "SUCCESS"),
        }


def parse_inference_lines(lines: Iterable[str]) -> Iterable[InferenceRecord]:
    for line in lines:
        line = line.strip()
        if not line:
            continue
        start = line.find("{")
        if start == -1:
            continue
        try:
            row = json.loads(line[start:])
        except json.JSONDecodeError:
            continue
        if row.get("type") != "bnn_inference":
            continue
        yield InferenceRecord(
            timestamp=float(row.get("timestamp_us", 0)) / 1_000_000.0,
            spike_vector=str(row["spike"]),
            decision=int(row["decision"]),
            latency_ns=int(row["latency_ns"]),
            status=str(row.get("status", "SUCCESS")),
        )


def run_self_test() -> int:
    monitor = BNNTradingMonitor(latency_sla_ns=300, audit_path=Path("monitoring/test_bnn_audit.jsonl"))
    for i in range(10):
        monitor.log_inference(
            InferenceRecord(
                timestamp=time.time(),
                spike_vector=f"0x{i:04x}",
                decision=i % 3,
                latency_ns=220 + i,
            )
        )
    metrics = monitor.get_bnn_metrics()
    print(json.dumps(metrics, indent=2))
    return 0 if metrics["total_inferences"] == 10 else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="BNN compliance/audit monitor")
    parser.add_argument("--input", type=Path, help="ESP32 serial log file; omit to read stdin")
    parser.add_argument("--audit", type=Path, default=Path("monitoring/bnn_audit.jsonl"))
    parser.add_argument("--sla-ns", type=int, default=300)
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    if args.test:
        return run_self_test()

    monitor = BNNTradingMonitor(latency_sla_ns=args.sla_ns, audit_path=args.audit)
    lines = args.input.read_text().splitlines() if args.input else sys.stdin
    for record in parse_inference_lines(lines):
        monitor.log_inference(record)
    print(json.dumps(monitor.get_bnn_metrics(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
