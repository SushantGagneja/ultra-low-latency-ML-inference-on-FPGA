#!/usr/bin/env python3
"""BNN Institutional Trading Monitor with PnL and Risk Management."""

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
    price: float = 0.0
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
        
        # PnL & Risk Model
        self.position = 0          # 1 for Long, -1 for Short, 0 for Flat
        self.entry_price = 0.0
        self.realized_pnl = 0.0    # Absolute monetary PnL (simplified unit sizing)
        self.trade_count = 0
        self.peak_equity = 0.0
        self.max_drawdown = 0.0
        
        # Risk Limits
        self.max_position_size = 1 # E.g., max 1 BTC

    def update_pnl(self, record: InferenceRecord):
        price = record.price
        if price <= 0.0:
            return
            
        fee_rate = 0.0004 # 4 bps taker
        
        if record.decision == 0: # BUY
            if self.position <= 0:
                if self.position < 0:
                    # Close Short
                    trade_pnl = (self.entry_price - price) / self.entry_price - fee_rate
                    self.realized_pnl += trade_pnl
                    self.trade_count += 1
                # Open Long
                self.position = self.max_position_size
                self.entry_price = price
                
        elif record.decision == 2: # SELL
            if self.position >= 0:
                if self.position > 0:
                    # Close Long
                    trade_pnl = (price - self.entry_price) / self.entry_price - fee_rate
                    self.realized_pnl += trade_pnl
                    self.trade_count += 1
                # Open Short
                self.position = -self.max_position_size
                self.entry_price = price
                
        # Mark to market equity
        equity = self.realized_pnl
        if self.position > 0:
            equity += (price - self.entry_price) / self.entry_price - fee_rate
        elif self.position < 0:
            equity += (self.entry_price - price) / self.entry_price - fee_rate
            
        if equity > self.peak_equity:
            self.peak_equity = equity
        dd = self.peak_equity - equity
        if dd > self.max_drawdown:
            self.max_drawdown = dd


    def log_inference(self, record: InferenceRecord) -> bool:
        self.history.append(record)
        self.audit.log_inference(record)
        
        if record.status == "SUCCESS":
            self.update_pnl(record)
            
        return record.status == "SUCCESS" and record.latency_ns <= self.latency_sla_ns

    def get_bnn_metrics(self) -> dict:
        if not self.history:
            return {}

        latencies = [r.latency_ns for r in self.history]
        counts = Counter(r.decision for r in self.history)
        sorted_latencies = sorted(latencies)
        p99_idx = int(0.99 * (len(sorted_latencies) - 1))
        
        # Per-class latency breakdown
        latency_by_class = {}
        for class_name, class_id in [("BUY", 0), ("HOLD", 1), ("SELL", 2)]:
            class_lats = [r.latency_ns for r in self.history if r.decision == class_id]
            if class_lats:
                latency_by_class[class_name] = {
                    "mean": round(statistics.mean(class_lats), 1),
                    "min": min(class_lats),
                    "max": max(class_lats)
                }
            else:
                latency_by_class[class_name] = None
        
        return {
            "total_inferences": len(self.history),
            "decision_distribution": {
                "BUY": counts.get(0, 0),
                "HOLD": counts.get(1, 0),
                "SELL": counts.get(2, 0),
                "INVALID": counts.get(3, 0),
            },
            "trading_performance": {
                "current_position": self.position,
                "entry_price": self.entry_price,
                "realized_pnl_pct": round(self.realized_pnl * 100, 4),
                "max_drawdown_pct": round(self.max_drawdown * 100, 4),
                "trade_count": self.trade_count
            },
            "latency_ns": {
                "aggregate": {
                    "min": min(latencies),
                    "mean": round(statistics.mean(latencies), 1),
                    "p99": sorted_latencies[p99_idx],
                    "max": max(latencies),
                },
                "by_class": latency_by_class,
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
            price=float(row.get("price", 0.0)),
            status=str(row.get("status", "SUCCESS")),
        )


def run_self_test() -> int:
    monitor = BNNTradingMonitor(latency_sla_ns=300, audit_path=Path("monitoring/test_bnn_audit.jsonl"))
    price = 60000.0
    for i in range(10):
        decision = i % 3
        if decision == 0:
            price += 100.0
        elif decision == 2:
            price -= 100.0
            
        monitor.log_inference(
            InferenceRecord(
                timestamp=time.time(),
                spike_vector=f"0x{i:04x}",
                decision=decision,
                latency_ns=220 + i,
                price=price
            )
        )
    metrics = monitor.get_bnn_metrics()
    print(json.dumps(metrics, indent=2))
    return 0 if metrics["total_inferences"] == 10 else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="BNN institutional compliance & PnL monitor")
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
