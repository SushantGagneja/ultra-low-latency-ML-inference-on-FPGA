# Phase 2 ESP32-S3 Temporal Engine

This firmware owns the temporal part of the design:

- Maintains fixed-size rolling market state without heap allocation in the hot path.
- Computes RSI, momentum, volume ratio, and volatility in O(1) per tick.
- Converts indicators into the exact 16-bit bipolar spike vector used by Phase 1.
- Sends `control[1:0] + spike[15:0]` to the FPGA over SPI.
- Ingests live BTCUSDT Binance `bookTicker` snapshots over WiFi/WebSocket.

`bookTicker` quantity is used as a deterministic liquidity/volume proxy. If we train on true trade
volume later, switch the producer to `aggTrade` or combine both streams before retraining.

## Build

```sh
cd esp32_firmware
idf.py set-target esp32s3
idf.py menuconfig
idf.py build
```

Set WiFi credentials under `BNN Trading Firmware` in `menuconfig`.

Pin assignments in `main.c` are placeholders until the Shrike FPGA/ESP32 IOMUX pins are confirmed.
