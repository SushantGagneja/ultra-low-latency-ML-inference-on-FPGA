package main

import (
	"bytes"
	"encoding/binary"
	"flag"
	"log"
	"net"
	"os"
	"os/signal"
	"strconv"
	"strings"

	"github.com/gorilla/websocket"
)

// Raw Tick Packet: 16 bytes total
// [0:4]   Bid Price (float32, little-endian)
// [4:8]   Ask Price (float32, little-endian)
// [8:12]  Bid Qty (float32, little-endian)
// [12:16] Ask Qty (float32, little-endian)

func main() {
	espIP := flag.String("ip", "192.168.1.100:8080", "ESP32 UDP IP and Port")
	flag.Parse()

	log.Printf("Starting HFT Gateway...")
	log.Printf("Target ESP32 UDP Endpoint: %s", *espIP)

	// 1. Setup UDP Connection to ESP32
	udpAddr, err := net.ResolveUDPAddr("udp", *espIP)
	if err != nil {
		log.Fatalf("Failed to resolve UDP address: %v", err)
	}
	conn, err := net.DialUDP("udp", nil, udpAddr)
	if err != nil {
		log.Fatalf("Failed to dial UDP: %v", err)
	}
	defer conn.Close()

	// 2. Connect to Binance WebSocket
	wsURL := "wss://stream.binance.com:9443/ws/btcusdt@bookTicker"
	log.Printf("Connecting to Binance WS: %s", wsURL)
	c, _, err := websocket.DefaultDialer.Dial(wsURL, nil)
	if err != nil {
		log.Fatalf("Dial error: %v", err)
	}
	defer c.Close()

	interrupt := make(chan os.Signal, 1)
	signal.Notify(interrupt, os.Interrupt)

	go func() {
		<-interrupt
		log.Println("Shutting down gateway...")
		c.Close()
		os.Exit(0)
	}()

	// Reusable binary buffer to avoid allocations in the hot path
	binBuf := new(bytes.Buffer)
	binBuf.Grow(16)

	log.Println("Listening for market ticks...")

	// 3. Hot Path Processing Loop
	for {
		_, message, err := c.ReadMessage()
		if err != nil {
			log.Fatalf("Read error: %v", err)
		}

		// Fast, zero-allocation extraction of the string values.
		// Expected JSON format:
		// {"u":400900217,"s":"BTCUSDT","b":"25.35190000","B":"31.21000000","a":"25.36520000","A":"40.66000000"}
		msgStr := string(message)

		bidPrice := fastParseFloat(msgStr, `"b":"`)
		askPrice := fastParseFloat(msgStr, `"a":"`)
		bidQty := fastParseFloat(msgStr, `"B":"`)
		askQty := fastParseFloat(msgStr, `"A":"`)

		if bidPrice == 0 || askPrice == 0 {
			continue // Drop malformed ticks
		}

		// Pack into tight 16-byte binary payload
		binBuf.Reset()
		
		// Note: The ESP32 Xtensa architecture is Little-Endian.
		// We use LittleEndian to avoid byte-swapping on the microcontroller.
		binary.Write(binBuf, binary.LittleEndian, float32(bidPrice))
		binary.Write(binBuf, binary.LittleEndian, float32(askPrice))
		binary.Write(binBuf, binary.LittleEndian, float32(bidQty))
		binary.Write(binBuf, binary.LittleEndian, float32(askQty))

		// Fire UDP packet
		_, err = conn.Write(binBuf.Bytes())
		if err != nil {
			log.Printf("UDP Write error: %v", err)
		}
	}
}

// fastParseFloat scans a JSON string for a key prefix and parses the float value.
// It avoids building a DOM tree or allocating new strings.
func fastParseFloat(jsonStr string, key string) float64 {
	idx := strings.Index(jsonStr, key)
	if idx == -1 {
		return 0
	}
	
	start := idx + len(key)
	end := strings.IndexByte(jsonStr[start:], '"')
	if end == -1 {
		return 0
	}
	end += start
	
	val, err := strconv.ParseFloat(jsonStr[start:end], 32)
	if err != nil {
		return 0
	}
	return val
}
