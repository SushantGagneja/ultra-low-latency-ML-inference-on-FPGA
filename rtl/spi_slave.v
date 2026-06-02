`timescale 1ns/1ps

module spi_slave (
    input  wire        rst_n,
    
    // SPI Pins
    input  wire        sclk,
    input  wire        cs_n,
    input  wire        mosi,
    output wire        miso,
    
    // Output to BNN Core / BRAM (Synchronized to system clk)
    input  wire        sys_clk,
    output reg         bnn_start,
    output reg  [15:0] bnn_spike_vector,
    
    output reg         bram_we,
    output reg  [7:0]  bram_waddr,
    output reg  [7:0]  bram_wdata,
    
    // Result from BNN Core
    input  wire [1:0]  bnn_decision,
    
    // Phase 2: Output to Tick Parser
    output reg         tick_start,
    output reg [127:0] tick_payload
);

    // SCLK domain shift register
    reg [135:0] shift_rx;
    reg [7:0]   bit_count;
    reg [2:0]   tx_bit_count;
    reg [135:0] packet_sclk;
    reg         packet_is_136_sclk;
    reg         packet_toggle_sclk;

    // CDC FIX: Latch bnn_decision into sys_clk domain on CS_n falling edge.
    // bnn_decision is driven by bnn_core (sys_clk domain). Reading it directly
    // from the SCLK domain is a CDC violation. We synchronize CS_n into sys_clk
    // and latch the decision while the value is stable (before inference starts).
    reg [1:0] decision_latched;
    reg [2:0] cs_n_sync_for_latch;
    wire cs_n_falling_latch = (cs_n_sync_for_latch[2:1] == 2'b10);

    always @(posedge sys_clk or negedge rst_n) begin
        if (!rst_n) begin
            cs_n_sync_for_latch <= 3'b111;
            decision_latched <= 2'd0;
        end else begin
            cs_n_sync_for_latch <= {cs_n_sync_for_latch[1:0], cs_n};
            if (cs_n_falling_latch) begin
                decision_latched <= bnn_decision;
            end
        end
    end

    // TX data for MISO: latched decision (CDC-safe), padded to 8 bits.
    wire [7:0] tx_data = {6'd0, decision_latched};

    // MISO is read-only status. During 24-bit command frames the master ignores it;
    // during 8-bit read frames it receives {6'b0, decision} MSB-first.
    assign miso = (!cs_n) ? tx_data[3'd7 - tx_bit_count] : 1'b0;

    always @(negedge sclk or posedge cs_n or negedge rst_n) begin
        if (!rst_n) begin
            tx_bit_count <= 3'd0;
        end else if (cs_n) begin
            tx_bit_count <= 3'd0;
        end else if (tx_bit_count != 3'd7) begin
            tx_bit_count <= tx_bit_count + 1'b1;
        end
    end

    always @(posedge sclk or posedge cs_n or negedge rst_n) begin
        if (!rst_n) begin
            shift_rx <= 136'd0;
            bit_count <= 8'd0;
            packet_sclk <= 136'd0;
            packet_is_136_sclk <= 1'b0;
            packet_toggle_sclk <= 1'b0;
        end else if (cs_n) begin
            if (bit_count == 8'd24 || bit_count == 8'd136) begin
                packet_sclk <= shift_rx;
                packet_is_136_sclk <= (bit_count == 8'd136);
                packet_toggle_sclk <= ~packet_toggle_sclk;
            end
            shift_rx <= 136'd0;
            bit_count <= 8'd0;
        end else begin
            // Saturate bit_count at 136 to match the max protocol length
            if (bit_count < 8'd136) begin
                shift_rx <= {shift_rx[134:0], mosi};
                bit_count <= bit_count + 1'b1;
            end
        end
    end
    
    // Clock Domain Crossing (CDC): completed 24-bit packet to sys_clk.
    // packet_sclk is held stable until the next complete 24-bit command.
    reg [2:0] packet_toggle_sync;
    always @(posedge sys_clk or negedge rst_n) begin
        if (!rst_n) begin
            packet_toggle_sync <= 3'b000;
        end else begin
            packet_toggle_sync <= {packet_toggle_sync[1:0], packet_toggle_sclk};
        end
    end
    
    wire packet_ready = packet_toggle_sync[2] ^ packet_toggle_sync[1];
    
    always @(posedge sys_clk or negedge rst_n) begin
        if (!rst_n) begin
            bnn_start <= 1'b0;
            bnn_spike_vector <= 16'd0;
            bram_we <= 1'b0;
            bram_waddr <= 8'd0;
            bram_wdata <= 8'd0;
            tick_start <= 1'b0;
            tick_payload <= 128'd0;
        end else begin
            // Default pulse
            bnn_start <= 1'b0;
            bram_we <= 1'b0;
            tick_start <= 1'b0;
            
            if (packet_ready) begin
                if (!packet_is_136_sclk) begin
                    // 24-bit frame decode
                    case (packet_sclk[23:16])
                        8'h80: begin // BRAM Write
                            bram_we <= 1'b1;
                            bram_waddr <= packet_sclk[15:8];
                            bram_wdata <= packet_sclk[7:0];
                        end
                        8'h01: begin // Legacy Inference
                            bnn_start <= 1'b1;
                            bnn_spike_vector <= packet_sclk[15:0];
                        end
                        default: ; // Drop unknown
                    endcase
                end else begin
                    // 136-bit frame decode
                    case (packet_sclk[135:128])
                        8'h10: begin // Raw Tick Stream
                            tick_start <= 1'b1;
                            tick_payload <= packet_sclk[127:0];
                        end
                        default: ; // Drop unknown
                    endcase
                end
            end
        end
    end

endmodule
