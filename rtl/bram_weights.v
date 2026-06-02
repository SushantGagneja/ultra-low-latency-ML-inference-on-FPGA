`timescale 1ns/1ps

module bram_weights (
    input  wire        clk,
    
    // Write Port (Port A) - 8-bit wide, from SPI slave
    input  wire        we,
    input  wire [7:0]  waddr,   // 8-bit byte address (up to 256 bytes = 2048 bits)
    input  wire [7:0]  wdata,
    
    // Read Port (Port B) - 64-bit wide, to BNN core
    input  wire [4:0]  raddr,   // 5-bit word address (up to 32 words of 64 bits)
    output reg  [63:0] rdata
);

    // 64-bit wide x 32 deep RAM
    reg [63:0] ram [0:31];

    wire [4:0] word_waddr = waddr[7:3];
    wire [2:0] byte_sel   = waddr[2:0];

    always @(posedge clk) begin
        if (we) begin
            case (byte_sel)
                3'd0: ram[word_waddr][7:0]   <= wdata;
                3'd1: ram[word_waddr][15:8]  <= wdata;
                3'd2: ram[word_waddr][23:16] <= wdata;
                3'd3: ram[word_waddr][31:24] <= wdata;
                3'd4: ram[word_waddr][39:32] <= wdata;
                3'd5: ram[word_waddr][47:40] <= wdata;
                3'd6: ram[word_waddr][55:48] <= wdata;
                3'd7: ram[word_waddr][63:56] <= wdata;
            endcase
        end
        rdata <= ram[raddr];
    end

    // Initialization for simulation / immediate power-up readiness
    // The Phase 1 script generates weights.mem as 1 bit per line (1216 lines).
    // We parse this into our 64-bit memory layout.
    reg [0:0] temp_ram [0:2047];
    integer i, j;
    
    initial begin
        // Init temp_ram to 0
        for (i = 0; i < 2048; i = i + 1) begin
            temp_ram[i] = 1'b0;
        end
        
`ifdef FORMAL
        $readmemb("weights.mem", temp_ram, 0, 1215);
`else
        $readmemb("fpga_weights/weights.mem", temp_ram, 0, 1215);
`endif
        
        // Pack 1-bit lines into 64-bit words
        for (i = 0; i < 32; i = i + 1) begin
            ram[i] = 64'b0;
            for (j = 0; j < 64; j = j + 1) begin
                ram[i][j] = temp_ram[i*64 + j];
            end
        end
    end

endmodule
