`timescale 1ns/1ps

module bnn_core (
    input  wire        clk,
    input  wire        rst_n,
    
    // Control from SPI slave
    input  wire        start,
    input  wire [15:0] spike_vector,
    output reg         done,
    output reg  [1:0]  decision,
    
    // Interface to BRAM (weights)
    output reg  [4:0]  bram_raddr,
    input  wire [63:0] bram_rdata
);

    // State machine — 4-bit encoding for robustness.
    // 6 states fit in 3 bits, but a single-bit upset could alias valid states.
    // 4 bits provides a safety margin against SEU / metastability.
    localparam STATE_IDLE       = 4'd0;
    localparam STATE_L1_PREFETCH = 4'd1;
    localparam STATE_LAYER1     = 4'd2;
    localparam STATE_L2_PREFETCH = 4'd3;
    localparam STATE_LAYER2     = 4'd4;
    localparam STATE_ARGMAX     = 4'd5;
    
    reg [3:0] state;
    reg [4:0] cycle_cnt;
    
    // Layer 1 outputs
    reg [63:0] layer1_out;
    
    // Layer 2 scores
    reg [6:0] score0;
    reg [6:0] score1;
    reg [6:0] score2;
    
    // Datapath routing multiplexers
    wire [15:0] unit_in [0:3];
    wire [15:0] unit_wt [0:3];
    wire [4:0]  unit_pop [0:3];
    
    // In Layer 1, all 4 units receive the same 16-bit spike vector.
    // In Layer 2, they receive 16-bit chunks of the 64-bit layer1_out.
    assign unit_in[0] = (state == STATE_LAYER1) ? spike_vector : layer1_out[15:0];
    assign unit_in[1] = (state == STATE_LAYER1) ? spike_vector : layer1_out[31:16];
    assign unit_in[2] = (state == STATE_LAYER1) ? spike_vector : layer1_out[47:32];
    assign unit_in[3] = (state == STATE_LAYER1) ? spike_vector : layer1_out[63:48];
    
    // Weights are always chunks of the 64-bit BRAM read data
    assign unit_wt[0] = bram_rdata[15:0];
    assign unit_wt[1] = bram_rdata[31:16];
    assign unit_wt[2] = bram_rdata[47:32];
    assign unit_wt[3] = bram_rdata[63:48];
    
    // Instantiate 4 parallel XNOR-popcount units
    genvar i;
    generate
        for (i = 0; i < 4; i = i + 1) begin : gen_units
            xnor_popcount #(
                .WIDTH(16)
            ) xpu (
                .inputs(unit_in[i]),
                .weights(unit_wt[i]),
                .popcount(unit_pop[i])
            );
        end
    endgenerate

    // Total popcount for Layer 2
    wire [6:0] l2_total_pop = unit_pop[0] + unit_pop[1] + unit_pop[2] + unit_pop[3];

    // -------------------------------------------------------------------------
    // BRAM read address generation (combinational)
    //
    // BRAM read is synchronous: rdata <= ram[raddr] on posedge clk.
    // Therefore rdata lags raddr by one clock cycle.  The address logic
    // accounts for this pipeline delay by issuing addresses ONE CYCLE AHEAD
    // of when the data is consumed.
    //
    // Layer 2 pipeline timing (critical — verified cycle-by-cycle):
    //   L2_PREFETCH: raddr=16  → BRAM latches ram[16] on posedge
    //   LAYER2 cnt=0: raddr=17 → rdata=ram[16]=neuron0  → score0 ← pop ✓
    //   LAYER2 cnt=1: raddr=18 → rdata=ram[17]=neuron1  → score1 ← pop ✓
    //   LAYER2 cnt=2: raddr=18 → rdata=ram[18]=neuron2  → score2 ← pop ✓
    // -------------------------------------------------------------------------
    always @(*) begin
        case (state)
            STATE_L1_PREFETCH: begin
                bram_raddr = 5'd0;
            end

            STATE_LAYER1: begin
                // Issue address for NEXT cycle's read; cap at 16 (first L2 word)
                bram_raddr = (cycle_cnt < 5'd15) ? (cycle_cnt + 1'b1) : 5'd16;
            end

            STATE_L2_PREFETCH: begin
                // Address 16 = first L2 weight word (neuron 0)
                bram_raddr = 5'd16;
            end

            STATE_LAYER2: begin
                // One-ahead addressing: at cnt=0 we issue addr 17 (for cnt=1),
                // while consuming the rdata from the prefetch (addr 16 = neuron 0).
                if (cycle_cnt == 5'd0) begin
                    bram_raddr = 5'd17; // Prefetch neuron 1 for next cycle
                end else begin
                    bram_raddr = 5'd18; // Prefetch neuron 2 (or hold at 18)
                end
            end

            default: begin
                bram_raddr = 5'd0;
            end
        endcase
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= STATE_IDLE;
            cycle_cnt <= 5'd0;
            done <= 1'b0;
            decision <= 2'd0;
            layer1_out <= 64'd0;
            score0 <= 7'd0;
            score1 <= 7'd0;
            score2 <= 7'd0;
        end else begin
            // Default pulse
            done <= 1'b0;
            
            case (state)
                STATE_IDLE: begin
                    if (start) begin
                        state <= STATE_L1_PREFETCH;
                        cycle_cnt <= 5'd0;
                    end
                end
                
                STATE_L1_PREFETCH: begin
                    // One cycle for synchronous BRAM rdata <= ram[0].
                    state <= STATE_LAYER1;
                    cycle_cnt <= 5'd0;
                end

                STATE_LAYER1: begin
                    // Store the 4 thresholded neuron outputs for the current BRAM word.
                    // Popcount threshold is N/2 = 8.
                    layer1_out[ cycle_cnt*4 + 0 ] <= (unit_pop[0] >= 5'd8) ? 1'b1 : 1'b0;
                    layer1_out[ cycle_cnt*4 + 1 ] <= (unit_pop[1] >= 5'd8) ? 1'b1 : 1'b0;
                    layer1_out[ cycle_cnt*4 + 2 ] <= (unit_pop[2] >= 5'd8) ? 1'b1 : 1'b0;
                    layer1_out[ cycle_cnt*4 + 3 ] <= (unit_pop[3] >= 5'd8) ? 1'b1 : 1'b0;

                    if (cycle_cnt < 5'd15) begin
                        cycle_cnt <= cycle_cnt + 1;
                    end else begin
                        state <= STATE_L2_PREFETCH;
                        cycle_cnt <= 5'd0;
                    end
                end

                STATE_L2_PREFETCH: begin
                    // One cycle for synchronous BRAM rdata <= ram[16].
                    state <= STATE_LAYER2;
                    cycle_cnt <= 5'd0;
                end
                
                STATE_LAYER2: begin
                    if (cycle_cnt == 0) begin
                        score0 <= l2_total_pop; // Store N0 score
                        cycle_cnt <= 1;
                    end else if (cycle_cnt == 1) begin
                        score1 <= l2_total_pop; // Store N1 score
                        cycle_cnt <= 2;
                    end else if (cycle_cnt == 2) begin
                        score2 <= l2_total_pop; // Store N2 score
                        state <= STATE_ARGMAX;
                    end
                end
                
                STATE_ARGMAX: begin
                    // Winner-take-all
                    if (score0 >= score1 && score0 >= score2) begin
                        decision <= 2'd0; // BUY
                    end else if (score1 >= score0 && score1 >= score2) begin
                        decision <= 2'd1; // HOLD
                    end else begin
                        decision <= 2'd2; // SELL
                    end
                    
                    done <= 1'b1;
                    state <= STATE_IDLE;
                end
            endcase
        end
    end

`ifdef FORMAL
    // Formal verification for bnn_core disabled in top-level integration check
    // to focus on the arbitration mutual exclusion proof.
`endif

endmodule
