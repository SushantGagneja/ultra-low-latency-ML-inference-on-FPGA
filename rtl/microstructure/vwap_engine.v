`timescale 1ns/1ps

module vwap_engine (
    input  wire         clk,
    input  wire         rst_n,
    
    // Interface from Tick Parser
    input  wire         tick_valid,
    input  wire [31:0]  bid_price_q17_15,
    input  wire [31:0]  ask_price_q17_15,
    input  wire [31:0]  bid_qty_q16_16,
    input  wire [31:0]  ask_qty_q16_16,
    
    // Outputs to Quantizer / Top
    output reg          vwap_valid,
    output reg  [33:0]  vwap_q18_15,
    output reg          vwap_busy
);

    // --------------------------------------------------------
    // FSM States
    // --------------------------------------------------------
    localparam ST_IDLE    = 3'd0;
    localparam ST_CYCLE_1 = 3'd1;
    localparam ST_CYCLE_2 = 3'd2;
    localparam ST_DIVWAIT = 3'd3;

    reg [2:0] state;

    // --------------------------------------------------------
    // Internal State & Accumulators
    // --------------------------------------------------------
    reg [4:0]  write_ptr;
    wire [4:0] write_ptr_curr = write_ptr; // Explicit pre-increment pointer
    
    reg [4:0]  tick_count;
    
    reg [71:0] sum_pv;
    reg [47:0] sum_v;
    
    // Pipeline registers from Cycle 0
    reg [71:0] new_pv_reg;
    reg [47:0] new_v_reg;
    
    wire [31:0] mid_price = (bid_price_q17_15 + ask_price_q17_15) >> 1;
    wire [31:0] total_qty = bid_qty_q16_16 + ask_qty_q16_16;

    // --------------------------------------------------------
    // BRAM Interface
    // --------------------------------------------------------
    // PIPELINE NOTE: bram_we and bram_addr are combinational wires, not registered.
    // Non-blocking assignment of bram_we inside ST_CYCLE_1 would delay assertion
    // to the ST_CYCLE_1 -> ST_CYCLE_2 clock edge, at which point write_ptr has
    // already incremented. This would write new_pv/new_v to ram[write_ptr+1]
    // instead of ram[write_ptr], corrupting the ring buffer eviction logic.
    // Combinational assignment ensures BRAM sees write_ptr_curr on the correct edge.
    wire         bram_we   = (state == ST_CYCLE_1);
    wire [4:0]   bram_addr = write_ptr_curr;
    wire [119:0] bram_din  = {new_pv_reg, new_v_reg};
    wire [119:0] bram_dout;

    bram_microstructure u_bram (
        .clk(clk),
        .we(bram_we),
        .addr(bram_addr),
        .din(bram_din),
        .dout(bram_dout)
    );

    wire [71:0] oldest_pv = bram_dout[119:48];
    wire [47:0] oldest_v  = bram_dout[47:0];

    // --------------------------------------------------------
    // Divider Interface
    // --------------------------------------------------------
    reg         div_start;
    wire        div_done;
    wire [33:0] div_quotient;

    restoring_divider u_divider (
        .clk(clk),
        .rst_n(rst_n),
        .start(div_start),
        .dividend(sum_pv),
        .divisor(sum_v),
        .done(div_done),
        .quotient(div_quotient)
    );

    // --------------------------------------------------------
    // Engine FSM
    // --------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state       <= ST_IDLE;
            write_ptr   <= 5'd0;
            tick_count  <= 5'd0;
            sum_pv      <= 72'd0;
            sum_v       <= 48'd0;
            new_pv_reg  <= 72'd0;
            new_v_reg   <= 48'd0;
            vwap_valid  <= 1'b0;
            vwap_q18_15 <= 34'd0;
            vwap_busy   <= 1'b0;
            
            div_start   <= 1'b0;
        end else begin
            // Default pulse clears
            vwap_valid <= 1'b0;
            div_start  <= 1'b0;
            
            case (state)
                ST_IDLE: begin
                    if (tick_valid && !vwap_busy) begin
                        vwap_busy <= 1'b1;
                        
                        // CYCLE 0:
                        // Compute new PV and V in parallel
                        new_pv_reg <= mid_price * total_qty;
                        new_v_reg  <= {16'd0, total_qty};
                        
                        state <= ST_CYCLE_1;
                    end
                end
                
                ST_CYCLE_1: begin
                    // CYCLE 1:
                    // BRAM read data is valid now (oldest_pv, oldest_v)
                    // 1. Update accumulators
                    // Note: If tick_count < 20, BRAM outputs 0 (because it's initialized to 0)
                    sum_pv <= sum_pv - oldest_pv + new_pv_reg;
                    sum_v  <= sum_v  - oldest_v  + new_v_reg;
                    
                    // 2. Write new_pv and new_v to BRAM at write_ptr
                    // bram_we is combinational. bram_din is also combinational.
                    
                    // 3. Increment pointers safely
                    if (write_ptr == 5'd19)
                        write_ptr <= 5'd0;
                    else
                        write_ptr <= write_ptr + 1'b1;
                        
                    // Saturating tick counter
                    if (tick_count < 5'd20)
                        tick_count <= tick_count + 1'b1;
                        
                    state <= ST_CYCLE_2;
                end
                
                ST_CYCLE_2: begin
                    // CYCLE 2:
                    if (tick_count >= 5'd20) begin
                        // Trigger divider
                        div_start <= 1'b1;
                        state     <= ST_DIVWAIT;
                    end else begin
                        // Still in warmup
                        vwap_busy <= 1'b0;
                        state     <= ST_IDLE;
                    end
                end
                
                ST_DIVWAIT: begin
                    // CYCLE N: Divider done
                    if (div_done) begin
                        vwap_q18_15 <= div_quotient;
                        vwap_valid  <= 1'b1;
                        vwap_busy   <= 1'b0;
                        state       <= ST_IDLE;
                    end
                end
                
                default: state <= ST_IDLE;
            endcase
        end
    end

endmodule
