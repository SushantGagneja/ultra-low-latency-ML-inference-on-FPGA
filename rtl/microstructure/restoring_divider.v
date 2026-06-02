`timescale 1ns/1ps

/*
 * Restoring Binary Divider
 *
 * Dividend: 72 bits (sum_pv)
 * Divisor:  48 bits (sum_v)
 * Quotient: 34 bits (vwap Q18.15)
 * Latency:  34 iteration cycles + 1 setup + 1 done = 36 cycles total
 *
 * State Transition Diagram:
 * IDLE -> (start & divisor == 0) -> ZERO_GUARD
 * IDLE -> (start & divisor != 0) -> CALC
 * CALC -> (count == 34) -> DONE
 * ZERO_GUARD -> DONE
 * DONE -> IDLE
 */

module restoring_divider (
    input  wire         clk,
    input  wire         rst_n,
    
    input  wire         start,
    input  wire [71:0]  dividend,
    input  wire [47:0]  divisor,
    
    output reg          done,
    output reg  [33:0]  quotient
);

    localparam STATE_IDLE       = 2'd0;
    localparam STATE_CALC       = 2'd1;
    localparam STATE_DONE       = 2'd2;
    localparam STATE_ZERO_GUARD = 2'd3;

    reg [1:0]  state;
    reg [5:0]  count;
    
    // R holds the remainder (up to 49 bits to accommodate shifted D)
    reg [48:0] R;
    // A holds the bottom bits of dividend, shifted out to become quotient
    reg [33:0] A;
    
    // Divisor latch
    reg [48:0] D;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state    <= STATE_IDLE;
            count    <= 6'd0;
            R        <= 49'd0;
            A        <= 34'd0;
            D        <= 49'd0;
            done     <= 1'b0;
            quotient <= 34'd0;
        end else begin
            // Default: done is a 1-cycle pulse
            done <= 1'b0;
            
            case (state)
                STATE_IDLE: begin
                    if (start) begin
                        if (divisor == 48'd0) begin
                            state <= STATE_ZERO_GUARD;
                        end else begin
                            // Initialize algorithm
                            // R gets the top 38 bits of the 72-bit dividend
                            R <= {11'd0, dividend[71:34]};
                            // A gets the bottom 34 bits of the dividend
                            A <= dividend[33:0];
                            // Latch divisor
                            D <= {1'b0, divisor};
                            
                            count <= 6'd0;
                            state <= STATE_CALC;
                        end
                    end
                end
                
                STATE_CALC: begin
                    if (count < 6'd34) begin
                        // 1. Shift (R, A) left by 1
                        // wire [48:0] next_R = {R[47:0], A[33]};
                        // wire [33:0] next_A = {A[32:0], 1'b0};
                        // 2. If next_R >= D, next_R -= D and next_A[0] = 1
                        
                        if ({R[47:0], A[33]} >= D) begin
                            R <= {R[47:0], A[33]} - D;
                            A <= {A[32:0], 1'b1};
                        end else begin
                            R <= {R[47:0], A[33]};
                            A <= {A[32:0], 1'b0};
                        end
                        
                        count <= count + 1'b1;
                    end else begin
                        state <= STATE_DONE;
                    end
                end
                
                STATE_ZERO_GUARD: begin
                    A <= 34'd0;
                    state <= STATE_DONE;
                end
                
                STATE_DONE: begin
                    quotient <= A;
                    done     <= 1'b1;
                    state    <= STATE_IDLE;
                end
                
                default: state <= STATE_IDLE;
            endcase
        end
    end

endmodule
