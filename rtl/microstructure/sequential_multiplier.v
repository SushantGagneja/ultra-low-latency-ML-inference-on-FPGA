`timescale 1ns/1ps

module sequential_multiplier (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start,
    input  wire [31:0] a,
    input  wire [31:0] b,
    output reg         done,
    output reg  [63:0] product
);

    reg [31:0] a_reg;
    reg [63:0] p_reg;
    reg [5:0]  count;
    reg        busy;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            done    <= 1'b0;
            product <= 64'd0;
            a_reg   <= 32'd0;
            p_reg   <= 64'd0;
            count   <= 6'd0;
            busy    <= 1'b0;
        end else begin
            done <= 1'b0; // Default pulse

            if (start && !busy) begin
                busy  <= 1'b1;
                a_reg <= a;
                p_reg <= {32'd0, b};
                count <= 6'd32;
            end else if (busy) begin
                if (count > 0) begin
                    if (p_reg[0]) begin
                        p_reg <= { (p_reg[63:32] + a_reg), p_reg[31:1] };
                    end else begin
                        p_reg <= { 1'b0, p_reg[63:1] };
                    end
                    count <= count - 1'b1;
                end else begin
                    busy    <= 1'b0;
                    done    <= 1'b1;
                    product <= p_reg;
                end
            end
        end
    end

endmodule
