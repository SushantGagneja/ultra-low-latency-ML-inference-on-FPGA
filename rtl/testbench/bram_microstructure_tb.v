`timescale 1ns/1ps

module bram_microstructure_tb;

    reg clk;
    reg we;
    reg [4:0] addr;
    reg [119:0] din;
    wire [119:0] dout;

    bram_microstructure u_bram (
        .clk(clk),
        .we(we),
        .addr(addr),
        .din(din),
        .dout(dout)
    );

    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    initial begin
        $dumpfile("bram_microstructure_tb.vcd");
        $dumpvars(0, bram_microstructure_tb);
        
        we = 0;
        addr = 0;
        din = 0;
        
        #20;
        
        // Write to address 5
        @(posedge clk);
        we <= 1;
        addr <= 5;
        din <= 120'hAAAA_BBBB_CCCC_DDDD_EEEE_FFFF_1234_5678;
        
        @(posedge clk);
        we <= 0;
        
        // Read from address 5
        @(posedge clk);
        addr <= 5;
        
        @(posedge clk);
        // Data should be valid here (1-cycle read latency)
        #1;
        if (dout !== 120'hAAAA_BBBB_CCCC_DDDD_EEEE_FFFF_1234_5678) begin
            $display("ERROR: BRAM Read Failed. Expected AAAA_BBBB_CCCC_DDDD_EEEE_FFFF_1234_5678, got %h", dout);
            $finish;
        end
        
        // Write to address 19
        @(posedge clk);
        we <= 1;
        addr <= 19;
        din <= 120'h1111_2222_3333_4444_5555_6666_7777_8888;
        
        @(posedge clk);
        we <= 0;
        addr <= 19; // Keep reading 19
        
        @(posedge clk);
        #1;
        if (dout !== 120'h1111_2222_3333_4444_5555_6666_7777_8888) begin
            $display("ERROR: BRAM Read Failed. Expected 1111_2222_3333_4444_5555_6666_7777_8888, got %h", dout);
            $finish;
        end
        
        $display("BRAM tests passed.");
        $finish;
    end

endmodule
