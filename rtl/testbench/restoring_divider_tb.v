`timescale 1ns/1ps

module restoring_divider_tb;

    reg clk;
    reg rst_n;
    
    reg start;
    reg [71:0] dividend;
    reg [47:0] divisor;
    
    wire done;
    wire [33:0] quotient;

    restoring_divider u_div (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .dividend(dividend),
        .divisor(divisor),
        .done(done),
        .quotient(quotient)
    );

    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    initial begin
        $dumpfile("restoring_divider_tb.vcd");
        $dumpvars(0, restoring_divider_tb);
        
        rst_n = 0;
        start = 0;
        dividend = 0;
        divisor = 0;
        
        #20;
        rst_n = 1;
        #20;
        
        // Test 1: Normal Division
        // 1000 / 5 = 200
        @(posedge clk);
        start <= 1;
        dividend <= 72'd1000;
        divisor <= 48'd5;
        
        @(posedge clk);
        start <= 0;
        
        wait(done);
        if (quotient !== 34'd200) begin
            $display("ERROR: 1000 / 5 failed. Got %d", quotient);
            $finish;
        end
        $display("Test 1 Passed: 1000 / 5 = 200");
        
        #20;
        
        // Test 2: sum_v = 0 (Zero Guard)
        @(posedge clk);
        start <= 1;
        dividend <= 72'd50000;
        divisor <= 48'd0;
        
        @(posedge clk);
        start <= 0;
        
        wait(done);
        if (quotient !== 34'd0) begin
            $display("ERROR: Zero guard failed. Got %d", quotient);
            $finish;
        end
        $display("Test 2 Passed: 50000 / 0 = 0 (Guard)");
        
        #20;
        
        // Test 3: Minimum non-zero divisor (1)
        @(posedge clk);
        start <= 1;
        dividend <= 72'd987654321;
        divisor <= 48'd1;
        
        @(posedge clk);
        start <= 0;
        
        wait(done);
        if (quotient !== 34'd987654321) begin
            $display("ERROR: Divisor=1 failed. Got %d", quotient);
            $finish;
        end
        $display("Test 3 Passed: 987654321 / 1 = 987654321");
        
        #20;
        
        // Test 4: Maximum Quotient (2^34 - 1)
        @(posedge clk);
        start <= 1;
        dividend <= {38'd0, 34'h3FFFFFFFF}; // (1<<34)-1
        divisor <= 48'd1;
        
        @(posedge clk);
        start <= 0;
        
        wait(done);
        if (quotient !== 34'h3FFFFFFFF) begin
            $display("ERROR: Max quotient failed. Got %h", quotient);
            $finish;
        end
        $display("Test 4 Passed: (2^34 - 1) / 1 = 2^34 - 1");
        
        $display("All restoring_divider tests passed.");
        $finish;
    end

endmodule
