`timescale 1ns/1ps

module bnn_top_tb;

    reg sys_clk;
    reg rst_n;
    
    reg spi_sclk;
    reg spi_cs_n;
    reg spi_mosi;
    wire spi_miso;
    wire fpga_done;
    
    bnn_top dut (
        .sys_clk(sys_clk),
        .rst_n(rst_n),
        .spi_sclk(spi_sclk),
        .spi_cs_n(spi_cs_n),
        .spi_mosi(spi_mosi),
        .spi_miso(spi_miso),
        .fpga_done(fpga_done)
    );
    
    initial begin
        sys_clk = 0;
        forever #5 sys_clk = ~sys_clk; // 100 MHz
    end
    
    // SPI packet: 0x10 command + 128 bit payload
    // Tick 20 values from golden model: 
    // bid_price = 1966014897, ask_price = 1966014897 + 10 = 1966014907 (just as an example)
    // Actually, any valid payload where ofi_valid and vwap_valid trigger is fine, but to make VWAP valid immediately, we need tick_count >= 20.
    
    task send_spi_tick;
        input [127:0] payload;
        integer i;
        reg [135:0] frame;
        begin
            frame = {8'h10, payload};
            spi_cs_n = 0;
            #10;
            for (i = 135; i >= 0; i = i - 1) begin
                spi_mosi = frame[i];
                #12.5; spi_sclk = 1; // 40 MHz = 25ns period
                #12.5; spi_sclk = 0;
            end
            #10;
            spi_cs_n = 1;
            #2000; // wait 2us for BNN inference and settling
        end
    endtask
    
    initial begin
        $dumpfile("bnn_top_trace.vcd");
        $dumpvars(0, bnn_top_tb);
        
        rst_n = 0;
        spi_sclk = 0;
        spi_cs_n = 1;
        spi_mosi = 0;
        
        #100 rst_n = 1;
        #100;
        
        // Send 20 ticks to warm up VWAP ring buffer
        // Payload layout:
        // [127:96] bid_price_q17_15
        // [95:64] ask_price_q17_15
        // [63:32] bid_qty_q16_16
        // [31:0] ask_qty_q16_16
        repeat (20) begin
            send_spi_tick(128'h00075300_00075310_00010000_00010000);
        end
        
        #5000;
        $finish;
    end

endmodule
