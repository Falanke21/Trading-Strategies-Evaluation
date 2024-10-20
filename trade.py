import asyncio
from alpaca.trading.client import TradingClient
from alpaca.data.live import StockDataStream
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# Replace these with your actual Alpaca API credentials
API_KEY = ""
SECRET_KEY = ""
PAPER = True  # Set to False for live trading
SYMBOL = 'AAPL'  # Replace with your desired stock symbol

# Initialize the Trading Client
trading_client = TradingClient(API_KEY, SECRET_KEY, paper=PAPER)

# Variable to store the latest price
latest_price = None

# Define the trade handler as an async function
async def on_trade(trade):
    global latest_price
    if trade.symbol == SYMBOL:
        latest_price = trade.price
        print(f"Received live trade update for {SYMBOL}: ${latest_price}")

# Define the trading logic
async def trade_logic():
    global latest_price
    if latest_price is None:
        print("No price data available yet.")
        return

    # Define simple buy/sell rules
    if latest_price < 150:  # Buy if the price is below $150
        try:
            order_data = MarketOrderRequest(
                symbol=SYMBOL,
                qty=1,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.GTC
            )
            trading_client.submit_order(order_data)
            print(f"Bought 1 share of {SYMBOL} at ${latest_price}")
        except Exception as e:
            print(f"Error buying {SYMBOL}: {e}")

    elif latest_price > 160:  # Sell if the price is above $160
        try:
            order_data = MarketOrderRequest(
                symbol=SYMBOL,
                qty=1,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.GTC
            )
            trading_client.submit_order(order_data)
            print(f"Sold 1 share of {SYMBOL} at ${latest_price}")
        except Exception as e:
            print(f"Error selling {SYMBOL}: {e}")

# Main function to run the data stream and trading logic
async def main():
    # Initialize the Stock Data Stream
    data_stream = StockDataStream(API_KEY, SECRET_KEY)
    
    # Register the trade handler as a coroutine
    data_stream.subscribe_trades(on_trade, SYMBOL)
    
    # Start the data stream in a separate task
    data_stream_task = asyncio.create_task(data_stream._run_forever())

    print(f"Subscribed to trade updates for {SYMBOL}. Starting trading logic...")

    # Continuously execute the trading logic at defined intervals
    try:
        while True:
            await trade_logic()
            await asyncio.sleep(60)  # Wait 1 minute before the next check
    finally:
        # Ensure the data stream is stopped if the program is interrupted
        data_stream_task.cancel()

# Run the main event loop
if __name__ == "__main__":
    asyncio.run(main())