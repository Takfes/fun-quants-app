"""
# ==============================================================
# PnL Calculator
# ==============================================================
"""


def calculate_pnl(entry_price, exit_price, original_amount, borrowed_amount):
    # Total position size is the sum of original and borrowed amounts
    total_position_size = original_amount + borrowed_amount
    # Calculate the profit or loss
    pnl = (exit_price - entry_price) / entry_price * total_position_size
    # Calculate the Return on Equity (ROE)
    roe = (pnl / original_amount) if original_amount != 0 else 0
    return pnl, roe


# Example usage
entry_price = 1000  # For example, 100 USDT
exit_price = 1100  # For example, 110 USDT
original_amount = 100  # Your capital
borrowed_amount = 900  # Borrowed funds

# Borrowed Amount -> Leverage
calculate_leverage = lambda o, b: o / (o + b)
leverage = calculate_leverage(original_amount, borrowed_amount)
print(f"Leverage: {leverage} | {leverage:.1%} | 1:{leverage*100:.0f}")

# # Leverage -> Calcualted Borrowed Funds
# leverage = 0.1
# borrowed_amount = (original_amount / leverage) - original_amount

pct_change = (exit_price - entry_price) / entry_price
final_amount = round(original_amount * (1 + pct_change), 4)
pnl_wom = final_amount - original_amount
roe_wom = pnl_wom / original_amount
print(f"Profit/Loss w/o Leverage: {pnl_wom:,.2f}")
print(f"Return on Equity w/o Leverage: {roe_wom:.2%}")

pnl, roe = calculate_pnl(entry_price, exit_price, original_amount, borrowed_amount)
print(f"Profit/Loss: {pnl:,.2f}")
print(f"Return on Equity: {roe:.2%}")


"""
# ==============================================================
# Target Price Calculator
# ==============================================================
"""


def calculate_target_price(
    entry_price, original_amount, borrowed_amount, desired_roe, position_type
):
    # Calculate the total position size
    total_position_size = original_amount + borrowed_amount
    # Calculate leverage
    leverage = original_amount / total_position_size
    # Calculate desired profit
    desired_profit = (1 + (desired_roe / 100)) * entry_price - entry_price
    # Desired profit based on leverage
    desired_profit_leveraged = desired_profit * leverage
    if position_type == "long":
        # Calculate the target exit price for a long position
        target_price = float(entry_price) + desired_profit_leveraged
    elif position_type == "short":
        # Calculate the target exit price for a short position
        target_price = float(entry_price) - desired_profit_leveraged
    else:
        raise ValueError("Position type must be 'long' or 'short'")
    return target_price


# Example usage:
entry_price = 25  # For example, 100 USDT
desired_roe = 33  # Desired ROE in percentage
original_amount = 5  # Your capital
borrowed_amount = 45  # Borrowed funds
position_type = "short"  # Position type: 'long' or 'short'

target_price_wom = (
    (1 + desired_roe / 100) * entry_price
    if position_type == "long"
    else (1 - desired_roe / 100) * entry_price
)
print(
    f"The target price to exit the {position_type} bet for {desired_roe}% ROE w/o Leverage: {target_price_wom:.2f}"
)

target_price = calculate_target_price(
    entry_price, original_amount, borrowed_amount, desired_roe, position_type
)
print(
    f"The target price to exit the {position_type} bet for {desired_roe}% ROE: {target_price:.2f}"
)
