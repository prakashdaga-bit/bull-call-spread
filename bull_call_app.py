import streamlit as st
import yfinance as yf
import pandas as pd
import datetime
import time
import random
import math
from dataclasses import dataclass
from typing import List, Optional, Dict, Union

# ==========================================
# GLOBAL CONFIGURATION
# ==========================================
st.set_page_config(page_title="Options Strategy Master", page_icon="📈", layout="wide")

# ==========================================
# SHARED HELPER FUNCTIONS (Real Data)
# ==========================================

def get_monthly_expirations(ticker_obj, limit=3):
    """Filters the list of expiration dates to find the next 'limit' distinct months."""
    try:
        expirations = ticker_obj.options
    except:
        return []
        
    if not expirations:
        return []
    dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in expirations]
    unique_months = []
    seen_months = set()
    for date in dates:
        month_key = (date.year, date.month)
        if month_key not in seen_months:
            unique_months.append(date.strftime('%Y-%m-%d'))
            seen_months.add(month_key)
        if len(unique_months) >= limit:
            break
    return unique_months

def filter_tradeable_options(chain):
    """Filters the option chain to keep only rows where Ask > 0 or LastPrice > 0."""
    if chain.empty: return chain
    cols = chain.columns
    has_ask = 'ask' in cols
    has_last = 'lastPrice' in cols
    if not has_ask and not has_last: return pd.DataFrame() 
    mask = pd.Series(False, index=chain.index)
    if has_ask: mask |= (chain['ask'] > 0)
    if has_last: mask |= (chain['lastPrice'] > 0)
    return chain[mask]

def find_closest_strike(chain, price_target):
    if chain.empty: return None
    chain = chain.copy()
    chain['abs_diff'] = (chain['strike'] - price_target).abs()
    return chain.sort_values('abs_diff').iloc[0]

def get_price(option_row, price_type='mid'):
    """
    Returns the price for the option.
    price_type: 'ask' (buying), 'bid' (selling), or 'mid' (valuing)
    """
    bid = option_row.get('bid', 0)
    ask = option_row.get('ask', 0)
    last = option_row.get('lastPrice', 0)
    
    if price_type == 'mid':
        if bid > 0 and ask > 0: return (bid + ask) / 2
        return last
    elif price_type == 'ask':
        return ask if ask > 0 else last
    elif price_type == 'bid':
        return bid if bid > 0 else (last * 0.95) # Fallback if no bid
    return last

def get_option_chain_with_retry(stock, date, retries=3):
    for i in range(retries):
        try:
            return stock.option_chain(date)
        except Exception as e:
            if i == retries - 1: raise e
            time.sleep((2 ** (i + 1)) + random.uniform(0.5, 1.5))
    return None

# ==========================================
# PART 1: SIMPLE ANALYSIS LOGIC
# ==========================================

@st.cache_data(ttl=600, show_spinner=False)
def fetch_and_analyze_ticker(ticker_symbol, strategy_type):
    """Performs option strategy analysis for a single ticker (Real Data)."""
    try:
        stock = yf.Ticker(ticker_symbol)
        try:
            cmp = stock.fast_info['last_price']
            if cmp is None: raise ValueError("Fast info returned None")
        except:
            hist = stock.history(period='1d')
            if hist.empty: return None, None, f"No price data found for {ticker_symbol}."
            cmp = hist['Close'].iloc[-1]

        target_dates = get_monthly_expirations(stock, limit=3)
        if not target_dates: return None, None, f"No options data found for {ticker_symbol}."

        analysis_rows = []
        summary_returns = {"Stock": ticker_symbol}

        for i, date in enumerate(target_dates):
            time.sleep(random.uniform(1.0, 2.0)) # Rate limit protection
            try:
                opt_chain = get_option_chain_with_retry(stock, date)
                calls = opt_chain.calls
                puts = opt_chain.puts
                
                if calls.empty and puts.empty: continue

                if strategy_type == "Bull Call Spread":
                    if calls.empty: continue
                    valid_calls = filter_tradeable_options(calls)
                    if valid_calls.empty: continue
                    
                    target_price = cmp * 1.05
                    long_leg = find_closest_strike(valid_calls, cmp)
                    short_leg = find_closest_strike(valid_calls, target_price)

                    if long_leg is None or short_leg is None: continue
                    if long_leg['strike'] == short_leg['strike']:
                        higher_strikes = valid_calls[valid_calls['strike'] > long_leg['strike']]
                        if not higher_strikes.empty: short_leg = higher_strikes.iloc[0]
                        else: continue 

                    buy_strike = long_leg['strike']
                    sell_strike = short_leg['strike']
                    long_ask = get_price(long_leg, 'ask')
                    short_bid = get_price(short_leg, 'bid')

                    if long_ask == 0: continue
                    net_cost = long_ask - short_bid
                    spread_width = sell_strike - buy_strike
                    max_gain = spread_width - net_cost
                    breakeven = buy_strike + net_cost
                    
                    ret_pct = 0
                    if net_cost > 0: ret_pct = (max_gain / net_cost) * 100

                    analysis_rows.append({
                        "Expiration": date, "Buy Strike": buy_strike, "Sell Strike": sell_strike,
                        "Net Cost": net_cost, "Max Gain": max_gain, "Return %": ret_pct, "Breakeven": breakeven
                    })
                    summary_returns[date] = f"{ret_pct:.1f}%"

                elif strategy_type == "Long Straddle":
                    if calls.empty or puts.empty: continue
                    valid_calls = filter_tradeable_options(calls)
                    valid_puts = filter_tradeable_options(puts)
                    if valid_calls.empty or valid_puts.empty: continue

                    common_strikes = set(valid_calls['strike']).intersection(set(valid_puts['strike']))
                    if not common_strikes: continue
                    available_strikes = pd.DataFrame({'strike': list(common_strikes)})
                    closest_row = find_closest_strike(available_strikes, cmp)
                    if closest_row is None: continue
                    
                    strike = closest_row['strike']
                    atm_call = valid_calls[valid_calls['strike'] == strike].iloc[0]
                    atm_put = valid_puts[valid_puts['strike'] == strike].iloc[0]
                    call_ask = get_price(atm_call, 'ask')
                    put_ask = get_price(atm_put, 'ask')

                    if call_ask == 0 or put_ask == 0: continue
                    net_cost = call_ask + put_ask
                    breakeven_low = strike - net_cost
                    breakeven_high = strike + net_cost
                    move_pct = (net_cost / cmp) * 100

                    analysis_rows.append({
                        "Expiration": date, "Strike": strike, "Call Cost": call_ask, "Put Cost": put_ask,
                        "Net Cost": net_cost, "BE Low": breakeven_low, "BE High": breakeven_high, "Move Needed": move_pct
                    })
                    summary_returns[date] = f"±{move_pct:.1f}%"

            except Exception as e: continue

        if not analysis_rows: return None, None, "Could not construct valid spreads."
        return summary_returns, pd.DataFrame(analysis_rows), None
    except Exception as e: return None, None, str(e)


# ==========================================
# PART 2: CUSTOM 4-LEG STRATEGY (REAL DATA)
# ==========================================

@st.cache_data(ttl=300, show_spinner=False)
def analyze_custom_strategy(ticker, sentiment, volatility_high):
    """
    Constructs a 4-leg strategy based on specific strike percentages using Real Data.
    
    Logic:
    1. Fetch real price and chain.
    2. Determine Strikes: -15%, -8%, +8%, +15%.
    3. Determine Action (Buy/Sell) based on Sentiment & Volatility.
    """
    try:
        # 1. Fetch Data
        stock = yf.Ticker(ticker)
        try:
            current_price = stock.fast_info['last_price']
        except:
            hist = stock.history(period='1d')
            if hist.empty: return None, "No price data found."
            current_price = hist['Close'].iloc[-1]
            
        # Get next valid monthly expiration (approx 30 days out)
        dates = get_monthly_expirations(stock, limit=2)
        if not dates: return None, "No option chain found."
        
        # Prefer the second month if available (more time for strategy), else first
        target_date = dates[0] 
        
        chain = get_option_chain_with_retry(stock, target_date)
        calls = filter_tradeable_options(chain.calls)
        puts = filter_tradeable_options(chain.puts)
        
        if calls.empty or puts.empty: return None, "Chain data incomplete (missing calls or puts)."

        # 2. Identify Strikes
        # Targets: -15% (0.85), -8% (0.92), +8% (1.08), +15% (1.15)
        k_put_far_target = current_price * 0.85
        k_put_near_target = current_price * 0.92
        k_call_near_target = current_price * 1.08
        k_call_far_target = current_price * 1.15
        
        # Find actual contracts
        put_far = find_closest_strike(puts, k_put_far_target)
        put_near = find_closest_strike(puts, k_put_near_target)
        call_near = find_closest_strike(calls, k_call_near_target)
        call_far = find_closest_strike(calls, k_call_far_target)
        
        if any(leg is None for leg in [put_far, put_near, call_near, call_far]):
            return None, "Could not find options at required strike depths (liquidity issue)."
            
        # 3. Determine Strategy Structure
        # Default Template (<10% Move):
        #   Put Side: Buy Far (-15%), Sell Near (-8%) -> Bull Put Spread (Credit)
        #   Call Side: Buy Near (+8%), Sell Far (+15%) -> Bull Call Spread (Debit)
        #   Result: Net Bullish Bias.
        
        legs = []
        
        # --- LOGIC MATRIX ---
        # Sentiment: Bullish vs Bearish
        # Volatility: <10% (Low) vs >10% (High)
        
        if sentiment == "Bullish":
            if not volatility_high: # < 10% Move
                # Logic: Buy P(-15%), Sell P(-8%), Buy C(+8%), Sell C(+15%)
                legs = [
                    {"type": "Put", "action": "Buy", "row": put_far, "desc": "Far OTM Put (-15%)"},
                    {"type": "Put", "action": "Sell", "row": put_near, "desc": "Near OTM Put (-8%)"},
                    {"type": "Call", "action": "Buy", "row": call_near, "desc": "Near OTM Call (+8%)"},
                    {"type": "Call", "action": "Sell", "row": call_far, "desc": "Far OTM Call (+15%)"},
                ]
            else: # > 10% Move (Opposite)
                # Logic: Sell P(-15%), Buy P(-8%), Sell C(+8%), Buy C(+15%)
                legs = [
                    {"type": "Put", "action": "Sell", "row": put_far, "desc": "Far OTM Put (-15%)"},
                    {"type": "Put", "action": "Buy", "row": put_near, "desc": "Near OTM Put (-8%)"},
                    {"type": "Call", "action": "Sell", "row": call_near, "desc": "Near OTM Call (+8%)"},
                    {"type": "Call", "action": "Buy", "row": call_far, "desc": "Far OTM Call (+15%)"},
                ]
        else: # Bearish (Mirroring the Bullish Logic)
            if not volatility_high: # < 10% Move
                # Mirror of Bullish/<10%:
                # Put Side becomes Bear Put Spread (Debit): Buy Near P(-8%), Sell Far P(-15%)
                # Call Side becomes Bear Call Spread (Credit): Sell Near C(+8%), Buy Far C(+15%)
                legs = [
                    {"type": "Put", "action": "Sell", "row": put_far, "desc": "Far OTM Put (-15%)"},
                    {"type": "Put", "action": "Buy", "row": put_near, "desc": "Near OTM Put (-8%)"},
                    {"type": "Call", "action": "Sell", "row": call_near, "desc": "Near OTM Call (+8%)"},
                    {"type": "Call", "action": "Buy", "row": call_far, "desc": "Far OTM Call (+15%)"},
                ]
            else: # > 10% Move (Opposite of Bearish/<10%)
                legs = [
                    {"type": "Put", "action": "Buy", "row": put_far, "desc": "Far OTM Put (-15%)"},
                    {"type": "Put", "action": "Sell", "row": put_near, "desc": "Near OTM Put (-8%)"},
                    {"type": "Call", "action": "Buy", "row": call_near, "desc": "Near OTM Call (+8%)"},
                    {"type": "Call", "action": "Sell", "row": call_far, "desc": "Far OTM Call (+15%)"},
                ]

        # 4. Calculate Financials
        net_premium = 0.0 # Positive = Credit, Negative = Debit
        trade_details = []
        
        for leg in legs:
            strike = leg['row']['strike']
            # Use Ask if Buying, Bid if Selling
            price = get_price(leg['row'], 'ask' if leg['action'] == "Buy" else 'bid')
            
            impact = -price if leg['action'] == "Buy" else price
            net_premium += impact
            
            trade_details.append({
                "Action": leg['action'],
                "Type": leg['type'],
                "Strike": strike,
                "Price": price,
                "Description": leg['desc']
            })

        # 5. P&L Analysis
        # Determine Max Profit/Loss and Breakevens
        # We simulate P&L at expiration across a range
        sim_prices = sorted([l['row']['strike'] for l in legs] + [current_price])
        min_p = current_price * 0.5
        max_p = current_price * 1.5
        sim_range = [min_p] + sim_prices + [max_p]
        
        profits = []
        for p in sim_range:
            p_l = net_premium
            for leg in legs:
                strike = leg['row']['strike']
                is_call = leg['type'] == "Call"
                is_buy = leg['action'] == "Buy"
                
                val_at_expiry = max(0, p - strike) if is_call else max(0, strike - p)
                leg_pnl = val_at_expiry if is_buy else -val_at_expiry
                p_l += leg_pnl
            profits.append(p_l)
            
        max_upside = max(profits)
        max_loss = min(profits)
        
        # Heuristic for breakevens (finding zero crossings)
        breakevens = []
        # Simple scan
        if min(profits) < 0 < max(profits):
            breakevens.append("Dynamic (Depends on expiry price)")
        else:
            breakevens.append("None (Always Profit/Loss)" if net_premium != 0 else "Flat")

        return {
            "current_price": current_price,
            "expiry": target_date,
            "net_premium": net_premium,
            "max_upside": max_upside,
            "max_loss": max_loss,
            "legs": trade_details
        }, None

    except Exception as e:
        return None, str(e)

# ==========================================
# PART 3: MAIN APP INTERFACE
# ==========================================

def main():
    st.title("🛡️ Options Strategy Master")
    
    # --- MASTER SWITCH ---
    mode = st.sidebar.radio(
        "Select Analysis Mode:", 
        ["Simple Analysis (Standard)", "Custom Strategy Generator (Iron Condor Style)"],
        index=0
    )
    st.sidebar.markdown("---")

    # ==========================================
    # MODE A: SIMPLE ANALYSIS
    # ==========================================
    if mode == "Simple Analysis (Standard)":
        st.subheader("📈 Multi-Stock Real-Time Analysis")
        st.caption("Fetches live option chains from Yahoo Finance. Standard Spreads/Straddles.")
        
        # Inputs
        strategy = st.radio("Strategy Type:", ("Bull Call Spread", "Long Straddle"), horizontal=True)
        default_tickers = "NKE, AAPL, AMD, TSLA"
        ticker_input = st.text_input("Enter Tickers (comma-separated):", value=default_tickers)
        
        if st.button("Analyze Real-Time Data"):
            if not ticker_input:
                st.error("Please enter at least one ticker.")
            else:
                tickers = [t.strip().upper() for t in ticker_input.split(',') if t.strip()]
                all_summaries = []
                all_details = {} 
                errors = []
                
                progress_bar = st.progress(0)
                with st.spinner(f"Fetching data from Yahoo Finance..."):
                    for i, ticker in enumerate(tickers):
                        summary, df, error = fetch_and_analyze_ticker(ticker, strategy)
                        if error: errors.append(f"{ticker}: {error}")
                        else:
                            all_summaries.append(summary)
                            all_details[ticker] = df
                        progress_bar.progress((i + 1) / len(tickers))

                # Output
                st.divider()
                if all_summaries:
                    st.header("Summary")
                    summary_df = pd.DataFrame(all_summaries)
                    cols = ["Stock"] + sorted([c for c in summary_df.columns if c != "Stock"])
                    st.dataframe(summary_df[cols], hide_index=True, use_container_width=True)
                else:
                    st.warning("No valid data found.")

                if all_details:
                    st.header("Detailed Breakdown")
                    for ticker, df in all_details.items():
                        with st.expander(f"{ticker} Details", expanded=False):
                            st.dataframe(df, use_container_width=True)

                if errors:
                    with st.expander("Errors"):
                        for e in errors: st.write(f"- {e}")

    # ==========================================
    # MODE B: CUSTOM 4-LEG STRATEGY
    # ==========================================
    else:
        st.subheader("🤖 Custom 4-Leg Strategy Generator")
        st.caption("Constructs a 4-leg structure (Iron Condor style) based on Sentiment & Expected Move using **Real-Time Data**.")
        
        # 1. Inputs
        c1, c2, c3 = st.columns(3)
        ticker = c1.text_input("Stock Ticker", "TSLA").upper()
        sentiment = c2.selectbox("Your Sentiment", ["Bullish", "Bearish"])
        vol_expect = c3.selectbox("Expect >10% Move?", ["No (Range Bound / <10%)", "Yes (Breakout / >10%)"])
        
        is_high_vol = "Yes" in vol_expect
        
        st.info(f"""
        **Strategy Logic:**
        - **Strikes:** Targets -15%, -8%, +8%, +15% relative to Spot Price.
        - **Structure:** Adapts leg direction (Buy/Sell) based on {sentiment} sentiment and {'High' if is_high_vol else 'Low'} Volatility expectation.
        """)

        if st.button("Generate Strategy"):
            with st.spinner(f"Fetching Real-Time Chain for {ticker}..."):
                res, error = analyze_custom_strategy(ticker, sentiment, is_high_vol)
                
                if error:
                    st.error(f"Analysis Failed: {error}")
                else:
                    st.success(f"Strategy Generated for {ticker} (Expiry: {res['expiry']})")
                    st.metric("Current Price", f"${res['current_price']:.2f}")
                    
                    # Display Legs
                    st.subheader("1. Trade Structure")
                    legs_df = pd.DataFrame(res['legs'])
                    st.dataframe(legs_df.style.format({"Price": "${:.2f}", "Strike": "${:.2f}"}), use_container_width=True)
                    
                    # Display Financials
                    st.subheader("2. Risk Profile")
                    col1, col2, col3 = st.columns(3)
                    
                    net = res['net_premium']
                    lbl = "Net Credit" if net > 0 else "Net Debit"
                    col1.metric(lbl, f"${abs(net):.2f}")
                    
                    col2.metric("Max Potential Profit", f"${res['max_upside']:.2f}")
                    col3.metric("Max Potential Loss", f"${res['max_loss']:.2f}")
                    
                    # P&L Explanation
                    st.caption(f"Note: Max Profit/Loss are theoretical estimates at expiration based on the 4 legs. 'Net Credit' means you receive money upfront; 'Net Debit' means you pay upfront.")

if __name__ == "__main__":
    main()
