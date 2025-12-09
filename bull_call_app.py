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
# WITH OPTIMIZATION ENGINE
# ==========================================

def calculate_strategy_metrics(legs, current_price):
    """Calculates Net Premium, Max Profit, Max Loss for a given set of legs."""
    net_premium = 0.0
    for leg in legs:
        strike = leg['row']['strike']
        # Ask if Buying, Bid if Selling
        price = get_price(leg['row'], 'ask' if leg['action'] == "Buy" else 'bid')
        impact = -price if leg['action'] == "Buy" else price
        net_premium += impact

    # Simulation for Max Upside/Loss
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
        
    return {
        "net_premium": net_premium,
        "max_upside": max(profits),
        "max_loss": min(profits)
    }

@st.cache_data(ttl=300, show_spinner=False)
def analyze_custom_strategy(ticker, sentiment, volatility_high, optimize=False):
    """
    1. Fetches Chain.
    2. Builds 'Base' Strategy (Fixed %).
    3. If optimize=True, scans neighbors to find Max Profit > Max Loss (Sweet Spot).
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
            
        dates = get_monthly_expirations(stock, limit=2)
        if not dates: return None, "No option chain found."
        target_date = dates[0] 
        
        chain = get_option_chain_with_retry(stock, target_date)
        # We need tradeable options, but sorted by strike
        calls = filter_tradeable_options(chain.calls).sort_values('strike').reset_index(drop=True)
        puts = filter_tradeable_options(chain.puts).sort_values('strike').reset_index(drop=True)
        
        if calls.empty or puts.empty: return None, "Chain data incomplete."

        # --- BASE STRATEGY CONSTRUCTION ---
        targets = {
            "pf": current_price * 0.85, # Put Far (15% cheaper)
            "pn": current_price * 0.92, # Put Near (8% cheaper)
            "cn": current_price * 1.08, # Call Near (8% expensive)
            "cf": current_price * 1.15  # Call Far (15% expensive)
        }
        
        # Helper to build legs from specific rows
        def build_legs(pf_row, pn_row, cn_row, cf_row):
            l = []
            
            # --- STRUCTURE DEFINITION ---
            # Volatility LOW (<10% move): Spread structures
            # Volatility HIGH (>10% move): Diagonal/Long structures
            
            if not volatility_high:
                # Low Volatility (Credit/Debit Bias)
                if sentiment == "Bullish":
                    # Buy P(-15%), Sell P(-8%), Buy C(+8%), Sell C(+15%) -> Net Bullish Condor
                    l = [
                        {"type": "Put", "action": "Buy", "row": pf_row, "desc": "Far Put (Long)"},
                        {"type": "Put", "action": "Sell", "row": pn_row, "desc": "Near Put (Short)"},
                        {"type": "Call", "action": "Buy", "row": cn_row, "desc": "Near Call (Long)"},
                        {"type": "Call", "action": "Sell", "row": cf_row, "desc": "Far Call (Short)"},
                    ]
                else: # Bearish
                    # Sell P(-15%), Buy P(-8%), Sell C(+8%), Buy C(+15%) -> Net Bearish Condor
                    l = [
                        {"type": "Put", "action": "Sell", "row": pf_row, "desc": "Far Put (Short)"},
                        {"type": "Put", "action": "Buy", "row": pn_row, "desc": "Near Put (Long)"},
                        {"type": "Call", "action": "Sell", "row": cn_row, "desc": "Near Call (Short)"},
                        {"type": "Call", "action": "Buy", "row": cf_row, "desc": "Far Call (Long)"},
                    ]
            else: 
                # High Volatility (Inverted/Explosive Structure)
                if sentiment == "Bullish":
                    # Inverted of Bullish/Low Vol: Sell P(-15%), Buy P(-8%), Sell C(+8%), Buy C(+15%)
                    l = [
                        {"type": "Put", "action": "Sell", "row": pf_row, "desc": "Far Put (Short)"},
                        {"type": "Put", "action": "Buy", "row": pn_row, "desc": "Near Put (Long)"},
                        {"type": "Call", "action": "Sell", "row": cn_row, "desc": "Near Call (Short)"},
                        {"type": "Call", "action": "Buy", "row": cf_row, "desc": "Far Call (Long)"},
                    ]
                else: # Bearish
                    # Inverted of Bearish/Low Vol: Buy P(-15%), Sell P(-8%), Buy C(+8%), Sell C(+15%)
                    l = [
                        {"type": "Put", "action": "Buy", "row": pf_row, "desc": "Far Put (Long)"},
                        {"type": "Put", "action": "Sell", "row": pn_row, "desc": "Near Put (Short)"},
                        {"type": "Call", "action": "Buy", "row": cn_row, "desc": "Near Call (Long)"},
                        {"type": "Call", "action": "Sell", "row": cf_row, "desc": "Far Call (Short)"},
                    ]
            return l

        # Find Base Rows
        pf_base = find_closest_strike(puts, targets["pf"])
        pn_base = find_closest_strike(puts, targets["pn"])
        cn_base = find_closest_strike(calls, targets["cn"])
        cf_base = find_closest_strike(calls, targets["cf"])
        
        if any(x is None for x in [pf_base, pn_base, cn_base, cf_base]):
            return None, "Could not find base strikes at required depth."

        base_legs = build_legs(pf_base, pn_base, cn_base, cf_base)
        base_metrics = calculate_strategy_metrics(base_legs, current_price)
        
        result_payload = {
            "ticker": ticker,
            "current_price": current_price,
            "expiry": target_date,
            "base": {
                "metrics": base_metrics,
                "legs": base_legs
            },
            "optimized": None # Initialize optimized slot
        }

        # --- OPTIMIZATION LOOP (THE SWEET SPOT SCANNER) ---
        if optimize:
            # We will scan neighbors of the base strikes to find MaxProfit > MaxLoss
            
            # Helper to safely get index
            def get_idx(df, strike): 
                indices = df.index[df['strike'] == strike].tolist()
                return indices[0] if indices else -1

            pf_idx = get_idx(puts, pf_base['strike'])
            pn_idx = get_idx(puts, pn_base['strike'])
            cn_idx = get_idx(calls, cn_base['strike'])
            cf_idx = get_idx(calls, cf_base['strike'])
            
            if any(i == -1 for i in [pf_idx, pn_idx, cn_idx, cf_idx]):
                 # If base indices weren't found precisely (shouldn't happen if base was successful)
                 return result_payload, None 

            best_ratio = -1.0
            best_config = None
            
            # Scan range: +/- 2 strikes around the targets
            range_scan = range(-2, 3) 
            
            # Limit iterations for safety
            
            for i1 in range_scan: # Put Far Offset
                for i2 in range_scan: # Put Near Offset
                    for i3 in range_scan: # Call Near Offset
                        for i4 in range_scan: # Call Far Offset
                            
                            # Safety bounds
                            if not (0 <= pf_idx+i1 < len(puts)): continue
                            if not (0 <= pn_idx+i2 < len(puts)): continue
                            if not (0 <= cn_idx+i3 < len(calls)): continue
                            if not (0 <= cf_idx+i4 < len(calls)): continue
                            
                            # Construct candidate rows
                            pf_cand = puts.iloc[pf_idx+i1]
                            pn_cand = puts.iloc[pn_idx+i2]
                            cn_cand = calls.iloc[cn_idx+i3]
                            cf_cand = calls.iloc[cf_idx+i4]
                            
                            # Logical Check: Strikes must maintain order
                            if pf_cand['strike'] >= pn_cand['strike']: continue
                            if cn_cand['strike'] >= cf_cand['strike']: continue
                            
                            cand_legs = build_legs(pf_cand, pn_cand, cn_cand, cf_cand)
                            cand_metrics = calculate_strategy_metrics(cand_legs, current_price)
                            
                            # Metric: Profit / Loss Ratio
                            max_loss_abs = abs(cand_metrics['max_loss'])
                            if max_loss_abs < 0.01: 
                                ratio = 0
                            else:
                                ratio = cand_metrics['max_upside'] / max_loss_abs
                                
                            if ratio > best_ratio:
                                best_ratio = ratio
                                best_config = {
                                    "metrics": cand_metrics,
                                    "legs": cand_legs,
                                    "ratio": ratio
                                }
            
            result_payload["optimized"] = best_config

        return result_payload, None

    except Exception as e:
        return None, str(e)

# ==========================================
# PART 3: MAIN APP INTERFACE
# ==========================================

def display_strategy_details(data, label, current_price, initial_sentiment, initial_vol):
    """Generates the detailed expander content for a single strategy result."""
    
    st.markdown(f"**{label}**")
    m = data['metrics']
    
    # Financials
    col1, col2, col3, col4 = st.columns(4)
    net = m['net_premium']
    lbl = "Net Credit" if net > 0 else "Net Debit"
    col1.metric("Spot Price", f"${current_price:.2f}")
    col2.metric(lbl, f"${abs(net):.2f}")
    col3.metric("Max Profit", f"${m['max_upside']:.2f}")
    col4.metric("Max Loss", f"${abs(m['max_loss']):.2f}")
    
    # Reward/Risk Ratio
    ratio = m['max_upside'] / abs(m['max_loss']) if abs(m['max_loss']) > 0 else 0
    st.markdown(f"**Reward/Risk Ratio:** `{ratio:.2f}`")
    
    if ratio > 1.0:
        st.success(f"🌟 Sweet Spot: Max Profit > Max Loss!")
    elif net > 0 and ratio > 0.3:
        st.info("High probability, low Risk/Reward (Typical Credit Strategy).")
    
    # Legs Table
    legs_simple = []
    for l in data['legs']:
        legs_simple.append({
            "Action": l['action'], "Type": l['type'], 
            "Strike": l['row']['strike'], "Price": get_price(l['row'], 'ask' if l['action']=="Buy" else 'bid')
        })
    st.dataframe(pd.DataFrame(legs_simple).style.format({"Price": "${:.2f}", "Strike": "${:.2f}"}), use_container_width=True)


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
        st.caption("Constructs an optimized 4-leg structure based on your market view.")
        
        # 1. Inputs
        c1, c2 = st.columns(2)
        default_tickers = "TSLA, AAPL, AMD"
        ticker_input = c1.text_input("Stock Tickers (comma-separated)", default_tickers).upper()
        sentiment = c1.selectbox("Your Sentiment (Applied to All)", ["Bullish", "Bearish"])
        vol_expect = c2.selectbox("Expect >10% Move?", ["No (Range Bound / <10%)", "Yes (Breakout / >10%)"])
        
        is_high_vol = "Yes" in vol_expect
        
        st.info(f"""
        **Optimization Goal:** Find the highest Reward/Risk ratio for the defined structure and sentiment.
        - **Target Strikes:** -15%, -8%, +8%, +15% from spot price.
        """)

        if st.button("Generate & Optimize Strategies"):
            if not ticker_input:
                st.error("Please enter at least one ticker.")
                return

            tickers = [t.strip().upper() for t in ticker_input.split(',') if t.strip()]
            all_results = {}
            all_summaries = []
            errors = []

            progress_bar = st.progress(0)
            
            with st.spinner(f"Fetching data and optimizing across {len(tickers)} tickers..."):
                for i, ticker in enumerate(tickers):
                    res, error = analyze_custom_strategy(ticker, sentiment, is_high_vol, optimize=True)
                    
                    if error:
                        errors.append(f"{ticker}: {error}")
                    else:
                        all_results[ticker] = res
                        
                        # Use the OPTIMIZED metrics for the summary table
                        metrics = res['optimized']['metrics']
                        ratio = res['optimized']['ratio']
                        
                        summary_data = {
                            "Stock": ticker,
                            "Spot Price": f"${res['current_price']:.2f}",
                            "Net Premium": f"${metrics['net_premium']:.2f}",
                            "Max Profit": f"${metrics['max_upside']:.2f}",
                            "Max Loss": f"${abs(metrics['max_loss']):.2f}",
                            "Reward/Risk": f"{ratio:.2f}"
                        }
                        all_summaries.append(summary_data)
                    
                    progress_bar.progress((i + 1) / len(tickers))

            # --- OUTPUT ---
            st.divider()
            
            if all_summaries:
                st.header("1. Strategy Summary (Optimized Results)")
                summary_df = pd.DataFrame(all_summaries)
                st.dataframe(summary_df, hide_index=True, use_container_width=True)

                st.header("2. Detailed Trade Structure & P&L")
                
                for ticker, res in all_results.items():
                    with st.expander(f"✨ {ticker} Optimized Strategy (R/R {res['optimized']['ratio']:.2f})", expanded=False):
                        
                        # Check if optimized is significantly better than base
                        base_ratio = res['base']['metrics']['max_upside'] / abs(res['base']['metrics']['max_loss'])
                        opt_ratio = res['optimized']['ratio']

                        if opt_ratio > base_ratio:
                            st.info(f"Optimization improved R/R from {base_ratio:.2f} to {opt_ratio:.2f}.")

                        display_strategy_details(
                            res['optimized'], 
                            f"Recommended Optimized Strategy (R/R {opt_ratio:.2f})", 
                            res['current_price'],
                            sentiment,
                            is_high_vol
                        )
                        
                        # Show base for comparison
                        with st.expander("Show Base Strategy (Fixed Targets)"):
                            display_strategy_details(
                                res['base'], 
                                f"Base Strategy (R/R {base_ratio:.2f})", 
                                res['current_price'],
                                sentiment,
                                is_high_vol
                            )

            if errors:
                with st.expander("Errors / Skipped Tickers"):
                    for err in errors:
                        st.write(f"- {err}")

if __name__ == "__main__":
    main()
