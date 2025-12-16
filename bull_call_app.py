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

def get_expirations_within_days(ticker_obj, days_limit=30):
    """Returns all expiration dates within the next X days."""
    try:
        expirations = ticker_obj.options
    except:
        return []
        
    if not expirations:
        return []
        
    valid_dates = []
    today = datetime.date.today()
    limit_date = today + datetime.timedelta(days=days_limit)
    
    for date_str in expirations:
        try:
            exp_date = datetime.datetime.strptime(date_str, '%Y-%m-%d').date()
            if today <= exp_date <= limit_date:
                valid_dates.append(date_str)
        except:
            continue
            
    return valid_dates

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

        # Use updated helper to get next few months roughly
        target_dates = get_expirations_within_days(stock, days_limit=90)[:3] 
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
    # Expand sim range to cover wings
    strikes = [l['row']['strike'] for l in legs]
    sim_prices = sorted(strikes + [current_price])
    
    # Dynamic range based on strikes
    min_s, max_s = min(strikes), max(strikes)
    range_width = max_s - min_s
    min_p = min_s - (range_width * 0.5)
    max_p = max_s + (range_width * 0.5)
    
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
def analyze_custom_strategy(ticker, sentiment, slab1_pct, slab2_pct, days_window, optimize=False):
    """
    Fetches expirations within 'days_window'.
    Builds strategy based on Slab 1 (Near) and Slab 2 (Far).
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
            
        # Get Expirations based on Window
        dates = get_expirations_within_days(stock, days_limit=days_window)
        if not dates: return None, f"No option chain found within {days_window} days."
        
        results_list = []
        errors = []

        # LOOP THROUGH EACH EXPIRATION
        for target_date in dates:
            try:
                chain = get_option_chain_with_retry(stock, target_date)
                calls = filter_tradeable_options(chain.calls).sort_values('strike').reset_index(drop=True)
                puts = filter_tradeable_options(chain.puts).sort_values('strike').reset_index(drop=True)
                
                if calls.empty or puts.empty: continue

                # --- TARGET CALCULATION BASED ON SLABS ---
                # Slab 1 = Near (e.g., 6%), Slab 2 = Far (e.g., 10%)
                s1 = slab1_pct / 100.0
                s2 = slab2_pct / 100.0
                
                targets = {
                    "pf": current_price * (1 - s2), # Put Far (Lower)
                    "pn": current_price * (1 - s1), # Put Near (Higher)
                    "cn": current_price * (1 + s1), # Call Near (Lower)
                    "cf": current_price * (1 + s2)  # Call Far (Higher)
                }
                
                # Helper to build legs
                def build_legs(pf_row, pn_row, cn_row, cf_row):
                    l = []
                    # Standard Directional Bias Logic (Low Vol assumption)
                    if sentiment == "Bullish":
                        # Bull Put Spread (Credit) + Bull Call Spread (Debit)
                        # Buy Far Put, Sell Near Put, Buy Near Call, Sell Far Call
                        l = [
                            {"type": "Put", "action": "Buy", "row": pf_row, "desc": f"Put Long (-{slab2_pct}%)"},
                            {"type": "Put", "action": "Sell", "row": pn_row, "desc": f"Put Short (-{slab1_pct}%)"},
                            {"type": "Call", "action": "Buy", "row": cn_row, "desc": f"Call Long (+{slab1_pct}%)"},
                            {"type": "Call", "action": "Sell", "row": cf_row, "desc": f"Call Short (+{slab2_pct}%)"},
                        ]
                    else: # Bearish
                        # Bearish Logic: Inverted Bullish
                        l = [
                            {"type": "Put", "action": "Sell", "row": pf_row, "desc": f"Put Short (-{slab2_pct}%)"},
                            {"type": "Put", "action": "Buy", "row": pn_row, "desc": f"Put Long (-{slab1_pct}%)"},
                            {"type": "Call", "action": "Sell", "row": cn_row, "desc": f"Call Short (+{slab1_pct}%)"},
                            {"type": "Call", "action": "Buy", "row": cf_row, "desc": f"Call Long (+{slab2_pct}%)"},
                        ]
                    return l

                # Find Base Rows
                pf_base = find_closest_strike(puts, targets["pf"])
                pn_base = find_closest_strike(puts, targets["pn"])
                cn_base = find_closest_strike(calls, targets["cn"])
                cf_base = find_closest_strike(calls, targets["cf"])
                
                if any(x is None for x in [pf_base, pn_base, cn_base, cf_base]):
                    continue 

                base_legs = build_legs(pf_base, pn_base, cn_base, cf_base)
                base_metrics = calculate_strategy_metrics(base_legs, current_price)
                
                payload = {
                    "ticker": ticker,
                    "current_price": current_price,
                    "expiry": target_date,
                    "base": {
                        "metrics": base_metrics,
                        "legs": base_legs
                    },
                    "optimized": None 
                }

                # --- OPTIMIZATION LOOP ---
                if optimize:
                    def get_idx(df, strike): 
                        indices = df.index[df['strike'] == strike].tolist()
                        return indices[0] if indices else -1

                    pf_idx = get_idx(puts, pf_base['strike'])
                    pn_idx = get_idx(puts, pn_base['strike'])
                    cn_idx = get_idx(calls, cn_base['strike'])
                    cf_idx = get_idx(calls, cf_base['strike'])
                    
                    if not any(i == -1 for i in [pf_idx, pn_idx, cn_idx, cf_idx]):
                        best_ratio = -1.0
                        best_config = None
                        
                        # Scan range: +/- 1 strike around targets to keep it fast for many expiries
                        range_scan = range(-1, 2) 
                        
                        for i1 in range_scan: # Put Far
                            for i2 in range_scan: # Put Near
                                for i3 in range_scan: # Call Near
                                    for i4 in range_scan: # Call Far
                                        if not (0 <= pf_idx+i1 < len(puts)): continue
                                        if not (0 <= pn_idx+i2 < len(puts)): continue
                                        if not (0 <= cn_idx+i3 < len(calls)): continue
                                        if not (0 <= cf_idx+i4 < len(calls)): continue
                                        
                                        pf_cand = puts.iloc[pf_idx+i1]
                                        pn_cand = puts.iloc[pn_idx+i2]
                                        cn_cand = calls.iloc[cn_idx+i3]
                                        cf_cand = calls.iloc[cf_idx+i4]
                                        
                                        if pf_cand['strike'] >= pn_cand['strike']: continue
                                        if cn_cand['strike'] >= cf_cand['strike']: continue
                                        
                                        cand_legs = build_legs(pf_cand, pn_cand, cn_cand, cf_cand)
                                        cand_metrics = calculate_strategy_metrics(cand_legs, current_price)
                                        
                                        max_loss_abs = abs(cand_metrics['max_loss'])
                                        ratio = 0 if max_loss_abs < 0.01 else cand_metrics['max_upside'] / max_loss_abs
                                            
                                        if ratio > best_ratio:
                                            best_ratio = ratio
                                            best_config = {
                                                "metrics": cand_metrics,
                                                "legs": cand_legs,
                                                "ratio": ratio
                                            }
                        
                        payload["optimized"] = best_config
                
                results_list.append(payload)

            except Exception as e:
                errors.append(f"Date {target_date}: {str(e)}")
                continue

        if not results_list:
             return None, "Could not build strategies for selected time window."

        return results_list, None

    except Exception as e:
        return None, str(e)

# ==========================================
# PART 3: MAIN APP INTERFACE
# ==========================================

def display_strategy_details(data, label, current_price):
    """Generates the detailed expander content for a single strategy result."""
    
    st.markdown(f"**{label}**")
    m = data['metrics']
    
    col1, col2, col3, col4 = st.columns(4)
    net = m['net_premium']
    lbl = "Net Credit" if net > 0 else "Net Debit"
    col1.metric("Spot Price", f"${current_price:.2f}")
    col2.metric(lbl, f"${abs(net):.2f}")
    col3.metric("Max Profit", f"${m['max_upside']:.2f}")
    col4.metric("Max Loss", f"${abs(m['max_loss']):.2f}")
    
    ratio = m['max_upside'] / abs(m['max_loss']) if abs(m['max_loss']) > 0 else 0
    st.markdown(f"**Reward/Risk Ratio:** `{ratio:.2f}`")
    
    if ratio > 1.0:
        st.success(f"🌟 Sweet Spot: Max Profit > Max Loss!")
    
    legs_simple = []
    for l in data['legs']:
        legs_simple.append({
            "Action": l['action'], "Type": l['type'], 
            "Strike": l['row']['strike'], "Price": get_price(l['row'], 'ask' if l['action']=="Buy" else 'bid'),
            "Description": l['desc']
        })
    st.dataframe(pd.DataFrame(legs_simple).style.format({"Price": "${:.2f}", "Strike": "${:.2f}"}), use_container_width=True)


def main():
    st.title("🛡️ Options Strategy Master")
    
    mode = st.sidebar.radio(
        "Select Analysis Mode:", 
        ["Simple Analysis (Standard)", "Custom Strategy Generator (Slab Based)"],
        index=0
    )
    st.sidebar.markdown("---")

    # ==========================================
    # MODE A: SIMPLE ANALYSIS
    # ==========================================
    if mode == "Simple Analysis (Standard)":
        st.subheader("📈 Multi-Stock Real-Time Analysis")
        st.caption("Fetches live option chains from Yahoo Finance. Standard Spreads/Straddles.")
        
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
    # MODE B: CUSTOM SLAB STRATEGY
    # ==========================================
    else:
        st.subheader("🤖 Custom Slab-Based Strategy Generator")
        st.caption("Generates strategies for all expirations within the selected window based on your Strike Slabs.")
        
        # 1. Inputs
        c1, c2 = st.columns(2)
        default_tickers = "TSLA, AAPL, AMD"
        ticker_input = c1.text_input("Stock Tickers (comma-separated)", default_tickers).upper()
        sentiment = c1.selectbox("Your Sentiment", ["Bullish", "Bearish"])
        
        c3, c4 = st.columns(2)
        days_select = c3.selectbox("Expiration Window", ["Next 30 Days", "Next 60 Days", "Next 90 Days"])
        days_map = {"Next 30 Days": 30, "Next 60 Days": 60, "Next 90 Days": 90}
        days_window = days_map[days_select]
        
        c5, c6 = st.columns(2)
        slab1 = c5.number_input("Slab 1 (Near Strike %)", min_value=1.0, max_value=20.0, value=6.0, step=0.5)
        slab2 = c6.number_input("Slab 2 (Far Strike %)", min_value=2.0, max_value=30.0, value=10.0, step=0.5)
        
        if slab1 >= slab2:
            st.error("Error: Slab 1 (Near) must be smaller than Slab 2 (Far).")
            stop = True
        else:
            stop = False

        st.info(f"Generating strategies for **{sentiment}** outlook. Strikes: **±{slab1}%** and **±{slab2}%**.")

        if st.button("Generate Strategies") and not stop:
            if not ticker_input:
                st.error("Please enter at least one ticker.")
                return

            tickers = [t.strip().upper() for t in ticker_input.split(',') if t.strip()]
            all_results = {}
            all_summaries = []
            errors = []

            progress_bar = st.progress(0)
            
            with st.spinner(f"Scanning expirations for {len(tickers)} tickers..."):
                for i, ticker in enumerate(tickers):
                    results_list, error = analyze_custom_strategy(ticker, sentiment, slab1, slab2, days_window, optimize=True)
                    
                    if error:
                        errors.append(f"{ticker}: {error}")
                    else:
                        all_results[ticker] = results_list
                        
                        for res in results_list:
                            metrics = res['optimized']['metrics'] if res['optimized'] else res['base']['metrics']
                            ratio = res['optimized']['ratio'] if res['optimized'] else (res['base']['metrics']['max_upside'] / abs(res['base']['metrics']['max_loss']))
                            
                            summary_data = {
                                "Stock": ticker,
                                "Expiry": res['expiry'],
                                "Spot": f"${res['current_price']:.2f}",
                                "Net Premium": f"${metrics['net_premium']:.2f}",
                                "Max Profit": f"${metrics['max_upside']:.2f}",
                                "Reward/Risk": f"{ratio:.2f}"
                            }
                            all_summaries.append(summary_data)
                    
                    progress_bar.progress((i + 1) / len(tickers))

            # --- OUTPUT ---
            st.divider()
            
            if all_summaries:
                st.header("1. Strategy Summary (Optimized)")
                summary_df = pd.DataFrame(all_summaries)
                st.dataframe(summary_df, hide_index=True, use_container_width=True)

                st.header("2. Detailed Trade Analysis")
                
                for ticker, results_list in all_results.items():
                    st.markdown(f"### {ticker}")
                    for res in results_list:
                        opt_data = res['optimized'] if res['optimized'] else res['base']
                        ratio = opt_data['metrics']['max_upside'] / abs(opt_data['metrics']['max_loss'])
                        
                        with st.expander(f"📅 {res['expiry']} | R/R {ratio:.2f}", expanded=False):
                            display_strategy_details(opt_data, "Recommended Strategy", res['current_price'])

            if errors:
                with st.expander("Errors / Skipped Tickers"):
                    for err in errors: st.write(f"- {err}")

if __name__ == "__main__":
    main()
