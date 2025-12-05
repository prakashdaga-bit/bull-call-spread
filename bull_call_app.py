import streamlit as st
import yfinance as yf
import pandas as pd
import datetime
import time
import random

# --- Configuration ---
st.set_page_config(page_title="Multi-Stock Strategy Analyzer", page_icon="📈", layout="wide")

# --- Helper Functions ---
def get_monthly_expirations(ticker_obj, limit=3):
    """
    Filters the list of expiration dates to find the next 'limit' distinct months.
    """
    expirations = ticker_obj.options
    if not expirations:
        return []

    # Convert strings to datetime objects
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
    """
    Filters the option chain to keep only rows where Ask > 0 or LastPrice > 0.
    """
    if chain.empty:
        return chain
    
    cols = chain.columns
    has_ask = 'ask' in cols
    has_last = 'lastPrice' in cols
    
    if not has_ask and not has_last:
        return pd.DataFrame() 
        
    mask = pd.Series(False, index=chain.index)
    if has_ask:
        mask |= (chain['ask'] > 0)
    if has_last:
        mask |= (chain['lastPrice'] > 0)
        
    return chain[mask]

def find_closest_strike(chain, price_target):
    """
    Finds the option row with the strike price closest to the price_target.
    """
    if chain.empty:
        return None
        
    chain = chain.copy()
    chain['abs_diff'] = (chain['strike'] - price_target).abs()
    return chain.sort_values('abs_diff').iloc[0]

def get_price(option_row, price_type='ask'):
    """
    Robust price fetcher. Falls back to 'lastPrice' if ask/bid is 0.
    """
    price = option_row.get(price_type, 0)
    if price == 0:
        return option_row.get('lastPrice', 0)
    return price

def get_option_chain_with_retry(stock, date, retries=3):
    """
    Fetches option chain with exponential backoff to handle rate limits.
    """
    for i in range(retries):
        try:
            return stock.option_chain(date)
        except Exception as e:
            if i == retries - 1: # Last attempt failed
                raise e
            
            # Exponential backoff with jitter: 2s, 4s, 8s... + random
            sleep_time = (2 ** (i + 1)) + random.uniform(0.5, 1.5)
            time.sleep(sleep_time)
    return None

# --- Core Analysis Logic (Cached) ---
# We cache this function so switching strategies doesn't re-trigger API calls
# TTL (Time To Live) is set to 600 seconds (10 minutes)
@st.cache_data(ttl=600, show_spinner=False)
def fetch_and_analyze_ticker(ticker_symbol, strategy_type):
    """
    Performs option strategy analysis for a single ticker.
    """
    try:
        stock = yf.Ticker(ticker_symbol)
        
        # 1. Get Current Market Price (CMP)
        try:
            cmp = stock.fast_info['last_price']
            if cmp is None: raise ValueError("Fast info returned None")
        except:
            hist = stock.history(period='1d')
            if hist.empty:
                return None, None, f"No price data found for {ticker_symbol}."
            cmp = hist['Close'].iloc[-1]

        # 2. Get Expirations
        target_dates = get_monthly_expirations(stock, limit=3)
        if not target_dates:
            return None, None, f"No options data found for {ticker_symbol}."

        analysis_rows = []
        summary_returns = {"Stock": ticker_symbol}

        for i, date in enumerate(target_dates):
            # Anti-Rate Limit: Random sleep between 1.5 to 3 seconds
            # This randomness helps avoid detection patterns
            time.sleep(random.uniform(1.5, 3.0)) 
            
            try:
                # Use the robust fetcher with retries
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
                        if not higher_strikes.empty:
                            short_leg = higher_strikes.iloc[0]
                        else:
                            continue 

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
                    if net_cost > 0:
                        ret_pct = (max_gain / net_cost) * 100

                    analysis_rows.append({
                        "Expiration": date,
                        "Buy Strike": buy_strike,
                        "Sell Strike": sell_strike,
                        "Net Cost": net_cost,
                        "Max Gain": max_gain,
                        "Return %": ret_pct,
                        "Breakeven": breakeven
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
                        "Expiration": date,
                        "Strike": strike,
                        "Call Cost": call_ask,
                        "Put Cost": put_ask,
                        "Net Cost": net_cost,
                        "BE Low": breakeven_low,
                        "BE High": breakeven_high,
                        "Move Needed": move_pct
                    })
                    summary_returns[date] = f"±{move_pct:.1f}%"

            except Exception as e:
                continue

        if not analysis_rows:
            return None, None, "Could not construct valid spreads (liquidity/data issues)."
            
        return summary_returns, pd.DataFrame(analysis_rows), None

    except Exception as e:
        return None, None, str(e)

# --- Main App Interface ---
st.title("📈 Multi-Stock Strategy Analyzer")
st.markdown("Analyze options strategies for multiple stocks simultaneously.")

# 1. Strategy Selector
strategy = st.radio(
    "Select Strategy:",
    ("Bull Call Spread", "Long Straddle"),
    horizontal=True,
    help="Bull Call: Bullish Directional (Target +5%).\nLong Straddle: Volatility Play (Betting on big move either way)."
)

# 2. Ticker Input
default_tickers = "NKE, AAPL, AMD, TSLA"
ticker_input = st.text_input(
    "Enter Stock Tickers (comma-separated):", 
    value=default_tickers,
    help="US: AAPL, TSLA. India NSE: RELIANCE.NS, TCS.NS. India BSE: 500325.BO"
)

if st.button("Analyze All"):
    if not ticker_input:
        st.error("Please enter at least one ticker.")
    else:
        # Parse inputs
        tickers = [t.strip().upper() for t in ticker_input.split(',') if t.strip()]
        
        all_summaries = []
        all_details = {} 
        errors = []

        progress_bar = st.progress(0)
        
        with st.spinner(f"Fetching data (Requests spaced out to avoid Rate Limiting)..."):
            for i, ticker in enumerate(tickers):
                # Call cached function
                summary, df, error = fetch_and_analyze_ticker(ticker, strategy)
                
                if error:
                    errors.append(f"{ticker}: {error}")
                else:
                    all_summaries.append(summary)
                    all_details[ticker] = df
                
                progress_bar.progress((i + 1) / len(tickers))

        # --- 1. Summary Table Output ---
        st.divider()
        st.header("1. Summary Table")
        
        if strategy == "Bull Call Spread":
            st.info("Values represent **Return on Investment (ROI)** if stock hits target.")
        else:
            st.info("Values represent **% Move Required** to break even (Lower is better).")
        
        if all_summaries:
            summary_df = pd.DataFrame(all_summaries)
            
            # Sort columns
            date_cols = sorted([c for c in summary_df.columns if c != "Stock"])
            cols = ["Stock"] + date_cols
            summary_df = summary_df[cols]
            
            # HTML Link Logic
            summary_df['Stock'] = summary_df['Stock'].apply(
                lambda x: f'<a href="#{x}" target="_self" style="text-decoration: none; font-weight: bold;">{x}</a>'
            )
            
            html_table = summary_df.to_html(escape=False, index=False)
            
            st.markdown("""
                <style>
                table { width: 100%; border-collapse: collapse; }
                th, td { text-align: left; padding: 8px; border-bottom: 1px solid #444; }
                tr:hover {background-color: rgba(255, 255, 255, 0.1);}
                </style>
                """, unsafe_allow_html=True)

            st.markdown(html_table, unsafe_allow_html=True)
            st.caption("Click a stock ticker above to jump to its detailed analysis.")

        else:
            st.warning("No valid data found.")

        # --- 2. Detailed Outputs ---
        st.divider()
        st.header("2. Detailed Analysis")

        if all_details:
            for ticker, df in all_details.items():
                st.markdown(f"<div id='{ticker}' style='padding-top: 20px; margin-top: -20px;'></div>", unsafe_allow_html=True)
                
                with st.expander(f"{ticker} Analysis ({strategy})", expanded=True):
                    if strategy == "Bull Call Spread":
                        format_dict = {
                            "Net Cost": "${:.2f}",
                            "Max Gain": "${:.2f}",
                            "Breakeven": "${:.2f}",
                            "Return %": "{:.1f}%"
                        }
                    else: 
                        format_dict = {
                            "Call Cost": "${:.2f}",
                            "Put Cost": "${:.2f}",
                            "Net Cost": "${:.2f}",
                            "BE Low": "${:.2f}",
                            "BE High": "${:.2f}",
                            "Move Needed": "{:.1f}%"
                        }

                    st.dataframe(
                        df.style.format(format_dict),
                        use_container_width=True
                    )
        
        if errors:
            with st.expander("Errors / Skipped Tickers"):
                for err in errors:
                    st.write(f"- {err}")
