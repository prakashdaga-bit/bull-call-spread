import streamlit as st
import yfinance as yf
import pandas as pd
import datetime

# --- Configuration ---
st.set_page_config(page_title="Bull Call Analyzer", page_icon="📈")

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
        # Create a key (Year, Month)
        month_key = (date.year, date.month)
        
        # We only want one expiration per month to simulate "Monthly" selection
        if month_key not in seen_months:
            unique_months.append(date.strftime('%Y-%m-%d'))
            seen_months.add(month_key)
        
        if len(unique_months) >= limit:
            break
            
    return unique_months

def find_closest_strike(chain, price_target):
    """
    Finds the option row with the strike price closest to the price_target.
    """
    # Calculate absolute difference between all strikes and target
    chain = chain.copy()
    chain['abs_diff'] = (chain['strike'] - price_target).abs()
    
    # Sort by difference and take the top one
    closest_row = chain.sort_values('abs_diff').iloc[0]
    return closest_row

# --- Main App Interface ---
st.title("📈 Bull Call Spread Analyzer")
st.markdown("Generate equity research-style spread analysis for any US stock.")

# Input Section
ticker_symbol = st.text_input("Enter Stock Ticker (e.g., NKE, AAPL):", value="NKE").upper().strip()

if st.button("Analyze Spreads"):
    if not ticker_symbol:
        st.error("Please enter a valid ticker symbol.")
    else:
        with st.spinner(f"Fetching data for {ticker_symbol}..."):
            try:
                # 1. Get Data
                stock = yf.Ticker(ticker_symbol)
                
                # Get Current Market Price (CMP)
                try:
                    cmp = stock.fast_info['last_price']
                except:
                    # Fallback
                    hist = stock.history(period='1d')
                    if hist.empty:
                        raise ValueError("No price data found. Symbol might be delisted or incorrect.")
                    cmp = hist['Close'].iloc[-1]

                target_price = cmp * 1.05
                
                # Display Metrics
                col1, col2 = st.columns(2)
                col1.metric("Current Price", f"${cmp:.2f}")
                col2.metric("Target Price (+5%)", f"${target_price:.2f}")

                # 2. Get Expirations
                target_dates = get_monthly_expirations(stock, limit=3)
                
                if not target_dates:
                    st.warning("No options data found for this ticker.")
                else:
                    analysis_data = []

                    # 3. Iterate through expirations
                    for date in target_dates:
                        try:
                            # Get Option Chain
                            opt_chain = stock.option_chain(date)
                            calls = opt_chain.calls
                            
                            if calls.empty:
                                continue

                            # --- SPREAD CONSTRUCTION ---
                            long_leg = find_closest_strike(calls, cmp)
                            short_leg = find_closest_strike(calls, target_price)

                            # Handle overlapping strikes
                            if long_leg['strike'] == short_leg['strike']:
                                idx = calls[calls['strike'] > long_leg['strike']].index
                                if not idx.empty:
                                    short_leg = calls.loc[idx[0]]
                                else:
                                    continue 

                            # --- CALCULATIONS ---
                            buy_strike = long_leg['strike']
                            sell_strike = short_leg['strike']
                            
                            long_ask = long_leg['ask']
                            short_bid = short_leg['bid']
                            
                            # Liquidity check
                            if long_ask == 0 or short_bid == 0:
                                continue

                            net_cost = long_ask - short_bid
                            spread_width = sell_strike - buy_strike
                            max_gain = spread_width - net_cost
                            breakeven = buy_strike + net_cost
                            
                            if net_cost > 0:
                                ret_pct = (max_gain / net_cost) * 100
                            else:
                                ret_pct = 0

                            analysis_data.append({
                                "Expiration": date,
                                "Buy Strike": buy_strike,
                                "Sell Strike": sell_strike,
                                "Net Cost": net_cost,
                                "Max Gain": max_gain,
                                "Return %": ret_pct,
                                "Breakeven": breakeven
                            })

                        except Exception as e:
                            st.warning(f"Could not process date {date}")
                            continue

                    # 4. Output Table
                    if not analysis_data:
                        st.error("Could not construct valid spreads (possibly low liquidity or data gaps).")
                    else:
                        df = pd.DataFrame(analysis_data)
                        
                        # Format for display (Round floats)
                        st.subheader("Strategy Analysis")
                        st.dataframe(
                            df.style.format({
                                "Net Cost": "${:.2f}",
                                "Max Gain": "${:.2f}",
                                "Breakeven": "${:.2f}",
                                "Return %": "{:.1f}%"
                            }),
                            use_container_width=True
                        )
                        st.caption("Note: Net Cost calculated using (Long Ask - Short Bid). Execution prices may vary.")

            except Exception as e:
                st.error(f"An error occurred: {e}")
