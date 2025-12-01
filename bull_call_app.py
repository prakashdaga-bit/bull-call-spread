import streamlit as st
import yfinance as yf
import pandas as pd
import datetime

# --- Configuration ---
st.set_page_config(page_title="Multi-Stock Bull Call Analyzer", page_icon="📈", layout="wide")

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
        # This filters out weekly options by picking the first available date of a new month
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
    chain = chain.copy()
    chain['abs_diff'] = (chain['strike'] - price_target).abs()
    # Sort by difference and take the top one
    return chain.sort_values('abs_diff').iloc[0]

def analyze_ticker(ticker_symbol):
    """
    Performs the bull call analysis for a single ticker.
    Returns:
        dict: Summary data (returns per month)
        DataFrame: Detailed analysis table
        str: Error message if any
    """
    try:
        stock = yf.Ticker(ticker_symbol)
        
        # Get Current Market Price (CMP)
        try:
            # fast_info is generally faster for live prices
            cmp = stock.fast_info['last_price']
        except:
            # Fallback to history if fast_info fails
            hist = stock.history(period='1d')
            if hist.empty:
                return None, None, f"No price data found for {ticker_symbol}."
            cmp = hist['Close'].iloc[-1]

        target_price = cmp * 1.05
        target_dates = get_monthly_expirations(stock, limit=3)
        
        if not target_dates:
            return None, None, f"No options data found for {ticker_symbol}."

        analysis_rows = []
        
        # UPDATED: Just use the plain ticker symbol. 
        # We will construct the HTML link manually in the display section.
        summary_returns = {"Stock": ticker_symbol}

        for i, date in enumerate(target_dates):
            try:
                # Get Option Chain for specific date
                opt_chain = stock.option_chain(date)
                calls = opt_chain.calls
                
                if calls.empty:
                    continue

                # --- SPREAD CONSTRUCTION ---
                
                # Long Leg (Buy) @ CMP
                long_leg = find_closest_strike(calls, cmp)
                
                # Short Leg (Sell) @ Target Price
                short_leg = find_closest_strike(calls, target_price)

                # Ensure strikes are different
                if long_leg['strike'] == short_leg['strike']:
                    # Force short leg to be the next strike up if they collided
                    idx = calls[calls['strike'] > long_leg['strike']].index
                    if not idx.empty:
                        short_leg = calls.loc[idx[0]]
                    else:
                        continue # Cannot build spread

                # --- CALCULATIONS ---
                
                # Prices
                buy_strike = long_leg['strike']
                sell_strike = short_leg['strike']
                
                # Cost to Buy Long (Ask Price)
                # Cost to Sell Short (Bid Price)
                long_ask = long_leg['ask']
                short_bid = short_leg['bid']
                
                # Validation: If liquidity is zero (no Ask or no Bid), skip
                if long_ask == 0 or short_bid == 0:
                    continue

                # Net Cost (Debit)
                net_cost = long_ask - short_bid
                
                # Max Gain: (Width of Spread) - Net Cost
                spread_width = sell_strike - buy_strike
                max_gain = spread_width - net_cost
                
                # Breakeven
                breakeven = buy_strike + net_cost
                
                # Return %
                ret_pct = 0
                if net_cost > 0:
                    ret_pct = (max_gain / net_cost) * 100

                # Data for Detailed Table
                analysis_rows.append({
                    "Expiration": date,
                    "Buy Strike": buy_strike,
                    "Sell Strike": sell_strike,
                    "Net Cost": net_cost,
                    "Max Gain": max_gain,
                    "Return %": ret_pct,
                    "Breakeven": breakeven
                })

                # Data for Summary Table
                # UPDATED: Use the specific date as key instead of "Month X"
                summary_returns[date] = f"{ret_pct:.1f}%"

            except Exception as e:
                # Skip individual dates if they fail
                continue

        if not analysis_rows:
            return None, None, "Could not construct valid spreads (liquidity/data issues)."
            
        return summary_returns, pd.DataFrame(analysis_rows), None

    except Exception as e:
        return None, None, str(e)

# --- Main App Interface ---
st.title("📈 Multi-Stock Bull Call Analyzer")
st.markdown("Analyze Bull Call Spreads (Target +5%) for multiple stocks simultaneously.")

# Input Section
default_tickers = "NKE, AAPL, AMD, TSLA"
ticker_input = st.text_input("Enter Stock Tickers (comma-separated):", value=default_tickers)

if st.button("Analyze All"):
    if not ticker_input:
        st.error("Please enter at least one ticker.")
    else:
        # Parse inputs
        tickers = [t.strip().upper() for t in ticker_input.split(',') if t.strip()]
        
        all_summaries = []
        all_details = {} # Map ticker -> DataFrame
        errors = []

        # Create a progress bar
        progress_bar = st.progress(0)
        
        with st.spinner("Fetching data and calculating strategies..."):
            for i, ticker in enumerate(tickers):
                # Analyze each ticker
                summary, df, error = analyze_ticker(ticker)
                
                if error:
                    errors.append(f"{ticker}: {error}")
                else:
                    all_summaries.append(summary)
                    all_details[ticker] = df
                
                # Update progress
                progress_bar.progress((i + 1) / len(tickers))

        # --- 1. Summary Table Output ---
        st.divider()
        st.header("1. Summary: Return Potential")
        
        if all_summaries:
            summary_df = pd.DataFrame(all_summaries)
            
            # Reorder columns to ensure Stock is first, then Dates chronologically
            # Filter out 'Stock' column, sort the date strings, then combine
            date_cols = sorted([c for c in summary_df.columns if c != "Stock"])
            cols = ["Stock"] + date_cols
            
            summary_df = summary_df[cols]
            
            # Convert Stock column to HTML links with target="_self"
            # This prevents opening a new tab
            summary_df['Stock'] = summary_df['Stock'].apply(
                lambda x: f'<a href="#{x}" target="_self" style="text-decoration: none; font-weight: bold;">{x}</a>'
            )
            
            # Convert DataFrame to HTML
            # escape=False is needed to render the <a> tags
            html_table = summary_df.to_html(escape=False, index=False)
            
            # Display HTML Table
            st.markdown(
                f"""
                <style>
                table {{
                    width: 100%;
                    border-collapse: collapse;
                }}
                th, td {{
                    text-align: left;
                    padding: 8px;
                    border-bottom: 1px solid #444;
                }}
                tr:hover {{background-color: rgba(255, 255, 255, 0.1);}}
                </style>
                {html_table}
                """, 
                unsafe_allow_html=True
            )
            st.caption("Click a stock ticker above to jump to its detailed analysis.")

        else:
            st.warning("No valid data found for any of the provided tickers.")

        # --- 2. Detailed Outputs ---
        st.divider()
        st.header("2. Detailed Strategy Breakdown")

        if all_details:
            for ticker, df in all_details.items():
                # Inject a hidden HTML anchor for the link to jump to
                # The ID must match the href in the link (e.g., href='#NKE' -> id='NKE')
                st.markdown(f"<div id='{ticker}' style='padding-top: 20px; margin-top: -20px;'></div>", unsafe_allow_html=True)
                
                with st.expander(f"{ticker} Analysis", expanded=True):
                    # Format and display with style
                    st.dataframe(
                        df.style.format({
                            "Net Cost": "${:.2f}",
                            "Max Gain": "${:.2f}",
                            "Breakeven": "${:.2f}",
                            "Return %": "{:.1f}%"
                        }),
                        use_container_width=True
                    )
        
        # Show errors if any occurred
        if errors:
            with st.expander("Errors / Skipped Tickers"):
                for err in errors:
                    st.write(f"- {err}")
