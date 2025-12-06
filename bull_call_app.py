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
# PART 1: SIMPLE ANALYSIS LOGIC (From bull_call_app.py)
# Uses yfinance for Real Data
# ==========================================

def get_monthly_expirations(ticker_obj, limit=3):
    """Filters the list of expiration dates to find the next 'limit' distinct months."""
    expirations = ticker_obj.options
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

def get_price(option_row, price_type='ask'):
    price = option_row.get(price_type, 0)
    if price == 0: return option_row.get('lastPrice', 0)
    return price

def get_option_chain_with_retry(stock, date, retries=3):
    for i in range(retries):
        try:
            return stock.option_chain(date)
        except Exception as e:
            if i == retries - 1: raise e
            time.sleep((2 ** (i + 1)) + random.uniform(0.5, 1.5))
    return None

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
# PART 2: CONSTRAINT ALGO LOGIC (Iron Condor Algo)
# Uses Mock Data / Logic Classes
# ==========================================

@dataclass
class IronCondorConfig:
    min_daily_volume: int
    min_open_interest: int
    min_iv_rank: float
    avoid_earnings: bool
    target_dte_min: int
    target_dte_max: int
    target_delta: float
    delta_tolerance: float
    min_credit_to_width_ratio: float
    min_pop: float
    min_ror: float

@dataclass
class BullCallSpreadConfig:
    min_daily_volume: int
    min_open_interest: int
    max_iv_rank: float
    target_dte_min: int
    target_dte_max: int
    long_call_delta: float
    short_call_delta: float
    delta_tolerance: float
    max_debit_to_width_ratio: float
    min_profit_to_risk_ratio: float

@dataclass
class LongStraddleConfig:
    min_daily_volume: int
    min_open_interest: int
    max_iv_rank: float 
    target_dte_min: int
    target_dte_max: int
    delta_tolerance: float 
    max_cost_pct: float    

@dataclass
class OptionContract:
    symbol: str; strike: float; option_type: str; expiry_days: int
    bid: float; ask: float; delta: float; open_interest: int
    @property
    def mid_price(self): return round((self.bid + self.ask) / 2, 2)

@dataclass
class StockData:
    symbol: str; price: float; avg_daily_volume: int; iv_rank: float; days_to_earnings: int

class StrategyBase:
    def __init__(self): self.logs = []
    def log(self, msg): self.logs.append(msg)

class IronCondorStrategy(StrategyBase):
    def __init__(self, config: IronCondorConfig):
        super().__init__(); self.config = config
    def check_filters(self, stock: StockData) -> bool:
        if stock.avg_daily_volume < self.config.min_daily_volume: self.log(f"❌ FAIL: Low Volume"); return False
        if stock.iv_rank < self.config.min_iv_rank: self.log(f"❌ FAIL: Low IV Rank"); return False
        if self.config.avoid_earnings and stock.days_to_earnings < self.config.target_dte_max: self.log(f"❌ FAIL: Earnings Imminent"); return False
        self.log("✅ Phase 1: Universe Filters Passed"); return True
    def find_strikes(self, stock: StockData, chain: List[OptionContract]):
        valid = [o for o in chain if self.config.target_dte_min <= o.expiry_days <= self.config.target_dte_max]
        if not valid: self.log("❌ FAIL: No DTE match"); return None
        wing_width = 10.0 if stock.price > 200 else 5.0
        def get_leg(target_delta, type_):
            cands = [o for o in valid if o.option_type == type_ and o.open_interest >= self.config.min_open_interest]
            if not cands: return None
            best = min(cands, key=lambda x: abs(abs(x.delta) - abs(target_delta)))
            return best if abs(abs(best.delta) - abs(target_delta)) <= self.config.delta_tolerance else None
        
        sp = get_leg(-self.config.target_delta, 'put')
        sc = get_leg(self.config.target_delta, 'call')
        if not sp or not sc: self.log("❌ FAIL: No Short Legs"); return None
        
        lp = next((o for o in valid if o.option_type == 'put' and o.strike == sp.strike - wing_width), None)
        lc = next((o for o in valid if o.option_type == 'call' and o.strike == sc.strike + wing_width), None)
        if not lp or not lc: self.log("❌ FAIL: No Wings"); return None
        
        self.log(f"✅ Phase 2: Geometry Found (Width ${wing_width})"); 
        return {"short_put": sp, "long_put": lp, "short_call": sc, "long_call": lc, "width": wing_width}
    
    def validate_trade(self, legs):
        sp, lp, sc, lc, width = legs['short_put'], legs['long_put'], legs['short_call'], legs['long_call'], legs['width']
        credit = round((sp.mid_price + sc.mid_price) - (lp.mid_price + lc.mid_price), 2)
        max_risk = width - credit
        pop = 1.0 - (abs(sc.delta) + abs(sp.delta))
        
        if (credit / width) < self.config.min_credit_to_width_ratio: self.log(f"⛔ ABORT: Low Credit"); return None
        if pop < self.config.min_pop: self.log(f"⛔ ABORT: Low POP"); return None
        if (credit / max_risk) < self.config.min_ror: self.log(f"⛔ ABORT: Low ROR"); return None
        return {"type": "Iron Condor", "strikes": f"P:{lp.strike}/{sp.strike} | C:{sc.strike}/{lc.strike}", "credit": credit, "max_risk": round(max_risk, 2), "pop": pop}

class BullCallSpreadStrategy(StrategyBase):
    def __init__(self, config: BullCallSpreadConfig):
        super().__init__(); self.config = config
    def check_filters(self, stock: StockData) -> bool:
        if stock.iv_rank > self.config.max_iv_rank: self.log(f"❌ FAIL: High IV"); return False
        self.log("✅ Phase 1: Filters Passed"); return True
    def find_strikes(self, stock: StockData, chain: List[OptionContract]):
        valid = [o for o in chain if self.config.target_dte_min <= o.expiry_days <= self.config.target_dte_max]
        def get_leg(delta_target):
            cands = [o for o in valid if o.option_type == 'call' and o.open_interest >= self.config.min_open_interest]
            if not cands: return None
            best = min(cands, key=lambda x: abs(x.delta - delta_target))
            return best if abs(best.delta - delta_target) <= self.config.delta_tolerance else None
        lc = get_leg(self.config.long_call_delta)
        sc = get_leg(self.config.short_call_delta)
        if not lc or not sc or lc.strike >= sc.strike: self.log("❌ FAIL: Strike Issues"); return None
        self.log("✅ Phase 2: Geometry Found"); return {"long_call": lc, "short_call": sc}
    def validate_trade(self, legs):
        lc, sc = legs['long_call'], legs['short_call']
        width = sc.strike - lc.strike
        debit = round(lc.mid_price - sc.mid_price, 2)
        if (debit/width) > self.config.max_debit_to_width_ratio: self.log("⛔ ABORT: Expensive"); return None
        return {"type": "Bull Call Spread", "strikes": f"Buy {lc.strike} / Sell {sc.strike}", "debit": debit, "max_profit": width - debit}

class LongStraddleStrategy(StrategyBase):
    def __init__(self, config: LongStraddleConfig):
        super().__init__(); self.config = config
    def check_filters(self, stock: StockData) -> bool:
        if stock.iv_rank > self.config.max_iv_rank: self.log(f"❌ FAIL: High IV"); return False
        self.log("✅ Phase 1: Filters Passed"); return True
    def find_strikes(self, stock: StockData, chain: List[OptionContract]):
        valid = [o for o in chain if self.config.target_dte_min <= o.expiry_days <= self.config.target_dte_max]
        if not valid: return None
        closest = min(valid, key=lambda x: abs(x.strike - stock.price)).strike
        c = next((o for o in valid if o.strike == closest and o.option_type == 'call'), None)
        p = next((o for o in valid if o.strike == closest and o.option_type == 'put'), None)
        if not c or not p: self.log("❌ FAIL: Missing legs"); return None
        self.log(f"✅ Phase 2: ATM Strike {closest}"); return {"long_call": c, "long_put": p, "stock_price": stock.price}
    def validate_trade(self, legs):
        debit = round(legs['long_call'].mid_price + legs['long_put'].mid_price, 2)
        if (debit / legs['stock_price']) > self.config.max_cost_pct: self.log("⛔ ABORT: Too Expensive"); return None
        return {"type": "Long Straddle", "strikes": f"Straddle {legs['long_call'].strike}", "debit": debit, "breakevens": f"±{debit}"}

def generate_mock_chain(ticker, spot_price, expiry_days, is_high_iv=False):
    chain = []
    vol_factor = 0.05 if is_high_iv else 0.02
    strikes = [spot_price + i*5 for i in range(-10, 11)] 
    for strike in strikes:
        moneyness = (spot_price - strike) / spot_price
        call_delta = max(0.01, min(0.99, 0.5 + (moneyness * 2)))
        put_delta = call_delta - 1.0
        dist = abs(spot_price - strike)
        base_premium = max(0.05, (spot_price * vol_factor) - (dist * 0.1))
        chain.append(OptionContract(ticker, strike, 'call', expiry_days, round(base_premium, 2), round(base_premium*1.1, 2), round(call_delta, 2), 5000))
        chain.append(OptionContract(ticker, strike, 'put', expiry_days, round(base_premium, 2), round(base_premium*1.1, 2), round(put_delta, 2), 5000))
    return chain

# ==========================================
# PART 3: MAIN APP INTERFACE
# ==========================================

def main():
    st.title("🛡️ Options Strategy Master")
    
    # --- MASTER SWITCH ---
    mode = st.sidebar.radio(
        "Select Analysis Mode:", 
        ["Simple Analysis (Real Data)", "Constraint-Based Algo (Simulation)"],
        index=0
    )
    st.sidebar.markdown("---")

    # ==========================================
    # MODE A: SIMPLE ANALYSIS (yfinance)
    # ==========================================
    if mode == "Simple Analysis (Real Data)":
        st.subheader("📈 Multi-Stock Real-Time Analysis")
        st.caption("Fetches live option chains from Yahoo Finance. No filters, just math.")
        
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
                    # Reorder cols
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
    # MODE B: CONSTRAINT ALGO (Mock Data)
    # ==========================================
    else:
        st.subheader("🤖 Algorithmic Constraint Analysis")
        st.caption("Uses strict logical gates (IV Rank, Delta, POP, ROR) to Approve/Reject trades.")
        
        # Sidebar Configs
        algo_strat = st.sidebar.selectbox("Algo Strategy", ["Iron Condor", "Bull Call Spread", "Long Straddle"])
        
        config = None
        if algo_strat == "Iron Condor":
            st.sidebar.subheader("Iron Condor Settings")
            config = IronCondorConfig(
                min_daily_volume=1000000, min_open_interest=500, 
                min_iv_rank=st.sidebar.number_input("Min IV Rank", 50, 100, 50),
                avoid_earnings=True, target_dte_min=30, target_dte_max=45,
                target_delta=st.sidebar.slider("Target Delta", 0.1, 0.3, 0.16, 0.01),
                delta_tolerance=0.04, min_credit_to_width_ratio=0.33,
                min_pop=st.sidebar.slider("Min POP", 0.5, 0.9, 0.60), min_ror=0.15
            )
        elif algo_strat == "Bull Call Spread":
            st.sidebar.subheader("Bull Call Settings")
            config = BullCallSpreadConfig(
                min_daily_volume=1000000, min_open_interest=500, 
                max_iv_rank=st.sidebar.number_input("Max IV Rank", 0, 100, 50),
                target_dte_min=45, target_dte_max=90, long_call_delta=0.55, short_call_delta=0.30,
                delta_tolerance=0.05, max_debit_to_width_ratio=0.50, min_profit_to_risk_ratio=1.0
            )
        else:
            st.sidebar.subheader("Straddle Settings")
            config = LongStraddleConfig(
                min_daily_volume=1000000, min_open_interest=500, 
                max_iv_rank=st.sidebar.number_input("Max IV Rank", 0, 100, 40),
                target_dte_min=14, target_dte_max=60, delta_tolerance=0.10, max_cost_pct=0.10
            )

        # Inputs
        c1, c2, c3, c4 = st.columns(4)
        ticker = c1.text_input("Ticker", "TSLA").upper()
        price = c2.number_input("Price", value=250.0)
        iv_rank = c3.number_input("IV Rank", value=65.0)
        earnings = c4.number_input("Days to Earnings", value=60)

        if st.button("Run Algo Simulation"):
            # Mock Data Generation
            stock_data = StockData(ticker, price, 50000000, iv_rank, earnings)
            expiry = 35 if algo_strat == "Iron Condor" else 60
            chain = generate_mock_chain(ticker, price, expiry, iv_rank > 50)
            
            # Init & Run
            if algo_strat == "Iron Condor": engine = IronCondorStrategy(config)
            elif algo_strat == "Bull Call Spread": engine = BullCallSpreadStrategy(config)
            else: engine = LongStraddleStrategy(config)

            c_log, c_res = st.columns([1,1])
            with c_log:
                st.write("**Logic Check:**")
                passed = engine.check_filters(stock_data)
                legs = engine.find_strikes(stock_data, chain) if passed else None
                res = engine.validate_trade(legs) if legs else None
                for l in engine.logs:
                    if "FAIL" in l or "ABORT" in l: st.error(l)
                    else: st.success(l)
            
            with c_res:
                st.write("**Outcome:**")
                if res:
                    st.balloons()
                    st.success(f"APPROVED: {res['type']}")
                    st.code(res['strikes'])
                    st.json(res)
                elif not passed: st.warning("Failed Universe Filters")
                elif not legs: st.warning("Failed Geometry/Strikes")
                else: st.warning("Failed Safety Constraints")

if __name__ == "__main__":
    main()
