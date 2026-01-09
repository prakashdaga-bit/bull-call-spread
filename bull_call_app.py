import streamlit as st
import yfinance as yf
import pandas as pd
import datetime
import time
import random
import math
import requests
import os
from dataclasses import dataclass
from typing import List, Optional, Dict, Union
from io import StringIO

# ==========================================
# GLOBAL CONFIGURATION
# ==========================================
st.set_page_config(page_title="Options Strategy Master", page_icon="📈", layout="wide")

# ==========================================
# F&O DATA HELPER (ZERODHA SOURCE)
# ==========================================
@st.cache_data(ttl=86400) # Cache for 1 day
def get_fno_info_zerodha():
    try:
        url = "https://api.kite.trade/instruments"
        df = pd.read_csv(url)
        fno_df = df[(df['exchange'] == 'NFO') & (df['instrument_type'] == 'FUT')]
        unique_df = fno_df.drop_duplicates(subset=['name'])
        result = dict(zip(unique_df['name'], unique_df['lot_size']))
        return result
    except Exception:
        return {
            "RELIANCE": 250, "TCS": 175, "HDFCBANK": 550, "INFY": 400,
            "NIFTY": 75, "BANKNIFTY": 30
        }

# ==========================================
# TICKER PRESETS
# ==========================================
def get_ticker_presets(region="USA"):
    presets = {"Custom / Manual Input": ""}
    if region == "USA":
        presets.update({
            "Magnificent 7": "NVDA, MSFT, AAPL, GOOGL, AMZN, META, TSLA",
            "ARK Innovation (Top 25)": "TSLA, COIN, ROKU, PLTR, SQ, RBLX, CRSP, PATH, SHOP, U, DKNG, TDOC, HOOD, ZM, TWLO, NTLA, EXAS, BEAM, PACB, VCYT, DNA, RXRX, PD, ADPT, TXG",
            "NASDAQ 100 (Top 25)": "AAPL, MSFT, NVDA, AMZN, GOOGL, META, AVGO, TSLA, GOOG, COST, AMD, NFLX, PEP, ADBE, LIN, CSCO, TMUS, QCOM, INTC, AMGN, INTU, TXN, CMCSA, AMAT, HON"
        })
    else:
        fno_data = get_fno_info_zerodha()
        fo_list = ", ".join(sorted(fno_data.keys())) if fno_data else "RELIANCE, TCS, HDFCBANK, INFY"
        presets.update({
            "NIFTY 50 Top 10": "RELIANCE, TCS, HDFCBANK, ICICIBANK, INFY, ITC, SBIN, BHARTIARTL, HINDUNILVR, LTIM",
            "Indices": "NIFTY, BANKNIFTY, FINNIFTY",
            "Auto Sector": "TATAMOTORS, M&M, MARUTI, BAJAJ-AUTO, HEROMOTOCO, EICHERMOT",
            "All F&O Stocks": fo_list
        })
    return presets

# ==========================================
# HELPER: LOAD TOKENS FROM FILE
# ==========================================
def load_zerodha_tokens():
    if os.path.exists("zerodha_token.txt"):
        try:
            with open("zerodha_token.txt", "r") as f:
                content = f.read().strip()
                if "," in content:
                    parts = content.split(",", 1)
                    return parts[0].strip(), parts[1].strip()
        except: pass
    return None, None

def get_token_file_info():
    if os.path.exists("zerodha_token.txt"):
        timestamp = os.path.getmtime("zerodha_token.txt")
        return datetime.datetime.fromtimestamp(timestamp)
    return None

# ==========================================
# MARKET ADAPTERS
# ==========================================

class NSEMarketAdapter:
    BASE_URL = "https://www.nseindia.com"
    INDICES = ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"]
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Mozilla/5.0", "Accept-Language": "en-US,en;q=0.9"})
        try: self.session.get(self.BASE_URL, timeout=5)
        except: pass
    def get_spot_price(self, ticker):
        yf_ticker = f"{ticker}.NS" if ticker not in self.INDICES else {"NIFTY": "^NSEI", "BANKNIFTY": "^NSEBANK", "FINNIFTY": "NIFTY_FIN_SERVICE.NS"}.get(ticker, ticker)
        try:
            stock = yf.Ticker(yf_ticker)
            price = stock.fast_info['last_price'] or stock.history(period="1d")['Close'].iloc[-1]
            return float(price)
        except: return None
    def get_lot_size(self, ticker):
        return int(get_fno_info_zerodha().get(ticker, 1))
    def fetch_option_chain_raw(self, ticker):
        url = f"https://www.nseindia.com/api/option-chain-{'indices' if ticker in self.INDICES else 'equities'}?symbol={ticker}"
        try:
            response = self.session.get(url, timeout=10)
            if response.status_code == 200: return response.json()
        except: pass
        return None
    def get_expirations(self, ticker, days_limit=90):
        data = self.fetch_option_chain_raw(ticker)
        if not data or 'records' not in data: return [], None
        valid = [d for d in data['records']['expiryDates'] if datetime.datetime.strptime(d, "%d-%b-%Y").date() >= datetime.date.today()]
        return valid[:3], data
    def parse_chain(self, raw_data, expiry):
        if not raw_data: return pd.DataFrame(), pd.DataFrame()
        data = [item for item in raw_data['records']['data'] if item['expiryDate'] == expiry]
        c, p = [], []
        for i in data:
            if 'CE' in i: c.append({'strike': float(i['CE']['strikePrice']), 'lastPrice': float(i['CE'].get('lastPrice', 0)), 'bid': float(i['CE'].get('bidprice', 0)), 'ask': float(i['CE'].get('askPrice', 0))})
            if 'PE' in i: p.append({'strike': float(i['PE']['strikePrice']), 'lastPrice': float(i['PE'].get('lastPrice', 0)), 'bid': float(i['PE'].get('bidprice', 0)), 'ask': float(i['PE'].get('askPrice', 0))})
        return pd.DataFrame(c), pd.DataFrame(p)

class ZerodhaMarketAdapter:
    def __init__(self, api_key, access_token):
        self.api_key, self.access_token = api_key.strip(), access_token.strip()
        self.kite, self.instruments = None, None
    def connect(self):
        try:
            from kiteconnect import KiteConnect
            self.kite = KiteConnect(api_key=self.api_key)
            self.kite.set_access_token(self.access_token)
            return True
        except: return False
    @st.cache_data(ttl=3600)
    def get_instruments(_self): return pd.DataFrame(_self.kite.instruments("NFO"))
    def get_spot_price(self, ticker):
        yf_t = {"NIFTY": "^NSEI", "BANKNIFTY": "^NSEBANK", "FINNIFTY": "NIFTY_FIN_SERVICE.NS"}.get(ticker, f"{ticker}.NS")
        try: return float(yf.Ticker(yf_t).fast_info['last_price'])
        except: return None
    def get_chain_for_symbol(self, ticker, days_limit=90):
        if self.instruments is None: self.instruments = self.get_instruments()
        name = ticker
        subset = self.instruments[self.instruments['name'] == name].copy()
        if subset.empty: return [], {}
        subset['expiry'] = pd.to_datetime(subset['expiry']).dt.date
        dates = sorted([d for d in subset['expiry'].unique() if d >= datetime.date.today()])[:3]
        return dates, subset
    def get_lot_size(self, ticker):
        if self.instruments is None: self.instruments = self.get_instruments()
        sub = self.instruments[self.instruments['name'] == ticker]
        return int(sub.iloc[0]['lot_size']) if not sub.empty else 1
    def get_margin_for_basket(self, legs, lot_size=1):
        if not self.kite: return 0.0
        orders = []
        for leg in legs:
            token = leg['row'].get('instrument_token')
            if self.instruments is not None and token:
                row = self.instruments[self.instruments['instrument_token'] == token]
                if not row.empty:
                    orders.append({"exchange": "NFO", "tradingsymbol": row.iloc[0]['tradingsymbol'], "transaction_type": self.kite.TRANSACTION_TYPE_BUY if leg['action'] == "Buy" else self.kite.TRANSACTION_TYPE_SELL, "variety": "regular", "product": "NRML", "order_type": "MARKET", "quantity": lot_size})
        try:
            resp = self.kite.basket_order_margins(orders)
            return float(resp['initial'].get('total', 0.0))
        except: return 0.0
    def parse_chain(self, valid_instr, expiry):
        sub = valid_instr[valid_instr['expiry'] == expiry]
        if sub.empty: return pd.DataFrame(), pd.DataFrame()
        quotes = self.kite.quote(sub['instrument_token'].tolist())
        c, p = [], []
        for _, row in sub.iterrows():
            q = quotes.get(str(row['instrument_token']))
            if not q: continue
            d = {'strike': float(row['strike']), 'lastPrice': float(q['last_price']), 'bid': float(q['depth']['buy'][0]['price']), 'ask': float(q['depth']['sell'][0]['price']), 'instrument_token': row['instrument_token']}
            if row['instrument_type'] == 'CE': c.append(d)
            else: p.append(d)
        return pd.DataFrame(c), pd.DataFrame(p)

# ==========================================
# CORE LOGIC
# ==========================================

def get_price(row, p_type='mid'):
    b, a, l = float(row.get('bid', 0)), float(row.get('ask', 0)), float(row.get('lastPrice', 0))
    if p_type == 'mid': return (b + a) / 2 if (b > 0 and a > 0) else l
    return a if p_type == 'ask' and a > 0 else (b if p_type == 'bid' and b > 0 else l)

def filter_tradeable_options(chain):
    if chain.empty: return chain
    mask = pd.Series(False, index=chain.index)
    if 'ask' in chain.columns: mask |= (chain['ask'] > 0)
    if 'lastPrice' in chain.columns: mask |= (chain['lastPrice'] > 0)
    return chain[mask]

def find_closest_strike(chain, price_target):
    if chain.empty: return None
    chain = chain.copy()
    chain['abs_diff'] = (chain['strike'] - price_target).abs()
    return chain.sort_values('abs_diff').iloc[0]

def get_monthly_expirations(ticker_obj, limit=3):
    try:
        expirations = ticker_obj.options
        if not expirations: return []
        dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in expirations]
        unique_months, seen_months = [], set()
        for date in dates:
            month_key = (date.year, date.month)
            if month_key not in seen_months:
                unique_months.append(date.strftime('%Y-%m-%d'))
                seen_months.add(month_key)
            if len(unique_months) >= limit: break
        return unique_months
    except: return []

@st.cache_data(ttl=600)
def fetch_and_analyze_ticker_hybrid_v23(ticker, strategy_type, region="USA", source="Yahoo", z_api=None, z_token=None, pct_1=0.0, pct_2=5.0, pct_3=0.0, expiry_idx=0):
    adapter = None
    if region == "India":
        if source == "Zerodha (API)":
            adapter = ZerodhaMarketAdapter(z_api, z_token)
            if not adapter.connect(): return None, None, "Zerodha Connection Failed"
        else: adapter = NSEMarketAdapter()
    
    try:
        current_price = adapter.get_spot_price(ticker) if adapter else float(yf.Ticker(ticker).fast_info['last_price'])
        lot_size = adapter.get_lot_size(ticker) if adapter else 1
        if not current_price: return None, None, "Price Fetch Failed"

        if region == "India":
            valid_dates, raw_data = (adapter.get_chain_for_symbol(ticker) if source == "Zerodha (API)" else adapter.get_expirations(ticker))
            valid_dates = [valid_dates[expiry_idx]] if expiry_idx < len(valid_dates) else [valid_dates[-1]]
        else:
            stock = yf.Ticker(ticker)
            valid_dates = get_monthly_expirations(stock, limit=3)

        rows = []
        for date_obj in valid_dates:
            calls, puts = pd.DataFrame(), pd.DataFrame()
            if isinstance(date_obj, datetime.date): date_str = date_obj.strftime('%Y-%m-%d')
            else: date_str = date_obj

            if region == "India": calls, puts = adapter.parse_chain(raw_data, date_obj if source == "Zerodha (API)" else date_str)
            else:
                chain = stock.option_chain(date_str)
                calls, puts = chain.calls, chain.puts
            
            calls, puts = filter_tradeable_options(calls), filter_tradeable_options(puts)
            if calls.empty or puts.empty: continue
            
            p1, p2, p3 = current_price * (1+pct_1/100), current_price * (1+pct_2/100), current_price * (1+pct_3/100)
            legs, res = [], {}
            
            # --- Strategy Logic ---
            buy_strike, sell_strike = 0.0, 0.0
            buy_prem, sell_prem = 0.0, 0.0
            
            if strategy_type == "Bull Call Spread":
                l1, l2 = find_closest_strike(calls, p1), find_closest_strike(calls, p2)
                if l1 is None or l2 is None or l1['strike'] >= l2['strike']: continue
                legs = [{'row': l1, 'action': 'Buy', 'type': 'Call'}, {'row': l2, 'action': 'Sell', 'type': 'Call'}]
                buy_strike, sell_strike = l1['strike'], l2['strike']
                buy_prem, sell_prem = get_price(l1, 'ask'), get_price(l2, 'bid')
                net_cost = buy_prem - sell_prem
                max_gain = l2['strike'] - l1['strike'] # Absolute Gain (Width)
                breakeven = l1['strike'] + net_cost

            elif strategy_type == "Bear Call Spread":
                l1, l2 = find_closest_strike(calls, p1), find_closest_strike(calls, p2)
                if l1 is None or l2 is None or l1['strike'] >= l2['strike']: continue
                legs = [{'row': l1, 'action': 'Sell', 'type': 'Call'}, {'row': l2, 'action': 'Buy', 'type': 'Call'}]
                buy_strike, sell_strike = l2['strike'], l1['strike']
                buy_prem, sell_prem = get_price(l2, 'ask'), get_price(l1, 'bid')
                net_cost = buy_prem - sell_prem # Credit
                max_gain = abs(net_cost) # Credit received is max gain
                breakeven = l1['strike'] + max_gain

            elif strategy_type == "Bull Put Spread":
                l1, l2 = find_closest_strike(puts, p1), find_closest_strike(puts, p2)
                if l1 is None or l2 is None or l1['strike'] >= l2['strike']: continue
                legs = [{'row': l1, 'action': 'Buy', 'type': 'Put'}, {'row': l2, 'action': 'Sell', 'type': 'Put'}]
                buy_strike, sell_strike = l1['strike'], l2['strike']
                buy_prem, sell_prem = get_price(l1, 'ask'), get_price(l2, 'bid')
                net_cost = buy_prem - sell_prem # Credit
                max_gain = abs(net_cost)
                breakeven = l2['strike'] - max_gain

            elif strategy_type == "Bear Put Spread":
                l1, l2 = find_closest_strike(puts, p1), find_closest_strike(puts, p2)
                if l1 is None or l2 is None or l1['strike'] >= l2['strike']: continue
                legs = [{'row': l2, 'action': 'Buy', 'type': 'Put'}, {'row': l1, 'action': 'Sell', 'type': 'Put'}]
                buy_strike, sell_strike = l2['strike'], l1['strike']
                buy_prem, sell_prem = get_price(l2, 'ask'), get_price(l1, 'bid')
                net_cost = buy_prem - sell_prem
                max_gain = l2['strike'] - l1['strike']
                breakeven = l2['strike'] - net_cost

            elif strategy_type == "Leveraged Bull Call Spread":
                lc, sc, sp = find_closest_strike(calls, p1), find_closest_strike(calls, p2), find_closest_strike(puts, p3)
                if any(x is None for x in [lc, sc, sp]) or lc['strike'] >= sc['strike']: continue
                legs = [{'row': lc, 'action': 'Buy', 'type': 'Call'}, {'row': sc, 'action': 'Sell', 'type': 'Call'}, {'row': sp, 'action': 'Sell', 'type': 'Put'}]
                buy_strike, sell_strike = lc['strike'], sc['strike']
                buy_prem = get_price(lc, 'ask')
                sell_prem = get_price(sc, 'bid') + get_price(sp, 'bid') # Combined Sell Premium
                net_cost = buy_prem - sell_prem
                max_gain = (sc['strike'] - lc['strike']) - net_cost # Adjusted gain for leveraged
                breakeven = sp['strike'] + net_cost if net_cost > 0 else sp['strike'] - abs(net_cost)
            
            # --- MARGIN & METRICS ---
            if legs:
                margin = 0.0
                if region == "India" and adapter and hasattr(adapter, 'get_margin_for_basket'):
                    margin = adapter.get_margin_for_basket(legs, lot_size)
                
                # Fallback Margin
                if margin == 0:
                    if net_cost > 0: margin = net_cost * lot_size 
                    else: margin = abs(buy_strike - sell_strike) * lot_size 
                
                # Brokerage
                brokerage = (20 * len(legs)) if region == "India" else (0.65 * len(legs))
                
                # Net Max Profit
                total_gain_val = (max_gain * lot_size) - brokerage
                if strategy_type in ["Bear Call Spread", "Bull Put Spread"]:
                    total_gain_val = (max_gain * lot_size) - brokerage
                elif strategy_type == "Leveraged Bull Call Spread":
                     # Upside Capped at spread width, adjusted for premium/cost
                     # Max Gain calculated above is per share
                     total_gain_val = (max_gain * lot_size) - brokerage
                else: # Debit Spread
                    total_gain_val = ((max_gain - net_cost) * lot_size) - brokerage
                
                rom = (total_gain_val / margin * 100) if margin > 0 else 0
                roc = (total_gain_val / (net_cost * lot_size) * 100) if net_cost > 0 and strategy_type not in ["Bear Call Spread", "Bull Put Spread"] else 0
                if net_cost <= 0: roc = None

                res = {
                    "Expiration": str(date_str), "CMP": current_price, 
                    "Buy Strike": buy_strike, "Buy Premium": buy_prem, 
                    "Sell Strike": sell_strike, "Sell Premium": sell_prem,
                    "Margin Required": margin, "Lot Size": lot_size,
                    "Net Cost": net_cost, "Net Max Profit": total_gain_val,
                    "Breakeven": breakeven, "Cost/CMP %": (net_cost/current_price)*100 if current_price>0 else 0,
                    "Return on Margin %": rom, "Return on Cost %": roc,
                    "Est Brokerage": brokerage, "Max Gain": max_gain
                }
                rows.append(res)
        
        df = pd.DataFrame(rows)
        for c in df.columns:
            if c != "Expiration": df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0.0)
            
        # Reorder per user request
        order = ["Expiration", "CMP", "Buy Strike", "Buy Premium", "Sell Strike", "Sell Premium", 
                 "Margin Required", "Lot Size", "Net Cost", "Net Max Profit", "Breakeven", 
                 "Cost/CMP %", "Return on Margin %", "Return on Cost %", "Est Brokerage"]
        
        final_df = df[order] if not df.empty else df
        return {}, final_df, None

    except Exception as e: return None, None, str(e)

# ==========================================
# MAIN INTERFACE
# ==========================================

def display_strategy_details(data, label, current_price):
    st.markdown(f"**{label}**")
    m = data['metrics']
    c1, c2, c3, c4 = st.columns(4)
    net = m['net_premium']
    lbl = "Net Credit (Total)" if net > 0 else "Net Debit (Total)"
    total_prem = net * m['lot_size']
    c1.metric("Margin (1 Lot)", f"${m['capital_required']:,.0f}")
    c2.metric("Net Max Profit", f"${m['net_max_profit']:,.0f}")
    c3.metric(f"ROI %", f"{m['roi']:.1f}%")
    c4.metric(lbl, f"${abs(total_prem):,.0f}")
    
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
    if st.sidebar.button("🔄 Clear Cache & Restart"):
        st.cache_data.clear()
        st.rerun()

    region = st.sidebar.selectbox("Market Region", ["USA", "India"])
    region_key = "USA" if "USA" in region else "India"
    source = st.sidebar.radio("Data Source", ["NSE Website (Free/Flaky)", "Zerodha (API)"]) if region_key == "India" else "Yahoo"
    z_api, z_token = None, None
    if source == "Zerodha (API)":
        saved_api, saved_token = load_zerodha_tokens()
        st.sidebar.info("Requires Kite Connect subscription.")
        if saved_api and saved_api != st.session_state.get("z_api_input", ""):
             st.session_state["z_api_input"] = saved_api
        if saved_token and saved_token != st.session_state.get("z_token_input", ""):
             st.session_state["z_token_input"] = saved_token
        z_api = st.sidebar.text_input("API Key", key="z_api_input", type="password")
        z_token = st.sidebar.text_input("Access Token", key="z_token_input", type="password")
        
        file_dt = get_token_file_info()
        if file_dt:
             hours_old = (datetime.datetime.now() - file_dt).total_seconds() / 3600
             if hours_old > 12: st.sidebar.warning(f"⚠️ Token file is {hours_old:.1f} hours old.")
             else: st.sidebar.success(f"✅ Token updated: {file_dt.strftime('%H:%M')}")

    mode = st.sidebar.radio("Select Analysis Mode:", ["Simple Analysis (Standard)", "Custom Strategy Generator (Slab Based)"], index=0)
    st.sidebar.markdown("---")
    
    if "input_simple" not in st.session_state: st.session_state["input_simple"] = ""
    if "input_custom" not in st.session_state: st.session_state["input_custom"] = ""

    presets = get_ticker_presets(region_key)

    if mode == "Simple Analysis (Standard)":
        st.subheader(f"📈 {region_key} Market Real-Time Analysis")
        st.caption("Fetches live option chains. Standard Spreads/Straddles.")
        strategy = st.radio("Strategy Type:", ("Bull Call Spread", "Bear Call Spread", "Bull Put Spread", "Bear Put Spread", "Long Straddle", "Leveraged Bull Call Spread"), horizontal=True)
        
        pct_1, pct_2, pct_3 = 0.0, 5.0, -5.0
        expiry_idx = 0
        if region_key == "India":
            c_exp, _ = st.columns([1,3])
            exp_opts = ["Current Month", "Next Month", "Far Month"]
            exp_sel = c_exp.selectbox("Select Expiry (India Only)", exp_opts)
            try: expiry_idx = exp_opts.index(exp_sel)
            except: expiry_idx = 0
        
        if strategy != "Long Straddle":
            c1, c2, c3 = st.columns(3)
            pct_1 = c1.number_input("Strike 1 (% from Spot)", value=0.0)
            pct_2 = c2.number_input("Strike 2 (% from Spot)", value=5.0)
            if strategy == "Leveraged Bull Call Spread":
                pct_3 = c3.number_input("Sell Put % (Leveraged Only)", value=-5.0)
        
        def on_preset_simple_change():
            sel = st.session_state.preset_simple
            if sel != "Custom / Manual Input": st.session_state.input_simple = presets[sel]

        c1, c2 = st.columns([1, 2])
        c1.selectbox("Quick Load Preset", list(presets.keys()), key="preset_simple", on_change=on_preset_simple_change)
        ticker_input = c2.text_input("Enter Tickers (comma-separated):", key="input_simple")
        
        if st.button("Analyze Real-Time Data"):
            if not ticker_input: st.error("Please enter at least one ticker.")
            elif region_key == "India" and source == "Zerodha (API)" and (not z_api or not z_token):
                st.error("Please provide Zerodha API credentials in sidebar.")
            else:
                tickers = [t.strip().upper() for t in ticker_input.split(',') if t.strip()]
                all_dfs = []
                with st.spinner(f"Fetching data..."):
                    for t in tickers:
                        _, df, err = fetch_and_analyze_ticker_hybrid_v23(t, strategy, region_key, source, z_api, z_token, pct_1, pct_2, pct_3, expiry_idx)
                        if err: st.error(f"{t}: {err}")
                        elif not df.empty:
                            df.insert(0, "Stock", t)
                            all_dfs.append(df)
                
                if all_dfs:
                    full_df = pd.concat(all_dfs)
                    fmt = {
                        "CMP": "${:,.2f}", "Buy Strike": "{:,.1f}", "Buy Premium": "${:,.2f}",
                        "Sell Strike": "{:,.1f}", "Sell Premium": "${:,.2f}",
                        "Margin Required": "${:,.0f}", "Net Cost": "${:,.2f}",
                        "Net Max Profit": "${:,.0f}", "Breakeven": "${:,.2f}",
                        "Cost/CMP %": "{:.2f}%", "Return on Margin %": "{:.1f}%",
                        "Return on Cost %": "{:.1f}%", "Est Brokerage": "${:,.2f}"
                    }
                    st.dataframe(full_df.style.format(fmt, na_rep="N/A"), use_container_width=True, hide_index=True)

    else:
        st.subheader(f"🤖 {region_key} Slab-Based Strategy Generator")
        c1, c2 = st.columns(2)
        def on_preset_custom_change():
            sel = st.session_state.preset_custom
            if sel != "Custom / Manual Input": st.session_state.input_custom = presets[sel]

        c1.selectbox("Quick Load Preset", list(presets.keys()), key="preset_custom", on_change=on_preset_custom_change)
        ticker_input = c1.text_input("Stock Tickers (comma-separated)", key="input_custom").upper()
        view = c1.selectbox("Your View", ["Neutral", "Volatile"])
        c3, c4 = st.columns(2)
        days_select = c3.selectbox("Expiration Window", ["Next 30 Days", "Next 60 Days", "Next 90 Days"])
        days_map = {"Next 30 Days": 30, "Next 60 Days": 60, "Next 90 Days": 90}
        days_window = days_map[days_select]
        c5, c6 = st.columns(2)
        slab1 = c5.number_input("Slab 1 (Near Strike %)", min_value=1.0, max_value=20.0, value=6.0, step=0.5)
        slab2 = c6.number_input("Slab 2 (Far Strike %)", min_value=2.0, max_value=30.0, value=10.0, step=0.5)
        if slab1 >= slab2: st.error("Error: Slab 1 (Near) must be smaller than Slab 2 (Far)."); stop = True
        else: stop = False

        if st.button("Generate Strategies") and not stop:
            if not ticker_input: st.error("Please enter at least one ticker.")
            elif region_key == "India" and source == "Zerodha (API)" and (not z_api or not z_token):
                st.error("Please provide Zerodha API credentials in sidebar.")
            else:
                tickers = [t.strip().upper() for t in ticker_input.split(',') if t.strip()]
                with st.spinner(f"Scanning expirations..."):
                    for i, ticker in enumerate(tickers):
                        results_list, error = analyze_custom_strategy(ticker, view, slab1, slab2, days_window, region_key, source, z_api, z_token, optimize=True)
                        if error: st.error(f"{ticker}: {error}")
                        else:
                            for res in results_list:
                                opt_data = res['optimized'] if res['optimized'] else res['base']
                                ratio = opt_data['metrics']['max_upside'] / abs(opt_data['metrics']['max_loss'])
                                roi = opt_data['metrics']['roi']
                                with st.expander(f"{ticker} | 📅 {res['expiry']} | ROI {roi:.1f}% | R/R {ratio:.2f}", expanded=False):
                                    display_strategy_details(opt_data, "Recommended Strategy", res['current_price'])

if __name__ == "__main__":
    main()
