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
# NSE HELPERS (LOT SIZES)
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
def fetch_and_analyze_ticker_hybrid_v22(ticker, strategy_type, region="USA", source="Yahoo", z_api=None, z_token=None, pct_1=0.0, pct_2=5.0, pct_3=0.0, expiry_idx=0):
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
                # Priority: Zerodha API for India
                if region == "India" and adapter and hasattr(adapter, 'get_margin_for_basket'):
                    margin = adapter.get_margin_for_basket(legs, lot_size)
                
                # Fallback: USA or India w/o API
                if margin == 0:
                    if net_cost > 0: margin = net_cost * lot_size # Debit Spread (Capital = Cost)
                    else: margin = abs(buy_strike - sell_strike) * lot_size # Credit Spread (Margin = Width)
                
                # Brokerage
                brokerage = (20 * len(legs)) if region == "India" else (0.65 * len(legs))
                
                # Net Max Profit (Realizable)
                # If Debit: (Max Gain * Lot) - (Net Cost * Lot) - Brokerage? No, Net Cost already paid.
                # Max Gain (Absolute) was width. 
                # Profit = (Width - Cost) * Lot - Brokerage
                
                # Standardize Profit Calc
                total_gain_val = 0.0
                if strategy_type in ["Bear Call Spread", "Bull Put Spread"]:
                    # Credit Spreads: Profit is Net Credit (which is stored in max_gain) * Lot
                    total_gain_val = (max_gain * lot_size) - brokerage
                elif strategy_type == "Leveraged Bull Call Spread":
                    # Logic: Upside is capped by spread, minus cost (or plus credit)
                    total_gain_val = (max_gain * lot_size) - brokerage
                else:
                    # Debit Spreads: (Width - Cost) * Lot
                    total_gain_val = ((max_gain - net_cost) * lot_size) - brokerage
                
                rom = (total_gain_val / margin * 100) if margin > 0 else 0
                roc = (max_gain / net_cost * 100) if net_cost > 0 and strategy_type not in ["Bear Call Spread", "Bull Put Spread"] else 0
                if net_cost <= 0: roc = None # N/A for credit

                res = {
                    "Expiration": str(date_str), "CMP": current_price, 
                    "Buy Strike": buy_strike, "Buy Premium": buy_prem, 
                    "Sell Strike": sell_strike, "Sell Premium": sell_prem,
                    "Margin Required": margin, "Lot Size": lot_size,
                    "Net Cost": net_cost, "Net Max Profit": total_gain_val,
                    "Breakeven": breakeven, "Cost / CMP %": (net_cost/current_price)*100 if current_price>0 else 0,
                    "Return on Margin %": rom, "Return on Cost %": roc,
                    "Est Brokerage": brokerage
                }
                rows.append(res)
        
        df = pd.DataFrame(rows)
        # Final formatting cleanup to ensure numeric types
        for c in df.columns:
            if c != "Expiration": df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0.0)
            
        # Enforce Column Order
        order = ["Expiration", "CMP", "Buy Strike", "Buy Premium", "Sell Strike", "Sell Premium", 
                 "Margin Required", "Lot Size", "Net Cost", "Net Max Profit", "Breakeven", 
                 "Cost / CMP %", "Return on Margin %", "Return on Cost %", "Est Brokerage"]
        
        final_df = df[order] if not df.empty else df
        return {}, final_df, None

    except Exception as e: return None, None, str(e)

# ==========================================
# MAIN INTERFACE
# ==========================================

def main():
    st.title("🛡️ Options Strategy Master")
    if st.sidebar.button("🔄 Clear Cache & Restart"):
        st.cache_data.clear()
        st.rerun()

    region = st.sidebar.selectbox("Market Region", ["USA", "India"])
    region_key = "USA" if "USA" in region else "India"
    source = st.sidebar.radio("Data Source", ["Yahoo", "Zerodha (API)"]) if region_key == "India" else "Yahoo"
    z_api, z_token = load_zerodha_tokens() if source == "Zerodha (API)" else (None, None)

    strategy = st.radio("Strategy", ["Bull Call Spread", "Bear Call Spread", "Bull Put Spread", "Bear Put Spread", "Leveraged Bull Call Spread"], horizontal=True)
    
    c1, c2, c3 = st.columns(3)
    pct1 = c1.number_input("Strike 1 (% from Spot)", value=0.0)
    pct2 = c2.number_input("Strike 2 (% from Spot)", value=5.0)
    pct3 = c3.number_input("Sell Put % (Leveraged Only)", value=-5.0)
    
    expiry_idx = 0
    if region_key == "India":
        expiry_idx = ["Current", "Next", "Far"].index(st.selectbox("Expiry", ["Current", "Next", "Far"]))

    if st.button("Analyze"):
        ticker_input = st.text_input("Tickers", "RELIANCE")
        tickers = [t.strip().upper() for t in ticker_input.split(',')]
        
        all_dfs = []
        for t in tickers:
            _, df, err = fetch_and_analyze_ticker_hybrid_v22(t, strategy, region_key, source, z_api, z_token, pct1, pct2, pct3, expiry_idx)
            if err: st.error(f"{t}: {err}")
            elif not df.empty:
                df.insert(0, "Stock", t)
                all_dfs.append(df)
        
        if all_dfs:
            full_df = pd.concat(all_dfs)
            
            # Custom Formatting
            fmt = {
                "CMP": "${:,.2f}", "Buy Strike": "{:,.1f}", "Buy Premium": "${:,.2f}",
                "Sell Strike": "{:,.1f}", "Sell Premium": "${:,.2f}",
                "Margin Required": "${:,.0f}", "Net Cost": "${:,.2f}",
                "Net Max Profit": "${:,.0f}", "Breakeven": "${:,.2f}",
                "Cost / CMP %": "{:.2f}%", "Return on Margin %": "{:.1f}%",
                "Return on Cost %": "{:.1f}%", "Est Brokerage": "${:,.2f}"
            }
            
            st.dataframe(full_df.style.format(fmt, na_rep="N/A"), use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()
