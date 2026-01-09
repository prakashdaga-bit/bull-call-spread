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
    """
    Fetches F&O stock list and Lot Sizes from Zerodha's public instrument dump.
    Returns: Dictionary {Symbol: LotSize}
    """
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

def get_ticker_presets(region="USA"):
    presets = {
        "Custom / Manual Input": ""
    }
    
    if region == "USA":
        presets.update({
            "Magnificent 7": "NVDA, MSFT, AAPL, GOOGL, AMZN, META, TSLA",
            "ARK Innovation (Top 25)": "TSLA, COIN, ROKU, PLTR, SQ, RBLX, CRSP, PATH, SHOP, U, DKNG, TDOC, HOOD, ZM, TWLO, NTLA, EXAS, BEAM, PACB, VCYT, DNA, RXRX, PD, ADPT, TXG",
            "NASDAQ 100 (Top 25)": "AAPL, MSFT, NVDA, AMZN, GOOGL, META, AVGO, TSLA, GOOG, COST, AMD, NFLX, PEP, ADBE, LIN, CSCO, TMUS, QCOM, INTC, AMGN, INTU, TXN, CMCSA, AMAT, HON"
        })
    else:
        # Fetch Zerodha list for keys
        fno_data = get_fno_info_zerodha()
        if fno_data:
            fo_list = ", ".join(sorted(fno_data.keys()))
        else:
            fo_list = "RELIANCE, TCS, HDFCBANK, ICICIBANK, INFY, ITC, SBIN, BHARTIARTL, HINDUNILVR, LTIM"
        
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
    """Reads api_key and access_token from local file if it exists."""
    if os.path.exists("zerodha_token.txt"):
        try:
            with open("zerodha_token.txt", "r") as f:
                content = f.read().strip()
                if "," in content:
                    parts = content.split(",", 1)
                    return parts[0].strip(), parts[1].strip()
        except:
            pass
    return None, None

def get_token_file_info():
    """Returns formatted string of token age."""
    if os.path.exists("zerodha_token.txt"):
        timestamp = os.path.getmtime("zerodha_token.txt")
        return datetime.datetime.fromtimestamp(timestamp)
    return None

# ==========================================
# MARKET ADAPTERS
# ==========================================

class NSEMarketAdapter:
    """
    Fetches Option Chain data directly from NSE India website.
    """
    BASE_URL = "https://www.nseindia.com"
    INDICES = ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"]
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br"
        })
        try:
            self.session.get(self.BASE_URL, timeout=5)
        except: pass

    def get_spot_price(self, ticker):
        yf_ticker = f"{ticker}.NS" if ticker not in self.INDICES else {"NIFTY": "^NSEI", "BANKNIFTY": "^NSEBANK", "FINNIFTY": "NIFTY_FIN_SERVICE.NS"}.get(ticker, ticker)
        try:
            stock = yf.Ticker(yf_ticker)
            price = stock.fast_info['last_price']
            if price is None:
                hist = stock.history(period="1d")
                if not hist.empty: price = hist['Close'].iloc[-1]
            return float(price) if price is not None else None
        except: return None

    def fetch_option_chain_raw(self, ticker):
        url = f"https://www.nseindia.com/api/option-chain-indices?symbol={ticker}" if ticker in self.INDICES else f"https://www.nseindia.com/api/option-chain-equities?symbol={ticker}"
        try:
            response = self.session.get(url, timeout=10)
            if response.status_code == 401:
                self.session.get(self.BASE_URL)
                response = self.session.get(url, timeout=10)
            if response.status_code == 200: return response.json()
        except: return None
        return None

    def get_expirations(self, ticker, days_limit=90):
        data = self.fetch_option_chain_raw(ticker)
        if not data or 'records' not in data: return [], None
        expiry_dates = data['records']['expiryDates']
        valid_dates = []
        today = datetime.date.today()
        # limit = today + datetime.timedelta(days=days_limit) # Original logic
        for d_str in expiry_dates:
            try:
                d = datetime.datetime.strptime(d_str, "%d-%b-%Y").date()
                if d >= today: # Get all future expirations, filter by index later
                     valid_dates.append(d_str)
            except: continue
        return valid_dates, data

    def parse_chain(self, raw_data, expiry_date_str):
        if not raw_data or 'records' not in raw_data: return pd.DataFrame(), pd.DataFrame()
        data = raw_data['records']['data']
        calls_list, puts_list = [], []
        for item in data:
            if item['expiryDate'] != expiry_date_str: continue
            if 'CE' in item:
                calls_list.append({
                    'strike': float(item['CE']['strikePrice']),
                    'lastPrice': float(item['CE'].get('lastPrice', 0)),
                    'bid': float(item['CE'].get('bidprice', 0)),
                    'ask': float(item['CE'].get('askPrice', 0)),
                    'openInterest': float(item['CE'].get('openInterest', 0))
                })
            if 'PE' in item:
                puts_list.append({
                    'strike': float(item['PE']['strikePrice']),
                    'lastPrice': float(item['PE'].get('lastPrice', 0)),
                    'bid': float(item['PE'].get('bidprice', 0)),
                    'ask': float(item['PE'].get('askPrice', 0)),
                    'openInterest': float(item['PE'].get('openInterest', 0))
                })
        return pd.DataFrame(calls_list), pd.DataFrame(puts_list)

class ZerodhaMarketAdapter:
    def __init__(self, api_key, access_token):
        self.api_key = api_key.strip()
        self.access_token = access_token.strip()
        self.kite = None
        self.instruments = None
        
    def connect(self):
        try:
            from kiteconnect import KiteConnect
            self.kite = KiteConnect(api_key=self.api_key)
            self.kite.set_access_token(self.access_token)
            return True
        except ImportError:
            st.error("Please install kiteconnect: `pip install kiteconnect`")
            return False
        except Exception as e:
            st.error(f"Zerodha Connection Error: {e}")
            return False

    @st.cache_data(ttl=3600)
    def get_instruments(_self):
        return pd.DataFrame(_self.kite.instruments("NFO"))

    def get_spot_price(self, ticker):
        yf_t = {"NIFTY": "^NSEI", "BANKNIFTY": "^NSEBANK", "FINNIFTY": "NIFTY_FIN_SERVICE.NS"}.get(ticker, f"{ticker}.NS")
        try: return float(yf.Ticker(yf_t).fast_info['last_price'])
        except: return None

    def get_chain_for_symbol(self, ticker, days_limit=90):
        if self.instruments is None: self.instruments = self.get_instruments()
        
        name = ticker
        if ticker == "NIFTY": name = "NIFTY"
        elif ticker == "BANKNIFTY": name = "BANKNIFTY"
        elif ticker == "FINNIFTY": name = "FINNIFTY"
        
        subset = self.instruments[self.instruments['name'] == name].copy()
        if subset.empty: return [], {}
        
        subset['expiry'] = pd.to_datetime(subset['expiry']).dt.date
        dates = sorted([d for d in subset['expiry'].unique() if d >= datetime.date.today()])[:3] # Default fetch
        return dates, subset

    def fetch_quotes(self, instrument_tokens):
        try:
            quotes = {}
            chunk_size = 500
            for i in range(0, len(instrument_tokens), chunk_size):
                batch = instrument_tokens[i:i + chunk_size]
                batch_quotes = self.kite.quote(batch)
                quotes.update(batch_quotes)
                time.sleep(0.1)
            return quotes
        except Exception as e:
            st.error(f"Quote Fetch Error: {e}")
            return {}

    def get_lot_size(self, ticker):
        if self.instruments is None: self.instruments = self.get_instruments()
        sub = self.instruments[self.instruments['name'] == ticker]
        if not sub.empty: return int(sub.iloc[0]['lot_size'])
        lots = get_fno_info_zerodha()
        return int(lots.get(ticker, 1))

    def get_margin_for_basket(self, legs, lot_size=1):
        if not self.kite: return 0.0
        orders = []
        for leg in legs:
            token = leg['row'].get('instrument_token')
            if self.instruments is not None and token:
                row = self.instruments[self.instruments['instrument_token'] == token]
                if not row.empty:
                    ts = row.iloc[0]['tradingsymbol']
                    txn_type = self.kite.TRANSACTION_TYPE_BUY if leg['action'] == "Buy" else self.kite.TRANSACTION_TYPE_SELL
                    orders.append({
                        "exchange": "NFO", "tradingsymbol": ts, "transaction_type": txn_type,
                        "variety": "regular", "product": "NRML", "order_type": "MARKET", "quantity": lot_size
                    })
        if not orders: return 0.0
        try:
            response = self.kite.basket_order_margins(orders)
            if response and 'initial' in response:
                return response['initial'].get('total', 0.0)
            return 0.0
        except: return 0.0

    def parse_chain(self, valid_instr, expiry):
        sub = valid_instr[valid_instr['expiry'] == expiry]
        if sub.empty: return pd.DataFrame(), pd.DataFrame()
        quotes = self.fetch_quotes(sub['instrument_token'].tolist())
        c, p = [], []
        for _, row in sub.iterrows():
            q = quotes.get(str(row['instrument_token']))
            if not q: continue
            d = {'strike': float(row['strike']), 'lastPrice': float(q['last_price']), 'bid': float(q['depth']['buy'][0]['price']), 'ask': float(q['depth']['sell'][0]['price']), 'instrument_token': row['instrument_token']}
            if row['instrument_type'] == 'CE': c.append(d)
            else: p.append(d)
        return pd.DataFrame(c), pd.DataFrame(p)

# ==========================================
# LOGIC
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
def fetch_and_analyze_v26(ticker, strategy, region, source, z_api, z_token, p1_pct, p2_pct, p3_pct, exp_idx):
    adapter = None
    if region == "India":
        if source == "Zerodha (API)":
            adapter = ZerodhaMarketAdapter(z_api, z_token)
            if not adapter.connect(): return None, None, "Zerodha Conn Failed"
        else: adapter = NSEMarketAdapter()
    
    try:
        # Data Fetch
        current_price = adapter.get_spot_price(ticker) if adapter else float(yf.Ticker(ticker).fast_info['last_price'])
        lot_size = adapter.get_lot_size(ticker) if adapter else 1
        
        valid_dates, raw_data = [], None
        if region == "India":
            valid_dates, raw_data = (adapter.get_chain_for_symbol(ticker) if source == "Zerodha (API)" else adapter.get_expirations(ticker))
            if valid_dates:
                 valid_dates = [valid_dates[exp_idx]] if exp_idx < len(valid_dates) else [valid_dates[-1]]
        else:
            stock = yf.Ticker(ticker)
            valid_dates = get_monthly_expirations(stock, limit=3)

        if not valid_dates: return None, None, "No Expiry Found"

        rows = []
        for date_obj in valid_dates:
            calls, puts = pd.DataFrame(), pd.DataFrame()
            d_str = date_obj.strftime('%Y-%m-%d') if isinstance(date_obj, datetime.date) else date_obj
            
            if region == "India":
                calls, puts = adapter.parse_chain(raw_data, date_obj if source == "Zerodha (API)" else d_str)
            else:
                try:
                    chain = stock.option_chain(d_str)
                    calls, puts = chain.calls, chain.puts
                except: continue
            
            # Filter and sort
            calls = filter_tradeable_options(calls)
            puts = filter_tradeable_options(puts)
            if calls.empty or puts.empty: continue

            # Strategy Logic
            pr1 = current_price * (1 + p1_pct/100)
            pr2 = current_price * (1 + p2_pct/100)
            
            legs_list = []

            if strategy == "Bull Call Spread":
                l1, l2 = find_closest_strike(calls, pr1), find_closest_strike(calls, pr2)
                if l1 is not None and l2 is not None and l1['strike'] != l2['strike']:
                    buy_leg = l1 if l1['strike'] < l2['strike'] else l2
                    sell_leg = l2 if l1['strike'] < l2['strike'] else l1
                    legs_list = [{'row': buy_leg, 'action': 'Buy', 'type': 'Call'}, {'row': sell_leg, 'action': 'Sell', 'type': 'Call'}]

            elif strategy == "Bear Put Spread":
                l1, l2 = find_closest_strike(puts, pr1), find_closest_strike(puts, pr2)
                if l1 is not None and l2 is not None and l1['strike'] != l2['strike']:
                    buy_leg = l1 if l1['strike'] > l2['strike'] else l2
                    sell_leg = l2 if l1['strike'] > l2['strike'] else l1
                    legs_list = [{'row': buy_leg, 'action': 'Buy', 'type': 'Put'}, {'row': sell_leg, 'action': 'Sell', 'type': 'Put'}]
            
            elif strategy == "Bear Call Spread":
                l1, l2 = find_closest_strike(calls, pr1), find_closest_strike(calls, pr2)
                if l1 is not None and l2 is not None and l1['strike'] != l2['strike']:
                    sell_leg = l1 if l1['strike'] < l2['strike'] else l2
                    buy_leg = l2 if l1['strike'] < l2['strike'] else l1
                    legs_list = [{'row': sell_leg, 'action': 'Sell', 'type': 'Call'}, {'row': buy_leg, 'action': 'Buy', 'type': 'Call'}]
            
            elif strategy == "Bull Put Spread":
                l1, l2 = find_closest_strike(puts, pr1), find_closest_strike(puts, pr2)
                if l1 is not None and l2 is not None and l1['strike'] != l2['strike']:
                    buy_leg = l1 if l1['strike'] < l2['strike'] else l2
                    sell_leg = l2 if l1['strike'] < l2['strike'] else l1
                    legs_list = [{'row': buy_leg, 'action': 'Buy', 'type': 'Put'}, {'row': sell_leg, 'action': 'Sell', 'type': 'Put'}]

            elif strategy == "Leveraged Bull Call Spread":
                l1, l2 = find_closest_strike(calls, pr1), find_closest_strike(calls, pr2)
                l3 = find_closest_strike(puts, current_price * (1 + p3_pct/100))
                if l1 is not None and l2 is not None and l3 is not None and l1['strike'] != l2['strike']:
                     buy_leg = l1 if l1['strike'] < l2['strike'] else l2
                     sell_leg = l2 if l1['strike'] < l2['strike'] else l1
                     legs_list = [{'row': buy_leg, 'action': 'Buy', 'type': 'Call'}, {'row': sell_leg, 'action': 'Sell', 'type': 'Call'}, {'row': l3, 'action': 'Sell', 'type': 'Put'}]

            if legs_list:
                # Premiums & Cost
                b_prem_sum = sum(get_price(l['row'], 'ask') for l in legs_list if l['action'] == 'Buy')
                s_prem_sum = sum(get_price(l['row'], 'bid') for l in legs_list if l['action'] == 'Sell')
                net_cost = b_prem_sum - s_prem_sum
                
                # Strikes for calculation
                b_strikes = [l['row']['strike'] for l in legs_list if l['action']=='Buy']
                s_strikes = [l['row']['strike'] for l in legs_list if l['action']=='Sell']
                
                # Margin
                margin = 0.0
                if adapter and hasattr(adapter, 'get_margin_for_basket'):
                    margin = adapter.get_margin_for_basket(legs_list, lot_size)
                
                if margin == 0:
                    if strategy in ["Bull Call Spread", "Bear Put Spread"]: margin = net_cost * lot_size if net_cost > 0 else 0
                    elif strategy in ["Bear Call Spread", "Bull Put Spread"]: margin = abs(b_strikes[0]-s_strikes[0]) * lot_size if (b_strikes and s_strikes) else 0
                    elif strategy == "Leveraged Bull Call Spread": margin = (legs_list[-1]['row']['strike'] * 0.15 * lot_size) # Put strike

                # Logic for Gain/Breakeven
                gain, be = 0.0, 0.0
                if strategy in ["Bull Call Spread", "Bear Put Spread"]:
                    width = abs(s_strikes[0] - b_strikes[0])
                    gain = width
                    be = b_strikes[0] + net_cost if strategy == "Bull Call Spread" else b_strikes[0] - net_cost
                elif strategy in ["Bear Call Spread", "Bull Put Spread"]:
                    gain = abs(net_cost)
                    be = s_strikes[0] + gain if strategy == "Bear Call Spread" else s_strikes[0] - gain
                elif strategy == "Leveraged Bull Call Spread":
                    width = abs(s_strikes[0] - b_strikes[0]) # Call Spread Width
                    gain = width - net_cost
                    # BE on downside = Put Strike + Cost/Credit
                    # legs_list: 0=BuyCall, 1=SellCall, 2=SellPut
                    put_st = legs_list[2]['row']['strike']
                    be = put_st + net_cost 
                
                # Brokerage & Net Profit
                brokerage = (20 * len(legs_list)) if region == "India" else (0.65 * len(legs_list))
                
                total_gain_lot = 0.0
                if strategy in ["Bear Call Spread", "Bull Put Spread"]:
                    total_gain_lot = (gain * lot_size) - brokerage
                elif strategy == "Leveraged Bull Call Spread":
                    total_gain_lot = (gain * lot_size) - brokerage
                else: # Debit
                    total_gain_lot = ((gain - net_cost) * lot_size) - brokerage

                rom = (total_gain_lot / margin * 100) if margin > 0 else 0
                roc = (total_gain_lot / (net_cost * lot_size) * 100) if net_cost > 0 and strategy not in ["Bear Call Spread", "Bull Put Spread"] else 0
                
                cost_cmp = 0.0
                if margin > 0: cost_cmp = (margin/lot_size)/current_price * 100
                
                # Columns mapping
                res = {
                    "Expiration": str(d_str), "CMP": current_price,
                    "Buy Strike": b_strikes[0] if b_strikes else 0, "Buy Premium": b_prem_sum,
                    "Sell Strike": s_strikes[0] if s_strikes else 0, "Sell Premium": s_prem_sum,
                    "Margin Required": margin, "Lot Size": lot_size,
                    "Net Cost": net_cost, "Net Max Profit": total_gain_lot,
                    "Breakeven": be, "Cost / CMP %": cost_cmp,
                    "Return on Margin %": rom, "Return on Cost %": roc if net_cost > 0 else None,
                    "Est Brokerage": brokerage, "Max Gain": gain
                }
                rows.append(res)

        df = pd.DataFrame(rows)
        # Enforce Order
        cols = ["Expiration", "CMP", "Buy Strike", "Buy Premium", "Sell Strike", "Sell Premium", 
                "Margin Required", "Lot Size", "Net Cost", "Net Max Profit", "Breakeven", 
                "Cost / CMP %", "Return on Margin %", "Return on Cost %", "Est Brokerage"]
        
        return {}, df[cols] if not df.empty else df, None
    except Exception as e: return None, None, str(e)

# ==========================================
# MAIN INTERFACE
# ==========================================

def main():
    st.title("🛡️ Options Strategy Master")
    if st.sidebar.button("🔄 Clear Cache & Restart"): st.cache_data.clear(); st.rerun()

    region = st.sidebar.selectbox("Market Region", ["USA", "India"])
    region_key = "USA" if "USA" in region else "India"
    
    source = "Yahoo"
    z_api, z_token = None, None
    saved_api, saved_token = load_zerodha_tokens()
    
    if region_key == "India":
        source = st.sidebar.radio("Data Source", ["NSE Website (Free/Flaky)", "Zerodha (API)"])
        if source == "Zerodha (API)":
            z_api = st.sidebar.text_input("API Key", value=saved_api or "", type="password")
            z_token = st.sidebar.text_input("Access Token", value=saved_token or "", type="password")
            ft = get_token_file_info()
            if ft: st.sidebar.success(f"Token: {ft.strftime('%H:%M')}")

    strategy = st.radio("Strategy", ["Bull Call Spread", "Bear Put Spread", "Bear Call Spread", "Bull Put Spread", "Leveraged Bull Call Spread"], horizontal=True)
    
    c1, c2, c3 = st.columns(3)
    p1 = c1.number_input("Strike 1 (% from Spot)", value=0.0)
    p2 = c2.number_input("Strike 2 (% from Spot)", value=5.0)
    p3 = 0.0
    if strategy == "Leveraged Bull Call Spread":
        p3 = c3.number_input("Sell Put %", value=-5.0)
    
    exp_idx = 0
    if region_key == "India":
        exp_idx = ["Current", "Next", "Far"].index(st.selectbox("Expiry", ["Current", "Next", "Far"]))

    if st.button("Analyze"):
        t_in = st.text_input("Tickers", "RELIANCE")
        tickers = [x.strip().upper() for x in t_in.split(',')]
        
        dfs = []
        with st.spinner("Analyzing..."):
            for t in tickers:
                _, df, err = fetch_and_analyze_v26(t, strategy, region_key, source, z_api, z_token, p1, p2, p3, exp_idx)
                if not df.empty:
                    df.insert(0, "Stock", t)
                    dfs.append(df)
                elif err: st.error(f"{t}: {err}")
        
        if dfs:
            full = pd.concat(dfs)
            
            # Format columns
            fmt = {
                "CMP": "${:,.2f}", "Buy Strike": "{:,.1f}", "Buy Premium": "${:,.2f}",
                "Sell Strike": "{:,.1f}", "Sell Premium": "${:,.2f}",
                "Margin Required": "${:,.0f}", "Net Cost": "${:,.2f}",
                "Net Max Profit": "${:,.0f}", "Breakeven": "${:,.2f}",
                "Cost / CMP %": "{:.2f}%", "Return on Margin %": "{:.1f}%",
                "Return on Cost %": "{:.1f}%", "Est Brokerage": "${:,.2f}"
            }
            
            # Group by Expiry
            unique_exps = full['Expiration'].unique()
            for exp in unique_exps:
                st.subheader(f"Expiry: {exp}")
                subset = full[full['Expiration'] == exp].drop(columns=['Expiration'])
                st.dataframe(subset.style.format(fmt, na_rep="N/A"), hide_index=True, use_container_width=True)

if __name__ == "__main__":
    main()
