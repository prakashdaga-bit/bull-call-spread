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
def get_nse_lot_sizes():
    """Fetches dictionary of {Symbol: LotSize} from NSE."""
    try:
        url = "https://nsearchives.nseindia.com/content/fo/fo_mktlots.csv"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            # Clean text to handle potential bad formatting
            csv_data = response.text.replace('\r', '')
            df = pd.read_csv(StringIO(csv_data))
            df.columns = [c.strip().upper() for c in df.columns]
            
            # Look for SYMBOL and the column containing LOT size (usually matches 'LOT')
            lot_col = next((c for c in df.columns if "LOT" in c), None)
            sym_col = next((c for c in df.columns if "SYMBOL" in c), None)
            
            if lot_col and sym_col:
                # Create a cleaned dictionary
                # Remove rows where Symbol is missing or Lot is not a number
                df = df.dropna(subset=[sym_col, lot_col])
                
                # Helper to clean lot strings (remove commas, spaces)
                def clean_lot(val):
                    try:
                        return int(str(val).replace(',', '').strip())
                    except:
                        return 1
                        
                return dict(zip(df[sym_col].str.strip(), df[lot_col].apply(clean_lot)))
    except:
        pass
    return {}

# ==========================================
# TICKER PRESETS
# ==========================================
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
        # Fetch NSE list for keys
        lots = get_nse_lot_sizes()
        url = "https://api.kite.trade/instruments"
        dfno = pd.read_csv(url)

        # Filter for NSE segment and Futures (only F&O stocks have futures)
        fno_stocks = dfno[(dfno['exchange'] == 'NFO') & (dfno['instrument_type'] == 'FUT')]['name'].unique().tolist()
        fo_list = ", ".join(sorted(lots.keys())) if lots else "RELIANCE, TCS, HDFCBANK, ICICIBANK, INFY, ITC, SBIN, BHARTIARTL, HINDUNILVR, LTIM"
        
        presets.update({
            "NIFTY 50 Top 10": "RELIANCE, TCS, HDFCBANK, ICICIBANK, INFY, ITC, SBIN, BHARTIARTL, HINDUNILVR, LTIM",
            "Indices": "NIFTY, BANKNIFTY, FINNIFTY",
            "Auto Sector": "TATAMOTORS, M&M, MARUTI, BAJAJ-AUTO, HEROMOTOCO, EICHERMOT",
            "All F&O Stocks": fno_stocks
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
        dt = datetime.datetime.fromtimestamp(timestamp)
        return dt
    return None

# ==========================================
# MARKET ADAPTERS
# ==========================================

class NSEMarketAdapter:
    """
    Fetches Option Chain data directly from NSE India website.
    Uses yfinance for Spot Price and Earnings.
    """
    BASE_URL = "https://www.nseindia.com"
    INDICES = ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"]
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "*/*",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Referer": "https://www.nseindia.com/option-chain"
        })
        try:
            self.session.get(self.BASE_URL, timeout=5)
        except: pass

    def get_spot_price(self, ticker):
        yf_ticker = ticker
        if ticker not in self.INDICES and not ticker.endswith(".NS"):
            yf_ticker = f"{ticker}.NS"
        elif ticker in self.INDICES:
            idx_map = {"NIFTY": "^NSEI", "BANKNIFTY": "^NSEBANK", "FINNIFTY": "NIFTY_FIN_SERVICE.NS"}
            yf_ticker = idx_map.get(ticker, ticker)
        try:
            stock = yf.Ticker(yf_ticker)
            price = stock.fast_info['last_price']
            if price is None:
                hist = stock.history(period="1d")
                if not hist.empty: price = hist['Close'].iloc[-1]
            return float(price) if price is not None else None
        except: return None
        
    def get_lot_size(self, ticker):
        """Fetches lot size from NSE Archives."""
        lots = get_nse_lot_sizes()
        return lots.get(ticker, 1)

    def fetch_option_chain_raw(self, ticker):
        if ticker in self.INDICES:
            url = f"https://www.nseindia.com/api/option-chain-indices?symbol={ticker}"
        else:
            url = f"https://www.nseindia.com/api/option-chain-equities?symbol={ticker.upper()}"
            
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
        limit = today + datetime.timedelta(days=days_limit)
        for d_str in expiry_dates:
            try:
                d = datetime.datetime.strptime(d_str, "%d-%b-%Y").date()
                if today <= d <= limit: valid_dates.append(d_str)
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
    """
    Uses Kite Connect API to fetch option chains.
    """
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
        idx_map = {"NIFTY": "^NSEI", "BANKNIFTY": "^NSEBANK", "FINNIFTY": "NIFTY_FIN_SERVICE.NS"}
        yf_ticker = idx_map.get(ticker, f"{ticker}.NS")
        try:
            stock = yf.Ticker(yf_ticker)
            price = stock.fast_info['last_price']
            return float(price) if price is not None else None
        except: return None

    def get_chain_for_symbol(self, ticker, days_limit=90):
        if self.instruments is None:
            self.instruments = self.get_instruments()
            
        df = self.instruments
        name = ticker
        if ticker == "NIFTY": name = "NIFTY"
        elif ticker == "BANKNIFTY": name = "BANKNIFTY"
        elif ticker == "FINNIFTY": name = "FINNIFTY"
        
        subset = df[df['name'] == name].copy()
        if subset.empty: return [], {}
        
        subset['expiry'] = pd.to_datetime(subset['expiry']).dt.date
        today = datetime.date.today()
        limit = today + datetime.timedelta(days=days_limit)
        
        valid_subset = subset[(subset['expiry'] >= today) & (subset['expiry'] <= limit)]
        unique_dates = sorted(valid_subset['expiry'].unique())
        unique_dates = unique_dates[:3]
        
        return unique_dates, valid_subset

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
            return {}

    def get_lot_size(self, ticker):
        if self.instruments is None:
            self.instruments = self.get_instruments()
        name = ticker
        if ticker == "NIFTY": name = "NIFTY"
        elif ticker == "BANKNIFTY": name = "BANKNIFTY"
        subset = self.instruments[self.instruments['name'] == name]
        if not subset.empty:
            return int(subset.iloc[0]['lot_size'])
        return 1

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

    def parse_chain(self, valid_instruments, expiry_date):
        expiry_subset = valid_instruments[valid_instruments['expiry'] == expiry_date]
        if expiry_subset.empty: return pd.DataFrame(), pd.DataFrame()
        
        tokens = expiry_subset['instrument_token'].tolist()
        quotes = self.fetch_quotes(tokens)
        
        calls_list, puts_list = [], []
        
        for _, row in expiry_subset.iterrows():
            token = row['instrument_token']
            q = quotes.get(token) or quotes.get(str(token))
            
            if not q: continue
            
            depth = q.get('depth', {})
            buy = depth.get('buy', [{}])[0]
            sell = depth.get('sell', [{}])[0]
            
            data = {
                'strike': float(row['strike']),
                'lastPrice': float(q.get('last_price', 0.0)),
                'bid': float(buy.get('price', 0.0)),
                'ask': float(sell.get('price', 0.0)),
                'openInterest': float(q.get('oi', 0.0)),
                'instrument_token': token
            }
            if row['instrument_type'] == 'CE': calls_list.append(data)
            elif row['instrument_type'] == 'PE': puts_list.append(data)
                
        return pd.DataFrame(calls_list), pd.DataFrame(puts_list)

# ==========================================
# SHARED HELPER FUNCTIONS
# ==========================================

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

def get_expirations_within_days(ticker_obj, days_limit=30):
    try:
        expirations = ticker_obj.options
    except: return []
    if not expirations: return []
    valid_dates = []
    today = datetime.date.today()
    limit_date = today + datetime.timedelta(days=days_limit)
    for date_str in expirations:
        try:
            exp_date = datetime.datetime.strptime(date_str, '%Y-%m-%d').date()
            if today <= exp_date <= limit_date: valid_dates.append(date_str)
        except: continue
    return valid_dates

def get_next_earnings_date(ticker_obj):
    try:
        cal = ticker_obj.calendar
        if cal is not None and not isinstance(cal, list) and bool(cal):
            if isinstance(cal, dict) and 'Earnings Date' in cal:
                dates = cal['Earnings Date']
                if dates: return dates[0].strftime('%Y-%m-%d')
            elif isinstance(cal, pd.DataFrame):
                if 'Earnings Date' in cal.index:
                    vals = cal.loc['Earnings Date']
                    if hasattr(vals, 'iloc'): return vals.iloc[0].strftime('%Y-%m-%d')
        dates_df = ticker_obj.get_earnings_dates(limit=4)
        if dates_df is not None and len(dates_df) > 0:
            future_dates = dates_df[dates_df.index > pd.Timestamp.now()]
            if not future_dates.empty: return future_dates.index[-1].strftime('%Y-%m-%d')
        return "N/A"
    except: return "N/A"

def filter_tradeable_options(chain):
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
    try:
        bid = float(option_row.get('bid', 0.0))
        ask = float(option_row.get('ask', 0.0))
        last = float(option_row.get('lastPrice', 0.0))
    except: return 0.0
    
    if price_type == 'mid': return (bid + ask) / 2 if (bid > 0 and ask > 0) else last
    elif price_type == 'ask': return ask if ask > 0 else last
    elif price_type == 'bid': return bid if bid > 0 else (last * 0.95) 
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
# MARKET FACTORY & LOGIC
# ==========================================

@st.cache_data(ttl=600, show_spinner=False)
def fetch_and_analyze_ticker_hybrid_v13(ticker, strategy_type, region="USA", source="Yahoo", z_api=None, z_token=None, pct_1=0.0, pct_2=5.0, expiry_idx=0):
    """Handles logic for USA (Yahoo) and India (NSE Scraper OR Zerodha)."""
    
    # 1. Setup Adapter
    adapter = None
    if region == "India":
        if source == "Zerodha (API)":
            adapter = ZerodhaMarketAdapter(z_api, z_token)
            if not adapter.connect(): return None, None, "Zerodha Connection Failed"
        else:
            adapter = NSEMarketAdapter()
    
    try:
        # 2. Get Spot Price
        lot_size = 1
        if region == "India":
            # For NSE scraper, sometimes .NS is added, strip it
            clean_ticker = ticker.replace(".NS", "")
            current_price = adapter.get_spot_price(clean_ticker)
            if adapter: # Zerodha or NSE
                if hasattr(adapter, 'get_lot_size'):
                    lot_size = adapter.get_lot_size(clean_ticker)
                else:
                    # Fallback for NSE Adapter
                    lots = get_nse_lot_sizes()
                    lot_size = lots.get(clean_ticker, 1)
        else:
            stock = yf.Ticker(ticker)
            try:
                current_price = stock.fast_info['last_price']
            except:
                hist = stock.history(period='1d')
                if not hist.empty: current_price = hist['Close'].iloc[-1]
                else: current_price = None

        if not current_price: return None, None, f"Could not fetch spot price for {ticker}"

        # 3. Get Expirations
        valid_dates = []
        raw_data = None # For NSE scraper
        valid_instruments = None # For Zerodha
        
        if region == "India":
            if source == "Zerodha (API)":
                valid_dates, valid_instruments = adapter.get_chain_for_symbol(clean_ticker)
            else:
                valid_dates, raw_data = adapter.get_expirations(clean_ticker, days_limit=90)
                if valid_dates: valid_dates = valid_dates[:3] # Default max 3
        else:
            stock = yf.Ticker(ticker)
            try:
                # Use Monthly Logic for Simple Analysis
                valid_dates = get_monthly_expirations(stock, limit=3)
            except: pass

        if not valid_dates: return None, None, "No valid expirations found."
        
        # --- EXPIRY FILTERING FOR INDIA ---
        if region == "India" and valid_dates:
            if expiry_idx < len(valid_dates):
                valid_dates = [valid_dates[expiry_idx]]
            else:
                valid_dates = [valid_dates[-1]] # Fallback to furthest if index out of bounds

        analysis_rows = []
        summary_returns = {"Stock": ticker}

        for date_obj in valid_dates:
            # 4. Get Option Chain
            calls, puts = pd.DataFrame(), pd.DataFrame()
            
            # Date Handling (Zerodha returns date objects, others strings)
            if isinstance(date_obj, datetime.date): date_str = date_obj.strftime('%Y-%m-%d')
            else: date_str = date_obj
            
            if region == "India":
                if source == "Zerodha (API)":
                    calls, puts = adapter.parse_chain(valid_instruments, date_obj) # Pass date object
                else:
                    calls, puts = adapter.parse_chain(raw_data, date_str)
            else:
                try:
                    chain = stock.option_chain(date_str)
                    calls, puts = chain.calls, chain.puts
                except: continue

            calls = filter_tradeable_options(calls)
            puts = filter_tradeable_options(puts)
            
            if calls.empty or puts.empty: continue

            # 5. Run Simple Strategy Logic
            try:
                # Common Vars
                buy_leg, sell_leg = None, None
                buy_strike, sell_strike = 0.0, 0.0
                buy_prem, sell_prem = 0.0, 0.0
                net_cost, max_gain, breakeven, ret_pct, margin = 0.0, 0.0, 0.0, 0.0, 0.0
                
                # Targets
                price_1 = current_price * (1 + pct_1/100.0)
                price_2 = current_price * (1 + pct_2/100.0)
                
                if strategy_type == "Bull Call Spread":
                    # Buy Low Call, Sell High Call (Debit)
                    leg_a = find_closest_strike(calls, price_1)
                    leg_b = find_closest_strike(calls, price_2)
                    
                    if leg_a is None or leg_b is None or leg_a['strike'] == leg_b['strike']: continue
                    
                    buy_leg = leg_a if leg_a['strike'] < leg_b['strike'] else leg_b
                    sell_leg = leg_b if leg_a['strike'] < leg_b['strike'] else leg_a
                    
                    buy_strike, sell_strike = buy_leg['strike'], sell_leg['strike']
                    buy_prem, sell_prem = get_price(buy_leg, 'ask'), get_price(sell_leg, 'bid')
                    if buy_prem == 0: continue
                    
                    net_cost = buy_prem - sell_prem
                    max_gain = sell_strike - buy_strike
                    breakeven = buy_strike + net_cost
                    margin = net_cost
                    if net_cost > 0: ret_pct = ((max_gain - net_cost) / net_cost) * 100

                elif strategy_type == "Bear Call Spread":
                    # Sell Low Call, Buy High Call (Credit)
                    leg_a = find_closest_strike(calls, price_1)
                    leg_b = find_closest_strike(calls, price_2)
                    
                    if leg_a is None or leg_b is None or leg_a['strike'] == leg_b['strike']: continue
                    
                    sell_leg = leg_a if leg_a['strike'] < leg_b['strike'] else leg_b
                    buy_leg = leg_b if leg_a['strike'] < leg_b['strike'] else leg_a
                    
                    buy_strike, sell_strike = buy_leg['strike'], sell_leg['strike']
                    buy_prem, sell_prem = get_price(buy_leg, 'ask'), get_price(sell_leg, 'bid')
                    
                    net_cost = buy_prem - sell_prem # Credit
                    max_gain = abs(net_cost)
                    margin = (buy_strike - sell_strike)
                    breakeven = sell_strike + max_gain
                    if margin > 0: ret_pct = (max_gain / margin) * 100

                elif strategy_type == "Bull Put Spread":
                    # Buy Low Put, Sell High Put (Credit)
                    leg_a = find_closest_strike(puts, price_1)
                    leg_b = find_closest_strike(puts, price_2)
                    
                    if leg_a is None or leg_b is None or leg_a['strike'] == leg_b['strike']: continue
                    
                    buy_leg = leg_a if leg_a['strike'] < leg_b['strike'] else leg_b
                    sell_leg = leg_b if leg_a['strike'] < leg_b['strike'] else leg_a
                    
                    buy_strike, sell_strike = buy_leg['strike'], sell_leg['strike']
                    buy_prem, sell_prem = get_price(buy_leg, 'ask'), get_price(sell_leg, 'bid')
                    
                    net_cost = buy_prem - sell_prem # Credit
                    max_gain = abs(net_cost)
                    margin = (sell_strike - buy_strike)
                    breakeven = sell_strike - max_gain
                    if margin > 0: ret_pct = (max_gain / margin) * 100

                elif strategy_type == "Bear Put Spread":
                    # Sell Low Put, Buy High Put (Debit)
                    leg_a = find_closest_strike(puts, price_1)
                    leg_b = find_closest_strike(puts, price_2)
                    
                    if leg_a is None or leg_b is None or leg_a['strike'] == leg_b['strike']: continue
                    
                    sell_leg = leg_a if leg_a['strike'] < leg_b['strike'] else leg_b
                    buy_leg = leg_b if leg_a['strike'] < leg_b['strike'] else leg_a
                    
                    buy_strike, sell_strike = buy_leg['strike'], sell_leg['strike']
                    buy_prem, sell_prem = get_price(buy_leg, 'ask'), get_price(sell_leg, 'bid')
                    
                    if buy_prem == 0: continue
                    net_cost = buy_prem - sell_prem # Debit
                    max_gain = buy_strike - sell_strike
                    breakeven = buy_strike - net_cost
                    margin = net_cost
                    if net_cost > 0: ret_pct = ((max_gain - net_cost) / net_cost) * 100
                    
                # -- COMMON OUTPUT --
                if strategy_type != "Long Straddle":
                    cost_cmp_pct = 0.0
                    if margin > 0 and current_price > 0:
                        cost_cmp_pct = (margin / current_price) * 100

                    analysis_rows.append({
                        "Expiration": date_str, 
                        "Spot Price": float(current_price),
                        "Lot Size": lot_size,
                        "Buy Strike": buy_strike, 
                        "Buy Premium": buy_prem,
                        "Sell Strike": sell_strike, 
                        "Sell Premium": sell_prem, 
                        "Net Cost": net_cost,
                        "Cost/CMP %": cost_cmp_pct,
                        "Max Gain": max_gain, 
                        "Return %": ret_pct, 
                        "Breakeven": breakeven
                    })
                    summary_returns[date_str] = f"{ret_pct:.1f}%"
                
                # STRADDLE Logic preserved
                elif strategy_type == "Long Straddle":
                    common = set(calls['strike']).intersection(set(puts['strike']))
                    if not common: continue
                    avail = pd.DataFrame({'strike': list(common)})
                    closest = find_closest_strike(avail, current_price)
                    if closest is None: continue
                    strike = closest['strike']
                    c = calls[calls['strike'] == strike].iloc[0]
                    p = puts[puts['strike'] == strike].iloc[0]
                    c_ask, p_ask = get_price(c, 'ask'), get_price(p, 'ask')
                    if c_ask == 0 or p_ask == 0: continue
                    net_cost = c_ask + p_ask
                    
                    cost_cmp_pct = 0.0
                    if current_price > 0:
                        cost_cmp_pct = (net_cost / current_price) * 100
                        
                    analysis_rows.append({
                        "Expiration": date_str, "Spot Price": float(current_price), "Lot Size": lot_size,
                        "Strike": strike, "Call Cost": c_ask, "Put Cost": p_ask, "Net Cost": net_cost, 
                        "Cost/CMP %": cost_cmp_pct,
                        "BE Low": strike - net_cost, "BE High": strike + net_cost, 
                        "Move Needed": (net_cost / current_price) * 100
                    })
                    summary_returns[date_str] = f"±{ (net_cost / current_price) * 100:.1f}%"

            except: continue

        if not analysis_rows: return None, None, "Could not build strategies."
        
        # Convert to DataFrame
        df = pd.DataFrame(analysis_rows)
        
        # Data Cleaning: Force Numeric Types
        if not df.empty:
            cols_to_numeric = ["Spot Price", "Lot Size", "Buy Strike", "Buy Premium", "Sell Strike", "Sell Premium", "Net Cost", "Max Gain", "Breakeven", "Return %", "Cost/CMP %", "Strike", "Call Cost", "Put Cost", "BE Low", "BE High", "Move Needed", "Margin Required"]
            for col in cols_to_numeric:
                if col in df.columns:
                    # Coerce errors to NaN, then fill with 0.0. 
                    # Also replace infinite values if any logic caused division by zero.
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
                    df[col] = df[col].replace([float('inf'), float('-inf')], 0.0)
        
        # Reorder columns for visibility
        if not df.empty and strategy_type != "Long Straddle":
             cols = ["Expiration", "Spot Price", "Cost/CMP %", "Return %", "Net Cost", "Max Gain", "Buy Strike", "Sell Strike", "Buy Premium", "Sell Premium", "Breakeven", "Lot Size"]
             # Keep only cols that exist
             cols = [c for c in cols if c in df.columns]
             df = df[cols]

        return summary_returns, df, None

    except Exception as e:
        return None, None, str(e)

# ==========================================
# CUSTOM 4-LEG STRATEGY (HYBRID)
# ==========================================
def calculate_strategy_metrics(legs, current_price, view, lot_size=1, adapter=None):
    net_premium = 0.0
    strikes = []
    
    for leg in legs:
        strikes.append(leg['row']['strike'])
        price = get_price(leg['row'], 'ask' if leg['action'] == "Buy" else 'bid')
        impact = -price if leg['action'] == "Buy" else price
        net_premium += impact

    sim_prices = sorted(strikes + [current_price])
    range_width = max(strikes) - min(strikes)
    sim_range = [min(strikes) - range_width*0.5] + sim_prices + [max(strikes) + range_width*0.5]
    
    profits = []
    for p in sim_range:
        current_pnl = net_premium
        for leg in legs:
            strike = leg['row']['strike']
            is_call = leg['type'] == "Call"
            is_buy = leg['action'] == "Buy"
            intrinsic = max(0, p - strike) if is_call else max(0, strike - p)
            current_pnl += (intrinsic if is_buy else -intrinsic)
        profits.append(current_pnl)

    max_profit_per_share = max(profits)
    max_loss_per_share = min(profits)
    
    capital_required = 0.0
    if adapter and hasattr(adapter, 'get_margin_for_basket'):
        api_margin = adapter.get_margin_for_basket(legs, lot_size)
        if api_margin > 0: capital_required = api_margin
    
    if capital_required == 0:
        # Fallback logic
        total_premium_lot = net_premium * lot_size
        puts = sorted([l for l in legs if l['type'] == 'Put'], key=lambda x: x['row']['strike'])
        calls = sorted([l for l in legs if l['type'] == 'Call'], key=lambda x: x['row']['strike'])
        if len(puts) == 2 and len(calls) == 2:
            width = max(abs(puts[1]['row']['strike'] - puts[0]['row']['strike']), abs(calls[1]['row']['strike'] - calls[0]['row']['strike']))
            if view == "Neutral": capital_required = width * lot_size
            else: capital_required = abs(total_premium_lot) if total_premium_lot < 0 else 0.0

    brokerage = 20 * 4 if lot_size > 1 else 0.05 * 4
    net_max_profit = (max_profit_per_share * lot_size) - brokerage
    roi = (net_max_profit / capital_required * 100) if capital_required > 0 else 0.0

    return {
        "net_premium": net_premium, "max_upside": max_profit_per_share, "max_loss": max_loss_per_share,
        "capital_required": capital_required, "brokerage": brokerage, "net_max_profit": net_max_profit, "roi": roi, "lot_size": lot_size
    }

@st.cache_data(ttl=300, show_spinner=False)
def analyze_custom_strategy(ticker, view, slab1_pct, slab2_pct, days_window, region="USA", source="Yahoo", z_api=None, z_token=None, optimize=False):
    adapter = None
    if region == "India":
        if source == "Zerodha (API)":
            adapter = ZerodhaMarketAdapter(z_api, z_token)
            if not adapter.connect(): return None, "Zerodha Connection Failed"
        else: adapter = NSEMarketAdapter()
    
    try:
        lot_size = 1
        if region == "India":
            clean_ticker = ticker.replace(".NS", "")
            current_price = adapter.get_spot_price(clean_ticker)
            if source == "Zerodha (API)": lot_size = adapter.get_lot_size(clean_ticker)
            try:
                stock = yf.Ticker(f"{clean_ticker}.NS")
                earnings_date = get_next_earnings_date(stock)
            except: earnings_date = "N/A"
        else:
            stock = yf.Ticker(ticker)
            try:
                current_price = stock.fast_info['last_price']
            except:
                hist = stock.history(period='1d')
                current_price = hist['Close'].iloc[-1] if not hist.empty else None
            earnings_date = get_next_earnings_date(stock)

        if not current_price: return None, f"Could not fetch price for {ticker}"

        valid_dates, raw_data, valid_instruments = [], None, None
        if region == "India":
            if source == "Zerodha (API)":
                valid_dates, valid_instruments = adapter.get_chain_for_symbol(clean_ticker, days_limit=days_window)
            else:
                valid_dates, raw_data = adapter.get_expirations(clean_ticker, days_limit=days_window)
        else:
            stock = yf.Ticker(ticker)
            dates = get_expirations_within_days(stock, days_limit=days_window)
            valid_dates = dates

        if not valid_dates: return None, f"No expirations found."
        
        results_list = []
        errors = []

        for date_obj in valid_dates:
            try:
                calls, puts = pd.DataFrame(), pd.DataFrame()
                if isinstance(date_obj, datetime.date): date_str = date_obj.strftime('%Y-%m-%d')
                else: date_str = date_obj
                
                if region == "India":
                    if source == "Zerodha (API)": calls, puts = adapter.parse_chain(valid_instruments, date_obj)
                    else: calls, puts = adapter.parse_chain(raw_data, date_str)
                else:
                    chain = get_option_chain_with_retry(stock, date_str)
                    calls, puts = chain.calls, chain.puts

                calls = filter_tradeable_options(calls).sort_values('strike').reset_index(drop=True)
                puts = filter_tradeable_options(puts).sort_values('strike').reset_index(drop=True)
                if calls.empty or puts.empty: continue

                s1, s2 = slab1_pct / 100.0, slab2_pct / 100.0
                targets = {
                    "pf": current_price * (1 - s2), "pn": current_price * (1 - s1),
                    "cn": current_price * (1 + s1), "cf": current_price * (1 + s2)
                }
                
                def build_legs(pf_row, pn_row, cn_row, cf_row):
                    l = []
                    if view == "Neutral":
                        l = [
                            {"type": "Put", "action": "Buy", "row": pf_row, "desc": f"Put Long (-{slab2_pct}%)"},
                            {"type": "Put", "action": "Sell", "row": pn_row, "desc": f"Put Short (-{slab1_pct}%)"},
                            {"type": "Call", "action": "Sell", "row": cn_row, "desc": f"Call Short (+{slab1_pct}%)"},
                            {"type": "Call", "action": "Buy", "row": cf_row, "desc": f"Call Long (+{slab2_pct}%)"},
                        ]
                    else:
                        l = [
                            {"type": "Put", "action": "Sell", "row": pf_row, "desc": f"Put Short (-{slab2_pct}%)"},
                            {"type": "Put", "action": "Buy", "row": pn_row, "desc": f"Put Long (-{slab1_pct}%)"},
                            {"type": "Call", "action": "Buy", "row": cn_row, "desc": f"Call Long (+{slab1_pct}%)"},
                            {"type": "Call", "action": "Sell", "row": cf_row, "desc": f"Call Short (+{slab2_pct}%)"},
                        ]
                    return l

                pf, pn = find_closest_strike(puts, targets["pf"]), find_closest_strike(puts, targets["pn"])
                cn, cf = find_closest_strike(calls, targets["cn"]), find_closest_strike(calls, targets["cf"])
                if any(x is None for x in [pf, pn, cn, cf]): continue

                base_legs = build_legs(pf, pn, cn, cf)
                base_metrics = calculate_strategy_metrics(base_legs, current_price, view, lot_size, adapter)
                
                payload = {
                    "ticker": ticker, "current_price": current_price, "expiry": date_str, "earnings": earnings_date,
                    "base": {"metrics": base_metrics, "legs": base_legs}, "optimized": None 
                }

                if optimize:
                    def get_idx(df, strike): 
                        indices = df.index[df['strike'] == strike].tolist()
                        return indices[0] if indices else -1
                    
                    pf_idx, pn_idx = get_idx(puts, pf['strike']), get_idx(puts, pn['strike'])
                    cn_idx, cf_idx = get_idx(calls, cn['strike']), get_idx(calls, cf['strike'])
                    
                    if not any(i == -1 for i in [pf_idx, pn_idx, cn_idx, cf_idx]):
                        best_ratio = -1.0
                        best_config = None
                        range_scan = range(-1, 2)
                        
                        for i1 in range_scan:
                            for i2 in range_scan:
                                for i3 in range_scan:
                                    for i4 in range_scan:
                                        if not (0<=pf_idx+i1<len(puts) and 0<=pn_idx+i2<len(puts) and 0<=cn_idx+i3<len(calls) and 0<=cf_idx+i4<len(calls)): continue
                                        pf_cand = puts.iloc[pf_idx+i1]
                                        pn_cand = puts.iloc[pn_idx+i2]
                                        cn_cand = calls.iloc[cn_idx+i3]
                                        cf_cand = calls.iloc[cf_idx+i4]
                                        
                                        if pf_cand['strike']>=pn_cand['strike'] or cn_cand['strike']>=cf_cand['strike']: continue
                                        
                                        cand_legs = build_legs(pf_cand, pn_cand, cn_cand, cf_cand)
                                        cand_metrics = calculate_strategy_metrics(cand_legs, current_price, view, lot_size, None)
                                        loss = abs(cand_metrics['max_loss'])
                                        ratio = cand_metrics['max_upside'] / loss if loss > 0.01 else 0
                                        if ratio > best_ratio:
                                            best_ratio = ratio
                                            best_config = {"metrics": cand_metrics, "legs": cand_legs, "ratio": ratio}
                        
                        if best_config:
                            best_config['metrics'] = calculate_strategy_metrics(best_config['legs'], current_price, view, lot_size, adapter)
                            payload["optimized"] = best_config
                
                results_list.append(payload)
            except Exception as e:
                errors.append(f"Date {date_str}: {str(e)}")
                continue

        if not results_list: return None, "Could not build strategies."
        return results_list, None
    except Exception as e: return None, str(e)

# ==========================================
# PART 3: MAIN APP INTERFACE
# ==========================================

def display_strategy_details(data, label, current_price):
    st.markdown(f"**{label}**")
    m = data['metrics']
    c1, c2, c3, c4 = st.columns(4)
    net = m['net_premium']
    lbl = "Net Credit (Total)" if net > 0 else "Net Debit (Total)"
    
    # Financials are now TOTAL LOT
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
    
    region = st.sidebar.selectbox("Select Market Region", ["USA (Yahoo)", "India"])
    region_key = "USA" if "USA" in region else "India"
    
    source = "Yahoo"
    z_api, z_token = None, None
    saved_api, saved_token = load_zerodha_tokens()
    
    if region_key == "India":
        source = st.sidebar.radio("India Data Source", ["NSE Website (Free/Flaky)", "Zerodha (API)"])
        if source == "Zerodha (API)":
            st.sidebar.info("Requires Kite Connect subscription.")
            z_api = st.sidebar.text_input("API Key", value=saved_api if saved_api else "", type="password")
            z_token = st.sidebar.text_input("Access Token", value=saved_token if saved_token else "", type="password")
        
        # Check token file age
        file_dt = get_token_file_info()
        if file_dt:
             hours_old = (datetime.datetime.now() - file_dt).total_seconds() / 3600
             if hours_old > 12:
                 st.sidebar.warning(f"⚠️ Token file is {hours_old:.1f} hours old. Auto-login might have failed.")
             else:
                 st.sidebar.success(f"✅ Token updated: {file_dt.strftime('%H:%M')}")
    
    mode = st.sidebar.radio(
        "Select Analysis Mode:", 
        ["Simple Analysis (Standard)", "Custom Strategy Generator (Slab Based)"],
        index=0
    )
    st.sidebar.markdown("---")
    
    if "input_simple" not in st.session_state: st.session_state["input_simple"] = ""
    if "input_custom" not in st.session_state: st.session_state["input_custom"] = ""

    presets = get_ticker_presets(region_key)

    if mode == "Simple Analysis (Standard)":
        st.subheader(f"📈 {region_key} Market Real-Time Analysis")
        st.caption("Fetches live option chains. Standard Spreads/Straddles.")
        strategy = st.radio("Strategy Type:", ("Bull Call Spread", "Bear Call Spread", "Bull Put Spread", "Bear Put Spread", "Long Straddle"), horizontal=True)
        
        pct_1 = 0.0
        pct_2 = 5.0
        expiry_idx = 0
        
        if region_key == "India":
            c_exp, _ = st.columns([1,3])
            exp_opts = ["Current Month", "Next Month", "Far Month"]
            exp_sel = c_exp.selectbox("Select Expiry (India Only)", exp_opts)
            try: expiry_idx = exp_opts.index(exp_sel)
            except: expiry_idx = 0
        
        if strategy != "Long Straddle":
            c1, c2 = st.columns(2)
            pct_1 = c1.number_input("Strike 1 (% from Spot)", min_value=-50.0, max_value=50.0, value=0.0, step=0.5)
            pct_2 = c2.number_input("Strike 2 (% from Spot)", min_value=-50.0, max_value=50.0, value=5.0, step=0.5)
        
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
                all_summaries, all_details, errors = [], {}, []
                consolidated_data = [] 
                
                progress_bar = st.progress(0)
                with st.spinner(f"Fetching data..."):
                    for i, ticker in enumerate(tickers):
                        # Renamed function call to bust cache and force fresh data fetch
                        summary, df, error = fetch_and_analyze_ticker_hybrid_v13(ticker, strategy, region_key, source, z_api, z_token, pct_2, pct_1, expiry_idx)
                        if error: errors.append(f"{ticker}: {error}")
                        else:
                            all_summaries.append(summary)
                            
                            # Clean numeric columns to avoid string formatting crash
                            all_details[ticker] = df
                            if not df.empty:
                                df_summary = df.copy()
                                df_summary.insert(0, "Stock", ticker)
                                consolidated_data.append(df_summary)
                        progress_bar.progress((i + 1) / len(tickers))
                st.divider()
                if consolidated_data:
                    st.header("1. Strategy Summary")
                    full_df = pd.concat(consolidated_data, ignore_index=True)
                    unique_expirations = sorted(full_df['Expiration'].unique())
                    for exp in unique_expirations:
                        st.subheader(f"Expiry: {exp}")
                        subset = full_df[full_df['Expiration'] == exp].drop(columns=['Expiration'])
                        
                        if strategy != "Long Straddle":
                            format_dict = {
                                "Spot Price": "${:,.2f}", "Buy Premium": "${:,.2f}", "Sell Premium": "${:,.2f}",
                                "Net Cost": "${:,.2f}", "Cost/CMP %": "{:.2f}%", "Max Gain": "${:,.2f}",
                                "Breakeven": "${:,.2f}", "Return %": "{:.1f}%"
                            }
                        else:
                            format_dict = {
                                "Spot Price": "${:,.2f}", "Call Cost": "${:,.2f}", "Put Cost": "${:,.2f}",
                                "Net Cost": "${:,.2f}", "Cost/CMP %": "{:.2f}%", "BE Low": "${:,.2f}",
                                "BE High": "${:,.2f}", "Move Needed": "{:.1f}%"
                            }
                        
                        # Use try/except block for display robustness
                        try:
                            st.dataframe(subset.style.format(format_dict), hide_index=True, use_container_width=True)
                        except:
                            st.dataframe(subset, hide_index=True, use_container_width=True)

                elif not errors: st.warning("No valid data found.")

                if all_details:
                    st.header("2. Detailed Breakdown")
                    for ticker, df in all_details.items():
                        with st.expander(f"{ticker} Details", expanded=False):
                            
                            if strategy != "Long Straddle":
                                format_dict = {
                                    "Spot Price": "${:,.2f}", "Buy Premium": "${:,.2f}", "Sell Premium": "${:,.2f}",
                                    "Net Cost": "${:,.2f}", "Cost/CMP %": "{:.2f}%", "Max Gain": "${:,.2f}",
                                    "Breakeven": "${:,.2f}", "Return %": "{:.1f}%"
                                }
                            else:
                                format_dict = {
                                    "Spot Price": "${:,.2f}", "Call Cost": "${:,.2f}", "Put Cost": "${:,.2f}",
                                    "Net Cost": "${:,.2f}", "Cost/CMP %": "{:.2f}%", "BE Low": "${:,.2f}",
                                    "BE High": "${:,.2f}", "Move Needed": "{:.1f}%"
                                }
                            
                            if not df.empty:
                                try:
                                    st.dataframe(df.style.format(format_dict), use_container_width=True)
                                except Exception as e:
                                    st.error(f"⚠️ Formatting error. Showing raw data.")
                                    st.dataframe(df, use_container_width=True)
                if errors:
                    with st.expander("Errors"):
                        for e in errors: st.write(f"- {e}")

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
                all_results, all_summaries, errors = {}, [], []
                progress_bar = st.progress(0)
                with st.spinner(f"Scanning expirations..."):
                    for i, ticker in enumerate(tickers):
                        results_list, error = analyze_custom_strategy(ticker, view, slab1, slab2, days_window, region_key, source, z_api, z_token, optimize=True)
                        if error: errors.append(f"{ticker}: {error}")
                        else:
                            all_results[ticker] = results_list
                            for res in results_list:
                                metrics = res['optimized']['metrics'] if res['optimized'] else res['base']['metrics']
                                ratio = res['optimized']['ratio'] if res['optimized'] else (res['base']['metrics']['max_upside'] / abs(res['base']['metrics']['max_loss']))
                                summary_data = {
                                    "Stock": ticker, "Next Earnings": res['earnings'], "Expiry": res['expiry'],
                                    "Spot": f"${res['current_price']:.2f}",
                                    "Margin (1 Lot)": f"${metrics['capital_required']:,.0f}",
                                    "Net Max Profit": f"${metrics['net_max_profit']:,.0f}",
                                    "ROI": f"{metrics['roi']:.1f}%",
                                    "Est. Brokerage": f"${metrics['brokerage']:.2f}",
                                    "Lot Size": metrics['lot_size']
                                }
                                all_summaries.append(summary_data)
                        progress_bar.progress((i + 1) / len(tickers))
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
                            roi = opt_data['metrics']['roi']
                            with st.expander(f"📅 {res['expiry']} | ROI {roi:.1f}% | R/R {ratio:.2f}", expanded=False):
                                display_strategy_details(opt_data, "Recommended Strategy", res['current_price'])
                if errors:
                    with st.expander("Errors / Skipped Tickers"):
                        for err in errors: st.write(f"- {err}")

if __name__ == "__main__":
    main()
