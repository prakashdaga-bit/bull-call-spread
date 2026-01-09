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
# TICKER PRESETS
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
    Uses Kite Connect API.
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
            st.error("Please install kiteconnect")
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
        if self.instruments is None: self.instruments = self.get_instruments()
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
            st.error(f"Quote Fetch Error: {e}")
            return {}

    def get_lot_size(self, ticker):
        if self.instruments is None: self.instruments = self.get_instruments()
        name = ticker
        if ticker == "NIFTY": name = "NIFTY"
        elif ticker == "BANKNIFTY": name = "BANKNIFTY"
        subset = self.instruments[self.instruments['name'] == name]
        if not subset.empty:
            return int(subset.iloc[0]['lot_size'])
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
# MAIN ANALYSIS LOGIC
# ==========================================

@st.cache_data(ttl=600, show_spinner=False)
def fetch_and_analyze_ticker_hybrid_v27(ticker, strategy_type, region="USA", source="Yahoo", z_api=None, z_token=None, pct_1=0.0, pct_2=5.0, pct_3=0.0, expiry_idx=0):
    adapter = None
    if region == "India":
        if source == "Zerodha (API)":
            adapter = ZerodhaMarketAdapter(z_api, z_token)
            if not adapter.connect(): return None, None, "Zerodha Connection Failed"
        else: adapter = NSEMarketAdapter()
    
    try:
        # Fetch Spot
        lot_size = 1
        if region == "India":
            clean_ticker = ticker.replace(".NS", "")
            current_price = adapter.get_spot_price(clean_ticker)
            if adapter: 
                if hasattr(adapter, 'get_lot_size'): lot_size = adapter.get_lot_size(clean_ticker)
                else: lot_size = int(get_fno_info_zerodha().get(clean_ticker, 1))
        else:
            stock = yf.Ticker(ticker)
            try:
                current_price = stock.fast_info['last_price']
            except:
                hist = stock.history(period='1d')
                current_price = hist['Close'].iloc[-1] if not hist.empty else None

        if not current_price: return None, None, f"Could not fetch spot price for {ticker}"

        # Fetch Expiry
        valid_dates, raw_data, valid_instruments = [], None, None
        if region == "India":
            if source == "Zerodha (API)":
                valid_dates, valid_instruments = adapter.get_chain_for_symbol(clean_ticker)
            else:
                valid_dates, raw_data = adapter.get_expirations(clean_ticker, days_limit=90)
                if valid_dates: valid_dates = valid_dates[:3]
        else:
            stock = yf.Ticker(ticker)
            valid_dates = get_monthly_expirations(stock, limit=3)

        if not valid_dates: return None, None, "No valid expirations found."
        
        # Filter Expiry (India Only)
        if region == "India" and valid_dates:
             valid_dates = [valid_dates[expiry_idx]] if expiry_idx < len(valid_dates) else [valid_dates[-1]]

        analysis_rows = []

        for date_obj in valid_dates:
            calls, puts = pd.DataFrame(), pd.DataFrame()
            if isinstance(date_obj, datetime.date): date_str = date_obj.strftime('%Y-%m-%d')
            else: date_str = date_obj
            
            if region == "India":
                if source == "Zerodha (API)": calls, puts = adapter.parse_chain(valid_instruments, date_obj)
                else: calls, puts = adapter.parse_chain(raw_data, date_str)
            else:
                try:
                    chain = stock.option_chain(date_str)
                    calls, puts = chain.calls, chain.puts
                except: continue

            calls = filter_tradeable_options(calls)
            puts = filter_tradeable_options(puts)
            if calls.empty or puts.empty: continue

            try:
                buy_leg, sell_leg, put_leg = None, None, None
                legs_list = []
                
                # Targets
                price_1 = current_price * (1 + pct_1/100.0)
                price_2 = current_price * (1 + pct_2/100.0)
                
                # Strategy Construction
                if strategy_type == "Bull Call Spread":
                    la, lb = find_closest_strike(calls, price_1), find_closest_strike(calls, price_2)
                    if la is not None and lb is not None and la['strike'] != lb['strike']:
                        buy_leg = la if la['strike'] < lb['strike'] else lb
                        sell_leg = lb if la['strike'] < lb['strike'] else la
                        legs_list = [{'row': buy_leg, 'action': 'Buy', 'type': 'Call'}, {'row': sell_leg, 'action': 'Sell', 'type': 'Call'}]

                elif strategy_type == "Bear Put Spread":
                    la, lb = find_closest_strike(puts, price_1), find_closest_strike(puts, price_2)
                    if la is not None and lb is not None and la['strike'] != lb['strike']:
                        buy_leg = lb if la['strike'] < lb['strike'] else la # Higher Strike Put
                        sell_leg = la if la['strike'] < lb['strike'] else lb # Lower Strike Put
                        legs_list = [{'row': buy_leg, 'action': 'Buy', 'type': 'Put'}, {'row': sell_leg, 'action': 'Sell', 'type': 'Put'}]

                elif strategy_type == "Bear Call Spread":
                    la, lb = find_closest_strike(calls, price_1), find_closest_strike(calls, price_2)
                    if la is not None and lb is not None and la['strike'] != lb['strike']:
                        buy_leg = lb if la['strike'] < lb['strike'] else la # Higher Call
                        sell_leg = la if la['strike'] < lb['strike'] else lb # Lower Call
                        legs_list = [{'row': sell_leg, 'action': 'Sell', 'type': 'Call'}, {'row': buy_leg, 'action': 'Buy', 'type': 'Call'}]
                
                elif strategy_type == "Bull Put Spread":
                    la, lb = find_closest_strike(puts, price_1), find_closest_strike(puts, price_2)
                    if la is not None and lb is not None and la['strike'] != lb['strike']:
                        buy_leg = la if la['strike'] < lb['strike'] else lb # Lower Put
                        sell_leg = lb if la['strike'] < lb['strike'] else la # Higher Put
                        legs_list = [{'row': buy_leg, 'action': 'Buy', 'type': 'Put'}, {'row': sell_leg, 'action': 'Sell', 'type': 'Put'}]

                elif strategy_type == "Leveraged Bull Call Spread":
                    price_3 = current_price * (1 + pct_3/100.0)
                    la = find_closest_strike(calls, price_1)
                    lb = find_closest_strike(calls, price_2)
                    lc = find_closest_strike(puts, price_3)
                    
                    if all(x is not None for x in [la, lb, lc]) and la['strike'] != lb['strike']:
                        buy_leg = la if la['strike'] < lb['strike'] else lb
                        sell_leg = lb if la['strike'] < lb['strike'] else la
                        put_leg = lc
                        legs_list = [
                            {'row': buy_leg, 'action': 'Buy', 'type': 'Call'},
                            {'row': sell_leg, 'action': 'Sell', 'type': 'Call'},
                            {'row': put_leg, 'action': 'Sell', 'type': 'Put'}
                        ]
                
                elif strategy_type == "Long Straddle":
                    common = set(calls['strike']).intersection(set(puts['strike']))
                    if common:
                        avail = pd.DataFrame({'strike': list(common)})
                        closest = find_closest_strike(avail, current_price)
                        if closest is not None:
                            k = closest['strike']
                            c = calls[calls['strike'] == k].iloc[0]
                            p = puts[puts['strike'] == k].iloc[0]
                            legs_list = [{'row': c, 'action': 'Buy', 'type': 'Call'}, {'row': p, 'action': 'Buy', 'type': 'Put'}]

                if legs_list:
                    # Calculations
                    margin = 0.0
                    brokerage = 20 * len(legs_list) if region == "India" else 0.65 * len(legs_list)
                    
                    # Margin Fetch
                    if adapter and hasattr(adapter, 'get_margin_for_basket'):
                        api_margin = adapter.get_margin_for_basket(legs_list, lot_size)
                        if api_margin > 0: margin = api_margin
                    
                    # Premiums & Cost
                    buy_prem_tot = sum(get_price(l['row'], 'ask') for l in legs_list if l['action'] == 'Buy')
                    sell_prem_tot = sum(get_price(l['row'], 'bid') for l in legs_list if l['action'] == 'Sell')
                    net_cost = buy_prem_tot - sell_prem_tot
                    
                    # Fallback Margin
                    buy_strikes = [l['row']['strike'] for l in legs_list if l['action']=='Buy']
                    sell_strikes = [l['row']['strike'] for l in legs_list if l['action']=='Sell']
                    
                    if margin == 0:
                        if net_cost > 0: margin = net_cost * lot_size # Debit
                        else: # Credit
                             # Simplified width logic
                             margin = (abs(buy_strikes[0] - sell_strikes[0]) * lot_size) if (buy_strikes and sell_strikes) else 0

                    # Max Gain & Breakeven
                    max_gain_abs, breakeven = 0.0, 0.0
                    
                    if strategy_type in ["Bull Call Spread", "Bear Put Spread"]:
                        max_gain_abs = abs(sell_strikes[0] - buy_strikes[0])
                        breakeven = buy_strikes[0] + net_cost if strategy_type == "Bull Call Spread" else buy_strikes[0] - net_cost
                        
                    elif strategy_type in ["Bear Call Spread", "Bull Put Spread"]:
                        max_gain_abs = abs(net_cost) # Credit received
                        breakeven = sell_strikes[0] + max_gain_abs if strategy_type == "Bear Call Spread" else sell_strikes[0] - max_gain_abs

                    elif strategy_type == "Leveraged Bull Call Spread":
                         width = sell_strikes[0] - buy_strikes[0]
                         max_gain_abs = width - net_cost
                         breakeven = sell_strikes[1] + net_cost # Put Strike + Cost/Credit
                    
                    elif strategy_type == "Long Straddle":
                         max_gain_abs = 0 # Unlimited
                         breakeven = buy_strikes[0] + net_cost # Upper BE (Display one)
                    
                    # Net Profit & ROI
                    net_max_profit = 0.0
                    if strategy_type == "Long Straddle": 
                         net_max_profit = 0
                    else:
                         # For spreads: (Width - NetCost)*Lot OR Credit*Lot - Brokerage
                         if net_cost > 0: # Debit
                             net_max_profit = ((max_gain_abs - net_cost) * lot_size) - brokerage
                         else: # Credit
                             net_max_profit = (abs(net_cost) * lot_size) - brokerage
                    
                    rom = (net_max_profit / margin * 100) if margin > 0 else 0
                    roc = (net_max_profit / (net_cost * lot_size) * 100) if net_cost > 0 else 0

                    cost_cmp = 0.0
                    if margin > 0: cost_cmp = (margin/lot_size)/current_price * 100
                    
                    # Column Mapping
                    b_k = buy_strikes[0] if buy_strikes else 0
                    s_k = sell_strikes[0] if sell_strikes else 0
                    
                    row = {
                        "Expiration": date_str, "CMP": current_price,
                        "Buy Strike": b_k, "Buy Premium": buy_prem_tot,
                        "Sell Strike": s_k, "Sell Premium": sell_prem_tot,
                        "Margin Required": margin, "Lot Size": lot_size,
                        "Net Cost": net_cost, "Net Max Profit": net_max_profit,
                        "Breakeven": breakeven, "Cost/CMP %": cost_cmp,
                        "Return on Margin %": rom, "Return on Cost %": roc,
                        "Est. Brokerage": brokerage
                    }
                    analysis_rows.append(row)

            except: continue

        if not analysis_rows: return None, None, "Could not build strategies."
        
        df = pd.DataFrame(analysis_rows)
        cols = ["Expiration", "CMP", "Buy Strike", "Buy Premium", "Sell Strike", "Sell Premium", 
                "Margin Required", "Lot Size", "Net Cost", "Net Max Profit", "Breakeven", 
                "Cost/CMP %", "Return on Margin %", "Return on Cost %", "Est. Brokerage"]
        
        # Clean numeric
        for c in df.columns:
             if c != "Expiration": df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0.0)

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

    strategy = st.radio("Strategy", ["Bull Call Spread", "Bear Put Spread", "Bear Call Spread", "Bull Put Spread", "Leveraged Bull Call Spread", "Long Straddle"], horizontal=True)
    
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
                _, df, err = fetch_and_analyze_ticker_hybrid_v25(t, strategy, region_key, source, z_api, z_token, p1, p2, p3, exp_idx)
                if not df.empty:
                    df.insert(0, "Stock", t)
                    dfs.append(df)
                elif err: st.error(f"{t}: {err}")
        
        if dfs:
            full = pd.concat(dfs)
            fmt = {
                "CMP": "${:,.2f}", "Buy Premium": "${:,.2f}", "Sell Premium": "${:,.2f}", 
                "Net Cost": "${:,.2f}", "Net Max Profit": "${:,.0f}", "Breakeven": "${:,.2f}",
                "Margin Required": "${:,.0f}", "Est. Brokerage": "${:,.2f}",
                "Cost/CMP %": "{:.2f}%", "Return on Margin %": "{:.1f}%", "Return on Cost %": "{:.1f}%"
            }
            st.dataframe(full.style.format(fmt, na_rep="N/A"), use_container_width=True, hide_index=True)
            
            st.subheader("Detailed Breakdown")
            for t in tickers:
                 sub = full[full['Stock'] == t]
                 if not sub.empty:
                     with st.expander(f"{t} Details", expanded=False):
                         st.dataframe(sub.style.format(fmt, na_rep="N/A"), use_container_width=True)

if __name__ == "__main__":
    main()
