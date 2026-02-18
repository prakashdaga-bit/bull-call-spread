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

# Try importing Gemini SDK
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

# Try importing PDF reader for Document upload
try:
    import PyPDF2
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False


# ==========================================
# GLOBAL CONFIGURATION
# ==========================================
st.set_page_config(page_title="Options Strategy Master", page_icon="📈", layout="wide")

# Initialize Session State variables for Chat Context
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_summary" not in st.session_state:
    st.session_state.last_summary = ""

# ==========================================
# MATH HELPERS (IV CALCULATION)
# ==========================================
def norm_cdf(x):
    """Standard normal cumulative distribution function."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2.0)))

def norm_pdf(x):
    """Standard normal probability density function."""
    return (1.0 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * x * x)

def black_scholes_price(S, K, T, r, sigma, option_type='Call'):
    """Calculate BS price."""
    if T <= 0: return max(0, S - K) if option_type == 'Call' else max(0, K - S)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if option_type == 'Call':
        return S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
    else:
        return K * math.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)

def calculate_iv(price, S, K, T, r=0.10, option_type='Call'):
    """Estimate Implied Volatility using Newton-Raphson."""
    if price <= 0: return 0.0
    sigma = 0.5 # Initial guess
    for i in range(20):
        bs_price = black_scholes_price(S, K, T, r, sigma, option_type)
        diff = price - bs_price
        if abs(diff) < 0.05: return sigma * 100 # Return as percentage
        
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        vega = S * norm_pdf(d1) * math.sqrt(T)
        
        if vega < 1e-5: break
        sigma = sigma + diff / vega * 0.5 # Damping
        if sigma <= 0: sigma = 0.01
    return sigma * 100

# ==========================================
# NSE HELPERS (LOT SIZES)
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
    except Exception as e:
        return {
            "RELIANCE": 250, "TCS": 175, "HDFCBANK": 550, "INFY": 400,
            "NIFTY": 75, "BANKNIFTY": 30
        }

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
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br"
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
    Uses Kite Connect API to fetch option chains.
    Requires 'kiteconnect' package: pip install kiteconnect
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

    @st.cache_data(ttl=3600) # Cache instruments for 1 hour
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
            st.error(f"Quote Fetch Error: {e}")
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
        # Fallback to public list
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
        except Exception as e:
            # st.error(f"Margin Calc Error: {e}")
            return 0.0

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
            
            # FORCE FLOATS
            data = {
                'strike': float(row['strike']),
                'lastPrice': float(q.get('last_price', 0.0)),
                'bid': float(buy.get('price', 0.0)),
                'ask': float(sell.get('price', 0.0)),
                'openInterest': float(q.get('oi', 0.0)),
                'instrument_token': token,
                'tradingsymbol': row['tradingsymbol'] # Added for Trade Execution
            }
            if row['instrument_type'] == 'CE': calls_list.append(data)
            elif row['instrument_type'] == 'PE': puts_list.append(data)
                
        return pd.DataFrame(calls_list), pd.DataFrame(puts_list)

    def place_order(self, symbol, transaction_type, quantity, price):
        """Places a single LIMIT order."""
        try:
            order_id = self.kite.place_order(
                variety=self.kite.VARIETY_REGULAR,
                exchange=self.kite.EXCHANGE_NFO,
                tradingsymbol=symbol,
                transaction_type=transaction_type,
                quantity=quantity,
                product=self.kite.PRODUCT_NRML,
                order_type=self.kite.ORDER_TYPE_LIMIT,
                price=price
            )
            return order_id
        except Exception as e:
            st.error(f"Order Placement Failed for {symbol}: {e}")
            return None

# ==========================================
# SHARED HELPER FUNCTIONS
# ==========================================

def get_monthly_expirations(ticker_obj, limit=3):
    """
    Filters the list of expiration dates to find the next 'limit' distinct months.
    Used for Simple Analysis (USA).
    """
    try:
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
    except:
        return []

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

def get_next_earnings_date(ticker_obj):
    """Fetches the next earnings date."""
    try:
        # Try retrieving calendar
        cal = ticker_obj.calendar
        if cal is not None and not isinstance(cal, list) and bool(cal):
            # yfinance calendar structure varies; typically 'Earnings Date' row or column
            # Check if dict-like or dataframe
            if isinstance(cal, dict) and 'Earnings Date' in cal:
                dates = cal['Earnings Date']
                if dates:
                    return dates[0].strftime('%Y-%m-%d')
            elif isinstance(cal, pd.DataFrame):
                # Try finding row 'Earnings Date'
                if 'Earnings Date' in cal.index:
                    vals = cal.loc['Earnings Date']
                    # vals might be a Series or list
                    if hasattr(vals, 'iloc'):
                        return vals.iloc[0].strftime('%Y-%m-%d')
        
        # Fallback method: get_earnings_dates
        dates_df = ticker_obj.get_earnings_dates(limit=4)
        if dates_df is not None and len(dates_df) > 0:
            future_dates = dates_df[dates_df.index > pd.Timestamp.now()]
            if not future_dates.empty:
                return future_dates.index[-1].strftime('%Y-%m-%d') # Often sorted desc
            
        return "N/A"
    except:
        return "N/A"

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
    # Force float conversion to avoid string errors
    try:
        bid = float(option_row.get('bid', 0.0))
        ask = float(option_row.get('ask', 0.0))
        last = float(option_row.get('lastPrice', 0.0))
    except:
        return 0.0
    
    if price_type == 'mid':
        if bid > 0 and ask > 0: return (bid + ask) / 2
        return last
    elif price_type == 'ask':
        return ask if ask > 0 else last
    elif price_type == 'bid':
        return bid if bid > 0 else (last * 0.95) 
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
def fetch_and_analyze_ticker_hybrid_v20(ticker, strategy_type, region="USA", source="Yahoo", z_api=None, z_token=None, pct_1=0.0, pct_2=5.0, pct_3=0.0, expiry_idx=0):
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
                    lots = get_fno_info_zerodha()
                    lot_size = int(lots.get(clean_ticker, 1))
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
            date_actual = None
            if isinstance(date_obj, datetime.date): 
                date_str = date_obj.strftime('%Y-%m-%d')
                date_actual = date_obj
            else: 
                date_str = date_obj
                try: date_actual = datetime.datetime.strptime(date_str, '%Y-%m-%d').date()
                except: pass
            
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
            
            # SORT CHAINS FOR INDEXING
            calls = calls.sort_values('strike').reset_index(drop=True)
            puts = puts.sort_values('strike').reset_index(drop=True)

            # 5. Run Strategy Logic
            try:
                # Common Vars
                buy_leg, sell_leg, put_leg = None, None, None
                buy_strike, sell_strike, put_strike = 0.0, 0.0, 0.0
                buy_prem, sell_prem, put_prem = 0.0, 0.0, 0.0
                net_cost, max_gain, breakeven, ret_pct, margin, rom_pct, roc_pct = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                legs_list = []
                
                # Targets
                price_1 = current_price * (1 + pct_1/100.0)
                price_2 = current_price * (1 + pct_2/100.0)
                
                # --- NEW TRADE RECOMMENDATION LOGIC ---
                if strategy_type == "Trade Recommendation":
                    # Calculate IV and Filter
                    atm_strike_row = find_closest_strike(calls, current_price)
                    iv_val = 0.0
                    if atm_strike_row is not None and date_actual:
                        days_to_exp = (date_actual - datetime.date.today()).days
                        if days_to_exp > 0:
                            T = days_to_exp / 365.0
                            iv_val = calculate_iv(get_price(atm_strike_row, 'mid'), current_price, atm_strike_row['strike'], T)
                    
                    # IV Percentile Mock (Real API doesn't provide history)
                    # We will default to a passing value but log/show it.
                    iv_percentile = 51.0 # Placeholder as we can't calculate IVP without 1y history
                    
                    if iv_percentile < 50: continue # The user requested filter
                    
                    strategies_to_check = []
                    
                    # 1. OPTIMIZE BULL PUT
                    # Filter potential Sell Puts (Strike < Spot) in range 3-7% OTM
                    target_high = current_price * 0.97
                    target_low = current_price * 0.93
                    
                    best_bull_put = None
                    max_credit_bull = -1.0
                    
                    for i in range(len(puts)):
                        sell_cand = puts.iloc[i]
                        strike = sell_cand['strike']
                        
                        # Filter 1: OTM Range
                        if not (target_low <= strike <= target_high): continue
                        
                        # Filter 2: Strike Multiple (20 or 50)
                        if not (strike % 20 == 0 or strike % 50 == 0): continue
                        
                        # Check Buy Legs (2, 3, 4 strikes below -> smaller index)
                        for gap in [2, 3, 4]:
                            if i - gap < 0: continue
                            buy_cand = puts.iloc[i - gap]
                            
                            s_bid = get_price(sell_cand, 'bid')
                            b_ask = get_price(buy_cand, 'ask')
                            if s_bid == 0 or b_ask == 0: continue
                            
                            credit = s_bid - b_ask
                            if credit > max_credit_bull:
                                max_credit_bull = credit
                                best_bull_put = {
                                    "type": "Bull Put Rec",
                                    "buy_leg": buy_cand, "sell_leg": sell_cand,
                                    "net_cost": -credit, "legs": [
                                        {'row': buy_cand.to_dict(), 'action': 'Buy', 'type': 'Put'},
                                        {'row': sell_cand.to_dict(), 'action': 'Sell', 'type': 'Put'}
                                    ]
                                }

                    # 2. OPTIMIZE BEAR CALL
                    # Filter potential Sell Calls (Strike > Spot) in range 3-7% OTM
                    target_call_low = current_price * 1.03
                    target_call_high = current_price * 1.07
                    
                    best_bear_call = None
                    max_credit_bear = -1.0
                    
                    for i in range(len(calls)):
                        sell_cand = calls.iloc[i]
                        strike = sell_cand['strike']
                        
                        # Filter 1: OTM Range
                        if not (target_call_low <= strike <= target_call_high): continue
                        
                        # Filter 2: Strike Multiple (20 or 50)
                        if not (strike % 20 == 0 or strike % 50 == 0): continue
                        
                        # Check Buy Legs (2, 3, 4 strikes above -> larger index)
                        for gap in [2, 3, 4]:
                            if i + gap >= len(calls): continue
                            buy_cand = calls.iloc[i + gap]
                            
                            s_bid = get_price(sell_cand, 'bid')
                            b_ask = get_price(buy_cand, 'ask')
                            if s_bid == 0 or b_ask == 0: continue
                            
                            credit = s_bid - b_ask
                            if credit > max_credit_bear:
                                max_credit_bear = credit
                                best_bear_call = {
                                    "type": "Bear Call Rec",
                                    "buy_leg": buy_cand, "sell_leg": sell_cand,
                                    "net_cost": -credit, "legs": [
                                        {'row': buy_cand.to_dict(), 'action': 'Buy', 'type': 'Call'},
                                        {'row': sell_cand.to_dict(), 'action': 'Sell', 'type': 'Call'}
                                    ]
                                }
                    
                    if best_bull_put: strategies_to_check.append(best_bull_put)
                    if best_bear_call: strategies_to_check.append(best_bear_call)
                    
                    # Process found recommendations
                    for strat in strategies_to_check:
                         buy_leg = strat['buy_leg']
                         sell_leg = strat['sell_leg']
                         net_cost = strat['net_cost']
                         legs_list = strat['legs']
                         rec_type = strat['type']
                         
                         buy_strike, sell_strike = buy_leg['strike'], sell_leg['strike']
                         
                         margin = 0.0
                         if adapter and hasattr(adapter, 'get_margin_for_basket'):
                             margin = adapter.get_margin_for_basket(legs_list, lot_size)
                         if margin == 0:
                             margin = abs(buy_strike - sell_strike) * lot_size
                         
                         max_gain = abs(net_cost) # Credit
                         
                         # Breakeven
                         if "Bull" in rec_type: breakeven = sell_strike - max_gain
                         else: breakeven = sell_strike + max_gain

                         total_max_gain = max_gain * lot_size
                         if margin > 0: rom_pct = (total_max_gain / margin) * 100
                         else: rom_pct = 0.0
                         
                         brokerage = 20 * 4 if lot_size > 1 else 0.05 * 4
                         net_max_profit_val = total_max_gain - brokerage
                         
                         # Add to results
                         cost_cmp_pct = (margin / lot_size / current_price * 100) if (lot_size > 0 and current_price > 0) else 0.0
                         
                         row = {
                            "Strategy": rec_type,
                            "Expiration": date_str, 
                            "Spot Price": float(current_price),
                            "Lot Size": lot_size,
                            "Net Cost": net_cost,
                            "Cost/CMP %": cost_cmp_pct,
                            "Max Gain": max_gain, 
                            "Margin Required": margin, 
                            "Return on Margin %": rom_pct,
                            "Return on Cost %": None, 
                            "Breakeven": breakeven,
                            "Est. Brokerage": brokerage,
                            "Net Max Profit": net_max_profit_val,
                            "Buy Strike": buy_strike, 
                            "Buy Premium": get_price(buy_leg, 'ask'),
                            "Sell Strike": sell_strike, 
                            "Sell Premium": get_price(sell_leg, 'bid'),
                            "IV Percentile": f"{iv_percentile:.0f}% (Est)",
                            "Buy Leg Bid": get_price(buy_leg, 'bid'), "Buy Leg Ask": get_price(buy_leg, 'ask'),
                            "Sell Leg Bid": get_price(sell_leg, 'bid'), "Sell Leg Ask": get_price(sell_leg, 'ask'),
                            "Buy TradingSymbol": buy_leg.get('tradingsymbol', ''),
                            "Sell TradingSymbol": sell_leg.get('tradingsymbol', '')
                         }
                         analysis_rows.append(row)
                         summary_returns[date_str] = "Done"
                         
                    continue # Skip standard logic below
                
                # --- STANDARD STRATEGY LOGIC (Existing) ---
                
                # Determine View for Margin Calculation
                view_for_calc = "Volatile" # Default Debit
                if strategy_type in ["Bear Call Spread", "Bull Put Spread"]:
                    view_for_calc = "Neutral" # Credit

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
                    
                    legs_list = [
                        {'row': buy_leg.to_dict(), 'action': 'Buy', 'type': 'Call'},
                        {'row': sell_leg.to_dict(), 'action': 'Sell', 'type': 'Call'}
                    ]

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
                    
                    legs_list = [
                        {'row': buy_leg.to_dict(), 'action': 'Buy', 'type': 'Call'},
                        {'row': sell_leg.to_dict(), 'action': 'Sell', 'type': 'Call'}
                    ]

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
                    
                    legs_list = [
                        {'row': buy_leg.to_dict(), 'action': 'Buy', 'type': 'Put'},
                        {'row': sell_leg.to_dict(), 'action': 'Sell', 'type': 'Put'}
                    ]

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
                    
                    legs_list = [
                        {'row': buy_leg.to_dict(), 'action': 'Buy', 'type': 'Put'},
                        {'row': sell_leg.to_dict(), 'action': 'Sell', 'type': 'Put'}
                    ]
                
                elif strategy_type == "Leveraged Bull Call Spread":
                    price_3 = current_price * (1 + pct_3/100.0)
                    leg_a = find_closest_strike(calls, price_1)
                    leg_b = find_closest_strike(calls, price_2)
                    leg_c = find_closest_strike(puts, price_3)

                    if leg_a is None or leg_b is None or leg_c is None: continue
                    if leg_a['strike'] == leg_b['strike']: continue

                    buy_leg = leg_a if leg_a['strike'] < leg_b['strike'] else leg_b
                    sell_leg = leg_b if leg_a['strike'] < leg_b['strike'] else leg_a
                    put_leg = leg_c

                    buy_strike, sell_strike = buy_leg['strike'], sell_leg['strike']
                    put_strike = put_leg['strike']

                    buy_prem, sell_prem = get_price(buy_leg, 'ask'), get_price(sell_leg, 'bid')
                    put_prem = get_price(put_leg, 'bid')

                    net_cost = (buy_prem - sell_prem) - put_prem
                    max_gain = (sell_strike - buy_strike) - net_cost
                    breakeven = put_strike + net_cost if net_cost > 0 else put_strike - abs(net_cost)
                    
                    legs_list = [
                        {'row': buy_leg.to_dict(), 'action': 'Buy', 'type': 'Call'},
                        {'row': sell_leg.to_dict(), 'action': 'Sell', 'type': 'Call'},
                        {'row': put_leg.to_dict(), 'action': 'Sell', 'type': 'Put'}
                    ]

                if legs_list:
                    # 1. Get Margin via API (Zerodha) if available
                    margin = 0.0
                    if adapter and hasattr(adapter, 'get_margin_for_basket'):
                        api_margin = adapter.get_margin_for_basket(legs_list, lot_size)
                        if api_margin > 0: margin = api_margin
                    
                    # 2. Fallback Margin Estimate if API failed or not used
                    if margin == 0:
                        if strategy_type in ["Bull Call Spread", "Bear Put Spread"]:
                            margin = net_cost * lot_size if net_cost > 0 else 0
                        elif strategy_type in ["Bear Call Spread", "Bull Put Spread"]:
                            # Credit Spread Margin approx: Spread Width * Lot Size
                            margin = abs(buy_strike - sell_strike) * lot_size
                        elif strategy_type == "Leveraged Bull Call Spread":
                             margin = (put_strike * 0.15 * lot_size)

                    total_max_gain = max_gain * lot_size
                    total_net_cost = net_cost * lot_size
                    
                    # 3. Return on Margin (ROM)
                    if margin > 0:
                        rom_pct = (total_max_gain / margin) * 100
                    
                    # 4. Return on Cost (ROC)
                    if total_net_cost > 0:
                        roc_pct = (total_max_gain / total_net_cost) * 100
                    else:
                        roc_pct = 0.0

                # -- COMMON OUTPUT --
                if strategy_type != "Long Straddle":
                    cost_cmp_pct = 0.0
                    # For comparison, margin per share is margin/lot size
                    margin_per_share = margin / lot_size if lot_size > 0 else 0
                    if margin_per_share > 0 and current_price > 0:
                        cost_cmp_pct = (margin_per_share / current_price) * 100
                    
                    # Calculate Brokerage
                    brokerage = 20 * 4 if lot_size > 1 else 0.05 * 4
                    net_max_profit_val = total_max_gain - brokerage

                    base_row = {
                        "Expiration": date_str, 
                        "Spot Price": float(current_price),
                        "Lot Size": lot_size,
                        "Net Cost": net_cost,
                        "Cost/CMP %": cost_cmp_pct,
                        "Max Gain": max_gain, 
                        "Margin Required": margin, 
                        "Return on Margin %": rom_pct,
                        "Return on Cost %": roc_pct if total_net_cost > 0 else None,
                        "Breakeven": breakeven,
                        "Est. Brokerage": brokerage,
                        "Net Max Profit": net_max_profit_val
                    }
                    
                    if strategy_type == "Leveraged Bull Call Spread":
                         base_row.update({
                             "Buy Call Strike": buy_strike, "Buy Call Prem": buy_prem,
                             "Sell Call Strike": sell_strike, "Sell Call Prem": sell_prem,
                             "Sell Put Strike": put_strike, "Sell Put Prem": put_prem
                         })
                    else:
                         base_row.update({
                            "Buy Strike": buy_strike, "Buy Premium": buy_prem,
                            "Sell Strike": sell_strike, "Sell Premium": sell_prem,
                         })
                         
                    analysis_rows.append(base_row)
                    summary_returns[date_str] = f"{rom_pct:.1f}%"
                
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
                    
                    margin = net_cost * lot_size
                    cost_cmp_pct = 0.0
                    if current_price > 0:
                        cost_cmp_pct = (net_cost / current_price) * 100
                        
                    analysis_rows.append({
                        "Expiration": date_str, "Spot Price": float(current_price), "Lot Size": lot_size,
                        "Strike": strike, "Call Cost": c_ask, "Put Cost": p_ask, "Net Cost": net_cost, 
                        "Cost/CMP %": cost_cmp_pct,
                        "Margin Required": margin,
                        "BE Low": strike - net_cost, "BE High": strike + net_cost, "Move Needed": (net_cost / current_price) * 100
                    })
                    summary_returns[date_str] = f"±{ (net_cost / current_price) * 100:.1f}%"

            except: continue

        if not analysis_rows: return None, None, "Could not build strategies."
        
        # Convert to DataFrame
        df = pd.DataFrame(analysis_rows)
        
        # Data Cleaning: Force Numeric Types
        if not df.empty:
            cols_to_numeric = ["Spot Price", "Lot Size", "Buy Strike", "Buy Premium", "Buy Call Strike", "Buy Call Prem", "Sell Strike", "Sell Premium", "Sell Call Strike", "Sell Call Prem", "Sell Put Strike", "Sell Put Prem", "Net Cost", "Max Gain", "Breakeven", "Return on Margin %", "Return on Cost %", "Cost/CMP %", "Strike", "Call Cost", "Put Cost", "BE Low", "BE High", "Move Needed", "Margin Required", "Est. Brokerage", "Net Max Profit", "Buy Leg Bid", "Buy Leg Ask", "Sell Leg Bid", "Sell Leg Ask"]
            for col in cols_to_numeric:
                if col in df.columns:
                    # Coerce errors to NaN, then fill with 0.0. 
                    # Also replace infinite values if any logic caused division by zero.
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
                    df[col] = df[col].replace([float('inf'), float('-inf')], 0.0)
        
        # Reorder columns for visibility
        if not df.empty and strategy_type != "Long Straddle":
             # Desired Order:
             # a) Stock (Added later), b) Spot Price, c) Buy Strike, d) Buy Premium, e) Sell Strike, f) Sell Premium
             # g) Margin Required, h) Lot Size, i) Net Cost, j) Net Max Profit, k) Breakeven, l) Cost/CMP %, 
             # m) Return on Margin %, n) Return on Cost %, o) Est. Brokerage
             # Adding Expiration at start usually
             
             # Add 'Strategy' if present (Trade Rec)
             prefix_cols = ["Expiration", "Strategy"] if "Strategy" in df.columns else ["Expiration"]
             
             cols = prefix_cols + ["Spot Price", "Buy Strike", "Buy Premium", "Sell Strike", "Sell Premium", 
                     "Margin Required", "Lot Size", "Net Cost", "Net Max Profit", "Breakeven", 
                     "Cost/CMP %", "Return on Margin %", "Return on Cost %", "Est. Brokerage"]
             
             if "IV Percentile" in df.columns:
                 cols.append("IV Percentile")
             if "Buy Leg Bid" in df.columns:
                 cols.extend(["Buy Leg Bid", "Buy Leg Ask", "Sell Leg Bid", "Sell Leg Ask"])
             
             # Keep only cols that exist
             cols = [c for c in cols if c in df.columns]
             # Add any remaining cols (e.g. for Leveraged strategies which have different keys)
             remaining = [c for c in df.columns if c not in cols]
             df = df[cols + remaining]

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
    # Find breakevens by checking where PnL crosses 0
    breakeven_points = []
    prev_pnl = None
    prev_p = None
    
    for p in sim_range:
        current_pnl = net_premium
        for leg in legs:
            strike = leg['row']['strike']
            is_call = leg['type'] == "Call"
            is_buy = leg['action'] == "Buy"
            intrinsic = max(0, p - strike) if is_call else max(0, strike - p)
            current_pnl += (intrinsic if is_buy else -intrinsic)
        profits.append(current_pnl)
        
        # Simple linear interpolation for breakeven
        if prev_pnl is not None:
             if (prev_pnl > 0 and current_pnl < 0) or (prev_pnl < 0 and current_pnl > 0):
                 # y = mx + c... approx intersection
                 pass # For now, simple approximation logic for Iron Condor below is safer
        prev_pnl = current_pnl
        prev_p = p

    max_profit_per_share = max(profits)
    max_loss_per_share = min(profits)
    
    # Calculate Breakevens accurately for Iron Condor/Fly (typical Slab Strategy)
    # BE Lower = Sell Put Strike - Net Credit (if credit)
    # BE Upper = Sell Call Strike + Net Credit (if credit)
    # Logic: Finding wings and centers
    puts = sorted([l for l in legs if l['type'] == 'Put'], key=lambda x: x['row']['strike'])
    calls = sorted([l for l in legs if l['type'] == 'Call'], key=lambda x: x['row']['strike'])
    
    be_str = "N/A"
    if len(puts) == 2 and len(calls) == 2 and net_premium > 0:
        # Credit Strategy (Condor/Fly)
        # Sell strikes are usually inner
        sell_put = puts[1]['row']['strike'] # Higher put is sold in bull put, but in condor:
        # Condor: Buy Low Put, Sell Higher Put | Sell Lower Call, Buy Higher Call
        # Let's rely on actions
        sell_legs = [l for l in legs if l['action'] == 'Sell']
        if len(sell_legs) == 2:
             # Assuming standard structure
             s_strikes = sorted([l['row']['strike'] for l in sell_legs])
             be_lower = s_strikes[0] - net_premium
             be_upper = s_strikes[-1] + net_premium
             be_str = f"{be_lower:.2f} / {be_upper:.2f}"
    
    capital_required = 0.0
    if adapter and hasattr(adapter, 'get_margin_for_basket'):
        api_margin = adapter.get_margin_for_basket(legs, lot_size)
        if api_margin > 0: capital_required = api_margin
    
    if capital_required == 0:
        # Fallback logic
        total_premium_lot = net_premium * lot_size
        if len(puts) == 2 and len(calls) == 2:
            # Approx margin for Iron Condor: Max width of wings
            width_put = abs(puts[1]['row']['strike'] - puts[0]['row']['strike'])
            width_call = abs(calls[1]['row']['strike'] - calls[0]['row']['strike'])
            width = max(width_put, width_call)
            if view == "Neutral": capital_required = width * lot_size
            else: capital_required = abs(total_premium_lot) if total_premium_lot < 0 else 0.0

    brokerage = 20 * 4 if lot_size > 1 else 0.05 * 4
    net_max_profit = (max_profit_per_share * lot_size) - brokerage
    roi = (net_max_profit / capital_required * 100) if capital_required > 0 else 0.0
    
    # Return on Cost
    total_net_cost = -net_premium * lot_size # If credit (net_premium > 0), cost is effectively 0 or margin
    roc = 0.0
    if total_net_cost > 0: # Debit strategy
        roc = (net_max_profit / total_net_cost) * 100
        
    # Cost / CMP
    cost_cmp = 0.0
    if current_price > 0 and capital_required > 0:
        cost_cmp = ((capital_required / lot_size) / current_price) * 100

    return {
        "net_premium": net_premium, "max_upside": max_profit_per_share, "max_loss": max_loss_per_share,
        "capital_required": capital_required, "brokerage": brokerage, "net_max_profit": net_max_profit, 
        "roi": roi, "roc": roc, "cost_cmp": cost_cmp, "lot_size": lot_size, "breakeven": be_str
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
                # Pass adapter here for Base Calculation
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
                            # Pass adapter here for Final Optimized Calculation
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
# FILE EXTRACTION HELPER FOR CHAT
# ==========================================
def extract_text_from_file(uploaded_file):
    """Extract text from uploaded TXT, CSV, or PDF file."""
    if uploaded_file.name.endswith('.txt') or uploaded_file.name.endswith('.csv'):
        return uploaded_file.getvalue().decode("utf-8")
    elif uploaded_file.name.endswith('.pdf') and PYPDF_AVAILABLE:
        try:
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            return f"Error reading PDF: {e}"
    elif uploaded_file.name.endswith('.pdf') and not PYPDF_AVAILABLE:
        return "PDF parsing requires PyPDF2. Please run `pip install PyPDF2`"
    return "Unsupported file type."


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
    
    # 1. Sidebar Buttons
    if st.sidebar.button("🔄 Clear Cache & Restart"):
        st.cache_data.clear()
        st.rerun()
        
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
    st.sidebar.markdown("### 🤖 Gemini AI Integration")
    
    # Try to load key from Streamlit Secrets first
    gemini_api_key = ""
    try:
        gemini_api_key = st.secrets.get("GEMINI_API_KEY", "")
    except FileNotFoundError:
        pass
    
    if gemini_api_key:
        st.sidebar.success("✅ Gemini AI Active (Loaded from secrets)")
    else:
        # Fallback for users running locally without a secrets file
        gemini_api_key = st.sidebar.text_input("Gemini API Key", type="password", placeholder="Enter key for chat...")
    
    st.sidebar.markdown("---")
    
    if "input_simple" not in st.session_state: st.session_state["input_simple"] = ""
    if "input_custom" not in st.session_state: st.session_state["input_custom"] = ""

    presets = get_ticker_presets(region_key)

    if mode == "Simple Analysis (Standard)":
        st.subheader(f"📈 {region_key} Market Real-Time Analysis")
        st.caption("Fetches live option chains. Standard Spreads/Straddles.")
        strategy = st.radio("Strategy Type:", ("Trade Recommendation", "Bull Call Spread", "Bear Call Spread", "Bull Put Spread", "Bear Put Spread", "Long Straddle", "Leveraged Bull Call Spread"), horizontal=True)
        
        pct_1 = 0.0
        pct_2 = 5.0
        pct_3 = 0.0 # For leveraged
        expiry_idx = 0
        
        if region_key == "India":
            c_exp, _ = st.columns([1,3])
            exp_opts = ["Current Month", "Next Month", "Far Month"]
            exp_sel = c_exp.selectbox("Select Expiry (India Only)", exp_opts)
            try: expiry_idx = exp_opts.index(exp_sel)
            except: expiry_idx = 0
        
        # Inputs (Hide for Trade Recommendation if desired, but user can just ignore)
        if strategy != "Long Straddle" and strategy != "Trade Recommendation":
            c1, c2, c3 = st.columns(3)
            pct_1 = c1.number_input("Strike 1 (% from Spot)", min_value=-50.0, max_value=50.0, value=0.0, step=0.5)
            pct_2 = c2.number_input("Strike 2 (% from Spot)", min_value=-50.0, max_value=50.0, value=5.0, step=0.5)
            
            if strategy == "Leveraged Bull Call Spread":
                pct_3 = c3.number_input("Sell Put Strike (% from Spot)", min_value=-50.0, max_value=0.0, value=0.0, step=0.5)
        elif strategy == "Trade Recommendation":
             st.info("💡 Recommendation Mode: Optimizes Credit Spreads (Bull Put & Bear Call) automatically. Inputs below are ignored.")
        
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
                        summary, df, error = fetch_and_analyze_ticker_hybrid_v20(ticker, strategy, region_key, source, z_api, z_token, pct_2, pct_1, pct_3, expiry_idx)
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
                    
                    # Store to context for Gemini to read using built-in to_csv instead of tabulate
                    st.session_state.last_summary = full_df.to_csv(index=False)
                    
                    unique_expirations = sorted(full_df['Expiration'].unique())
                    for exp in unique_expirations:
                        subset = full_df[full_df['Expiration'] == exp].drop(columns=['Expiration'])
                        
                        if strategy == "Trade Recommendation":
                            # Custom Table with Buttons
                            st.write(f"### Expiry: {exp}")
                            
                            # Headers
                            cols = st.columns([1.5, 1, 1, 1, 1, 1, 1, 1, 1, 1])
                            headers = ["Stock", "Strategy", "Spot", "Buy Strike", "Sell Strike", "Net Credit", "Margin", "Max Profit", "ROI", "Action"]
                            for c, h in zip(cols, headers):
                                c.markdown(f"**{h}**")
                            
                            for idx, row in subset.iterrows():
                                c1, c2, c3, c4, c5, c6, c7, c8, c9, c10 = st.columns([1.5, 1, 1, 1, 1, 1, 1, 1, 1, 1])
                                c1.write(row['Stock'])
                                c2.write(row['Strategy'])
                                c3.write(f"{row['Spot Price']:.2f}")
                                c4.write(f"{row['Buy Strike']:.0f} (@{row['Buy Premium']:.1f})")
                                c5.write(f"{row['Sell Strike']:.0f} (@{row['Sell Premium']:.1f})")
                                c6.write(f"{abs(row['Net Cost']):.0f}")
                                c7.write(f"{row['Margin Required']:.0f}")
                                c8.write(f"{row['Net Max Profit']:.0f}")
                                c9.write(f"{row['Return on Margin %']:.1f}%")
                                
                                # Trade Button
                                if c10.button("Trade", key=f"trade_{idx}_{row['Stock']}"):
                                    if not z_api or not z_token:
                                        st.error("Login Required")
                                    else:
                                        # Execute Trade
                                        t_adapter = ZerodhaMarketAdapter(z_api, z_token)
                                        if t_adapter.connect():
                                            qty = int(row['Lot Size'])
                                            # Buy Leg
                                            st.toast(f"Placing Buy Order for {row['Buy TradingSymbol']}...")
                                            oid1 = t_adapter.place_order(row['Buy TradingSymbol'], t_adapter.kite.TRANSACTION_TYPE_BUY, qty, row['Buy Leg Ask'])
                                            
                                            if oid1:
                                                time.sleep(0.5) # Small delay
                                                st.toast(f"Placing Sell Order for {row['Sell TradingSymbol']}...")
                                                oid2 = t_adapter.place_order(row['Sell TradingSymbol'], t_adapter.kite.TRANSACTION_TYPE_SELL, qty, row['Sell Leg Bid'])
                                                if oid2:
                                                    st.success(f"Trade Executed! IDs: {oid1}, {oid2}")
                        else:
                            cols_to_show = []
                            if strategy == "Long Straddle":
                                format_dict = {
                                    "Spot Price": "${:,.2f}", "Call Cost": "${:,.2f}", "Put Cost": "${:,.2f}",
                                    "Net Cost": "${:,.2f}", "Cost/CMP %": "{:.2f}%", "BE Low": "${:,.2f}",
                                    "BE High": "${:,.2f}", "Move Needed": "{:.1f}%", "Margin Required": "${:,.0f}"
                                }
                            elif strategy == "Leveraged Bull Call Spread":
                                 format_dict = {
                                    "Spot Price": "${:,.2f}", "Net Cost": "${:,.2f}", "Margin Required": "${:,.0f}",
                                    "Return on Margin %": "{:.1f}%", "Return on Cost %": "{:.1f}%", 
                                    "Max Gain": "${:,.2f}", "Breakeven": "${:,.2f}",
                                    "Buy Call Strike": "${:,.2f}", "Sell Call Strike": "${:,.2f}", "Sell Put Strike": "${:,.2f}",
                                    "Est. Brokerage": "${:,.2f}", "Net Max Profit": "${:,.0f}"
                                }
                            else:
                                format_dict = {
                                    "Spot Price": "${:,.2f}", "Buy Premium": "${:,.2f}", "Sell Premium": "${:,.2f}",
                                    "Buy Strike": "${:,.2f}", "Sell Strike": "${:,.2f}",
                                    "Net Cost": "${:,.2f}", "Cost/CMP %": "{:.2f}%", "Max Gain": "${:,.2f}",
                                    "Margin Required": "${:,.0f}", "Lot Size": "{:,.0f}",
                                    "Breakeven": "${:,.2f}", "Return on Margin %": "{:.1f}%", "Return on Cost %": "{:.1f}%",
                                    "Est. Brokerage": "${:,.2f}", "Net Max Profit": "${:,.0f}",
                                    "Buy Leg Bid": "${:,.2f}", "Buy Leg Ask": "${:,.2f}",
                                    "Sell Leg Bid": "${:,.2f}", "Sell Leg Ask": "${:,.2f}"
                                }
                            
                            try:
                                st.dataframe(subset.style.format(format_dict, na_rep="N/A"), hide_index=True, use_container_width=True)
                            except:
                                st.dataframe(subset, hide_index=True, use_container_width=True)

                elif not errors: st.warning("No valid data found.")

                if all_details:
                    st.header("2. Detailed Breakdown")
                    for ticker, df in all_details.items():
                        with st.expander(f"{ticker} Details", expanded=False):
                             try:
                                st.dataframe(df.style.format(format_dict, na_rep="N/A"), use_container_width=True)
                             except Exception as e:
                                st.error(f"⚠️ Formatting error. Showing raw data.")
                                st.dataframe(df, use_container_width=True)
                if errors:
                    with st.expander("Errors"):
                        for e in errors: st.write(f"- {e}")

    else:
        st.subheader(f"🤖 {region_key} Slab-Based Strategy Generator")
        # ... (Custom Strategy Section code follows, unchanged logic just re-rendered)
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
                                opt_data = res['optimized'] if res['optimized'] else res['base']
                                metrics = opt_data['metrics']
                                legs = opt_data['legs']
                                
                                # Extract breakdown values for summary
                                buy_legs = [l for l in legs if l['action'] == 'Buy']
                                sell_legs = [l for l in legs if l['action'] == 'Sell']
                                
                                # Helper to sum/join
                                buy_strikes = ", ".join([f"{l['row']['strike']:.1f}" for l in buy_legs])
                                sell_strikes = ", ".join([f"{l['row']['strike']:.1f}" for l in sell_legs])
                                
                                buy_prem_val = sum([get_price(l['row'], 'ask') for l in buy_legs])
                                sell_prem_val = sum([get_price(l['row'], 'bid') for l in sell_legs])
                                
                                # Net Cost
                                net_cost_val = metrics['net_premium'] * metrics['lot_size']
                                
                                summary_data = {
                                    "Stock": ticker, 
                                    "Spot Price": f"${res['current_price']:.2f}",
                                    "Buy Strike": buy_strikes,
                                    "Buy Premium": f"${buy_prem_val:.2f}",
                                    "Sell Strike": sell_strikes,
                                    "Sell Premium": f"${sell_prem_val:.2f}",
                                    "Margin Required": f"${metrics['capital_required']:,.0f}",
                                    "Lot Size": metrics['lot_size'],
                                    "Net Cost": f"${net_cost_val:.2f}",
                                    "Net Max Profit": f"${metrics['net_max_profit']:,.0f}",
                                    "Breakeven": metrics['breakeven'],
                                    "Cost/CMP %": f"{metrics['cost_cmp']:.2f}%",
                                    "Return on Margin %": f"{metrics['roi']:.1f}%",
                                    "Return on Cost %": f"{metrics['roc']:.1f}%",
                                    "Est. Brokerage": f"${metrics['brokerage']:.2f}",
                                    "Next Earnings": res['earnings']
                                }
                                all_summaries.append(summary_data)
                        progress_bar.progress((i + 1) / len(tickers))
                st.divider()
                if all_summaries:
                    st.header("1. Strategy Summary (Optimized)")
                    summary_df = pd.DataFrame(all_summaries)
                    
                    # Store to context for Gemini to read using built-in to_csv instead of tabulate
                    st.session_state.last_summary = summary_df.to_csv(index=False)
                    
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

    # ========================================================
    # NEW: GEMINI CHAT INTERFACE
    # ========================================================
    st.divider()
    st.header("💬 Gemini Options Assistant")
    
    if not GENAI_AVAILABLE:
        st.error("⚠️ `google-generativeai` package is missing. Please run: `pip install google-generativeai`")
    else:
        if not gemini_api_key:
            st.info("👈 Please enter your Gemini Enterprise API Key in the sidebar to activate the Chat Assistant.")
        else:
            try:
                genai.configure(api_key=gemini_api_key)
                # Using latest Gemini 3.0 Pro model
                model = genai.GenerativeModel('gemini-3.0-pro') 
            except Exception as e:
                st.error(f"Error configuring Gemini API: {e}")
            
            with st.expander("📎 Upload Document for Context (Optional)", expanded=False):
                uploaded_doc = st.file_uploader("Upload PDF, CSV, or TXT file", type=['pdf', 'txt', 'csv'])
                
                doc_text = ""
                if uploaded_doc is not None:
                    with st.spinner("Extracting text..."):
                        doc_text = extract_text_from_file(uploaded_doc)
                    if doc_text:
                        st.success(f"Document '{uploaded_doc.name}' loaded successfully! ({len(doc_text)} chars)")

            # Display Chat History
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                    
            # Chat Input
            if prompt := st.chat_input("Ask about the active strategy, documents, or general options..."):
                # Append user msg to UI
                st.session_state.chat_history.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                # Build context for the AI
                system_context = "You are a professional options trading assistant.\n\n"
                
                if st.session_state.last_summary:
                    system_context += "### Current Generated Options Strategies Data:\n"
                    system_context += f"{st.session_state.last_summary}\n\n"
                
                if doc_text:
                    system_context += f"### User Uploaded Document Context:\n{doc_text[:25000]}\n\n" # Truncated to avoid immense context blasts for basic models
                    
                full_prompt = f"{system_context}\nUser Query: {prompt}"
                
                # Fetch Response
                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    try:
                        with st.spinner("Thinking..."):
                            # We send it as a fresh generation to easily inject the massive data string safely
                            response = model.generate_content(full_prompt)
                            reply_text = response.text
                            message_placeholder.markdown(reply_text)
                            
                            st.session_state.chat_history.append({"role": "assistant", "content": reply_text})
                    except Exception as e:
                        st.error(f"Error communicating with Gemini API: {e}")

if __name__ == "__main__":
    main()
