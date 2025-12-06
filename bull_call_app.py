import math
import streamlit as st
from dataclasses import dataclass
from typing import List, Optional, Dict, Union
import random

# ==========================================
# 1. USER CONFIGURATION (Data Classes)
# ==========================================

@dataclass
class IronCondorConfig:
    """Configuration for Iron Condor (Neutral / Short Volatility)"""
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
    """Configuration for Bull Call Spread (Bullish / Long Volatility)"""
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
    """Configuration for Long Straddle (Explosive Move / Long Volatility)"""
    min_daily_volume: int
    min_open_interest: int
    max_iv_rank: float  # Prefer buying when Vol is low (cheap)
    target_dte_min: int
    target_dte_max: int
    delta_tolerance: float # Tolerance for finding ATM strikes
    max_cost_pct: float    # Max cost as % of stock price (e.g. don't pay > 10% of stock price)

# ==========================================
# 2. DATA MODELS
# ==========================================
@dataclass
class OptionContract:
    symbol: str
    strike: float
    option_type: str # 'call' or 'put'
    expiry_days: int
    bid: float
    ask: float
    delta: float
    open_interest: int
    
    @property
    def mid_price(self):
        return round((self.bid + self.ask) / 2, 2)
    
    @property
    def spread(self):
        return round(self.ask - self.bid, 2)

@dataclass
class StockData:
    symbol: str
    price: float
    avg_daily_volume: int
    iv_rank: float
    days_to_earnings: int

# ==========================================
# 3. STRATEGY ENGINES
# ==========================================

class StrategyBase:
    def __init__(self):
        self.logs = []

    def log(self, msg):
        self.logs.append(msg)

class IronCondorStrategy(StrategyBase):
    def __init__(self, config: IronCondorConfig):
        super().__init__()
        self.config = config
        self.name = "Iron Condor"

    def check_filters(self, stock: StockData) -> bool:
        if stock.avg_daily_volume < self.config.min_daily_volume:
            self.log(f"❌ FAIL: Low Volume {stock.avg_daily_volume:,}")
            return False
        if stock.iv_rank < self.config.min_iv_rank:
            self.log(f"❌ FAIL: IV Rank {stock.iv_rank} < {self.config.min_iv_rank} (Need High Volatility)")
            return False
        if self.config.avoid_earnings and stock.days_to_earnings < self.config.target_dte_max:
            self.log(f"❌ FAIL: Earnings in {stock.days_to_earnings} days (Risk of binary event)")
            return False
        
        self.log("✅ Phase 1: Universe Filters Passed")
        return True

    def find_strikes(self, stock: StockData, chain: List[OptionContract]):
        valid_expiry = [
            opt for opt in chain 
            if self.config.target_dte_min <= opt.expiry_days <= self.config.target_dte_max
        ]
        
        if not valid_expiry:
            self.log("❌ FAIL: No options found in DTE range")
            return None

        # Dynamic Width: $5 for stocks < $200, $10 for stocks > $200
        wing_width = 10.0 if stock.price > 200 else 5.0
        
        def get_best_leg(options, target_delta, type_):
            candidates = [
                opt for opt in options if opt.option_type == type_ 
                and opt.open_interest >= self.config.min_open_interest
            ]
            if not candidates: return None
            
            # Find closest delta
            best = min(candidates, key=lambda x: abs(abs(x.delta) - abs(target_delta)))
            
            # Check tolerance
            if abs(abs(best.delta) - abs(target_delta)) > self.config.delta_tolerance:
                return None
            return best

        short_put = get_best_leg(valid_expiry, -self.config.target_delta, 'put')
        short_call = get_best_leg(valid_expiry, self.config.target_delta, 'call')

        if not short_put or not short_call:
            self.log(f"❌ FAIL: Could not find Short legs near {self.config.target_delta} Delta")
            return None

        target_long_put = short_put.strike - wing_width
        target_long_call = short_call.strike + wing_width

        long_put = next((o for o in valid_expiry if o.option_type == 'put' and o.strike == target_long_put), None)
        long_call = next((o for o in valid_expiry if o.option_type == 'call' and o.strike == target_long_call), None)

        if not long_put or not long_call:
            self.log(f"❌ FAIL: Could not find Wings at width ${wing_width}")
            return None

        self.log(f"✅ Phase 2: Geometry Found (Width ${wing_width})")
        return {
            "short_put": short_put, "long_put": long_put,
            "short_call": short_call, "long_call": long_call,
            "width": wing_width
        }

    def validate_trade(self, legs):
        sp, lp = legs['short_put'], legs['long_put']
        sc, lc = legs['short_call'], legs['long_call']
        width = legs['width']

        credit = (sp.mid_price + sc.mid_price) - (lp.mid_price + lc.mid_price)
        credit = round(credit, 2)
        max_risk = width - credit

        # Constraint 1: Credit > 1/3 Width
        if (credit / width) < self.config.min_credit_to_width_ratio:
            self.log(f"⛔ ABORT: Credit ${credit} is only {credit/width:.1%} of width (Target {self.config.min_credit_to_width_ratio:.0%})")
            return None

        # Constraint 2: POP
        pop = 1.0 - (abs(sc.delta) + abs(sp.delta))
        if pop < self.config.min_pop:
            self.log(f"⛔ ABORT: POP {pop:.1%} is below target {self.config.min_pop:.1%}")
            return None

        # Constraint 3: Return on Risk
        ror = credit / max_risk if max_risk > 0 else 0
        if ror < self.config.min_ror:
            self.log(f"⛔ ABORT: ROR {ror:.1%} is below target {self.config.min_ror:.1%}")
            return None

        return {
            "type": "Iron Condor",
            "strikes": f"P:{lp.strike}/{sp.strike} | C:{sc.strike}/{lc.strike}",
            "credit": credit, 
            "max_risk": round(max_risk, 2),
            "pop": pop, 
            "ror": ror
        }


class BullCallSpreadStrategy(StrategyBase):
    def __init__(self, config: BullCallSpreadConfig):
        super().__init__()
        self.config = config
        self.name = "Bull Call Spread"

    def check_filters(self, stock: StockData) -> bool:
        if stock.avg_daily_volume < self.config.min_daily_volume:
            self.log(f"❌ FAIL: Low Volume {stock.avg_daily_volume:,}")
            return False
        
        if stock.iv_rank > self.config.max_iv_rank:
            self.log(f"❌ FAIL: IV Rank {stock.iv_rank} is too high for buying spreads.")
            return False
            
        self.log("✅ Phase 1: Universe Filters Passed")
        return True

    def find_strikes(self, stock: StockData, chain: List[OptionContract]):
        valid_expiry = [
            opt for opt in chain 
            if self.config.target_dte_min <= opt.expiry_days <= self.config.target_dte_max
        ]
        
        def get_call_by_delta(delta_target):
            candidates = [o for o in valid_expiry if o.option_type == 'call' and o.open_interest >= self.config.min_open_interest]
            if not candidates: return None
            best = min(candidates, key=lambda x: abs(x.delta - delta_target))
            if abs(best.delta - delta_target) > self.config.delta_tolerance:
                return None
            return best

        long_call = get_call_by_delta(self.config.long_call_delta)
        short_call = get_call_by_delta(self.config.short_call_delta)

        if not long_call or not short_call:
            self.log("❌ FAIL: Could not find specific delta strikes")
            return None
        
        if long_call.strike >= short_call.strike:
            self.log("❌ FAIL: Inverted Strikes (Delta skew issue)")
            return None

        self.log("✅ Phase 2: Geometry Found")
        return {"long_call": long_call, "short_call": short_call}

    def validate_trade(self, legs):
        lc = legs['long_call']
        sc = legs['short_call']
        width = sc.strike - lc.strike

        debit_paid = lc.mid_price - sc.mid_price
        debit_paid = round(debit_paid, 2)
        max_profit = width - debit_paid
        
        if (debit_paid / width) > self.config.max_debit_to_width_ratio:
            self.log(f"⛔ ABORT: Debit ${debit_paid} is > {self.config.max_debit_to_width_ratio:.0%} of width")
            return None

        risk_reward = max_profit / debit_paid if debit_paid > 0 else 0
        if risk_reward < self.config.min_profit_to_risk_ratio:
            self.log(f"⛔ ABORT: Risk/Reward {risk_reward:.2f} < {self.config.min_profit_to_risk_ratio}")
            return None

        pop_estimate = lc.delta 

        return {
            "type": "Bull Call Spread",
            "strikes": f"Buy {lc.strike} / Sell {sc.strike} Call",
            "debit": debit_paid,
            "max_profit": round(max_profit, 2),
            "pop": pop_estimate,
            "risk_reward": risk_reward
        }

class LongStraddleStrategy(StrategyBase):
    def __init__(self, config: LongStraddleConfig):
        super().__init__()
        self.config = config
        self.name = "Long Straddle"

    def check_filters(self, stock: StockData) -> bool:
        if stock.avg_daily_volume < self.config.min_daily_volume:
            self.log(f"❌ FAIL: Low Volume {stock.avg_daily_volume:,}")
            return False
        
        # We want to buy when options are CHEAP (Low IV) before a move
        if stock.iv_rank > self.config.max_iv_rank:
            self.log(f"❌ FAIL: IV Rank {stock.iv_rank} is too high (Options expensive)")
            return False
            
        self.log("✅ Phase 1: Universe Filters Passed")
        return True

    def find_strikes(self, stock: StockData, chain: List[OptionContract]):
        valid_expiry = [
            opt for opt in chain 
            if self.config.target_dte_min <= opt.expiry_days <= self.config.target_dte_max
        ]
        
        # Find ATM options (closest to stock price, or Delta ~0.50)
        # Using simple distance to spot price here
        
        candidates = [o for o in valid_expiry if o.open_interest >= self.config.min_open_interest]
        if not candidates:
            self.log("❌ FAIL: No liquid options found")
            return None
            
        # Sort by distance from spot price
        closest_strike = min(candidates, key=lambda x: abs(x.strike - stock.price)).strike
        
        # Get the Call and Put at this strike
        atm_call = next((o for o in candidates if o.strike == closest_strike and o.option_type == 'call'), None)
        atm_put = next((o for o in candidates if o.strike == closest_strike and o.option_type == 'put'), None)
        
        if not atm_call or not atm_put:
             self.log(f"❌ FAIL: Could not find matching Call/Put at strike {closest_strike}")
             return None
             
        self.log(f"✅ Phase 2: Geometry Found (ATM Strike {closest_strike})")
        return {"long_call": atm_call, "long_put": atm_put, "stock_price": stock.price}

    def validate_trade(self, legs):
        lc = legs['long_call']
        lp = legs['long_put']
        stock_price = legs['stock_price']
        
        # Cost = Call Price + Put Price
        debit = lc.mid_price + lp.mid_price
        debit = round(debit, 2)
        
        # Breakevens
        be_upper = lc.strike + debit
        be_lower = lp.strike - debit
        
        # Constraint: Cost Efficiency
        # If the straddle costs 20% of the stock price, it's very hard to win.
        cost_pct = debit / stock_price
        if cost_pct > self.config.max_cost_pct:
            self.log(f"⛔ ABORT: Cost ${debit} is {cost_pct:.1%} of Stock Price (Max {self.config.max_cost_pct:.1%})")
            return None
            
        return {
            "type": "Long Straddle",
            "strikes": f"Buy {lc.strike} Call & Put",
            "debit": debit,
            "max_profit": "Unlimited",
            "breakevens": f"Below ${be_lower:.2f} / Above ${be_upper:.2f}",
            "cost_pct": f"{cost_pct:.1%}"
        }

# ==========================================
# 4. STREAMLIT UI IMPLEMENTATION
# ==========================================

def generate_mock_chain(ticker, spot_price, expiry_days, is_high_iv=False):
    """Helper to generate a fake option chain for testing logic."""
    chain = []
    # Volatility scalar (Higher IV = Higher prices)
    vol_factor = 0.05 if is_high_iv else 0.02
    
    # Generate strikes around spot
    strikes = [spot_price + i*5 for i in range(-10, 11)] # +/- $50 range
    
    for strike in strikes:
        # Crude Delta Approximation
        moneyness = (spot_price - strike) / spot_price
        
        # Call Delta
        call_delta = 0.5 + (moneyness * 2)
        call_delta = max(0.01, min(0.99, call_delta))
        
        # Put Delta
        put_delta = call_delta - 1.0
        
        # Price approximation
        dist = abs(spot_price - strike)
        base_premium = (spot_price * vol_factor) - (dist * 0.1)
        base_premium = max(0.05, base_premium)
        
        # Calls
        chain.append(OptionContract(
            symbol=ticker, strike=strike, option_type='call', expiry_days=expiry_days,
            bid=round(base_premium, 2), ask=round(base_premium*1.1, 2),
            delta=round(call_delta, 2), open_interest=random.randint(100, 10000)
        ))
        
        # Puts
        chain.append(OptionContract(
            symbol=ticker, strike=strike, option_type='put', expiry_days=expiry_days,
            bid=round(base_premium, 2), ask=round(base_premium*1.1, 2),
            delta=round(put_delta, 2), open_interest=random.randint(100, 10000)
        ))
    return chain

def main():
    st.set_page_config(page_title="Algo Options Strategy", layout="wide")
    st.title("🛡️ Automated Options Strategy Analyzer")
    st.markdown("Use this tool to validate if a trade meets your strict **Algorithmic Criteria**.")

    # --- SIDEBAR: STRATEGY CONFIG ---
    st.sidebar.header("Strategy Settings")
    strategy_choice = st.sidebar.selectbox("Select Strategy", ["Iron Condor", "Bull Call Spread", "Long Straddle"])

    config = None
    if strategy_choice == "Iron Condor":
        st.sidebar.subheader("Iron Condor Constraints")
        min_iv = st.sidebar.number_input("Min IV Rank", value=50, step=5)
        target_delta = st.sidebar.slider("Target Delta (Short Legs)", 0.10, 0.30, 0.16, 0.01)
        min_credit_ratio = st.sidebar.slider("Min Credit/Width Ratio", 0.20, 0.50, 0.33, 0.01)
        min_pop = st.sidebar.slider("Min Prob. of Profit", 0.50, 0.90, 0.60, 0.05)
        
        config = IronCondorConfig(
            min_daily_volume=1000000, min_open_interest=500, min_iv_rank=min_iv,
            avoid_earnings=True, target_dte_min=30, target_dte_max=45,
            target_delta=target_delta, delta_tolerance=0.04,
            min_credit_to_width_ratio=min_credit_ratio, min_pop=min_pop, min_ror=0.15
        )
    elif strategy_choice == "Bull Call Spread":
        st.sidebar.subheader("Bull Call Constraints")
        max_iv = st.sidebar.number_input("Max IV Rank (Buy Low Vol)", value=50, step=5)
        long_delta = st.sidebar.slider("Long Call Delta", 0.40, 0.80, 0.55, 0.05)
        short_delta = st.sidebar.slider("Short Call Delta", 0.10, 0.40, 0.30, 0.05)
        
        config = BullCallSpreadConfig(
            min_daily_volume=1000000, min_open_interest=500, max_iv_rank=max_iv,
            target_dte_min=45, target_dte_max=90, long_call_delta=long_delta,
            short_call_delta=short_delta, delta_tolerance=0.05,
            max_debit_to_width_ratio=0.50, min_profit_to_risk_ratio=1.0
        )
    else:
        st.sidebar.subheader("Long Straddle Constraints")
        max_iv = st.sidebar.number_input("Max IV Rank (Cheap Premium)", value=40, step=5)
        max_cost = st.sidebar.slider("Max Cost (% of Stock Price)", 0.02, 0.15, 0.08, 0.01)
        
        config = LongStraddleConfig(
            min_daily_volume=1000000, min_open_interest=500, max_iv_rank=max_iv,
            target_dte_min=14, target_dte_max=60, delta_tolerance=0.10,
            max_cost_pct=max_cost
        )

    # --- MAIN AREA: TICKER INPUT ---
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        ticker = st.text_input("Ticker Symbol", "TSLA").upper()
    with col2:
        price = st.number_input("Current Price", value=250.0)
    with col3:
        iv_rank = st.number_input("Current IV Rank", value=65.0)
    with col4:
        earnings_days = st.number_input("Days to Earnings", value=60)

    # --- GENERATE DATA & RUN ---
    if st.button("Run Algorithmic Analysis", type="primary"):
        # 1. Create Data Objects
        stock_data = StockData(ticker, price, 50000000, iv_rank, earnings_days)
        
        # 2. Mock Option Chain (Since we don't have an API here)
        # If IV is high, prices are higher.
        is_high_iv = iv_rank > 50
        expiry_days = 35 if strategy_choice == "Iron Condor" else 60
        chain = generate_mock_chain(ticker, price, expiry_days, is_high_iv)
        
        st.info(f"Generated {len(chain)} mock contracts for {ticker} (Expiry: {expiry_days} days)")

        # 3. Initialize Strategy
        if strategy_choice == "Iron Condor":
            algo = IronCondorStrategy(config)
        elif strategy_choice == "Bull Call Spread":
            algo = BullCallSpreadStrategy(config)
        else:
            algo = LongStraddleStrategy(config)

        # 4. Execute Logic
        col_log, col_res = st.columns([1, 1])
        
        with col_log:
            st.subheader("Algorithmic Decision Log")
            
            # Phase 1
            passed_filters = algo.check_filters(stock_data)
            
            legs = None
            if passed_filters:
                # Phase 2
                legs = algo.find_strikes(stock_data, chain)
            
            result = None
            if legs:
                # Phase 3
                result = algo.validate_trade(legs)
            
            # Print Logs
            for log in algo.logs:
                if "FAIL" in log or "ABORT" in log:
                    st.error(log)
                else:
                    st.success(log)

        with col_res:
            st.subheader("Final Recommendation")
            if result:
                st.balloons()
                st.success(f"✅ TRADE APPROVED: {result['type']}")
                st.markdown(f"**Structure:** `{result['strikes']}`")
                
                m1, m2, m3 = st.columns(3)
                if 'credit' in result:
                    m1.metric("Credit Received", f"${result['credit']}")
                    m2.metric("Max Risk", f"${result['max_risk']}")
                    m3.metric("POP", f"{result['pop']:.1%}")
                elif 'breakevens' in result:
                    m1.metric("Total Debit", f"${result['debit']}")
                    m2.metric("Cost %", result['cost_pct'])
                    m3.metric("Breakevens", "See Below")
                    st.info(f"Breakevens: {result['breakevens']}")
                else:
                    m1.metric("Debit Paid", f"${result['debit']}")
                    m2.metric("Max Profit", f"${result['max_profit']}")
                    m3.metric("Risk/Reward", f"{result['risk_reward']}")

            elif not passed_filters:
                st.warning("🚫 Trade Rejected at Phase 1 (Filters)")
            elif not legs:
                st.warning("🚫 Trade Rejected at Phase 2 (Geometry/Strikes)")
            else:
                st.warning("🚫 Trade Rejected at Phase 3 (Risk Constraints)")

if __name__ == "__main__":
    main()
