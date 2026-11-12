"""Training-code task bank: trading / market-making / crypto coding tasks with reference solutions
and inline execution checkers. DISJOINT from eval/eval_bots.py (which stays held-out for scoring).

Each task: id, prompt, ref (reference solution source), check (source defining check(m)->(ok,why)).
build_code_sft.py verifies every ref against its check (and that none share an id with eval_bots)
before emitting training examples, so every code target shipped is known-correct.
"""
import textwrap

TASKS = [
    {
        "id": "tp_sl_prices",
        "prompt": "Write `tp_sl(entry, tp_pct, sl_pct, side='long')` returning a dict "
                  "{'tp': price, 'sl': price}. For a long, tp is entry above by tp_pct percent and "
                  "sl below by sl_pct; for a short, mirrored. One ```python block, code only.",
        "ref": textwrap.dedent("""
            def tp_sl(entry, tp_pct, sl_pct, side='long'):
                if side == 'long':
                    return {'tp': entry*(1+tp_pct/100), 'sl': entry*(1-sl_pct/100)}
                return {'tp': entry*(1-tp_pct/100), 'sl': entry*(1+sl_pct/100)}
        """),
        "check": textwrap.dedent("""
            def check(m):
                r=m.tp_sl(100,10,5,'long')
                if abs(r['tp']-110)>1e-9 or abs(r['sl']-95)>1e-9: return False,f"long {r}"
                r=m.tp_sl(100,10,5,'short')
                if abs(r['tp']-90)>1e-9 or abs(r['sl']-105)>1e-9: return False,f"short {r}"
                return True,"ok"
        """),
    },
    {
        "id": "rsi",
        "prompt": "Write `rsi(prices, period=14)` returning the Wilder RSI of the last point as a "
                  "float, using simple averages of gains and losses over `period`. If there are no "
                  "losses return 100.0. One ```python block, code only.",
        "ref": textwrap.dedent("""
            def rsi(prices, period=14):
                gains=losses=0.0
                for i in range(1, period+1):
                    d=prices[-i]-prices[-i-1]
                    if d>=0: gains+=d
                    else: losses-=d
                avg_g=gains/period; avg_l=losses/period
                if avg_l==0: return 100.0
                rs=avg_g/avg_l
                return 100-100/(1+rs)
        """),
        "check": textwrap.dedent("""
            def check(m):
                up=list(range(1,30))
                if abs(m.rsi(up,14)-100.0)>1e-6: return False,"all-up should be 100"
                flatish=[100,101,100,101,100,101,100,101,100,101,100,101,100,101,100]
                v=m.rsi(flatish,14)
                if not (0<=v<=100): return False,f"out of range {v}"
                return True,"ok"
        """),
    },
    {
        "id": "vwap",
        "prompt": "Write `vwap(trades)` where trades is a list of (price, size) tuples; return the "
                  "volume-weighted average price as a float. One ```python block, code only.",
        "ref": textwrap.dedent("""
            def vwap(trades):
                pv=sum(p*s for p,s in trades); v=sum(s for _,s in trades)
                return pv/v if v else 0.0
        """),
        "check": textwrap.dedent("""
            def check(m):
                if abs(m.vwap([(100,1),(102,1)])-101.0)>1e-9: return False,"case1"
                if abs(m.vwap([(10,3),(20,1)])-12.5)>1e-9: return False,"case2"
                return True,"ok"
        """),
    },
    {
        "id": "max_drawdown",
        "prompt": "Write `max_drawdown(equity)` taking a list of equity values and returning the "
                  "maximum drawdown as a positive fraction (e.g. 0.2 for a 20% drop from a peak). "
                  "One ```python block, code only.",
        "ref": textwrap.dedent("""
            def max_drawdown(equity):
                peak=equity[0]; mdd=0.0
                for x in equity:
                    peak=max(peak,x)
                    dd=(peak-x)/peak if peak else 0.0
                    mdd=max(mdd,dd)
                return mdd
        """),
        "check": textwrap.dedent("""
            def check(m):
                if abs(m.max_drawdown([100,120,90,110])-0.25)>1e-9: return False,"case1"
                if abs(m.max_drawdown([100,100,100]))>1e-12: return False,"flat should be 0"
                return True,"ok"
        """),
    },
    {
        "id": "slippage_price",
        "prompt": "Write `fill_price(mid, side, slippage_bps)` returning the expected fill price when "
                  "crossing the spread: a buy fills above mid by slippage_bps, a sell below. "
                  "side is 'buy' or 'sell'. One ```python block, code only.",
        "ref": textwrap.dedent("""
            def fill_price(mid, side, slippage_bps):
                adj=mid*slippage_bps/10000
                return mid+adj if side=='buy' else mid-adj
        """),
        "check": textwrap.dedent("""
            def check(m):
                if abs(m.fill_price(100,'buy',10)-100.1)>1e-9: return False,"buy"
                if abs(m.fill_price(100,'sell',10)-99.9)>1e-9: return False,"sell"
                return True,"ok"
        """),
    },
    {
        "id": "rebalance",
        "prompt": "Write `rebalance(holdings_usd, targets)` where holdings_usd is a dict "
                  "{asset: usd_value} and targets is a dict {asset: weight} summing to 1. Return a "
                  "dict {asset: usd_delta} (positive = buy) to reach target weights of the current "
                  "total. One ```python block, code only.",
        "ref": textwrap.dedent("""
            def rebalance(holdings_usd, targets):
                total=sum(holdings_usd.values())
                return {a: targets.get(a,0.0)*total - holdings_usd.get(a,0.0)
                        for a in set(holdings_usd)|set(targets)}
        """),
        "check": textwrap.dedent("""
            def check(m):
                d=m.rebalance({'BTC':6000,'ETH':4000},{'BTC':0.5,'ETH':0.5})
                if abs(d['BTC']+1000)>1e-6 or abs(d['ETH']-1000)>1e-6: return False,f"{d}"
                return True,"ok"
        """),
    },
    {
        "id": "twap_slices",
        "prompt": "Write `twap_slices(total, n)` returning a list of n equal order sizes that sum to "
                  "total (last slice absorbs any rounding so the sum is exact). One ```python block.",
        "ref": textwrap.dedent("""
            def twap_slices(total, n):
                base=round(total/n, 8)
                out=[base]*(n-1)
                out.append(round(total-base*(n-1), 8))
                return out
        """),
        "check": textwrap.dedent("""
            def check(m):
                s=m.twap_slices(100,3)
                if len(s)!=3: return False,"len"
                if abs(sum(s)-100)>1e-6: return False,f"sum {sum(s)}"
                return True,"ok"
        """),
    },
    {
        "id": "pct_change",
        "prompt": "Write `pct_change(old, new)` returning the percent change from old to new as a "
                  "float (e.g. 100->150 gives 50.0). One ```python block, code only.",
        "ref": textwrap.dedent("""
            def pct_change(old, new):
                return (new-old)/old*100
        """),
        "check": textwrap.dedent("""
            def check(m):
                if abs(m.pct_change(100,150)-50.0)>1e-9: return False,"c1"
                if abs(m.pct_change(200,100)+50.0)>1e-9: return False,"c2"
                return True,"ok"
        """),
    },
    {
        "id": "dca_cost",
        "prompt": "Write `dca_avg(buys)` where buys is a list of (price, usd_spent); return the "
                  "average entry price across all buys as a float (total units = sum(usd/price)). "
                  "One ```python block, code only.",
        "ref": textwrap.dedent("""
            def dca_avg(buys):
                spent=sum(u for _,u in buys)
                units=sum(u/p for p,u in buys)
                return spent/units if units else 0.0
        """),
        "check": textwrap.dedent("""
            def check(m):
                v=m.dca_avg([(100,100),(50,100)])
                # 1 unit + 2 units = 3 units for $200 -> 66.666..
                if abs(v-200/3)>1e-6: return False,f"{v}"
                return True,"ok"
        """),
    },
    {
        "id": "liq_price_fn",
        "prompt": "Write `liq_price(entry, leverage, side='long')` returning the liquidation price "
                  "ignoring fees and maintenance margin: a `leverage`x position is wiped by a "
                  "1/leverage move against it. One ```python block, code only.",
        "ref": textwrap.dedent("""
            def liq_price(entry, leverage, side='long'):
                m=entry/leverage
                return entry-m if side=='long' else entry+m
        """),
        "check": textwrap.dedent("""
            def check(m):
                if abs(m.liq_price(100,10,'long')-90)>1e-9: return False,"long"
                if abs(m.liq_price(100,10,'short')-110)>1e-9: return False,"short"
                if abs(m.liq_price(100,4,'long')-75)>1e-9: return False,"4x"
                return True,"ok"
        """),
    },
    {
        "id": "orderbook_depth",
        "prompt": "Write `cost_to_buy(asks, qty)` where asks is a list of (price, size) sorted by "
                  "price ascending; return the total cost to buy `qty` units sweeping the book, or "
                  "None if the book is too thin. One ```python block, code only.",
        "ref": textwrap.dedent("""
            def cost_to_buy(asks, qty):
                cost=0.0; need=qty
                for p,s in asks:
                    take=min(s,need); cost+=take*p; need-=take
                    if need<=1e-12: return cost
                return None
        """),
        "check": textwrap.dedent("""
            def check(m):
                if abs(m.cost_to_buy([(100,1),(101,1)],1.5)-(100+0.5*101))>1e-6: return False,"c1"
                if m.cost_to_buy([(100,1)],5) is not None: return False,"thin should be None"
                return True,"ok"
        """),
    },
    {
        "id": "sharpe",
        "prompt": "Write `sharpe(returns)` returning the Sharpe ratio of a list of period returns "
                  "(mean/std, population std, risk-free 0). Return 0.0 if std is 0. "
                  "One ```python block, code only.",
        "ref": textwrap.dedent("""
            def sharpe(returns):
                n=len(returns)
                if n==0: return 0.0
                mean=sum(returns)/n
                var=sum((r-mean)**2 for r in returns)/n
                sd=var**0.5
                return mean/sd if sd else 0.0
        """),
        "check": textwrap.dedent("""
            def check(m):
                if m.sharpe([0.01,0.01,0.01])!=0.0: return False,"zero var -> 0"
                v=m.sharpe([0.02,-0.01,0.03,0.0])
                if not isinstance(v,float): return False,"type"
                return True,"ok"
        """),
    },
]
