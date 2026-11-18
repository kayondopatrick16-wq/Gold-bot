"""
XAUUSD 5-minute signal bot (Android / Termux version)  ->  emails every signal.
Data: Yahoo Finance chart API (no MT5 needed). Signals only, it does NOT place trades.
Set env vars: EMAIL_USER, EMAIL_PASS (Gmail app password), EMAIL_TO
Optional: SYMBOL (default XAUUSD=X spot gold; GC=F = gold futures), SMTP_HOST, SMTP_PORT
"""
import os, time, smtplib, ssl
from email.message import EmailMessage
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import json, urllib.request

# ---------------- CONFIG ----------------
SYMBOL = os.getenv("SYMBOL", "GC=F")
SERVER_UTC_OFFSET = int(os.getenv("SERVER_UTC_OFFSET", "2"))  # XM server = UTC+2 (+3 in US DST)
BARS = 1500
SWING_N = 3
ATR_SL_MULT = 1.0       # scalp: tighter stop
RR = 1.5                # scalp: nearer target
MIN_IND_SCORE = 3       # net indicator votes needed (max ~11)
MIN_ICT_SCORE = 2       # ICT/SMC points needed (max 6)
KILLZONES_UTC = [(6, 17)]   # London open through NY session. Set [] for all hours
COOLDOWN_BARS = 3       # same-direction signals at least 15 min apart
DAILY_TARGET = 3        # aim for at least this many signals per UTC day
# If behind target, thresholds relax step by step (see relax_level in main)

EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")
EMAIL_TO = os.getenv("EMAIL_TO", EMAIL_USER)
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "465"))


# ---------------- DATA ----------------
def get_data():
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{SYMBOL}?interval=5m&range=10d"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as r:
        res = json.load(r)["chart"]["result"][0]
    q = res["indicators"]["quote"][0]
    df = pd.DataFrame({"open": q["open"], "high": q["high"], "low": q["low"],
                       "close": q["close"], "vol": q["volume"]},
                      index=pd.to_datetime(res["timestamp"], unit="s"))
    df = df.dropna(subset=["open", "high", "low", "close"])
    df["vol"] = df["vol"].fillna(0).replace(0, 1)   # spot gold often has no volume
    return df.iloc[:-1]  # drop forming candle


# ---------------- INDICATORS ----------------
def ema(s, n): return s.ewm(span=n, adjust=False).mean()

def rma(s, n): return s.ewm(alpha=1 / n, adjust=False).mean()

def add_indicators(d):
    c, h, l = d.close, d.high, d.low
    for n in (9, 21, 50, 200): d[f"ema{n}"] = ema(c, n)
    # RSI
    delta = c.diff()
    rs = rma(delta.clip(lower=0), 14) / rma(-delta.clip(upper=0), 14).replace(0, np.nan)
    d["rsi"] = 100 - 100 / (1 + rs)
    # MACD
    d["macd"] = ema(c, 12) - ema(c, 26)
    d["macd_sig"] = ema(d["macd"], 9)
    d["macd_hist"] = d["macd"] - d["macd_sig"]
    # Bollinger
    ma, sd = c.rolling(20).mean(), c.rolling(20).std()
    d["bb_up"], d["bb_lo"], d["bb_mid"] = ma + 2 * sd, ma - 2 * sd, ma
    # ATR
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    d["atr"] = rma(tr, 14)
    # ADX / DI
    up, dn = h.diff(), -l.diff()
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    d["pdi"] = 100 * rma(pd.Series(pdm, d.index), 14) / d["atr"]
    d["mdi"] = 100 * rma(pd.Series(mdm, d.index), 14) / d["atr"]
    d["adx"] = rma(100 * (d["pdi"] - d["mdi"]).abs() / (d["pdi"] + d["mdi"]).replace(0, np.nan), 14)
    # Stochastic
    ll, hh = l.rolling(14).min(), h.rolling(14).max()
    d["stk"] = 100 * (c - ll) / (hh - ll).replace(0, np.nan)
    d["stk_sig"] = d["stk"].rolling(3).mean()
    # CCI
    tp = (h + l + c) / 3
    d["cci"] = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).apply(
        lambda x: np.abs(x - x.mean()).mean(), raw=True))
    # Daily-reset VWAP (tick volume)
    day = d.index.date
    pv = (tp * d.vol).groupby(day).cumsum()
    d["vwap"] = pv / d.vol.groupby(day).cumsum().replace(0, np.nan)
    # Supertrend (10, 3)
    atr10 = rma(tr, 10)
    hl2 = (h + l) / 2
    ub, lb = (hl2 + 3 * atr10).values, (hl2 - 3 * atr10).values
    st_dir = np.ones(len(d))
    fub, flb = ub.copy(), lb.copy()
    cv = c.values
    for i in range(1, len(d)):
        fub[i] = ub[i] if (ub[i] < fub[i-1] or cv[i-1] > fub[i-1]) else fub[i-1]
        flb[i] = lb[i] if (lb[i] > flb[i-1] or cv[i-1] < flb[i-1]) else flb[i-1]
        if st_dir[i-1] == 1:
            st_dir[i] = -1 if cv[i] < flb[i] else 1
        else:
            st_dir[i] = 1 if cv[i] > fub[i] else -1
    d["st_dir"] = st_dir
    # Ichimoku (cloud shifted 26 forward, evaluated at current bar)
    tenkan = (h.rolling(9).max() + l.rolling(9).min()) / 2
    kijun = (h.rolling(26).max() + l.rolling(26).min()) / 2
    d["tenkan"], d["kijun"] = tenkan, kijun
    d["spanA"] = ((tenkan + kijun) / 2).shift(26)
    d["spanB"] = ((h.rolling(52).max() + l.rolling(52).min()) / 2).shift(26)
    # OBV
    d["obv"] = (np.sign(c.diff()).fillna(0) * d.vol).cumsum()
    d["obv_ema"] = ema(d["obv"], 20)
    return d


def indicator_votes(d):
    """Returns (net score, notes). Positive = bullish."""
    x, p = d.iloc[-1], d.iloc[-2]
    v, notes = 0, []
    def vote(cond_up, cond_dn, name):
        nonlocal v
        if cond_up: v += 1; notes.append(f"+ {name}")
        elif cond_dn: v -= 1; notes.append(f"- {name}")
    vote(x.ema9 > x.ema21 > x.ema50, x.ema9 < x.ema21 < x.ema50, "EMA 9/21/50 stack")
    vote(x.close > x.ema200, x.close < x.ema200, "Price vs EMA200")
    vote(x.rsi > 55 and x.rsi < 75, x.rsi < 45 and x.rsi > 25, f"RSI {x.rsi:.0f}")
    vote(x.macd_hist > 0 and x.macd_hist > p.macd_hist, x.macd_hist < 0 and x.macd_hist < p.macd_hist, "MACD hist")
    vote(x.adx > 20 and x.pdi > x.mdi, x.adx > 20 and x.mdi > x.pdi, f"ADX {x.adx:.0f}/DI")
    vote(x.st_dir == 1, x.st_dir == -1, "Supertrend")
    vote(x.stk > x.stk_sig and x.stk < 80, x.stk < x.stk_sig and x.stk > 20, "Stochastic")
    vote(x.close > x.vwap, x.close < x.vwap, "VWAP")
    vote(x.close > max(x.spanA, x.spanB) and x.tenkan > x.kijun,
         x.close < min(x.spanA, x.spanB) and x.tenkan < x.kijun, "Ichimoku")
    vote(x.cci > 50, x.cci < -50, f"CCI {x.cci:.0f}")
    vote(x.obv > x.obv_ema, x.obv < x.obv_ema, "OBV")
    return v, notes


# ---------------- ICT / SMART MONEY ----------------
def swings(d, n=SWING_N):
    h, l = d.high.values, d.low.values
    sh, sl = [], []
    for i in range(n, len(d) - n):  # only confirmed swings (no lookahead)
        if h[i] == h[i-n:i+n+1].max(): sh.append((i, h[i]))
        if l[i] == l[i-n:i+n+1].min(): sl.append((i, l[i]))
    return sh, sl


def ict_analysis(d):
    """Returns dict with bull/bear points, notes and structural SL levels."""
    n = len(d)
    x = d.iloc[-1]
    atr = x.atr
    sh, sl = swings(d)
    bull, bear, nb, ns = 0, 0, [], []
    sl_bull, sl_bear = None, None
    if len(sh) < 2 or len(sl) < 2:
        return dict(bull=0, bear=0, nb=[], ns=[], sl_bull=None, sl_bear=None)

    last_sh, last_sl = sh[-1][1], sl[-1][1]
    prev_sh, prev_sl = sh[-2][1], sl[-2][1]

    # 1) Market structure: BOS / CHoCH in last 5 bars
    recent_close = d.close.iloc[-5:]
    hh_hl = last_sh > prev_sh and last_sl > prev_sl
    ll_lh = last_sh < prev_sh and last_sl < prev_sl
    if recent_close.max() > last_sh and sh[-1][0] < n - 5:
        bull += 1; nb.append("Bullish BOS" if hh_hl else "Bullish CHoCH")
    if recent_close.min() < last_sl and sl[-1][0] < n - 5:
        bear += 1; ns.append("Bearish BOS" if ll_lh else "Bearish CHoCH")

    # 2) Liquidity sweeps (swing points + previous day high/low) in last 3 bars
    day = d.index.date
    today = day[-1]
    prev_days = d[d.index.date < today]
    levels_lo, levels_hi = [last_sl, prev_sl], [last_sh, prev_sh]
    if len(prev_days):
        pd_day = prev_days.index.date[-1]
        pdd = prev_days[prev_days.index.date == pd_day]
        levels_lo.append(pdd.low.min()); levels_hi.append(pdd.high.max())
    last3 = d.iloc[-3:]
    for lv in levels_lo:  # sell-side liquidity taken, close back above -> bullish
        if last3.low.min() < lv and x.close > lv:
            bull += 1; nb.append("Sell-side liquidity sweep"); sl_bull = last3.low.min(); break
    for lv in levels_hi:
        if last3.high.max() > lv and x.close < lv:
            bear += 1; ns.append("Buy-side liquidity sweep"); sl_bear = last3.high.max(); break

    # 3) Fair Value Gap retest (unfilled, last 40 bars)
    got_b = got_s = False
    for i in range(max(2, n - 40), n - 2):
        if not got_b and d.low.iloc[i] > d.high.iloc[i-2]:
            bot, top = d.high.iloc[i-2], d.low.iloc[i]
            if d.close.iloc[i+1:n-1].min() > bot and x.low <= top and x.close > bot:
                bull += 1; nb.append(f"Bullish FVG retest {bot:.2f}-{top:.2f}"); got_b = True
        if not got_s and d.high.iloc[i] < d.low.iloc[i-2]:
            bot, top = d.high.iloc[i], d.low.iloc[i-2]
            if d.close.iloc[i+1:n-1].max() < top and x.high >= bot and x.close < top:
                bear += 1; ns.append(f"Bearish FVG retest {bot:.2f}-{top:.2f}"); got_s = True

    # 4) Order block retest: last opposite candle before a displacement candle
    got_b = got_s = False
    for j in range(n - 40, n - 2):
        body = abs(d.close.iloc[j] - d.open.iloc[j])
        if body < 1.5 * d.atr.iloc[j]: continue
        up_imp = d.close.iloc[j] > d.open.iloc[j]
        for k in range(j - 1, max(j - 4, 0), -1):
            is_down = d.close.iloc[k] < d.open.iloc[k]
            if up_imp and is_down and not got_b:
                lo, hi = d.low.iloc[k], d.high.iloc[k]
                if d.close.iloc[j+1:n-1].min() > lo and x.low <= hi and x.close > lo:
                    bull += 1; nb.append(f"Bullish order block {lo:.2f}-{hi:.2f}"); got_b = True
                    sl_bull = lo if sl_bull is None else min(sl_bull, lo)
                break
            if (not up_imp) and (not is_down) and not got_s:
                lo, hi = d.low.iloc[k], d.high.iloc[k]
                if d.close.iloc[j+1:n-1].max() < hi and x.high >= lo and x.close < hi:
                    bear += 1; ns.append(f"Bearish order block {lo:.2f}-{hi:.2f}"); got_s = True
                    sl_bear = hi if sl_bear is None else max(sl_bear, hi)
                break

    # 5) Premium / Discount of current dealing range
    mid = (last_sh + last_sl) / 2
    if x.close < mid: bull += 1; nb.append("Price in discount (<50% of range)")
    if x.close > mid: bear += 1; ns.append("Price in premium (>50% of range)")

    # 6) Higher-timeframe bias (1h EMA50 slope via 12x5min resample)
    h1 = d.close.resample("1h").last().dropna()
    if len(h1) > 55:
        e = ema(h1, 50)
        if h1.iloc[-1] > e.iloc[-1] and e.iloc[-1] > e.iloc[-2]: bull += 1; nb.append("1H bias bullish")
        if h1.iloc[-1] < e.iloc[-1] and e.iloc[-1] < e.iloc[-2]: bear += 1; ns.append("1H bias bearish")

    return dict(bull=bull, bear=bear, nb=nb, ns=ns, sl_bull=sl_bull, sl_bear=sl_bear,
                has_pd_bull=got_b or any("sweep" in s or "FVG" in s or "order block" in s for s in nb),
                has_pd_bear=got_s or any("sweep" in s or "FVG" in s or "order block" in s for s in ns))


# ---------------- PRICE ACTION ----------------
def price_action(d):
    x, p = d.iloc[-1], d.iloc[-2]
    rng = x.high - x.low
    if rng == 0: return 0, []
    body = abs(x.close - x.open)
    up_w = x.high - max(x.open, x.close)
    lo_w = min(x.open, x.close) - x.low
    notes, s = [], 0
    if x.close > x.open and p.close < p.open and x.close > p.open and x.open < p.close:
        s += 1; notes.append("Bullish engulfing")
    if x.close < x.open and p.close > p.open and x.close < p.open and x.open > p.close:
        s -= 1; notes.append("Bearish engulfing")
    if lo_w > 2 * body and lo_w > 0.6 * rng: s += 1; notes.append("Bullish pin bar")
    if up_w > 2 * body and up_w > 0.6 * rng: s -= 1; notes.append("Bearish pin bar")
    if body > 0.7 * rng and body > 1.2 * x.atr:
        s += 1 if x.close > x.open else -1
        notes.append("Strong momentum candle")
    return s, notes


# ---------------- SIGNAL ----------------
def in_killzone(ts):
    if not KILLZONES_UTC: return True
    return any(a <= ts.hour < b for a, b in KILLZONES_UTC)


def evaluate(d, relax=0):
    """relax 0 = normal, 1 = indicators -1, 2 = also ICT -1 (lower quality)."""
    ind_min = MIN_IND_SCORE - (1 if relax >= 1 else 0)
    ict_min = max(MIN_ICT_SCORE - (1 if relax >= 2 else 0), 1)
    d = add_indicators(d.copy())
    x = d.iloc[-1]
    if not in_killzone(d.index[-1]) or np.isnan(x.ema200) or np.isnan(x.spanB):
        return None
    ind, ind_notes = indicator_votes(d)
    ict = ict_analysis(d)
    pa, pa_notes = price_action(d)

    side = None
    if ind >= ind_min and ict["bull"] + max(pa, 0) >= ict_min and ict.get("has_pd_bull"):
        side = "BUY"
    elif -ind >= ind_min and ict["bear"] + max(-pa, 0) >= ict_min and ict.get("has_pd_bear"):
        side = "SELL"
    if not side: return None

    atr, price = x.atr, x.close
    if side == "BUY":
        sl = min(price - ATR_SL_MULT * atr, (ict["sl_bull"] - 0.2 * atr) if ict["sl_bull"] else 1e9)
        sl = min(sl, price - atr)  # never tighter than 1 ATR
        tp = price + RR * (price - sl)
        notes = ict["nb"] + [n for n in pa_notes if "Bullish" in n or "Strong" in n] + [n for n in ind_notes if n.startswith("+")]
    else:
        sl = max(price + ATR_SL_MULT * atr, (ict["sl_bear"] + 0.2 * atr) if ict["sl_bear"] else -1e9)
        sl = max(sl, price + atr)
        tp = price - RR * (sl - price)
        notes = ict["ns"] + [n for n in pa_notes if "Bearish" in n or "Strong" in n] + [n for n in ind_notes if n.startswith("-")]
    return dict(relax=relax, side=side, price=price, sl=sl, tp=tp, atr=atr, time=d.index[-1],
                ind=ind, ict=ict["bull"] if side == "BUY" else ict["bear"], notes=notes)


# ---------------- EMAIL ----------------
def send_email(sig):
    grade = ["A", "B", "C"][sig["relax"]]
    subj = f"[{grade}] {sig['side']} {SYMBOL} @ {sig['price']:.2f} (5m scalp)"
    body = (
        f"{sig['side']} {SYMBOL}  |  5-min  |  {sig['time']} UTC\n\n"
        f"Entry: {sig['price']:.2f}\nStop loss: {sig['sl']:.2f}\nTake profit: {sig['tp']:.2f} (RR 1:{RR})\n"
        f"ATR(14): {sig['atr']:.2f}\n\n"
        f"Grade {grade} (A = strict, B/C = relaxed to hit daily target)\n"
        f"Indicator score: {sig['ind']:+d}   ICT/SMC score: {sig['ict']}\n\nConfluences:\n"
        + "\n".join(f" - {n}" for n in sig["notes"])
        + "\n\nAutomated signal, not financial advice. Always verify before trading."
    )
    m = EmailMessage()
    m["Subject"], m["From"], m["To"] = subj, EMAIL_USER, EMAIL_TO
    m.set_content(body)
    with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, context=ssl.create_default_context()) as s:
        s.login(EMAIL_USER, EMAIL_PASS)
        s.send_message(m)
    print(subj)


# ---------------- SINGLE RUN (for GitHub Actions) ----------------
STATE_FILE = "state.json"

def load_state():
    try:
        with open(STATE_FILE) as f: return json.load(f)
    except Exception:
        return {}

def save_state(st):
    with open(STATE_FILE, "w") as f: json.dump(st, f)

def run_once():
    if not (EMAIL_USER and EMAIL_PASS):
        raise SystemExit("Set EMAIL_USER and EMAIL_PASS secrets.")
    df = get_data()
    now = datetime.now(timezone.utc)
    st = load_state()
    today = str(now.date())
    if st.get("day") != today:
        st = {"day": today, "sent": 0, "last_side": None, "last_bar": None}
    frac = min(max((now.hour + now.minute / 60 - 6) / 11, 0), 1)
    behind = st["sent"] < DAILY_TARGET * frac
    relax = 2 if (behind and frac > 0.65) else 1 if (behind and frac > 0.35) else 0
    sig = evaluate(df, relax)
    if sig:
        bar = str(sig["time"])
        age_min = (now.replace(tzinfo=None) - sig["time"].to_pydatetime()).total_seconds() / 60
        last = pd.Timestamp(st["last_bar"]) if st.get("last_bar") else None
        cooling = (last is not None and st["last_side"] == sig["side"]
                   and (sig["time"] - last).total_seconds() < COOLDOWN_BARS * 300)
        stale = age_min > 20            # delayed run: don't alert on an old candle
        if bar != st.get("last_bar") and not cooling and not stale:
            send_email(sig)
            st.update(sent=st["sent"] + 1, last_side=sig["side"], last_bar=bar)
    save_state(st)


if __name__ == "__main__":
    run_once()
