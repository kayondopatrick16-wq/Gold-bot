#!/usr/bin/env python3
"""
XAUUSD signal bot - Smart Money setups, trend-aligned. Emails signals, places NO trades.
Data: Yahoo Finance chart API (default GC=F gold futures).

MAIN STRATEGY (SWEEP=1, default): 4H bias + sweep + 15m entry
  1) 4H bias = trend of 4H market structure (4H candles are built from 1H data).
  2) Only between 07:00 and 17:00 (SERVER_UTC_OFFSET time, default UTC+2), Mon-Fri.
  3) Wait for price to SWEEP the previous 4H candle's low (bullish bias) or high (bearish bias).
  4) Then wait for a 15m BOS or CHOCH in the bias direction that leaves a 15m FVG.
  5) Signal when price returns to that 15m FVG, or to the 15m Order Block (last opposite
     candle before the impulse). SL beyond the sweep extreme. TP = 1:3 (RR_SWEEP).

OPTIONAL (CLASSIC=1): the earlier 1H/15m setups (FVG after BOS, Order Block, FVG after CHOCH).

Env vars: EMAIL_USER, EMAIL_PASS (Gmail app password), EMAIL_TO
Optional: SYMBOL, SERVER_UTC_OFFSET, WINDOW_START, WINDOW_END, RR_SWEEP, SL_BUFFER_ATR,
          SWEEP, CLASSIC, SMTP_HOST, SMTP_PORT
A manual "Run workflow" click also sends a status email, so you can confirm email works.
"""
import os
import sys
import json
import time
import ssl
import smtplib
import urllib.parse
import urllib.request
from email.message import EmailMessage

import numpy as np
import pandas as pd

# ---------------- CONFIG ----------------
SYMBOL = os.getenv("SYMBOL", "GC=F")
UTC_OFFSET = int(os.getenv("SERVER_UTC_OFFSET", "2"))   # only for the times shown in emails/logs
SWING_N = int(os.getenv("SWING_N", "3"))                # bars each side to confirm a swing point
RR = float(os.getenv("RR", "2.0"))                      # take profit = RR x risk
SL_BUFFER_ATR = float(os.getenv("SL_BUFFER_ATR", "0.25"))
ATR_N = 14
MIN_ZONE_ATR = 0.15      # ignore FVG / OB zones smaller than this many ATRs
FVG_WINDOW = 6           # an FVG counts as "after" a BOS/CHOCH if formed within this many bars of it
FVG_BACK = 3             # for order blocks: the FVG may sit up to this many bars before the BOS candle
MAX_EMAILS_PER_RUN = 3
STATE_FILE = "state.json"

TF_CFG = {
    "1h":  {"range": "90d", "minutes": 60, "max_age": 30, "recent": 2},
    "15m": {"range": "30d", "minutes": 15, "max_age": 60, "recent": 4},
}
# max_age: how many bars old a BOS/CHOCH may be. recent: a first touch of a zone counts as a
# live entry if it happened within this many bars (runs on GitHub can be delayed).


# ---------------- DATA ----------------
def utc_now():
    return pd.Timestamp.now(tz="UTC")


def fetch(interval, rng):
    sym = urllib.parse.quote(SYMBOL, safe="")
    last_err = None
    for attempt in range(3):
        for host in ("query1", "query2"):
            url = f"https://{host}.finance.yahoo.com/v8/finance/chart/{sym}?interval={interval}&range={rng}"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            try:
                with urllib.request.urlopen(req, timeout=20) as r:
                    j = json.loads(r.read().decode())
                res = j["chart"]["result"][0]
                q = res["indicators"]["quote"][0]
                df = pd.DataFrame({
                    "time": pd.to_datetime(res["timestamp"], unit="s", utc=True),
                    "open": q["open"], "high": q["high"], "low": q["low"], "close": q["close"],
                })
                df = df.dropna().reset_index(drop=True)
                if len(df) > 0:
                    return df
            except Exception as e:  # noqa
                last_err = e
        time.sleep(2)
    raise RuntimeError(f"data fetch failed ({SYMBOL} {interval}): {last_err}")


def split_forming(df, minutes):
    """Return (completed candles, latest price). Drops the candle that is still forming."""
    live = float(df["close"].iloc[-1])
    now = utc_now()
    if len(df) and now < df["time"].iloc[-1] + pd.Timedelta(minutes=minutes):
        return df.iloc[:-1].reset_index(drop=True), live
    return df, live


def add_atr(df, n=ATR_N):
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    df["atr"] = tr.rolling(n).mean()
    return df


# ---------------- STRUCTURE (swings, BOS, CHOCH) ----------------
def structure(df, n=SWING_N):
    """
    Walks the candles and returns (events, trend).
    A swing high/low is confirmed n bars after it forms. A candle CLOSE beyond the latest
    swing level is a break: with the trend = BOS, against the trend = CHOCH (trend flips).
    event = {idx, kind: 'BOS'|'CHOCH', dir: +1|-1, level, swing_idx}
    """
    H = df["high"].values
    L = df["low"].values
    C = df["close"].values
    events = []
    trend = 0
    sh = None  # latest unbroken swing high (price, idx)
    sl = None  # latest unbroken swing low
    for i in range(2 * n, len(df)):
        p = i - n
        if H[p] == H[p - n:p + n + 1].max():
            sh = (H[p], p)
        if L[p] == L[p - n:p + n + 1].min():
            sl = (L[p], p)
        if sh is not None and C[i] > sh[0]:
            kind = "CHOCH" if trend == -1 else "BOS"
            events.append({"idx": i, "kind": kind, "dir": 1, "level": sh[0], "swing_idx": sh[1]})
            trend = 1
            sh = None
        elif sl is not None and C[i] < sl[0]:
            kind = "CHOCH" if trend == 1 else "BOS"
            events.append({"idx": i, "kind": kind, "dir": -1, "level": sl[0], "swing_idx": sl[1]})
            trend = -1
            sl = None
    return events, trend


# ---------------- FVG / ORDER BLOCK ----------------
def find_fvgs(df):
    """3-candle gap. f = index of the 3rd candle (when the gap is confirmed)."""
    H = df["high"].values
    L = df["low"].values
    A = df["atr"].values
    out = []
    for i in range(2, len(df)):
        if np.isnan(A[i]):
            continue
        if L[i] > H[i - 2] and (L[i] - H[i - 2]) >= MIN_ZONE_ATR * A[i]:
            out.append({"f": i, "dir": 1, "bottom": float(H[i - 2]), "top": float(L[i])})
        elif H[i] < L[i - 2] and (L[i - 2] - H[i]) >= MIN_ZONE_ATR * A[i]:
            out.append({"f": i, "dir": -1, "bottom": float(H[i]), "top": float(L[i - 2])})
    return out


def find_ob(df, f, d):
    """Last opposite-colour candle just before the impulse whose FVG was confirmed at index f."""
    O = df["open"].values
    C = df["close"].values
    m = f - 1  # middle (impulse) candle of the FVG
    for j in range(m - 1, max(m - 7, -1), -1):
        if d == 1 and C[j] < O[j]:
            return j
        if d == -1 and C[j] > O[j]:
            return j
    return None


def zone_state(df, f, d, bottom, top, live, recent):
    """
    'invalid'  - price closed through the zone
    'used'     - zone was already touched earlier (not a fresh entry any more)
    'in_play'  - price is in the zone now, or first touched it within the last `recent` bars
    'waiting'  - zone is fresh and price has not reached it yet
    """
    H = df["high"].values
    L = df["low"].values
    C = df["close"].values
    last = len(df) - 1
    first_touch = None
    for j in range(f + 1, last + 1):
        if d == 1:
            if C[j] < bottom:
                return "invalid"
            if first_touch is None and L[j] <= top:
                first_touch = j
        else:
            if C[j] > top:
                return "invalid"
            if first_touch is None and H[j] >= bottom:
                first_touch = j
    if d == 1 and live < bottom:
        return "invalid"
    if d == -1 and live > top:
        return "invalid"
    inside_now = bottom <= live <= top
    if first_touch is None:
        return "in_play" if inside_now else "waiting"
    if first_touch >= last - (recent - 1):
        return "in_play"
    return "used"


def make_trade(d, bottom, top, live, atr_now):
    entry = min(max(live, bottom), top)
    buf = SL_BUFFER_ATR * atr_now
    if d == 1:
        sl = bottom - buf
        risk = entry - sl
    else:
        sl = top + buf
        risk = sl - entry
    tp = entry + d * RR * risk
    return entry, sl, tp


# ---------------- SETUP SCAN ----------------
def analyse(tf):
    cfg = TF_CFG[tf]
    raw = fetch(tf, cfg["range"])
    df, live = split_forming(raw, cfg["minutes"])
    if len(df) < 60:
        raise RuntimeError(f"not enough {tf} data ({len(df)} candles)")
    add_atr(df)
    events, trend = structure(df)
    return {"tf": tf, "df": df, "live": live, "events": events, "trend": trend,
            "fvgs": find_fvgs(df), "cfg": cfg}


def find_setups(A, htf_trend):
    df, cfg, tf = A["df"], A["cfg"], A["tf"]
    last = len(df) - 1
    live = A["live"]
    atr_now = float(df["atr"].iloc[-1])
    signals, pending = [], []
    counts = {"in_play": 0, "waiting": 0, "used": 0, "invalid": 0, "blocked_by_trend": 0}

    def handle(setup, d, bottom, top, formed_idx, ev):
        st = zone_state(df, formed_idx, d, bottom, top, live, cfg["recent"])
        counts[st] += 1
        item = {
            "tf": tf, "setup": setup, "dir": d, "bottom": bottom, "top": top,
            "formed": df["time"].iloc[formed_idx], "ev_kind": ev["kind"], "ev_time": df["time"].iloc[ev["idx"]],
            "key": f"{tf}|{setup}|{d}|{df['time'].iloc[formed_idx].isoformat()}",
        }
        if st == "in_play":
            item["entry"], item["sl"], item["tp"] = make_trade(d, bottom, top, live, atr_now)
            signals.append(item)
        elif st == "waiting":
            pending.append(item)

    for ev in A["events"]:
        if last - ev["idx"] > cfg["max_age"]:
            continue
        d = ev["dir"]
        if not (d == A["trend"] == htf_trend):
            counts["blocked_by_trend"] += 1
            continue
        # Setups 1 and 3: FVG formed after the BOS / CHOCH, in the trend direction
        setup = "FVG_BOS" if ev["kind"] == "BOS" else "FVG_CHOCH"
        for fg in A["fvgs"]:
            if fg["dir"] == d and ev["idx"] <= fg["f"] <= ev["idx"] + FVG_WINDOW:
                handle(setup, d, fg["bottom"], fg["top"], fg["f"], ev)
        # Setup 2: order block behind an impulse that left an FVG and broke structure (BOS only)
        if ev["kind"] == "BOS":
            seen = set()
            for fg in A["fvgs"]:
                if fg["dir"] == d and ev["idx"] - FVG_BACK <= fg["f"] <= ev["idx"] + 1:
                    j = find_ob(df, fg["f"], d)
                    if j is None or j in seen:
                        continue
                    seen.add(j)
                    lo, hi = float(df["low"].iloc[j]), float(df["high"].iloc[j])
                    if (hi - lo) < MIN_ZONE_ATR * atr_now:
                        continue
                    handle("OB_BOS", d, lo, hi, max(fg["f"], ev["idx"]), ev)
    return signals, pending, counts


# ---------------- 4H BIAS + SWEEP STRATEGY ----------------
SWEEP_ON = os.getenv("SWEEP", "1") == "1"
CLASSIC_ON = os.getenv("CLASSIC", "0") == "1"
WINDOW_START = int(os.getenv("WINDOW_START", "7"))
WINDOW_END = int(os.getenv("WINDOW_END", "17"))
RR_SWEEP = float(os.getenv("RR_SWEEP", "3.0"))
BIAS_SWING_N = int(os.getenv("BIAS_SWING_N", "2"))
SWEEP_MAX_BARS = int(os.getenv("SWEEP_MAX_BARS", "24"))   # 15m bars (24 = 6 hours)
SWEEP_RECENT_BARS = 4                                      # first touch of the zone within 4 x 15m bars


def local_ts(ts):
    return ts + pd.Timedelta(hours=UTC_OFFSET)


def in_window(ts):
    lt = local_ts(ts)
    return lt.weekday() < 5 and WINDOW_START <= lt.hour < WINDOW_END


def build_4h(df1h):
    """4H candles (aligned to 00/04/08/12/16/20 in SERVER_UTC_OFFSET time) from completed 1H candles."""
    off = pd.Timedelta(hours=UTC_OFFSET)
    d = df1h.copy()
    d["bucket"] = (d["time"] + off).dt.floor("4h")
    g = d.groupby("bucket").agg(open=("open", "first"), high=("high", "max"),
                                low=("low", "min"), close=("close", "last")).reset_index()
    g["time"] = g["bucket"] - off
    now = utc_now()
    g = g[g["time"] + pd.Timedelta(hours=4) <= now].reset_index(drop=True)   # completed 4H candles only
    g = g[["time", "open", "high", "low", "close"]]
    return add_atr(g)


def find_sweep_setups(A15, df4, bias):
    """Sweep of the previous 4H candle -> 15m BOS/CHOCH with FVG -> entry at the FVG or the OB."""
    df, live = A15["df"], A15["live"]
    last = len(df) - 1
    H, L, C = df["high"].values, df["low"].values, df["close"].values
    atr_now = float(df["atr"].iloc[-1])
    signals, pending, notes = [], [], []
    if bias == 0 or len(df4) < 10:
        return signals, pending, ["no clear 4H bias - nothing to trade"]

    off = pd.Timedelta(hours=UTC_OFFSET)
    lt = df["time"] + off
    bucket = lt.dt.floor("4h") - off
    pos = np.searchsorted(df4["time"].values, bucket.values)
    prev_i = pos - 1
    valid = prev_i >= 0
    prev_i = np.where(valid, prev_i, 0)
    prev_low = df4["low"].values[prev_i]
    prev_high = df4["high"].values[prev_i]
    inwin = ((lt.dt.hour >= WINDOW_START) & (lt.dt.hour < WINDOW_END) & (lt.dt.weekday < 5)).values
    swept = (L < prev_low) if bias == 1 else (H > prev_high)
    mask = swept & valid & inwin
    mask[: max(0, last - SWEEP_MAX_BARS)] = False

    sweeps, seen = [], set()
    for s in np.flatnonzero(mask):           # first sweep candle in each 4H bucket
        b = bucket.iloc[s]
        if b in seen:
            continue
        seen.add(b)
        sweeps.append(int(s))
    if not sweeps:
        side = "low" if bias == 1 else "high"
        notes.append(f"no sweep of the previous 4H {side} inside the {WINDOW_START:02d}:00-{WINDOW_END:02d}:00 window yet")
        return signals, pending, notes

    done_events = set()
    for s in sweeps:
        s_time = df["time"].iloc[s]
        level = float(prev_low[s]) if bias == 1 else float(prev_high[s])
        side = "low" if bias == 1 else "high"
        ev = next((e for e in A15["events"] if e["idx"] > s and e["dir"] == bias), None)
        if ev is None:
            notes.append(f"4H {side} {level:.2f} swept at {fmt_t(s_time)} - waiting for a 15m BOS/CHOCH")
            continue
        if ev["idx"] in done_events:      # same 15m break already handled via an earlier sweep
            continue
        done_events.add(ev["idx"])
        ext = float(L[s:ev["idx"] + 1].min()) if bias == 1 else float(H[s:ev["idx"] + 1].max())
        after = C[ev["idx"] + 1:]
        if len(after) and ((bias == 1 and after.min() < ext) or (bias == -1 and after.max() > ext)):
            notes.append(f"setup from sweep at {fmt_t(s_time)} invalidated (price closed beyond the sweep extreme)")
            continue
        cands = [fg for fg in A15["fvgs"] if fg["dir"] == bias and fg["f"] - 1 >= s
                 and ev["idx"] - FVG_BACK <= fg["f"] <= ev["idx"] + 2]
        zones = [("FVG", fg["bottom"], fg["top"], fg["f"]) for fg in cands]
        used_ob = set()
        for fg in cands:
            j = find_ob(df, fg["f"], bias)
            if j is None or j < s or j in used_ob:
                continue
            used_ob.add(j)
            lo, hi = float(L[j]), float(H[j])
            if (hi - lo) >= MIN_ZONE_ATR * atr_now:
                zones.append(("OB", lo, hi, max(fg["f"], ev["idx"])))
        if not zones:
            notes.append(f"{ev['kind']} after the sweep at {fmt_t(s_time)} but it left no FVG")
            continue
        for kind, bot, top, fidx in zones:
            st = zone_state(df, fidx, bias, bot, top, live, SWEEP_RECENT_BARS)
            entry = min(max(live, bot), top)
            buf = SL_BUFFER_ATR * atr_now
            sl = (min(ext, bot) - buf) if bias == 1 else (max(ext, top) + buf)
            risk = max(abs(entry - sl), 0.3 * atr_now)
            sl = entry - bias * risk
            tp = entry + bias * RR_SWEEP * risk
            item = {"tf": "15m", "zone_kind": kind, "dir": bias, "bottom": bot, "top": top,
                    "formed": df["time"].iloc[fidx], "ev_kind": ev["kind"], "ev_time": df["time"].iloc[ev["idx"]],
                    "sweep_time": s_time, "level": level, "ext": ext, "entry": entry, "sl": sl, "tp": tp,
                    "key": f"SWEEP|{bias}|{df['time'].iloc[ev['idx']].isoformat()}"}
            if st == "waiting":
                pending.append(item)
            elif st == "in_play":
                if (live - entry) * bias > risk:
                    notes.append(f"{kind} entry at {fmt_t(s_time)} sweep is already >1R away - skipped (late)")
                else:
                    signals.append(item)
    signals.sort(key=lambda x: 0 if x["zone_kind"] == "FVG" else 1)
    return signals, pending, notes


def sweep_text(s, bias, live):
    side = "BUY" if s["dir"] == 1 else "SELL"
    word = "low" if s["dir"] == 1 else "high"
    subject = (f"XAUUSD {side} | 4H {trend_name(bias)} | sweep + 15m {s['ev_kind']} | "
               f"{s['zone_kind']} entry {s['entry']:.2f}")
    body = (
        f"XAUUSD {side}  (4H bias {trend_name(bias)}, entry on 15m {s['zone_kind']})\n\n"
        f"Entry:  {s['entry']:.2f}\n"
        f"SL:     {s['sl']:.2f}\n"
        f"TP:     {s['tp']:.2f}   (1:{RR_SWEEP:g})\n\n"
        f"Sweep:  previous 4H {word} {s['level']:.2f} taken at {fmt_t(s['sweep_time'])} (extreme {s['ext']:.2f})\n"
        f"Then:   15m {s['ev_kind']} at {fmt_t(s['ev_time'])}\n"
        f"Zone:   15m {s['zone_kind']} {s['bottom']:.2f} - {s['top']:.2f} (formed {fmt_t(s['formed'])})\n"
        f"Price now: {live:.2f}\n\n"
        f"Times are UTC{UTC_OFFSET:+d}. Prices are {SYMBOL} (Yahoo); check levels on your broker chart "
        f"before entering. This is a signal only - no trade was placed.\n"
    )
    return subject, body


# ---------------- OUTPUT ----------------
def fmt_t(ts):
    return (ts + pd.Timedelta(hours=UTC_OFFSET)).strftime("%Y-%m-%d %H:%M")


def trend_name(t):
    return {1: "bullish", -1: "bearish", 0: "undetermined"}[t]


SETUP_NAME = {
    "FVG_BOS": "FVG after BOS",
    "OB_BOS": "Order Block after BOS (with FVG)",
    "FVG_CHOCH": "FVG after CHOCH",
}


def signal_text(s, A):
    side = "BUY" if s["dir"] == 1 else "SELL"
    subject = f"XAUUSD {side} | {s['tf'].upper()} | {SETUP_NAME[s['setup']]} | entry {s['entry']:.2f}"
    body = (
        f"XAUUSD {side}  ({s['tf'].upper()})\n"
        f"Setup:  {SETUP_NAME[s['setup']]}  ({s['ev_kind']} at {fmt_t(s['ev_time'])})\n\n"
        f"Entry:  {s['entry']:.2f}\n"
        f"SL:     {s['sl']:.2f}\n"
        f"TP:     {s['tp']:.2f}   (RR {RR:g})\n\n"
        f"Zone:   {s['bottom']:.2f} - {s['top']:.2f}  (formed {fmt_t(s['formed'])})\n"
        f"Price now: {A[s['tf']]['live']:.2f}\n"
        f"Trend:  1H {trend_name(A['1h']['trend'])}, 15m {trend_name(A['15m']['trend'])}\n\n"
        f"Times are UTC{UTC_OFFSET:+d}. Prices are {SYMBOL} (Yahoo); check levels on your broker chart "
        f"before entering. This is a signal only - no trade was placed.\n"
    )
    return subject, body


def send_email(subject, body):
    user = os.environ["EMAIL_USER"]
    pw = os.environ["EMAIL_PASS"].replace(" ", "")
    to = os.environ["EMAIL_TO"]
    host = os.getenv("SMTP_HOST", "smtp.gmail.com")
    port = int(os.getenv("SMTP_PORT", "465"))
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = user
    msg["To"] = to
    msg.set_content(body)
    with smtplib.SMTP_SSL(host, port, context=ssl.create_default_context(), timeout=30) as s:
        s.login(user, pw)
        s.send_message(msg)


def load_state():
    try:
        with open(STATE_FILE) as f:
            s = json.load(f)
        if isinstance(s, dict) and isinstance(s.get("sent"), dict):
            return s
    except Exception:
        pass
    return {"sent": {}}


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=1, sort_keys=True)


def run_once():
    state = load_state()
    now = time.time()
    state["sent"] = {k: v for k, v in state["sent"].items() if now - v < 3 * 86400}
    now_ts = utc_now()
    window_open = in_window(now_ts)

    A = {"1h": analyse("1h"), "15m": analyse("15m")}
    df4 = build_4h(A["1h"]["df"])
    ev4, bias = structure(df4, BIAS_SWING_N)

    print(f"=== {SYMBOL} | now UTC{UTC_OFFSET:+d} {fmt_t(now_ts)} | window {WINDOW_START:02d}:00-{WINDOW_END:02d}:00 "
          f"{'OPEN' if window_open else 'closed'} ===")
    e4 = ", ".join(f"{e['kind']} {'up' if e['dir'] == 1 else 'down'} @ {fmt_t(df4['time'].iloc[e['idx']])}"
                   for e in ev4[-3:]) or "none"
    print(f"4H bias: {trend_name(bias)} | last 4H events: {e4}")
    print(f"15m price {A['15m']['live']:.2f}")

    all_signals, all_pending, notes = [], [], []
    emails = []                                   # (key, subject, body)
    if SWEEP_ON:
        sigs, pend, notes = find_sweep_setups(A["15m"], df4, bias)
        all_pending += pend
        seen = set()
        for s in sigs:
            if s["key"] in state["sent"] or s["key"] in seen:
                continue
            seen.add(s["key"])
            subj, body = sweep_text(s, bias, A["15m"]["live"])
            emails.append((s["key"], subj, body))
        all_signals += sigs
        for n in notes:
            print("  note:", n)
        for p in pend:
            side = "BUY" if p["dir"] == 1 else "SELL"
            print(f"  waiting: 15m {p['zone_kind']} {side} zone {p['bottom']:.2f}-{p['top']:.2f}")

    if CLASSIC_ON:
        htf = A["1h"]["trend"]
        for tf in ("1h", "15m"):
            sigs, pend, counts = find_setups(A[tf], htf)
            print(f"[classic {tf}] trend {trend_name(A[tf]['trend'])} zones: {counts}")
            for s in sigs:
                if s["key"] not in state["sent"]:
                    subj, body = signal_text(s, A)
                    emails.append((s["key"], subj, body))
            all_signals += sigs
            all_pending += pend

    print(f"signals in play: {len(all_signals)} | new to send: {len(emails)}")
    errors = 0
    if emails and not window_open:
        print("outside the trading window - emails held back")
        emails = []
    for key, subj, body in emails[:MAX_EMAILS_PER_RUN]:
        try:
            send_email(subj, body)
            state["sent"][key] = now
            print("EMAIL SENT:", subj)
        except Exception as e:  # noqa
            errors += 1
            print("EMAIL FAILED:", repr(e))

    if os.getenv("GITHUB_EVENT_NAME") == "workflow_dispatch":
        lines = [f"Gold bot is running. Price {A['15m']['live']:.2f} ({SYMBOL}).",
                 f"4H bias: {trend_name(bias)}",
                 f"Trading window {WINDOW_START:02d}:00-{WINDOW_END:02d}:00 (UTC{UTC_OFFSET:+d}): "
                 f"{'open' if window_open else 'closed right now'}",
                 f"Signals in play: {len(all_signals)}   Fresh zones waiting: {len(all_pending)}", ""]
        lines += [f"- {n}" for n in notes]
        for p in all_pending:
            side = "BUY" if p["dir"] == 1 else "SELL"
            lines.append(f"- waiting: 15m {p.get('zone_kind', p.get('setup'))} {side} zone {p['bottom']:.2f}-{p['top']:.2f}")
        try:
            send_email("XAUUSD bot status (manual run)", "\n".join(lines))
            print("STATUS EMAIL SENT")
        except Exception as e:  # noqa
            errors += 1
            print("STATUS EMAIL FAILED:", repr(e))

    save_state(state)
    return errors


if __name__ == "__main__":
    sys.exit(1 if run_once() else 0)
