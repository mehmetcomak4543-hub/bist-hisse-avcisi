
from flask import Flask, jsonify, render_template
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from io import StringIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import threading, time, json, os, re

app = Flask(__name__)

SCAN_COUNT = 500
PERIOD = "1y"
INTERVAL = "1d"
CACHE_SECONDS = 300

state = {
    "running": False,
    "progress": 0,
    "total": SCAN_COUNT,
    "results": [],
    "updated": None,
    "error": None
}
lock = threading.Lock()
cache = {}

FALLBACK_TICKER_URL = (
    "https://raw.githubusercontent.com/ahmeterenodaci/"
    "Istanbul-Stock-Exchange--BIST--including-symbols-and-logos/main/bist.csv"
)
BIST500_PAGE = "https://www.halkaarztakvimi.com.tr/bist-500-endeks-xu500/"

def clean_symbol(x):
    x = str(x).strip().upper()
    x = x.replace(".IS", "")
    x = re.sub(r"[^A-Z0-9]", "", x)
    return x

def load_bist500():
    # 1) Try a page that publishes the current BIST 500 membership.
    try:
        tables = pd.read_html(BIST500_PAGE)
        for t in tables:
            cols = [str(c).lower() for c in t.columns]
            if any("bist kod" in c or "kod" == c for c in cols):
                col = t.columns[0]
                syms = [clean_symbol(x) for x in t[col].tolist()]
                syms = [x for x in syms if 2 <= len(x) <= 6]
                if len(syms) >= 450:
                    return list(dict.fromkeys(syms))[:SCAN_COUNT]
    except Exception:
        pass

    # 2) Fallback: KAP-derived public BIST symbol list.
    try:
        r = requests.get(FALLBACK_TICKER_URL, timeout=15)
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.text))
        col = "symbol" if "symbol" in df.columns else df.columns[0]
        syms = [clean_symbol(x) for x in df[col].tolist()]
        syms = [x for x in syms if 2 <= len(x) <= 6]
        return list(dict.fromkeys(syms))[:SCAN_COUNT]
    except Exception as e:
        raise RuntimeError(f"Hisse listesi alınamadı: {e}")

def rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0)
    dn = -d.clip(upper=0)
    au = up.ewm(alpha=1/n, adjust=False).mean()
    ad = dn.ewm(alpha=1/n, adjust=False).mean()
    rs = au / ad.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def score_stock(symbol, df):
    if df is None or df.empty or len(df) < 80:
        return None

    df = df.dropna(subset=["Close"]).copy()
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"].fillna(0)

    price = float(close.iloc[-1])
    prev = float(close.iloc[-2]) if len(close) > 1 else price
    daily = (price / prev - 1) * 100 if prev else 0

    e9 = close.ewm(span=9, adjust=False).mean()
    e21 = close.ewm(span=21, adjust=False).mean()
    e50 = close.ewm(span=50, adjust=False).mean()
    e200 = close.ewm(span=200, adjust=False).mean()

    rv = float(rsi(close).iloc[-1])
    vol20 = float(volume.rolling(20).mean().iloc[-1] or 0)
    vol5 = float(volume.rolling(5).mean().iloc[-1] or 0)
    vol_ratio = vol5 / vol20 if vol20 > 0 else 0

    resistance = float(high.rolling(20).max().shift(1).iloc[-1])
    support = float(low.rolling(20).min().shift(1).iloc[-1])
    high52 = float(high.tail(252).max())
    low52 = float(low.tail(252).min())

    ret20 = (price / float(close.iloc[-21]) - 1) * 100 if len(close) > 21 else 0
    ret60 = (price / float(close.iloc[-61]) - 1) * 100 if len(close) > 61 else 0

    std20 = float(close.pct_change().rolling(20).std().iloc[-1] or 0)
    mid = float(close.rolling(20).mean().iloc[-1])
    std = float(close.rolling(20).std().iloc[-1] or 0)
    upper = mid + 2 * std
    lower = mid - 2 * std
    band_width = (upper - lower) / mid if mid else 999

    range20 = (float(high.tail(20).max()) - float(low.tail(20).min())) / price if price else 999
    dist_res = ((resistance / price) - 1) * 100 if price else 999
    from_peak = ((price / high52) - 1) * 100 if high52 else 0

    # Sıkışma score 0-100
    squeeze = 0
    if std20 < 0.018: squeeze += 25
    elif std20 < 0.025: squeeze += 18
    elif std20 < 0.035: squeeze += 10

    if range20 < 0.08: squeeze += 25
    elif range20 < 0.12: squeeze += 18
    elif range20 < 0.16: squeeze += 10

    if band_width < 0.10: squeeze += 25
    elif band_width < 0.15: squeeze += 18
    elif band_width < 0.20: squeeze += 10

    if vol_ratio < 0.80 and vol_ratio < 1.0: squeeze += 15
    if 0 <= dist_res <= 5: squeeze += 10
    squeeze = min(100, squeeze)

    # Patlama hazırlık score
    breakout = 0
    if squeeze >= 70: breakout += 30
    elif squeeze >= 55: breakout += 20
    elif squeeze >= 40: breakout += 10
    if 45 <= rv <= 65: breakout += 15
    if e9.iloc[-1] > e21.iloc[-1]: breakout += 15
    if price > e21.iloc[-1]: breakout += 10
    if 0 <= dist_res <= 5: breakout += 20
    if vol_ratio >= 1.15: breakout += 10
    breakout = min(100, breakout)

    # Genel teknik score
    technical = 0
    if vol_ratio >= 2: technical += 30
    elif vol_ratio >= 1.5: technical += 22
    elif vol_ratio >= 1.2: technical += 12
    if daily > 0 and vol_ratio >= 1.2: technical += 15
    if e9.iloc[-1] > e21.iloc[-1]: technical += 12
    if e21.iloc[-1] > e50.iloc[-1]: technical += 10
    if e50.iloc[-1] > e200.iloc[-1]: technical += 8
    if 45 <= rv <= 65: technical += 12
    if price > resistance: technical += 15
    if -65 <= from_peak <= -45: technical += 8
    if rv > 72: technical -= 20
    technical = max(0, min(100, technical))

    # Yeni "kalite" skoru: teknik + sıkışma + kırılım + momentum
    momentum = 0
    if ret20 > 0: momentum += 8
    if ret20 > 5: momentum += 7
    if ret60 > 0: momentum += 5
    if price > e21.iloc[-1]: momentum += 5
    if price > e50.iloc[-1]: momentum += 5
    momentum = min(30, momentum)

    final_score = round(
        technical * 0.45 +
        breakout * 0.25 +
        squeeze * 0.15 +
        momentum * 0.15
    )

    if rv > 78:
        category = "⚠️ AŞIRI YÜKSELDİ"
    elif final_score >= 78 and vol_ratio >= 1.5:
        category = "🔥 GÜÇLÜ AL"
    elif breakout >= 75 and squeeze >= 65 and vol_ratio >= 1.2:
        category = "🚀 KIRILIM ADAYI"
    elif squeeze >= 78 and breakout >= 70:
        category = "🔒 GÜÇLÜ SIKIŞMA"
    elif final_score >= 55:
        category = "🟢 AL"
    elif squeeze >= 60:
        category = "🔒 SIKIŞMA"
    else:
        category = "👀 TAKİBE AL"

    entry_low = max(support, float(e21.iloc[-1]))
    entry_high = max(entry_low, price)
    risk = max(0.025, min(std20 * 1.5, 0.07))
    stop = entry_low * (1 - risk)
    target1 = max(resistance, price * 1.04)
    target2 = max(target1 * 1.04, price * 1.08)
    if resistance <= price:
        target1, target2 = price * 1.04, price * 1.08

    reasons = []
    if price > e21.iloc[-1]: reasons.append("EMA21 üzerinde")
    if e9.iloc[-1] > e21.iloc[-1]: reasons.append("Kısa trend pozitif")
    if vol_ratio >= 1.2: reasons.append("Hacim destekli")
    if 45 <= rv <= 65: reasons.append("RSI dengeli")
    if squeeze >= 70: reasons.append("Güçlü sıkışma")
    if breakout >= 70: reasons.append("Kırılım hazırlığı güçlü")
    if price > resistance: reasons.append("Direnç kırıldı")
    reasons = reasons[:5]

    return {
        "symbol": symbol,
        "price": round(price, 2),
        "daily": round(daily, 2),
        "technical": technical,
        "rsi": round(rv, 1),
        "volume_ratio": round(vol_ratio, 2),
        "squeeze": squeeze,
        "breakout": breakout,
        "score": final_score,
        "category": category,
        "support": round(support, 2),
        "resistance": round(resistance, 2),
        "entry_low": round(entry_low, 2),
        "entry_high": round(entry_high, 2),
        "stop": round(stop, 2),
        "target1": round(target1, 2),
        "target2": round(target2, 2),
        "from_peak": round(from_peak, 1),
        "reasons": reasons
    }

def normalize_download(data, symbol):
    try:
        if isinstance(data.columns, pd.MultiIndex):
            if symbol in data.columns.get_level_values(1):
                df = data.xs(symbol, axis=1, level=1)
            elif symbol in data.columns.get_level_values(0):
                df = data.xs(symbol, axis=1, level=0)
            else:
                return None
        else:
            df = data
        return df[["Open","High","Low","Close","Volume"]].dropna(how="all")
    except Exception:
        return None

def do_scan():
    global state
    try:
        symbols = load_bist500()
        symbols = symbols[:SCAN_COUNT]
        state.update({"running": True, "progress": 0, "total": len(symbols), "error": None})

        tickers = [s + ".IS" for s in symbols]
        frames = {}
        chunk_size = 75

        for start in range(0, len(tickers), chunk_size):
            chunk = tickers[start:start+chunk_size]
            try:
                data = yf.download(
                    chunk, period=PERIOD, interval=INTERVAL,
                    auto_adjust=False, progress=False, group_by="column",
                    threads=True
                )
                for s in chunk:
                    frames[s.replace(".IS","")] = normalize_download(data, s)
            except Exception:
                for s in chunk:
                    try:
                        one = yf.download(s, period=PERIOD, interval=INTERVAL,
                                          auto_adjust=False, progress=False)
                        frames[s.replace(".IS","")] = normalize_download(one, s)
                    except Exception:
                        frames[s.replace(".IS","")] = None
            state["progress"] = min(len(tickers), start + len(chunk))

        results = []
        for sym in symbols:
            r = score_stock(sym, frames.get(sym))
            if r:
                results.append(r)

        results.sort(key=lambda x: x["score"], reverse=True)

        # 3 + 3 + 3: categories have priority, then score.
        strong = [x for x in results if x["category"] == "🔥 GÜÇLÜ AL"][:3]
        buy = [x for x in results if x["category"] == "🟢 AL"][:3]
        squeeze = [x for x in results if x["category"] in ("🚀 KIRILIM ADAYI","🔒 GÜÇLÜ SIKIŞMA")] [:3]

        # Fill empty groups with best remaining candidates.
        used = {x["symbol"] for x in strong + buy + squeeze}
        if len(strong) < 3:
            for x in results:
                if x["symbol"] not in used and x["score"] >= 60:
                    strong.append(x); used.add(x["symbol"])
                    if len(strong) == 3: break
        if len(buy) < 3:
            for x in results:
                if x["symbol"] not in used and x["score"] >= 50:
                    buy.append(x); used.add(x["symbol"])
                    if len(buy) == 3: break
        if len(squeeze) < 3:
            for x in sorted(results, key=lambda z: (z["squeeze"], z["breakout"]), reverse=True):
                if x["symbol"] not in used:
                    squeeze.append(x); used.add(x["symbol"])
                    if len(squeeze) == 3: break

        gainers = sorted(results, key=lambda x: x["daily"], reverse=True)[:10]
        losers = sorted(results, key=lambda x: x["daily"])[:10]

        state["results"] = results
        state["updated"] = datetime.now().strftime("%d.%m.%Y %H:%M")
        state["running"] = False
        state["progress"] = len(symbols)

        cache["home"] = {
            "strong": strong[:3],
            "buy": buy[:3],
            "squeeze": squeeze[:3],
            "gainers": gainers,
            "losers": losers,
            "scanned": len(results),
            "updated": state["updated"]
        }
    except Exception as e:
        state["running"] = False
        state["error"] = str(e)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/scan", methods=["POST"])
def start_scan():
    if state["running"]:
        return jsonify({"ok": False, "message": "Tarama zaten devam ediyor."})
    threading.Thread(target=do_scan, daemon=True).start()
    return jsonify({"ok": True})

@app.route("/api/status")
def status():
    return jsonify(state)

@app.route("/api/home")
def home():
    return jsonify(cache.get("home", {
        "strong": [], "buy": [], "squeeze": [],
        "gainers": [], "losers": [], "scanned": 0, "updated": None
    }))

@app.route("/api/stock/<symbol>")
def stock(symbol):
    symbol = clean_symbol(symbol)
    now = time.time()
    if symbol in cache and now - cache[symbol]["time"] < CACHE_SECONDS:
        return jsonify(cache[symbol]["payload"])

    try:
        df = yf.download(symbol + ".IS", period="1y", interval="1d",
                         auto_adjust=False, progress=False)
        df = normalize_download(df, symbol + ".IS")
        if df is None or df.empty:
            return jsonify({"error": "Veri bulunamadı"}), 404

        analysis = score_stock(symbol, df)
        candles = []
        for idx, row in df.tail(220).iterrows():
            candles.append({
                "time": idx.strftime("%Y-%m-%d"),
                "open": round(float(row["Open"]), 4),
                "high": round(float(row["High"]), 4),
                "low": round(float(row["Low"]), 4),
                "close": round(float(row["Close"]), 4),
                "volume": int(row["Volume"]) if pd.notna(row["Volume"]) else 0
            })

        payload = {"analysis": analysis, "candles": candles}
        cache[symbol] = {"time": now, "payload": payload}
        return jsonify(payload)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
