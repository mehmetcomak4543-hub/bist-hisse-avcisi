from flask import Flask, render_template, jsonify
import yfinance as yf
import pandas as pd
from datetime import datetime, timezone

app = Flask(__name__)

# İlk test listesi. Sonraki sürümde otomatik tüm BIST listesine bağlanacak.
SYMBOLS = ["AKSEN","KARSN","MAGEN","RAYSG","HEKTS","LINK","ASGYO","EGEEN","THYAO","ASELS"]

def fetch_symbol(symbol):
    df = yf.download(
        symbol + ".IS", period="2y", interval="1d",
        auto_adjust=False, progress=False
    )
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna(subset=["Close"])
    if len(df) < 60:
        return None

    c = df["Close"].astype(float)
    v = df["Volume"].astype(float)

    ema9 = c.ewm(span=9, adjust=False).mean()
    ema21 = c.ewm(span=21, adjust=False).mean()
    ema50 = c.ewm(span=50, adjust=False).mean()
    ema200 = c.ewm(span=200, adjust=False).mean()

    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, pd.NA)
    rsi = (100 - 100 / (1 + rs)).fillna(50)

    avgvol = v.rolling(20).mean()
    volx = float(v.iloc[-1] / avgvol.iloc[-1]) if avgvol.iloc[-1] else 1.0

    price = float(c.iloc[-1])
    prev = float(c.iloc[-2])
    change = (price / prev - 1) * 100
    high20 = float(df["High"].tail(20).max())
    high52 = float(df["High"].tail(252).max())

    score = 0
    score += 15 if ema9.iloc[-1] > ema21.iloc[-1] else 0
    score += 15 if ema21.iloc[-1] > ema50.iloc[-1] else 0
    score += 10 if ema50.iloc[-1] > ema200.iloc[-1] else 0
    score += 15 if 50 <= rsi.iloc[-1] <= 70 else (8 if 45 <= rsi.iloc[-1] < 50 else 0)
    score += 20 if volx >= 3 else (14 if volx >= 2 else (7 if volx >= 1.5 else 0))
    score += 15 if change > 0 else 0
    score += 10 if price >= high20 else 0

    signal = "GÜÇLÜ AL" if score >= 85 else ("AL" if score >= 75 else ("İZLE" if score >= 60 else "BEKLE"))

    return {
        "symbol": symbol,
        "price": round(price, 2),
        "change": round(change, 2),
        "vol": round(volx, 2),
        "rsi": round(float(rsi.iloc[-1]), 1),
        "score": int(score),
        "signal": signal,
        "trend": "POZİTİF" if ema9.iloc[-1] > ema21.iloc[-1] else "ZAYIF",
        "breakout": "KIRILDI" if price >= high20 else "YOK",
        "distance52": round((price / high52 - 1) * 100, 2),
        "date": str(df.index[-1].date())
    }

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/scan")
def scan():
    results, errors = [], []
    for symbol in SYMBOLS:
        try:
            row = fetch_symbol(symbol)
            if row:
                results.append(row)
            else:
                errors.append(symbol)
        except Exception as exc:
            errors.append(symbol)
    results.sort(key=lambda x: x["score"], reverse=True)
    return jsonify({
        "updated": datetime.now(timezone.utc).isoformat(),
        "source": "Yahoo Finance / yfinance",
        "count": len(results),
        "data": results,
        "errors": errors
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
