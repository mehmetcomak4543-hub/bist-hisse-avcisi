from flask import Flask, render_template, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
import time
import os

app = Flask(__name__)

BIST_HISSELERI = [
    "ASELS.IS",
    "THYAO.IS",
    "TUPRS.IS",
    "EREGL.IS",
    "SISE.IS",
    "HEKTS.IS",
    "KCHOL.IS",
    "AKBNK.IS",
    "YKBNK.IS",
    "GARAN.IS",
    "ISCTR.IS",
    "PETKM.IS",
    "FROTO.IS",
    "TOASO.IS",
    "BIMAS.IS",
    "SAHOL.IS",
    "TCELL.IS",
    "PGSUS.IS",
    "ENKAI.IS",
    "ALARK.IS"
]


def rsi_hesapla(series, period=14):
    delta = series.diff()

    kazanc = delta.clip(lower=0)
    kayip = -delta.clip(upper=0)

    ort_kazanc = kazanc.rolling(period).mean()
    ort_kayip = kayip.rolling(period).mean()

    rs = ort_kazanc / ort_kayip.replace(0, np.nan)

    rsi = 100 - (100 / (1 + rs))

    return rsi


def yahoo_verisi_al(ticker, deneme=3):

    son_hata = "Yahoo veri döndürmedi"

    for i in range(deneme):

        try:

            df = yf.download(
                ticker,
                period="1y",
                interval="1d",
                auto_adjust=False,
                progress=False,
                threads=False,
                timeout=20
            )

            if df is not None and not df.empty:

                # MultiIndex problemi
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                if "Close" in df.columns and "Volume" in df.columns:
                    return df, None

                son_hata = "Close/Volume verisi bulunamadı"

            else:
                son_hata = "Yahoo boş veri döndürdü"

        except Exception as e:

            son_hata = str(e)

        time.sleep(2 * (i + 1))

    return None, son_hata


def hisse_analiz(ticker):

    try:

        df, hata = yahoo_verisi_al(ticker)

        if df is None:
            return None, hata

        df = df.copy()

        df["Close"] = pd.to_numeric(
            df["Close"],
            errors="coerce"
        )

        df["Volume"] = pd.to_numeric(
            df["Volume"],
            errors="coerce"
        )

        df = df.dropna(subset=["Close"])

        if len(df) < 60:
            return None, "Yeterli geçmiş veri yok"

        close = df["Close"]

        volume = df["Volume"].reindex(
            close.index
        ).fillna(0)

        # EMA
        ema9 = close.ewm(
            span=9,
            adjust=False
        ).mean()

        ema21 = close.ewm(
            span=21,
            adjust=False
        ).mean()

        ema50 = close.ewm(
            span=50,
            adjust=False
        ).mean()

        ema200 = close.ewm(
            span=200,
            adjust=False
        ).mean()

        # RSI
        rsi = rsi_hesapla(
            close,
            14
        )

        # Hacim
        volume20 = volume.rolling(20).mean()

        # 20 günlük direnç
        resistance20 = close.shift(1).rolling(20).max()

        fiyat = float(close.iloc[-1])

        onceki = float(close.iloc[-2])

        gunluk_degisim = (
            ((fiyat / onceki) - 1) * 100
            if onceki
            else 0
        )

        rsi_son = (
            float(rsi.iloc[-1])
            if pd.notna(rsi.iloc[-1])
            else 50
        )

        hacim_son = float(
            volume.iloc[-1]
        )

        hacim_ort = (
            float(volume20.iloc[-1])
            if pd.notna(volume20.iloc[-1])
            else 0
        )

        hacim_orani = (
            hacim_son / hacim_ort
            if hacim_ort > 0
            else 0
        )

        ema9_son = float(
            ema9.iloc[-1]
        )

        ema21_son = float(
            ema21.iloc[-1]
        )

        ema50_son = float(
            ema50.iloc[-1]
        )

        ema200_son = float(
            ema200.iloc[-1]
        )

        direnç = (
            float(resistance20.iloc[-1])
            if pd.notna(resistance20.iloc[-1])
            else fiyat
        )

        # -------------------------
        # TEKNİK PUAN
        # -------------------------

        puan = 0

        sinyaller = []

        # EMA9 > EMA21
        if ema9_son > ema21_son:

            puan += 15

            sinyaller.append(
                "EMA9 > EMA21"
            )

        # Fiyat EMA21 üzerinde
        if fiyat >= ema21_son:

            puan += 10

            sinyaller.append(
                "Fiyat EMA21 üzerinde"
            )

        # EMA21 > EMA50
        if ema21_son > ema50_son:

            puan += 10

            sinyaller.append(
                "EMA21 > EMA50"
            )

        # Fiyat EMA50 üzerinde
        if fiyat >= ema50_son:

            puan += 10

            sinyaller.append(
                "Fiyat EMA50 üzerinde"
            )

        # Uzun vadeli trend
        if fiyat > ema200_son:

            puan += 10

            sinyaller.append(
                "EMA200 üzerinde"
            )

        # RSI
        if 50 <= rsi_son <= 68:

            puan += 15

            sinyaller.append(
                "Sağlıklı RSI"
            )

        elif 40 <= rsi_son < 50:

            puan += 7

            sinyaller.append(
                "RSI toparlanıyor"
            )

        elif rsi_son > 70:

            puan -= 5

            sinyaller.append(
                "RSI yüksek"
            )

        # Hacim
        if hacim_orani >= 2:

            puan += 15

            sinyaller.append(
                "Çok güçlü hacim"
            )

        elif hacim_orani >= 1.5:

            puan += 10

            sinyaller.append(
                "Hacim artışı"
            )

        elif hacim_orani >= 1.2:

            puan += 5

            sinyaller.append(
                "Hacim destekli"
            )

        # Direnç kırılımı
        if fiyat > direnç:

            puan += 15

            sinyaller.append(
                "20G direnç kırılımı"
            )

        # Günlük momentum
        if gunluk_degisim > 3:

            puan += 5

            sinyaller.append(
                "Güçlü günlük momentum"
            )

        # 0-100 arası sınırla
        puan = max(
            0,
            min(100, puan)
        )

        # Hisse adını temizle
        isim = ticker.replace(
            ".IS",
            ""
        )

        sonuc = {

            "hisse": isim,

            "fiyat": round(
                fiyat,
                2
            ),

            "puan": int(puan),

            "rsi": round(
                rsi_son,
                1
            ),

            "gunluk_degisim": round(
                gunluk_degisim,
                2
            ),

            "hacim_orani": round(
                hacim_orani,
                2
            ),

            "ema9": round(
                ema9_son,
                2
            ),

            "ema21": round(
                ema21_son,
                2
            ),

            "ema50": round(
                ema50_son,
                2
            ),

            "ema200": round(
                ema200_son,
                2
            ),

            "direnc20": round(
                direnç,
                2
            ),

            "sinyaller": sinyaller
        }

        return sonuc, None

    except Exception as e:

        return None, str(e)


@app.route("/")
def index():

    return render_template(
        "index.html"
    )


@app.route("/api/scan")
def scan():

    sonuclar = []

    hatalar = []

    for ticker in BIST_HISSELERI:

        sonuc, hata = hisse_analiz(
            ticker
        )

        if sonuc:

            sonuclar.append(
                sonuc
            )

        else:

            hatalar.append({

                "hisse": ticker.replace(
                    ".IS",
                    ""
                ),

                "hata": hata
            })

        # Yahoo'ya aşırı hızlı istek göndermemek için
        time.sleep(0.8)

    # En yüksek puan önce
    sonuclar.sort(
        key=lambda x: (
            x["puan"],
            x["hacim_orani"],
            x["gunluk_degisim"]
        ),
        reverse=True
    )

    return jsonify({

        "basarili": True,

        "toplam": len(
            BIST_HISSELERI
        ),

        "sonuc_sayisi": len(
            sonuclar
        ),

        "sonuclar": sonuclar,

        "hatalar": hatalar
    })


@app.route("/health")
def health():

    return jsonify({

        "status": "ok",

        "uygulama":
            "BIST Hisse Avcısı V4"
    })


if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    app.run(

        host="0.0.0.0",

        port=port,

        debug=False
    )
