from flask import Flask, render_template, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
import time
import os
import json
from datetime import datetime

app = Flask(__name__)

HISTORY_FILE = "signal_history.json"

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

    return 100 - (100 / (1 + rs))


def veri_al(ticker):

    son_hata = "Veri alınamadı"

    for deneme in range(3):

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

                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                if (
                    "Close" in df.columns
                    and "Volume" in df.columns
                ):
                    return df, None

                son_hata = "Close veya Volume verisi yok"

        except Exception as e:

            son_hata = str(e)

        time.sleep(2 * (deneme + 1))

    return None, son_hata


def hisse_analiz(ticker):

    df, hata = veri_al(ticker)

    if df is None:
        return None, hata

    try:

        close = pd.to_numeric(
            df["Close"],
            errors="coerce"
        )

        volume = pd.to_numeric(
            df["Volume"],
            errors="coerce"
        )

        data = pd.DataFrame({
            "close": close,
            "volume": volume
        }).dropna()

        if len(data) < 60:
            return None, "Yeterli veri yok"

        close = data["close"]
        volume = data["volume"]

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
        rsi = rsi_hesapla(close)

        # Ortalama hacim
        volume20 = volume.rolling(20).mean()

        # 20 günlük direnç
        resistance20 = (
            close.shift(1)
            .rolling(20)
            .max()
        )

        # 52 haftalık zirve
        high52 = (
            close
            .rolling(252, min_periods=60)
            .max()
        )

        fiyat = float(close.iloc[-1])

        onceki_fiyat = float(close.iloc[-2])

        gunluk_degisim = (
            ((fiyat / onceki_fiyat) - 1) * 100
            if onceki_fiyat != 0
            else 0
        )

        rsi_son = (
            float(rsi.iloc[-1])
            if pd.notna(rsi.iloc[-1])
            else 50
        )

        ortalama_hacim = float(
            volume20.iloc[-1]
        )

        if ortalama_hacim > 0:

            hacim_orani = (
                float(volume.iloc[-1])
                / ortalama_hacim
            )

        else:

            hacim_orani = 0

        zirve = float(
            high52.iloc[-1]
        )

        if zirve > 0:

            zirveden_uzaklik = (
                (fiyat / zirve - 1) * 100
            )

        else:

            zirveden_uzaklik = 0

        direnç = float(
            resistance20.iloc[-1]
        )

        # -------------------------
        # GİZLİ PUANLAMA
        # -------------------------

        puan = 0

        nedenler = []

        # HACİM
        if hacim_orani >= 2:

            puan += 30

            nedenler.append(
                "Hacim 2x üzeri"
            )

        elif hacim_orani >= 1.5:

            puan += 22

            nedenler.append(
                "Hacim belirgin artıyor"
            )

        elif hacim_orani >= 1.2:

            puan += 12

            nedenler.append(
                "Hacim destekli"
            )

        # FİYAT + HACİM
        if (
            gunluk_degisim > 0
            and hacim_orani >= 1.2
        ):

            puan += 15

            nedenler.append(
                "Fiyat ve hacim birlikte yükseliyor"
            )

        # KISA TREND
        if ema9.iloc[-1] > ema21.iloc[-1]:

            puan += 12

            nedenler.append(
                "Kısa vadeli trend pozitif"
            )

        # ORTA TREND
        if ema21.iloc[-1] > ema50.iloc[-1]:

            puan += 10

            nedenler.append(
                "Orta vadeli trend pozitif"
            )

        # RSI
        if 45 <= rsi_son <= 65:

            puan += 12

            nedenler.append(
                "RSI sağlıklı bölgede"
            )

        elif 65 < rsi_son <= 70:

            puan += 5

            nedenler.append(
                "RSI yükselmiş"
            )

        # DİRENÇ
        if fiyat > direnç:

            puan += 15

            nedenler.append(
                "20 günlük direnç kırıldı"
            )

        # ZİRVE UZAKLIĞI
        if -65 <= zirveden_uzaklik <= -30:

            puan += 8

            nedenler.append(
                "Zirveden hâlâ uzak"
            )

        # AŞIRI ISINMA
        if rsi_son > 72:

            puan -= 15

            nedenler.append(
                "RSI aşırı yüksek"
            )

        if gunluk_degisim > 8:

            puan -= 10

            nedenler.append(
                "Günlük yükseliş aşırı"
            )

        # -------------------------
        # SİNYAL
        # -------------------------

        if (
            rsi_son > 72
            or gunluk_degisim > 8
        ):

            sinyal = "⚠️ AŞIRI YÜKSELDİ"

        elif (
            puan >= 70
            and hacim_orani >= 1.5
        ):

            sinyal = "🟢 AL"

        elif puan >= 55:

            sinyal = "🟢 ALIM İÇİN UYGUN"

        elif puan >= 38:

            sinyal = "👀 TAKİBE AL"

        elif puan >= 22:

            sinyal = "⏳ BEKLE"

        else:

            sinyal = "🔴 UZAK DUR"

        return {

            "hisse": ticker.replace(
                ".IS",
                ""
            ),

            "fiyat": round(
                fiyat,
                2
            ),

            "sinyal": sinyal,

            "rsi": round(
                rsi_son,
                1
            ),

            "hacim_orani": round(
                hacim_orani,
                2
            ),

            "gunluk_degisim": round(
                gunluk_degisim,
                2
            ),

            "zirveden_uzaklik": round(
                zirveden_uzaklik,
                1
            ),

            "ema9": round(
                float(ema9.iloc[-1]),
                2
            ),

            "ema21": round(
                float(ema21.iloc[-1]),
                2
            ),

            "ema50": round(
                float(ema50.iloc[-1]),
                2
            ),

            "ema200": round(
                float(ema200.iloc[-1]),
                2
            ),

            "direnc20": round(
                direnç,
                2
            ),

            "nedenler": nedenler,

            # Kullanıcıya göstermiyoruz.
            "puan": int(
                max(
                    0,
                    min(100, puan)
                )
            )

        }, None

    except Exception as e:

        return None, str(e)


def gecmis_oku():

    try:

        with open(
            HISTORY_FILE,
            "r",
            encoding="utf-8"
        ) as dosya:

            return json.load(dosya)

    except:

        return []


def gecmis_kaydet(veriler):

    with open(
        HISTORY_FILE,
        "w",
        encoding="utf-8"
    ) as dosya:

        json.dump(
            veriler[-1000:],
            dosya,
            ensure_ascii=False,
            indent=2
        )


@app.route("/")
def ana_sayfa():

    return render_template(
        "index.html"
    )


@app.route("/api/scan")
def tara():

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

                "hisse":
                    ticker.replace(
                        ".IS",
                        ""
                    ),

                "hata": hata
            })

        time.sleep(0.7)

    # En güçlü hisseler üstte
    sonuclar.sort(
        key=lambda x: x["puan"],
        reverse=True
    )

    # Sinyal geçmişine kaydet
    tarih = datetime.now().strftime(
        "%Y-%m-%d %H:%M"
    )

    gecmis = gecmis_oku()

    for sonuc in sonuclar:

        gecmis.append({

            "tarih": tarih,

            "hisse":
                sonuc["hisse"],

            "sinyal":
                sonuc["sinyal"],

            "fiyat":
                sonuc["fiyat"]

        })

    gecmis_kaydet(
        gecmis
    )

    return jsonify({

        "basarili": True,

        "sonuc_sayisi":
            len(sonuclar),

        "sonuclar":
            sonuclar,

        "hatalar":
            hatalar

    })


@app.route("/api/history")
def sinyal_gecmisi():

    gecmis = gecmis_oku()

    return jsonify(
        gecmis[-200:]
    )


@app.route("/health")
def saglik():

    return jsonify({

        "status": "ok",

        "uygulama":
            "BIST Hisse Avcısı V5"

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
