from flask import Flask, render_template, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime

app = Flask(__name__)

HISTORY_FILE = "signal_history.json"


# =========================================================
# BIST HİSSELERİ
# =========================================================

BIST_HISSELERI = [
    "AEFES.IS", "AGHOL.IS", "AKBNK.IS", "AKSA.IS", "AKSEN.IS",
    "ALARK.IS", "ARCLK.IS", "ASELS.IS", "ASTOR.IS", "AYDEM.IS",
    "BIMAS.IS", "BRSAN.IS", "DOAS.IS", "ECILC.IS", "EKGYO.IS",
    "ENKAI.IS", "EREGL.IS", "FROTO.IS", "GARAN.IS", "GUBRF.IS",
    "HEKTS.IS", "ISCTR.IS", "KCHOL.IS", "KONTR.IS", "KOZAA.IS",
    "KOZAL.IS", "MGROS.IS", "OYAKC.IS", "PETKM.IS", "PGSUS.IS",
    "SAHOL.IS", "SASA.IS", "SISE.IS", "TCELL.IS", "THYAO.IS",
    "TKFEN.IS", "TOASO.IS", "TSPOR.IS", "TUPRS.IS", "YKBNK.IS"
]


# =========================================================
# RSI
# =========================================================

def rsi_hesapla(series, period=14):

    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    rsi = 100 - (100 / (1 + rs))

    return rsi


# =========================================================
# VERİ ÇEK
# =========================================================

def veri_al(ticker):

    try:

        df = yf.download(
            ticker,
            period="1y",
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False
        )

        if df is None or df.empty:
            return None, "Veri bulunamadı"

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        if "Close" not in df.columns:
            return None, "Close verisi yok"

        if "Volume" not in df.columns:
            return None, "Volume verisi yok"

        return df, None

    except Exception as e:

        return None, str(e)


# =========================================================
# TEK HİSSE ANALİZ
# =========================================================

def analiz_et(ticker):

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
            return None, "Yetersiz veri"

        c = data["close"]
        v = data["volume"]

        # =================================================
        # EMA
        # =================================================

        ema9 = c.ewm(
            span=9,
            adjust=False
        ).mean()

        ema21 = c.ewm(
            span=21,
            adjust=False
        ).mean()

        ema50 = c.ewm(
            span=50,
            adjust=False
        ).mean()

        ema200 = c.ewm(
            span=200,
            adjust=False
        ).mean()

        # =================================================
        # RSI
        # =================================================

        rsi = rsi_hesapla(c)

        # =================================================
        # HACİM
        # =================================================

        volume20 = v.rolling(20).mean()

        # =================================================
        # DESTEK / DİRENÇ
        # =================================================

        resistance20 = (
            c.shift(1)
            .rolling(20)
            .max()
        )

        support20 = (
            c.shift(1)
            .rolling(20)
            .min()
        )

        # =================================================
        # 52 HAFTA
        # =================================================

        high52 = c.rolling(
            252,
            min_periods=60
        ).max()

        low52 = c.rolling(
            252,
            min_periods=60
        ).min()

        # =================================================
        # SON DEĞERLER
        # =================================================

        fiyat = float(c.iloc[-1])

        onceki = float(c.iloc[-2])

        if onceki != 0:

            gunluk = (
                (fiyat / onceki - 1) * 100
            )

        else:

            gunluk = 0

        rsi_son = float(
            rsi.iloc[-1]
        )

        hacim_ortalama = float(
            volume20.iloc[-1]
        )

        if hacim_ortalama > 0:

            hacim_orani = (
                float(v.iloc[-1])
                / hacim_ortalama
            )

        else:

            hacim_orani = 0

        e9 = float(ema9.iloc[-1])
        e21 = float(ema21.iloc[-1])
        e50 = float(ema50.iloc[-1])
        e200 = float(ema200.iloc[-1])

        direnç = float(
            resistance20.iloc[-1]
        )

        destek = float(
            support20.iloc[-1]
        )

        zirve = float(
            high52.iloc[-1]
        )

        dip = float(
            low52.iloc[-1]
        )

        if zirve > 0:

            zirveden_uzaklik = (
                (fiyat / zirve - 1) * 100
            )

        else:

            zirveden_uzaklik = 0

        # =================================================
        # PUANLAMA
        # =================================================

        puan = 0

        nedenler = []

        # HACİM

        if hacim_orani >= 2:

            puan += 30

            nedenler.append(
                "Hacim güçlü şekilde arttı"
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
            gunluk > 0
            and hacim_orani >= 1.2
        ):

            puan += 15

            nedenler.append(
                "Fiyat ve hacim birlikte yükseliyor"
            )

        # EMA

        if e9 > e21:

            puan += 12

            nedenler.append(
                "Kısa vadeli trend pozitif"
            )

        if e21 > e50:

            puan += 10

            nedenler.append(
                "Orta vadeli trend pozitif"
            )

        if e50 > e200:

            puan += 8

            nedenler.append(
                "Uzun vadeli trend pozitif"
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

        if (
            -65 <= zirveden_uzaklik <= -45
        ):

            puan += 8

            nedenler.append(
                "52 haftalık zirveden uzak"
            )

        # AŞIRI YÜKSELME

        if rsi_son > 72:

            puan -= 20

            nedenler.append(
                "RSI aşırı yüksek"
            )

        # =================================================
        # SİNYAL
        # =================================================

        if rsi_son > 72:

            sinyal = "⚠️ AŞIRI YÜKSELDİ"

        elif (
            puan >= 78
            and hacim_orani >= 1.5
        ):

            sinyal = "🔥 GÜÇLÜ AL"

        elif puan >= 55:

            sinyal = "🟢 AL"

        elif puan >= 35:

            sinyal = "👀 TAKİBE AL"

        else:

            sinyal = "⏳ BEKLE"

        # =================================================
        # GİRİŞ
        # =================================================

        giris_alt = max(
            destek,
            e21
        )

        giris_ust = max(
            giris_alt,
            fiyat
        )

        # =================================================
        # VOLATİLİTE
        # =================================================

        gunluk_degisimler = (
            c.pct_change()
            .rolling(14)
            .std()
        )

        volatilite = float(
            gunluk_degisimler.iloc[-1]
        )

        if np.isnan(volatilite):

            volatilite = 0.02

        # =================================================
        # STOP
        # =================================================

        stop_orani = max(
            0.025,
            min(
                volatilite * 1.5,
                0.07
            )
        )

        stop = giris_alt * (
            1 - stop_orani
        )

        # =================================================
        # HEDEFLER
        # =================================================

        hedef1 = max(
            direnç,
            fiyat * 1.04
        )

        hedef2 = max(
            hedef1 * 1.04,
            fiyat * 1.08
        )

        if direnç <= fiyat:

            hedef1 = fiyat * 1.04

            hedef2 = fiyat * 1.08

        # =================================================
        # RİSK GETİRİ
        # =================================================

        risk = fiyat - stop

        getiri1 = hedef1 - fiyat

        if risk > 0:

            risk_getiri = (
                getiri1 / risk
            )

        else:

            risk_getiri = 0

        # =================================================
        # GRAFİK
        # =================================================

        grafik = []

        son_veriler = data.tail(120)

        for tarih, row in son_veriler.iterrows():

            try:

                tarih_str = tarih.strftime(
                    "%Y-%m-%d"
                )

            except:

                tarih_str = str(tarih)

            grafik.append({

                "tarih": tarih_str,

                "fiyat": round(
                    float(row["close"]),
                    2
                )

            })

        # =================================================
        # SONUÇ
        # =================================================

        temiz_hisse = ticker.replace(
            ".IS",
            ""
        )

        sonuc = {

            "hisse": temiz_hisse,

            "fiyat": round(
                fiyat,
                2
            ),

            "sinyal": sinyal,

            "puan": puan,

            "rsi": round(
                rsi_son,
                1
            ),

            "hacim_orani": round(
                hacim_orani,
                2
            ),

            "gunluk_degisim": round(
                gunluk,
                2
            ),

            "zirveden_uzaklik": round(
                zirveden_uzaklik,
                1
            ),

            "ema9": round(
                e9,
                2
            ),

            "ema21": round(
                e21,
                2
            ),

            "ema50": round(
                e50,
                2
            ),

            "ema200": round(
                e200,
                2
            ),

            "direnc": round(
                direnç,
                2
            ),

            "destek": round(
                destek,
                2
            ),

            "zirve52": round(
                zirve,
                2
            ),

            "dip52": round(
                dip,
                2
            ),

            "giris_alt": round(
                giris_alt,
                2
            ),

            "giris_ust": round(
                giris_ust,
                2
            ),

            "stop": round(
                stop,
                2
            ),

            "hedef1": round(
                hedef1,
                2
            ),

            "hedef2": round(
                hedef2,
                2
            ),

            "risk_getiri": round(
                risk_getiri,
                2
            ),

            "nedenler": nedenler,

            "grafik": grafik,

            "tradingview":
                temiz_hisse
        }

        return sonuc, None

    except Exception as e:

        return None, str(e)


# =========================================================
# GEÇMİŞ
# =========================================================

def gecmis_oku():

    try:

        if not os.path.exists(
            HISTORY_FILE
        ):

            return []

        with open(
            HISTORY_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            return json.load(f)

    except:

        return []


def gecmis_kaydet(veriler):

    try:

        with open(
            HISTORY_FILE,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                veriler[-1000:],
                f,
                ensure_ascii=False,
                indent=2
            )

    except Exception as e:

        print(
            "Geçmiş kayıt hatası:",
            e
        )


# =========================================================
# ANA SAYFA
# =========================================================

@app.route("/")
def ana_sayfa():

    return render_template(
        "index.html"
    )


# =========================================================
# TARAMA
# =========================================================

@app.route("/api/scan")
def tara():

    print(
        "=== BIST TARAMASI BAŞLADI ==="
    )

    sonuclar = []

    hatalar = []

    # -----------------------------------------------------
    # ÖNEMLİ:
    # Hisseleri tek tek bekletmek yerine
    # mümkün olduğunca hızlı çekiyoruz.
    # -----------------------------------------------------

    for i, ticker in enumerate(
        BIST_HISSELERI
    ):

        print(
            f"{i + 1}/{len(BIST_HISSELERI)} "
            f"{ticker} taranıyor..."
        )

        try:

            sonuc, hata = analiz_et(
                ticker
            )

            if sonuc is not None:

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

                    "hata":
                        hata or
                        "Bilinmeyen hata"
                })

        except Exception as e:

            hatalar.append({

                "hisse":
                    ticker.replace(
                        ".IS",
                        ""
                    ),

                "hata":
                    str(e)
            })

    # =====================================================
    # SIRALAMA
    # =====================================================

    def siralama(x):

        sinyal = x.get(
            "sinyal",
            ""
        )

        puan = x.get(
            "puan",
            0
        )

        if "GÜÇLÜ AL" in sinyal:

            kategori = 0

        elif "🟢 AL" in sinyal:

            kategori = 1

        elif "TAKİBE" in sinyal:

            kategori = 2

        else:

            kategori = 3

        return (
            kategori,
            -puan
        )

    sonuclar.sort(
        key=siralama
    )

    # =====================================================
    # GEÇMİŞ
    # =====================================================

    try:

        tarih = datetime.now().strftime(
            "%Y-%m-%d %H:%M"
        )

        gecmis = gecmis_oku()

        for x in sonuclar:

            gecmis.append({

                "tarih":
                    tarih,

                "hisse":
                    x["hisse"],

                "sinyal":
                    x["sinyal"],

                "fiyat":
                    x["fiyat"]

            })

        gecmis_kaydet(
            gecmis
        )

    except Exception as e:

        print(
            "Geçmiş hatası:",
            e
        )

    print(
        "=== TARAMA TAMAMLANDI ==="
    )

    print(
        "Bulunan:",
        len(sonuclar)
    )

    print(
        "Hatalı:",
        len(hatalar)
    )

    # =====================================================
    # MUTLAKA JSON DÖNDÜR
    # =====================================================

    return jsonify({

        "basarili": True,

        "sonuc_sayisi":
            len(sonuclar),

        "sonuclar":
            sonuclar,

        "hatalar":
            hatalar,

        "mesaj":
            "Tarama tamamlandı."

    })


# =========================================================
# TEK HİSSE DETAY
# =========================================================

@app.route(
    "/api/hisse/<ticker>"
)
def hisse_detay(ticker):

    ticker = ticker.upper()

    if not ticker.endswith(
        ".IS"
    ):

        ticker += ".IS"

    sonuc, hata = analiz_et(
        ticker
    )

    if sonuc is None:

        return jsonify({

            "basarili": False,

            "hata":
                hata or
                "Hisse verisi alınamadı."

        }), 404

    return jsonify({

        "basarili": True,

        "hisse":
            sonuc

    })


# =========================================================
# GEÇMİŞ API
# =========================================================

@app.route("/api/history")
def sinyal_gecmisi():

    try:

        veriler = gecmis_oku()

        return jsonify(
            veriler[-200:]
        )

    except Exception as e:

        return jsonify([])


# =========================================================
# HEALTH
# =========================================================

@app.route("/health")
def health():

    return jsonify({

        "status":
            "ok",

        "uygulama":
            "BIST Hisse Avcısı V6",

        "hisse_sayisi":
            len(BIST_HISSELERI),

        "durum":
            "çalışıyor"

    })


# =========================================================
# RENDER
# =========================================================

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
