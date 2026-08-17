from flask import Flask, render_template, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
import os
import json
import threading
from datetime import datetime

app = Flask(__name__)

HISTORY_FILE = "signal_history.json"

# ============================================================
# BIST HİSSE LİSTESİ
# ============================================================

FALLBACK_HISSELER = """
AEFES AGHOL AKBNK AKSA AKSEN ALARK ARCLK ASELS ASTOR BIMAS BRSAN DOAS
ECILC EKGYO ENKAI EREGL FROTO GARAN GUBRF HEKTS ISCTR KCHOL KONTR KOZAA
KOZAL MGROS OYAKC PETKM PGSUS SAHOL SASA SISE TCELL THYAO TKFEN TOASO
TUPRS YKBNK KAREL JANTS TARKM YEOTK MIATK REEDR ALFAS CWENE GESAN ODAS
ZOREN ENJSA SMRTG GWIND CANTE KONKA LINK LIDFA LYDHO ASGYO AGYO AKGRT
AKFGY ALGYO AVHOL BAGFS BERA BINHO BINBT BNTAS BRYAT BUCIM CIMSA CLEBI
DOHOL DGNMO EGEEN ENERY FMIZP GLYHO HALKB ICBCT INDES IPEKE ISDMR ISFIN
ISGSY ISMEN IZMDC KARSN KCAER KERVT KLGYO KMPUR LOGO MAVI MEPET MPARK
NTHOL NUHCM OBASE ORGE OTKAR OYAYO PENTA POLHO QUAGR RALYH RYSAS SARKY
SELEC SKBNK SMART SOKM TATEN TATGD TAVHL TEZOL TKNSA TRCAS TRGYO TSKB
TTKOM TTRAK TUKAS ULKER ULUUN VAKBN VAKKO VESBE VESTL YATAS YIGIT YYLGD
""".split()

BIST_HISSELERI = sorted(
    set(
        hisse + ".IS"
        for hisse in FALLBACK_HISSELER
    )
)


# ============================================================
# TARMA DURUMU
# ============================================================

tarama = {
    "running": False,
    "progress": 0,
    "total": len(BIST_HISSELERI),
    "results": [],
    "errors": [],
    "message": "Hazır",
    "started": None,
    "finished": None
}

tarama_lock = threading.Lock()


# ============================================================
# RSI
# ============================================================

def rsi_hesapla(series, period=14):

    delta = series.diff()

    gain = delta.clip(lower=0)

    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()

    avg_loss = loss.rolling(period).mean()

    rs = (
        avg_gain /
        avg_loss.replace(0, np.nan)
    )

    return 100 - (
        100 /
        (1 + rs)
    )


# ============================================================
# DATA TEMİZLE
# ============================================================

def dataframe_temizle(df):

    if df is None or df.empty:
        return None

    if isinstance(
        df.columns,
        pd.MultiIndex
    ):

        df.columns = (
            df.columns
            .get_level_values(0)
        )

    if (
        "Close" not in df.columns
        or
        "Volume" not in df.columns
    ):

        return None

    data = pd.DataFrame({

        "close":
            pd.to_numeric(
                df["Close"],
                errors="coerce"
            ),

        "volume":
            pd.to_numeric(
                df["Volume"],
                errors="coerce"
            )

    }).dropna()

    if len(data) < 60:
        return None

    return data


# ============================================================
# ANALİZ
# ============================================================

def analiz_et(df, ticker):

    data = dataframe_temizle(df)

    if data is None:
        return None

    try:

        c = data["close"]

        v = data["volume"]

        # ----------------------------------------------------
        # EMA
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # RSI
        # ----------------------------------------------------

        rsi = rsi_hesapla(c)

        # ----------------------------------------------------
        # HACİM
        # ----------------------------------------------------

        volume20 = v.rolling(20).mean()

        volume5 = v.rolling(5).mean()

        # ----------------------------------------------------
        # DESTEK / DİRENÇ
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # 52 HAFTA
        # ----------------------------------------------------

        high52 = (
            c.rolling(
                252,
                min_periods=60
            ).max()
        )

        low52 = (
            c.rolling(
                252,
                min_periods=60
            ).min()
        )

        # ----------------------------------------------------
        # FİYAT
        # ----------------------------------------------------

        fiyat = float(c.iloc[-1])

        onceki = float(c.iloc[-2])

        gunluk_degisim = (
            (
                fiyat / onceki - 1
            ) * 100
            if onceki != 0
            else 0
        )

        rsi_son = float(
            rsi.iloc[-1]
        )

        if np.isnan(rsi_son):
            rsi_son = 50

        hacim_ortalama = float(
            volume20.iloc[-1]
        )

        if (
            np.isnan(
                hacim_ortalama
            )
            or
            hacim_ortalama <= 0
        ):

            hacim_ortalama = 1

        hacim5 = float(
            volume5.iloc[-1]
        )

        if np.isnan(hacim5):
            hacim5 = hacim_ortalama

        son_hacim = float(
            v.iloc[-1]
        )

        hacim_orani = (
            son_hacim /
            hacim_ortalama
        )

        kisa_hacim_orani = (
            hacim5 /
            hacim_ortalama
        )

        e9 = float(
            ema9.iloc[-1]
        )

        e21 = float(
            ema21.iloc[-1]
        )

        e50 = float(
            ema50.iloc[-1]
        )

        e200 = float(
            ema200.iloc[-1]
        )

        direnç = float(
            resistance20.iloc[-1]
        )

        destek = float(
            support20.iloc[-1]
        )

        if np.isnan(direnç):
            direnç = fiyat

        if np.isnan(destek):
            destek = fiyat

        zirve = float(
            high52.iloc[-1]
        )

        dip = float(
            low52.iloc[-1]
        )

        if np.isnan(zirve):
            zirve = fiyat

        if np.isnan(dip):
            dip = fiyat

        zirveden_uzaklik = (
            (
                fiyat / zirve - 1
            ) * 100
            if zirve > 0
            else 0
        )

        # ====================================================
        # SIKIŞMA
        # ====================================================

        getiriler = c.pct_change()

        volatilite20 = (
            getiriler
            .rolling(20)
            .std()
            .iloc[-1]
        )

        if pd.isna(volatilite20):
            volatilite20 = 0.03

        son20_yuksek = float(
            c.tail(20).max()
        )

        son20_dusuk = float(
            c.tail(20).min()
        )

        fiyat_aralik = (
            (
                son20_yuksek -
                son20_dusuk
            )
            / fiyat
            if fiyat > 0
            else 1
        )

        orta20 = c.rolling(20).mean()

        std20 = c.rolling(20).std()

        ust_band = (
            orta20 +
            2 * std20
        )

        alt_band = (
            orta20 -
            2 * std20
        )

        if (
            orta20.iloc[-1] > 0
        ):

            bant_genisligi = (
                (
                    ust_band.iloc[-1] -
                    alt_band.iloc[-1]
                )
                /
                orta20.iloc[-1]
            )

        else:

            bant_genisligi = 1

        # ----------------------------------------------------
        # SIKIŞMA PUANI
        # ----------------------------------------------------

        sikisma_puani = 0

        sikisma_nedenleri = []

        if volatilite20 < 0.018:

            sikisma_puani += 25

            sikisma_nedenleri.append(
                "20 günlük volatilite çok düşük"
            )

        elif volatilite20 < 0.025:

            sikisma_puani += 18

            sikisma_nedenleri.append(
                "Volatilite düşük"
            )

        elif volatilite20 < 0.035:

            sikisma_puani += 10

        if fiyat_aralik < 0.08:

            sikisma_puani += 25

            sikisma_nedenleri.append(
                "Fiyat son 20 günde dar bantta"
            )

        elif fiyat_aralik < 0.12:

            sikisma_puani += 18

        elif fiyat_aralik < 0.16:

            sikisma_puani += 10

        if bant_genisligi < 0.10:

            sikisma_puani += 25

            sikisma_nedenleri.append(
                "Bollinger bantları sıkıştı"
            )

        elif bant_genisligi < 0.15:

            sikisma_puani += 18

        elif bant_genisligi < 0.20:

            sikisma_puani += 10

        if (
            kisa_hacim_orani < 0.80
            and
            hacim_orani < 1
        ):

            sikisma_puani += 15

            sikisma_nedenleri.append(
                "Hacim kuruyor"
            )

        direnç_mesafe = (
            (
                direnç -
                fiyat
            )
            /
            fiyat *
            100
            if fiyat > 0
            else 999
        )

        if (
            0 <=
            direnç_mesafe <= 5
        ):

            sikisma_puani += 10

            sikisma_nedenleri.append(
                "Dirence çok yakın"
            )

        sikisma_puani = min(
            100,
            max(
                0,
                int(
                    sikisma_puani
                )
            )
        )

        # ====================================================
        # PATLAMA
        # ====================================================

        if sikisma_puani >= 70:

            patlama_puani = 30

        elif sikisma_puani >= 55:

            patlama_puani = 20

        elif sikisma_puani >= 40:

            patlama_puani = 10

        else:

            patlama_puani = 0

        patlama_nedenleri = []

        if 45 <= rsi_son <= 65:

            patlama_puani += 15

            patlama_nedenleri.append(
                "RSI aşırı alımda değil"
            )

        if e9 > e21:

            patlama_puani += 15

            patlama_nedenleri.append(
                "Kısa trend yukarı dönüyor"
            )

        if fiyat > e21:

            patlama_puani += 10

        if (
            0 <=
            direnç_mesafe <= 5
        ):

            patlama_puani += 20

            patlama_nedenleri.append(
                "Kırılmaya yakın direnç"
            )

        if kisa_hacim_orani >= 1.15:

            patlama_puani += 10

            patlama_nedenleri.append(
                "Hacimde kıpırdanma başladı"
            )

        patlama_puani = min(
            100,
            int(
                patlama_puani
            )
        )

        # ====================================================
        # NORMAL PUAN
        # ====================================================

        puan = 0

        nedenler = []

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

        if (
            gunluk_degisim > 0
            and
            hacim_orani >= 1.2
        ):

            puan += 15

            nedenler.append(
                "Fiyat ve hacim birlikte yükseliyor"
            )

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

        if 45 <= rsi_son <= 65:

            puan += 12

            nedenler.append(
                "RSI sağlıklı bölgede"
            )

        elif 65 < rsi_son <= 70:

            puan += 5

        if fiyat > direnç:

            puan += 15

            nedenler.append(
                "20 günlük direnç kırıldı"
            )

        if (
            -65 <=
            zirveden_uzaklik <=
            -45
        ):

            puan += 8

            nedenler.append(
                "52 haftalık zirveden uzak"
            )

        if rsi_son > 72:

            puan -= 20

            nedenler.append(
                "RSI aşırı yüksek"
            )

        puan = max(
            0,
            min(
                100,
                int(puan)
            )
        )

        # ====================================================
        # SİNYAL
        # ====================================================

        if rsi_son > 72:

            sinyal = "⚠️ AŞIRI YÜKSELDİ"

        elif (
            patlama_puani >= 75
            and
            sikisma_puani >= 65
            and
            hacim_orani >= 1.2
        ):

            sinyal = "🔥 SIKIŞMA ÇÖZÜLÜYOR"

        elif (
            puan >= 78
            and
            hacim_orani >= 1.5
        ):

            sinyal = "🔥 GÜÇLÜ AL"

        elif (
            sikisma_puani >= 75
            and
            patlama_puani >= 50
        ):

            sinyal = "🔒 GÜÇLÜ SIKIŞMA"

        elif puan >= 55:

            sinyal = "🟢 AL"

        elif sikisma_puani >= 60:

            sinyal = "🔒 SIKIŞMA"

        elif puan >= 35:

            sinyal = "👀 TAKİBE AL"

        else:

            sinyal = "⏳ BEKLE"

        # ====================================================
        # GİRİŞ / STOP / HEDEF
        # ====================================================

        gunluk_vol = (
            c.pct_change()
            .rolling(14)
            .std()
            .iloc[-1]
        )

        if pd.isna(gunluk_vol):

            gunluk_vol = 0.02

        giris_alt = max(
            destek,
            e21
        )

        giris_ust = max(
            giris_alt,
            fiyat
        )

        stop_yuzdesi = max(
            0.025,
            min(
                float(gunluk_vol) * 1.5,
                0.07
            )
        )

        stop = (
            giris_alt *
            (
                1 -
                stop_yuzdesi
            )
        )

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

        risk = fiyat - stop

        getiri1 = hedef1 - fiyat

        if risk > 0:

            risk_getiri = (
                getiri1 /
                risk
            )

        else:

            risk_getiri = 0

        # ====================================================
        # GRAFİK
        # ====================================================

        grafik = []

        for tarih, row in (
            data.tail(120).iterrows()
        ):

            try:

                tarih_text = (
                    tarih.strftime(
                        "%Y-%m-%d"
                    )
                )

            except:

                tarih_text = str(
                    tarih
                )

            grafik.append({

                "tarih":
                    tarih_text,

                "fiyat":
                    round(
                        float(
                            row["close"]
                        ),
                        2
                    )

            })

        return {

            "hisse":
                ticker.replace(
                    ".IS",
                    ""
                ),

            "fiyat":
                round(
                    fiyat,
                    2
                ),

            "sinyal":
                sinyal,

            "puan":
                puan,

            "rsi":
                round(
                    rsi_son,
                    1
                ),

            "hacim_orani":
                round(
                    hacim_orani,
                    2
                ),

            "gunluk_degisim":
                round(
                    gunluk_degisim,
                    2
                ),

            "zirveden_uzaklik":
                round(
                    zirveden_uzaklik,
                    1
                ),

            "ema9":
                round(
                    e9,
                    2
                ),

            "ema21":
                round(
                    e21,
                    2
                ),

            "ema50":
                round(
                    e50,
                    2
                ),

            "ema200":
                round(
                    e200,
                    2
                ),

            "direnc":
                round(
                    direnç,
                    2
                ),

            "destek":
                round(
                    destek,
                    2
                ),

            "zirve52":
                round(
                    zirve,
                    2
                ),

            "dip52":
                round(
                    dip,
                    2
                ),

            "giris_alt":
                round(
                    giris_alt,
                    2
                ),

            "giris_ust":
                round(
                    giris_ust,
                    2
                ),

            "stop":
                round(
                    stop,
                    2
                ),

            "hedef1":
                round(
                    hedef1,
                    2
                ),

            "hedef2":
                round(
                    hedef2,
                    2
                ),

            "risk_getiri":
                round(
                    risk_getiri,
                    2
                ),

            "sikisma_puani":
                sikisma_puani,

            "patlama_puani":
                patlama_puani,

            "volatilite20":
                round(
                    volatilite20 * 100,
                    2
                ),

            "fiyat_aralik":
                round(
                    fiyat_aralik * 100,
                    2
                ),

            "bant_genisligi":
                round(
                    bant_genisligi * 100,
                    2
                ),

            "direnc_mesafe":
                round(
                    direnç_mesafe,
                    2
                ),

            "sikisma_nedenleri":
                sikisma_nedenleri,

            "patlama_nedenleri":
                patlama_nedenleri,

            "nedenler":
                nedenler,

            "grafik":
                grafik,

            "tradingview":
                ticker.replace(
                    ".IS",
                    ""
                )

        }

    except Exception:

        return None


# ============================================================
# TOPLU VERİ İNDİRME
# ============================================================

def toplu_veri_al():

    try:

        df = yf.download(

            BIST_HISSELERI,

            period="1y",

            interval="1d",

            auto_adjust=False,

            progress=False,

            threads=True,

            group_by="column"

        )

        return df, None

    except Exception as e:

        return None, str(e)


# ============================================================
# ARKA PLAN TARAMA
# ============================================================

def tarama_worker():

    with tarama_lock:

        tarama["running"] = True

        tarama["progress"] = 0

        tarama["total"] = len(
            BIST_HISSELERI
        )

        tarama["results"] = []

        tarama["errors"] = []

        tarama["message"] = (
            "BIST verileri toplu indiriliyor..."
        )

        tarama["started"] = (
            datetime.now()
            .strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        )

        tarama["finished"] = None

    results = []

    errors = []

    raw, hata = toplu_veri_al()

    if raw is None:

        with tarama_lock:

            tarama["running"] = False

            tarama["message"] = (
                "Veri alınamadı: " +
                str(hata)
            )

            tarama["errors"] = [
                {
                    "hata":
                        str(hata)
                }
            ]

        return

    # ========================================================
    # HER HİSSEYİ ANALİZ ET
    # ========================================================

    for sira, ticker in enumerate(
        BIST_HISSELERI,
        start=1
    ):

        try:

            if isinstance(
                raw.columns,
                pd.MultiIndex
            ):

                # yfinance kolon yapısını bul
                hisse_df = None

                for level in range(
                    raw.columns.nlevels
                ):

                    try:

                        if ticker in (
                            raw.columns
                            .get_level_values(
                                level
                            )
                        ):

                            hisse_df = (
                                raw.xs(
                                    ticker,
                                    axis=1,
                                    level=level
                                )
                            )

                            break

                    except Exception:

                        pass

                if hisse_df is None:

                    raise Exception(
                        "Hisse verisi bulunamadı"
                    )

            else:

                hisse_df = raw

            sonuc = analiz_et(
                hisse_df,
                ticker
            )

            if sonuc:

                results.append(
                    sonuc
                )

            else:

                errors.append({

                    "hisse":
                        ticker.replace(
                            ".IS",
                            ""
                        ),

                    "hata":
                        "Yetersiz veri"

                })

        except Exception as e:

            errors.append({

                "hisse":
                    ticker.replace(
                        ".IS",
                        ""
                    ),

                "hata":
                    str(e)

            })

        # ====================================================
        # DURUMU GÜNCELLE
        # ====================================================

        with tarama_lock:

            tarama["progress"] = sira

            tarama["results"] = sorted(

                results,

                key=lambda x: (

                    x.get(
                        "patlama_puani",
                        0
                    ),

                    x.get(
                        "sikisma_puani",
                        0
                    ),

                    x.get(
                        "puan",
                        0
                    )

                ),

                reverse=True

            )

            tarama["errors"] = errors

            tarama["message"] = (
                f"{sira}/"
                f"{len(BIST_HISSELERI)} "
                "hisse analiz edildi"
            )

    # ========================================================
    # GEÇMİŞİ KAYDET
    # ========================================================

    tarih = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    try:

        with open(
            HISTORY_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            gecmis = json.load(f)

    except Exception:

        gecmis = []

    for sonuc in results:

        gecmis.append({

            "tarih":
                tarih,

            "hisse":
                sonuc["hisse"],

            "sinyal":
                sonuc["sinyal"],

            "fiyat":
                sonuc["fiyat"],

            "sikisma_puani":
                sonuc[
                    "sikisma_puani"
                ],

            "patlama_puani":
                sonuc[
                    "patlama_puani"
                ]

        })

    try:

        with open(
            HISTORY_FILE,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                gecmis[-1000:],
                f,
                ensure_ascii=False,
                indent=2
            )

    except Exception:

        pass

    with tarama_lock:

        tarama["running"] = False

        tarama["finished"] = tarih

        tarama["message"] = (
            "Tarama tamamlandı"
        )


# ============================================================
# ANA SAYFA
# ============================================================

@app.route("/")
def ana_sayfa():

    return render_template(
        "index.html"
    )


# ============================================================
# TARAMAYI BAŞLAT
# ============================================================

@app.route(
    "/api/scan"
)
def tara():

    with tarama_lock:

        if tarama["running"]:

            return jsonify({

                "basarili":
                    True,

                "running":
                    True,

                "message":
                    "Tarama zaten devam ediyor"

            })

        thread = threading.Thread(

            target=tarama_worker,

            daemon=True

        )

        thread.start()

    return jsonify({

        "basarili":
            True,

        "running":
            True,

        "message":
            "Tarama arka planda başlatıldı"

    })


# ============================================================
# TARMA DURUMU
# ============================================================

@app.route(
    "/api/status"
)
def durum():

    with tarama_lock:

        return jsonify({

            "basarili":
                True,

            "running":
                tarama["running"],

            "progress":
                tarama["progress"],

            "total":
                tarama["total"],

            "results":
                tarama["results"],

            "errors":
                tarama["errors"],

            "message":
                tarama["message"],

            "started":
                tarama["started"],

            "finished":
                tarama["finished"]

        })


# ============================================================
# TEK HİSSE
# ============================================================

@app.route(
    "/api/hisse/<ticker>"
)
def hisse_detay(ticker):

    ticker = ticker.upper()

    if not ticker.endswith(
        ".IS"
    ):

        ticker += ".IS"

    try:

        df = yf.download(

            ticker,

            period="1y",

            interval="1d",

            auto_adjust=False,

            progress=False,

            threads=False

        )

        sonuc = analiz_et(
            df,
            ticker
        )

        if sonuc is None:

            return jsonify({

                "basarili":
                    False,

                "hata":
                    "Veri alınamadı"

            }), 404

        return jsonify({

            "basarili":
                True,

            "hisse":
                sonuc

        })

    except Exception as e:

        return jsonify({

            "basarili":
                False,

            "hata":
                str(e)

        }), 500


# ============================================================
# GEÇMİŞ
# ============================================================

@app.route(
    "/api/history"
)
def sinyal_gecmisi():

    try:

        with open(
            HISTORY_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            gecmis = json.load(f)

        return jsonify(
            gecmis[-200:]
        )

    except Exception:

        return jsonify([])


# ============================================================
# SİSTEM BİLGİSİ
# ============================================================

@app.route(
    "/api/info"
)
def sistem_bilgisi():

    return jsonify({

        "status":
            "ok",

        "uygulama":
            "BIST Hisse Avcısı V9",

        "hisse_sayisi":
            len(
                BIST_HISSELERI
            ),

        "sikisma_sistemi":
            True,

        "patlama_sistemi":
            True,

        "batch_veri":
            True,

        "arka_plan_tarama":
            True

    })


# ============================================================
# HEALTH
# ============================================================

@app.route(
    "/health"
)
def health():

    return jsonify({

        "status":
            "ok",

        "uygulama":
            "BIST Hisse Avcısı V9",

        "hisse_sayisi":
            len(
                BIST_HISSELERI
            )

    })


# ============================================================
# ÇALIŞTIR
# ============================================================

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
