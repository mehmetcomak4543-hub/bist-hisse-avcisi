from flask import Flask, render_template, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
import os
import json
import time
import requests
import zipfile
import io
import re
from datetime import datetime, timedelta

app = Flask(__name__)

HISTORY_FILE = "signal_history.json"
UNIVERSE_FILE = "bist_universe.json"


# ============================================================
# FALLBACK LİSTE
# BIST resmi listesinden otomatik liste alınamazsa kullanılır.
# ============================================================

FALLBACK_HISSELER = [
    "AEFES","AGHOL","AKBNK","AKSA","AKSEN","ALARK","ARCLK",
    "ASELS","ASTOR","AYDEM","BIMAS","BRSAN","DOAS","ECILC",
    "EKGYO","ENKAI","EREGL","FROTO","GARAN","GUBRF","HEKTS",
    "ISCTR","KCHOL","KONTR","KOZAA","KOZAL","MGROS","OYAKC",
    "PETKM","PGSUS","SAHOL","SASA","SISE","TCELL","THYAO",
    "TKFEN","TOASO","TSPOR","TUPRS","YKBNK","KLNMA","KAREL",
    "JANTS","AGRO","AKGRT","LINK","ASGYO","LYDHO","BERA"
]


# ============================================================
# RSI
# ============================================================

def rsi_hesapla(series, period=14):

    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(
        0,
        np.nan
    )

    return 100 - (
        100 / (1 + rs)
    )


# ============================================================
# BIST EVRENİ
# ============================================================

def bist_listesi_al():

    # Önce daha önce kaydedilmiş liste
    try:

        if os.path.exists(UNIVERSE_FILE):

            with open(
                UNIVERSE_FILE,
                "r",
                encoding="utf-8"
            ) as f:

                liste = json.load(f)

                if len(liste) > 100:

                    return liste

    except:

        pass


    # Borsa İstanbul bülteninden otomatik almaya çalış
    bugun = datetime.now()

    for geri in range(10):

        tarih = bugun - timedelta(
            days=geri
        )

        tarih_str = tarih.strftime(
            "%Y%m%d"
        )

        yil = tarih.strftime("%Y")
        ay = tarih.strftime("%m")

        # BIST'in resmi bülten dosya yapısı
        temel_url = (
            "https://www.borsaistanbul.com/data/thm/"
            + yil
            + "/"
            + ay
            + "/"
        )

        for seans in ["G1", "G2", "1", "2"]:

            url = (
                temel_url
                + "thm"
                + tarih_str
                + seans
                + ".zip"
            )

            try:

                cevap = requests.get(
                    url,
                    timeout=15
                )

                if cevap.status_code != 200:
                    continue

                if len(cevap.content) < 100:
                    continue

                z = zipfile.ZipFile(
                    io.BytesIO(
                        cevap.content
                    )
                )

                semboller = []

                for dosya in z.namelist():

                    try:

                        ham = z.read(
                            dosya
                        )

                        metin = ham.decode(
                            "utf-8",
                            errors="ignore"
                        )

                    except:

                        continue


                    # Muhtemel BIST sembollerini bul
                    bulunanlar = re.findall(
                        r"\b[A-Z0-9]{2,7}\b",
                        metin
                    )

                    for sembol in bulunanlar:

                        # BIST sembol filtresi
                        if (
                            2 <= len(sembol) <= 7
                            and sembol.isalnum()
                            and not sembol.isdigit()
                        ):

                            # Tarih / gereksiz kodları ele
                            yasak = {
                                "BIST",
                                "ISIN",
                                "TRY",
                                "USD",
                                "EUR",
                                "NULL",
                                "DATE",
                                "CODE",
                                "PRICE",
                                "VOLUME",
                                "SESSION",
                                "MARKET"
                            }

                            if sembol not in yasak:

                                semboller.append(
                                    sembol
                                )


                # Temizle
                semboller = sorted(
                    list(
                        set(semboller)
                    )
                )


                # Çok fazla gereksiz veri yakalanmışsa
                # bu dosyayı kullanma
                if 100 <= len(semboller) <= 1500:

                    liste = semboller

                    with open(
                        UNIVERSE_FILE,
                        "w",
                        encoding="utf-8"
                    ) as f:

                        json.dump(
                            liste,
                            f,
                            ensure_ascii=False,
                            indent=2
                        )

                    return liste

            except Exception:
                continue


    # Son çare
    return FALLBACK_HISSELER


# ============================================================
# VERİ AL
# ============================================================

def veri_al(ticker):

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

            if (
                df is not None
                and not df.empty
            ):

                if isinstance(
                    df.columns,
                    pd.MultiIndex
                ):

                    df.columns = (
                        df.columns
                        .get_level_values(0)
                    )


                gerekli = [
                    "Close",
                    "High",
                    "Low",
                    "Volume"
                ]

                if all(
                    x in df.columns
                    for x in gerekli
                ):

                    return df, None

        except Exception as e:

            hata = str(e)

        time.sleep(0.5)


    return None, locals().get(
        "hata",
        "Veri alınamadı"
    )


# ============================================================
# NORMALİZE
# ============================================================

def normalize(value, low, high):

    if high == low:
        return 0

    sonuc = (
        (value - low)
        /
        (high - low)
    ) * 100

    return max(
        0,
        min(
            100,
            sonuc
        )
    )


# ============================================================
# ANA ANALİZ
# ============================================================

def analiz_et(ticker):

    if not ticker.endswith(".IS"):
        ticker += ".IS"


    df, hata = veri_al(
        ticker
    )

    if df is None:
        return None, hata


    try:

        close = pd.to_numeric(
            df["Close"],
            errors="coerce"
        )

        high = pd.to_numeric(
            df["High"],
            errors="coerce"
        )

        low = pd.to_numeric(
            df["Low"],
            errors="coerce"
        )

        volume = pd.to_numeric(
            df["Volume"],
            errors="coerce"
        )


        data = pd.DataFrame({
            "close": close,
            "high": high,
            "low": low,
            "volume": volume
        }).dropna()


        if len(data) < 80:
            return None, "Yetersiz veri"


        c = data["close"]
        h = data["high"]
        l = data["low"]
        v = data["volume"]


        # ====================================================
        # EMA
        # ====================================================

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


        # ====================================================
        # RSI
        # ====================================================

        rsi = rsi_hesapla(c)


        # ====================================================
        # BOLLINGER
        # ====================================================

        bb_mid = c.rolling(
            20
        ).mean()

        bb_std = c.rolling(
            20
        ).std()

        bb_upper = (
            bb_mid +
            2 * bb_std
        )

        bb_lower = (
            bb_mid -
            2 * bb_std
        )

        bb_width = (
            (bb_upper - bb_lower)
            /
            bb_mid
        )


        # ====================================================
        # ATR
        # ====================================================

        previous_close = c.shift(1)

        tr1 = h - l
        tr2 = abs(
            h - previous_close
        )
        tr3 = abs(
            l - previous_close
        )

        true_range = pd.concat(
            [
                tr1,
                tr2,
                tr3
            ],
            axis=1
        ).max(axis=1)

        atr = true_range.rolling(
            14
        ).mean()

        atr_pct = (
            atr / c
        )


        # ====================================================
        # HACİM
        # ====================================================

        volume20 = v.rolling(
            20
        ).mean()

        volume5 = v.rolling(
            5
        ).mean()

        volume_ratio = (
            v.iloc[-1]
            /
            volume20.iloc[-1]
            if volume20.iloc[-1] > 0
            else 0
        )

        volume_trend = (
            volume5.iloc[-1]
            /
            volume20.iloc[-1]
            if volume20.iloc[-1] > 0
            else 0
        )


        # ====================================================
        # DESTEK / DİRENÇ
        # ====================================================

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

        resistance50 = (
            c.shift(1)
            .rolling(50)
            .max()
        )

        support50 = (
            c.shift(1)
            .rolling(50)
            .min()
        )


        # ====================================================
        # 52 HAFTA
        # ====================================================

        high52 = c.rolling(
            252,
            min_periods=60
        ).max()

        low52 = c.rolling(
            252,
            min_periods=60
        ).min()


        # ====================================================
        # SON DEĞERLER
        # ====================================================

        fiyat = float(
            c.iloc[-1]
        )

        onceki = float(
            c.iloc[-2]
        )

        gunluk = (
            (
                fiyat / onceki
            ) - 1
        ) * 100


        rsi_son = float(
            rsi.iloc[-1]
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


        direnç20 = float(
            resistance20.iloc[-1]
        )

        destek20 = float(
            support20.iloc[-1]
        )

        direnç50 = float(
            resistance50.iloc[-1]
        )

        destek50 = float(
            support50.iloc[-1]
        )


        zirve = float(
            high52.iloc[-1]
        )

        dip = float(
            low52.iloc[-1]
        )


        zirveden_uzaklik = (
            (
                fiyat / zirve
            ) - 1
        ) * 100


        # ====================================================
        # SIKIŞMA ANALİZİ
        # ====================================================

        # Bollinger genişliği
        bb_now = float(
            bb_width.iloc[-1]
        )

        bb_series = (
            bb_width
            .dropna()
            .tail(120)
        )


        if len(bb_series) > 10:

            bb_percentile = (
                bb_series
                .rank(pct=True)
                .iloc[-1]
            )

        else:

            bb_percentile = 0.5


        bollinger_sikisma = (
            1 -
            bb_percentile
        ) * 100


        # ====================================================
        # 20 GÜNLÜK FİYAT SIKIŞMASI
        # ====================================================

        son20_max = float(
            c.tail(20).max()
        )

        son20_min = float(
            c.tail(20).min()
        )

        fiyat_aralik = (
            (
                son20_max -
                son20_min
            )
            /
            fiyat
        ) * 100


        # Dar aralık = yüksek skor
        fiyat_sikisma = max(
            0,
            min(
                100,
                100 -
                (
                    fiyat_aralik
                    * 8
                )
            )
        )


        # ====================================================
        # ATR SIKIŞMASI
        # ====================================================

        atr_series = (
            atr_pct
            .dropna()
            .tail(120)
        )


        if len(atr_series) > 10:

            atr_percentile = (
                atr_series
                .rank(pct=True)
                .iloc[-1]
            )

        else:

            atr_percentile = 0.5


        atr_sikisma = (
            1 -
            atr_percentile
        ) * 100


        # ====================================================
        # EMA YAKINLAŞMASI
        # ====================================================

        ema_spread = (
            abs(e9 - e21)
            /
            fiyat
        ) * 100


        ema_sikisma = max(
            0,
            min(
                100,
                100 -
                (
                    ema_spread
                    * 15
                )
            )
        )


        # ====================================================
        # HACİM KURUMASI
        # ====================================================

        son5_hacim = float(
            v.tail(5).mean()
        )

        onceki20_hacim = float(
            v.tail(20).mean()
        )

        hacim_kuruma = 0

        if onceki20_hacim > 0:

            oran = (
                son5_hacim
                /
                onceki20_hacim
            )

            if oran < 1:

                hacim_kuruma = (
                    1 - oran
                ) * 100

            else:

                hacim_kuruma = 0


        hacim_kuruma = max(
            0,
            min(
                100,
                hacim_kuruma
            )
        )


        # ====================================================
        # DİRENCE YAKINLIK
        # ====================================================

        if direnç20 > fiyat:

            direnç_mesafe = (
                (
                    direnç20 -
                    fiyat
                )
                /
                fiyat
            ) * 100

            direnç_yakinlik = max(
                0,
                min(
                    100,
                    100 -
                    direnç_mesafe * 10
                )
            )

        else:

            direnç_mesafe = 0
            direnç_yakinlik = 100


        # ====================================================
        # SIKIŞMA SKORU
        # ====================================================

        sikisma_skoru = (

            bollinger_sikisma * 0.25

            +

            fiyat_sikisma * 0.20

            +

            atr_sikisma * 0.15

            +

            hacim_kuruma * 0.15

            +

            ema_sikisma * 0.10

            +

            direnç_yakinlik * 0.15

        )


        sikisma_skoru = round(
            max(
                0,
                min(
                    100,
                    sikisma_skoru
                )
            ),
            1
        )


        # ====================================================
        # PATLAMA HAZIRLIK SKORU
        # ====================================================

        hacim_ivmesi = normalize(
            volume_trend,
            0.5,
            2.5
        )


        # RSI ideal bölge
        if 45 <= rsi_son <= 65:

            rsi_skor = 100

        elif (
            35 <= rsi_son < 45
            or
            65 < rsi_son <= 72
        ):

            rsi_skor = 70

        else:

            rsi_skor = 35


        # EMA trend
        ema_skor = 0

        if e9 > e21:
            ema_skor += 35

        if e21 > e50:
            ema_skor += 35

        if e50 > e200:
            ema_skor += 30


        # Hacim
        hacim_patlama = min(
            100,
            hacim_ivmesi
        )


        # Son fiyat hareketi
        fiyat_momentum = normalize(
            gunluk,
            -3,
            5
        )


        patlama_skoru = (

            sikisma_skoru * 0.30

            +

            hacim_patlama * 0.20

            +

            direnç_yakinlik * 0.15

            +

            rsi_skor * 0.10

            +

            ema_skor * 0.10

            +

            fiyat_momentum * 0.15

        )


        patlama_skoru = round(
            max(
                0,
                min(
                    100,
                    patlama_skoru
                )
            ),
            1
        )


        # ====================================================
        # KIRILIM DURUMU
        # ====================================================

        if (
            fiyat > direnç20
            and
            volume_ratio >= 1.5
        ):

            kirilim_durumu = (
                "🚀 KIRILIM BAŞLADI"
            )

        elif (
            sikisma_skoru >= 75
            and
            patlama_skoru >= 75
        ):

            kirilim_durumu = (
                "🔥 PATLAMAYA HAZIR"
            )

        elif (
            sikisma_skoru >= 65
        ):

            kirilim_durumu = (
                "⚡ SIKIŞMA YÜKSEK"
            )

        elif (
            sikisma_skoru >= 50
        ):

            kirilim_durumu = (
                "👀 SIKIŞMA OLUŞUYOR"
            )

        else:

            kirilim_durumu = (
                "⏳ SIKIŞMA ZAYIF"
            )


        # ====================================================
        # NORMAL TEKNİK PUAN
        # ====================================================

        puan = 0
        nedenler = []


        if volume_ratio >= 2:

            puan += 30

            nedenler.append(
                "Hacim çok güçlü"
            )

        elif volume_ratio >= 1.5:

            puan += 22

            nedenler.append(
                "Hacim belirgin artıyor"
            )

        elif volume_ratio >= 1.2:

            puan += 12

            nedenler.append(
                "Hacim destekli"
            )


        if gunluk > 0 and volume_ratio >= 1.2:

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


        if fiyat > direnç20:

            puan += 15

            nedenler.append(
                "20 günlük direnç kırıldı"
            )


        if sikisma_skoru >= 75:

            puan += 15

            nedenler.append(
                "Güçlü fiyat sıkışması tespit edildi"
            )

        elif sikisma_skoru >= 60:

            puan += 8

            nedenler.append(
                "Belirgin sıkışma var"
            )


        if patlama_skoru >= 80:

            puan += 20

            nedenler.append(
                "Patlama hazırlık skoru çok yüksek"
            )

        elif patlama_skoru >= 70:

            puan += 12

            nedenler.append(
                "Patlama hazırlık skoru yüksek"
            )


        if rsi_son > 72:

            puan -= 20

            nedenler.append(
                "RSI aşırı yüksek"
            )


        # ====================================================
        # SİNYAL
        # ====================================================

        if rsi_son > 72:

            sinyal = "⚠️ AŞIRI YÜKSELDİ"

        elif (
            puan >= 78
            and
            volume_ratio >= 1.5
        ):

            sinyal = "🔥 GÜÇLÜ AL"

        elif puan >= 55:

            sinyal = "🟢 AL"

        elif puan >= 35:

            sinyal = "👀 TAKİBE AL"

        else:

            sinyal = "⏳ BEKLE"


        # ====================================================
        # GİRİŞ / STOP / HEDEF
        # ====================================================

        volatilite = float(
            atr_pct.iloc[-1]
        )

        if (
            np.isnan(volatilite)
            or
            volatilite <= 0
        ):

            volatilite = 0.02


        giris_alt = max(
            destek20,
            e21
        )

        giris_ust = max(
            giris_alt,
            fiyat
        )


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


        hedef1 = max(
            direnç20,
            fiyat * 1.04
        )

        hedef2 = max(
            hedef1 * 1.04,
            fiyat * 1.08
        )


        if direnç20 <= fiyat:

            hedef1 = fiyat * 1.04

            hedef2 = fiyat * 1.08


        risk = fiyat - stop

        getiri1 = (
            hedef1 - fiyat
        )


        risk_getiri = (
            getiri1 / risk
            if risk > 0
            else 0
        )


        # ====================================================
        # GRAFİK
        # ====================================================

        grafik = []

        son_veriler = data.tail(
            120
        )

        for tarih, row in son_veriler.iterrows():

            grafik.append({

                "tarih":
                    tarih.strftime(
                        "%Y-%m-%d"
                    ),

                "fiyat":
                    round(
                        float(
                            row["close"]
                        ),
                        2
                    )

            })


        # ====================================================
        # SONUÇ
        # ====================================================

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
                round(
                    puan,
                    1
                ),

            "sikisma_skoru":
                sikisma_skoru,

            "patlama_skoru":
                patlama_skoru,

            "kirilim_durumu":
                kirilim_durumu,

            "bollinger_sikisma":
                round(
                    bollinger_sikisma,
                    1
                ),

            "fiyat_sikisma":
                round(
                    fiyat_sikisma,
                    1
                ),

            "atr_sikisma":
                round(
                    atr_sikisma,
                    1
                ),

            "ema_sikisma":
                round(
                    ema_sikisma,
                    1
                ),

            "hacim_kuruma":
                round(
                    hacim_kuruma,
                    1
                ),

            "hacim_orani":
                round(
                    volume_ratio,
                    2
                ),

            "hacim_trend":
                round(
                    volume_trend,
                    2
                ),

            "rsi":
                round(
                    rsi_son,
                    1
                ),

            "gunluk_degisim":
                round(
                    gunluk,
                    2
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
                    direnç20,
                    2
                ),

            "destek":
                round(
                    destek20,
                    2
                ),

            "direnc50":
                round(
                    direnç50,
                    2
                ),

            "destek50":
                round(
                    destek50,
                    2
                ),

            "direnc_mesafe":
                round(
                    direnç_mesafe,
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

            "zirveden_uzaklik":
                round(
                    zirveden_uzaklik,
                    1
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

            "nedenler":
                nedenler,

            "grafik":
                grafik,

            "tradingview":
                ticker.replace(
                    ".IS",
                    ""
                )

        }, None


    except Exception as e:

        return None, str(e)


# ============================================================
# GEÇMİŞ
# ============================================================

def gecmis_oku():

    try:

        with open(
            HISTORY_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            return json.load(f)

    except:

        return []


def gecmis_kaydet(veriler):

    with open(
        HISTORY_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            veriler[-2000:],
            f,
            ensure_ascii=False,
            indent=2
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
# BIST EVRENİ
# ============================================================

@app.route("/api/universe")
def universe():

    liste = bist_listesi_al()

    return jsonify({

        "basarili": True,

        "hisse_sayisi":
            len(liste),

        "hisseler":
            liste

    })


# ============================================================
# TARAMA
# ============================================================

@app.route("/api/scan")
def tara():

    liste = bist_listesi_al()

    sonuclar = []
    hatalar = []

    toplam = len(liste)


    for sayac, sembol in enumerate(
        liste
    ):

        ticker = sembol

        if not ticker.endswith(".IS"):

            ticker += ".IS"


        sonuc, hata = analiz_et(
            ticker
        )


        if sonuc:

            sonuclar.append(
                sonuc
            )

        else:

            hatalar.append({

                "hisse":
                    sembol,

                "hata":
                    hata

            })


        # Çok agresif istek göndermemek için
        time.sleep(0.15)


    # ========================================================
    # SIRALAMA
    # Öncelik:
    # Patlama + sıkışma
    # ========================================================

    sonuclar.sort(

        key=lambda x: (

            x.get(
                "patlama_skoru",
                0
            ),

            x.get(
                "sikisma_skoru",
                0
            ),

            x.get(
                "puan",
                0
            )

        ),

        reverse=True

    )


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
                x["fiyat"],

            "sikisma":
                x["sikisma_skoru"],

            "patlama":
                x["patlama_skoru"],

            "durum":
                x["kirilim_durumu"]

        })


    gecmis_kaydet(
        gecmis
    )


    return jsonify({

        "basarili":
            True,

        "sonuc_sayisi":
            len(sonuclar),

        "evren_sayisi":
            toplam,

        "sonuclar":
            sonuclar,

        "hatalar":
            hatalar

    })


# ============================================================
# HİSSE DETAY
# ============================================================

@app.route(
    "/api/hisse/<ticker>"
)
def hisse_detay(ticker):

    ticker = ticker.upper()


    if not ticker.endswith(".IS"):

        ticker += ".IS"


    sonuc, hata = analiz_et(
        ticker
    )


    if sonuc is None:

        return jsonify({

            "basarili":
                False,

            "hata":
                hata

        }), 404


    return jsonify({

        "basarili":
            True,

        "hisse":
            sonuc

    })


# ============================================================
# GEÇMİŞ
# ============================================================

@app.route("/api/history")
def sinyal_gecmisi():

    return jsonify(
        gecmis_oku()[-500:]
    )


# ============================================================
# SAĞLIK
# ============================================================

@app.route("/health")
def health():

    return jsonify({

        "status":
            "ok",

        "uygulama":
            "BIST Hisse Avcısı V7",

        "zaman":
            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
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
