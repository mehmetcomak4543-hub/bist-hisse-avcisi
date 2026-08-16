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
    "AEFES.IS","AGHOL.IS","AKBNK.IS","AKSA.IS","AKSEN.IS",
    "ALARK.IS","ARCLK.IS","ASELS.IS","ASTOR.IS","AYDEM.IS",
    "BIMAS.IS","BRSAN.IS","DOAS.IS","ECILC.IS","EKGYO.IS",
    "ENKAI.IS","EREGL.IS","FROTO.IS","GARAN.IS","GUBRF.IS",
    "HEKTS.IS","ISCTR.IS","KCHOL.IS","KONTR.IS","KOZAA.IS",
    "KOZAL.IS","MGROS.IS","OYAKC.IS","PETKM.IS","PGSUS.IS",
    "SAHOL.IS","SASA.IS","SISE.IS","TCELL.IS","THYAO.IS",
    "TKFEN.IS","TOASO.IS","TSPOR.IS","TUPRS.IS","YKBNK.IS"
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

    return 100 - (100 / (1 + rs))


# =========================================================
# VERİ
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

        if isinstance(
            df.columns,
            pd.MultiIndex
        ):

            df.columns = (
                df.columns
                .get_level_values(0)
            )

        if "Close" not in df.columns:

            return None, "Close verisi yok"

        if "Volume" not in df.columns:

            return None, "Volume verisi yok"

        return df, None

    except Exception as e:

        return None, str(e)


# =========================================================
# SIKIŞMA HESAPLA
# =========================================================

def sikisma_hesapla(c, v):

    try:

        # -------------------------------------------------
        # 1) 20 GÜNLÜK FİYAT BANT GENİŞLİĞİ
        # -------------------------------------------------

        high20 = (
            c.rolling(20).max()
        )

        low20 = (
            c.rolling(20).min()
        )

        bant20 = (
            (high20 - low20)
            / low20
        ) * 100

        bant_son = float(
            bant20.iloc[-1]
        )

        # -------------------------------------------------
        # 2) ÖNCEKİ BANTLA KARŞILAŞTIR
        # -------------------------------------------------

        bant_eski = float(
            bant20.iloc[-11]
        )

        if bant_eski > 0:

            bant_daralma = (
                1 -
                (
                    bant_son
                    / bant_eski
                )
            ) * 100

        else:

            bant_daralma = 0

        # -------------------------------------------------
        # 3) ATR BENZERİ VOLATİLİTE
        # -------------------------------------------------

        getiriler = c.pct_change()

        vol10 = (
            getiriler
            .rolling(10)
            .std()
        )

        vol30 = (
            getiriler
            .rolling(30)
            .std()
        )

        v10 = float(
            vol10.iloc[-1]
        )

        v30 = float(
            vol30.iloc[-1]
        )

        if v30 > 0:

            volatilite_orani = (
                v10 / v30
            )

        else:

            volatilite_orani = 1

        # -------------------------------------------------
        # 4) HACİM SIKIŞMASI
        # -------------------------------------------------

        hacim10 = (
            v.rolling(10).mean()
        )

        hacim30 = (
            v.rolling(30).mean()
        )

        h10 = float(
            hacim10.iloc[-1]
        )

        h30 = float(
            hacim30.iloc[-1]
        )

        if h30 > 0:

            hacim_sikisma_orani = (
                h10 / h30
            )

        else:

            hacim_sikisma_orani = 1

        # -------------------------------------------------
        # 5) BOLLINGER BAND DARALMASI
        # -------------------------------------------------

        orta = (
            c.rolling(20).mean()
        )

        std = (
            c.rolling(20).std()
        )

        bb_ust = orta + (
            2 * std
        )

        bb_alt = orta - (
            2 * std
        )

        bb_width = (
            (bb_ust - bb_alt)
            / orta
        ) * 100

        bb_son = float(
            bb_width.iloc[-1]
        )

        bb_eski = float(
            bb_width.iloc[-11]
        )

        if bb_eski > 0:

            bb_daralma = (
                1 -
                (
                    bb_son
                    / bb_eski
                )
            ) * 100

        else:

            bb_daralma = 0

        # -------------------------------------------------
        # 6) EMA'LERİN BİRBİRİNE YAKLAŞMASI
        # -------------------------------------------------

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

        fiyat = float(
            c.iloc[-1]
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

        ema_mesafe = (
            (
                abs(e9 - e21)
                / fiyat
            )
            +
            (
                abs(e21 - e50)
                / fiyat
            )
        ) * 100

        # -------------------------------------------------
        # 7) DİRENÇ
        # -------------------------------------------------

        direnç = float(
            c.shift(1)
            .rolling(20)
            .max()
            .iloc[-1]
        )

        direnç_uzaklik = (
            (
                direnç - fiyat
            )
            / fiyat
        ) * 100

        # -------------------------------------------------
        # 8) KIRILIM KONTROLÜ
        # -------------------------------------------------

        hacim20 = (
            v.rolling(20).mean()
        )

        son_hacim = float(
            v.iloc[-1]
        )

        ort_hacim = float(
            hacim20.iloc[-1]
        )

        if ort_hacim > 0:

            hacim_orani = (
                son_hacim
                / ort_hacim
            )

        else:

            hacim_orani = 0

        kirilim = (
            fiyat > direnç
            and hacim_orani >= 1.5
        )

        # -------------------------------------------------
        # SIKIŞMA PUANI
        # -------------------------------------------------

        puan = 0

        nedenler = []

        # Fiyat bandı

        if bant_son <= 8:

            puan += 20

            nedenler.append(
                "20 günlük fiyat bandı dar"
            )

        elif bant_son <= 12:

            puan += 12

        # Bant daralması

        if bant_daralma >= 35:

            puan += 20

            nedenler.append(
                "Fiyat bandı belirgin şekilde daralıyor"
            )

        elif bant_daralma >= 20:

            puan += 12

        # Volatilite

        if volatilite_orani <= 0.65:

            puan += 20

            nedenler.append(
                "Volatilite güçlü şekilde sıkışıyor"
            )

        elif volatilite_orani <= 0.80:

            puan += 12

        # Bollinger

        if bb_daralma >= 30:

            puan += 15

            nedenler.append(
                "Bollinger bantları daralıyor"
            )

        elif bb_daralma >= 15:

            puan += 8

        # Hacim

        if hacim_sikisma_orani <= 0.70:

            puan += 10

            nedenler.append(
                "Hacim sıkışması mevcut"
            )

        elif hacim_sikisma_orani <= 0.85:

            puan += 5

        # EMA

        if ema_mesafe <= 4:

            puan += 10

            nedenler.append(
                "EMA'lar birbirine yaklaşıyor"
            )

        elif ema_mesafe <= 7:

            puan += 5

        # Dirence yakınlık

        if (
            0 <= direnç_uzaklik <= 3
        ):

            puan += 15

            nedenler.append(
                "Fiyat dirence çok yakın"
            )

        elif (
            0 <= direnç_uzaklik <= 6
        ):

            puan += 8

        # -------------------------------------------------
        # 100'E SINIRLA
        # -------------------------------------------------

        puan = min(
            100,
            max(0, puan)
        )

        # -------------------------------------------------
        # SIKIŞMA DURUMU
        # -------------------------------------------------

        if puan >= 75:

            durum = "🔥 ÇOK SIKIŞIK"

        elif puan >= 60:

            durum = "🟠 SIKIŞMA GÜÇLÜ"

        elif puan >= 45:

            durum = "🟡 SIKIŞMA VAR"

        elif puan >= 30:

            durum = "🔵 HAFİF SIKIŞMA"

        else:

            durum = "⚪ SIKIŞMA YOK"

        # -------------------------------------------------
        # KIRILIM DURUMU
        # -------------------------------------------------

        if kirilim:

            kirilim_durumu = (
                "🚀 YUKARI KIRILIM"
            )

        elif (
            direnç_uzaklik >= 0
            and direnç_uzaklik <= 3
            and hacim_orani >= 1.2
        ):

            kirilim_durumu = (
                "⚡ KIRILIM HAZIRLIĞI"
            )

        else:

            kirilim_durumu = (
                "⏳ KIRILIM BEKLENİYOR"
            )

        return {

            "sikisma_puani":
                round(puan),

            "sikisma_durumu":
                durum,

            "kirilim_durumu":
                kirilim_durumu,

            "bant_genisligi":
                round(
                    bant_son,
                    2
                ),

            "bant_daralma":
                round(
                    bant_daralma,
                    1
                ),

            "volatilite_orani":
                round(
                    volatilite_orani,
                    2
                ),

            "hacim_sikisma":
                round(
                    hacim_sikisma_orani,
                    2
                ),

            "bollinger_daralma":
                round(
                    bb_daralma,
                    1
                ),

            "ema_mesafe":
                round(
                    ema_mesafe,
                    2
                ),

            "direnc_uzaklik":
                round(
                    direnç_uzaklik,
                    2
                ),

            "kirilim":
                bool(kirilim),

            "nedenler":
                nedenler

        }

    except Exception as e:

        return {

            "sikisma_puani": 0,

            "sikisma_durumu":
                "HESAPLANAMADI",

            "kirilim_durumu":
                "BİLİNMİYOR",

            "bant_genisligi": 0,

            "bant_daralma": 0,

            "volatilite_orani": 1,

            "hacim_sikisma": 1,

            "bollinger_daralma": 0,

            "ema_mesafe": 0,

            "direnc_uzaklik": 0,

            "kirilim": False,

            "nedenler": [
                str(e)
            ]

        }


# =========================================================
# ANA ANALİZ
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

        volume20 = (
            v.rolling(20).mean()
        )

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

        fiyat = float(
            c.iloc[-1]
        )

        onceki = float(
            c.iloc[-2]
        )

        gunluk = (
            (fiyat / onceki - 1)
            * 100
            if onceki != 0
            else 0
        )

        rsi_son = float(
            rsi.iloc[-1]
        )

        ort_hacim = float(
            volume20.iloc[-1]
        )

        hacim_orani = (
            float(v.iloc[-1])
            / ort_hacim
            if ort_hacim > 0
            else 0
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

        zirve = float(
            high52.iloc[-1]
        )

        dip = float(
            low52.iloc[-1]
        )

        zirveden_uzaklik = (
            (
                fiyat / zirve - 1
            ) * 100
            if zirve > 0
            else 0
        )

        # =================================================
        # SIKIŞMA
        # =================================================

        sikisma = sikisma_hesapla(
            c,
            v
        )

        # =================================================
        # PUAN
        # =================================================

        puan = 0

        nedenler = []

        # -------------------------------------------------
        # HACİM
        # -------------------------------------------------

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

        # -------------------------------------------------
        # FİYAT + HACİM
        # -------------------------------------------------

        if (
            gunluk > 0
            and hacim_orani >= 1.2
        ):

            puan += 15

            nedenler.append(
                "Fiyat ve hacim birlikte yükseliyor"
            )

        # -------------------------------------------------
        # EMA
        # -------------------------------------------------

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

        # -------------------------------------------------
        # RSI
        # -------------------------------------------------

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

        # -------------------------------------------------
        # DİRENÇ
        # -------------------------------------------------

        if fiyat > direnç:

            puan += 15

            nedenler.append(
                "20 günlük direnç kırıldı"
            )

        # -------------------------------------------------
        # ZİRVE
        # -------------------------------------------------

        if (
            -65 <= zirveden_uzaklik <= -45
        ):

            puan += 8

            nedenler.append(
                "52 haftalık zirveden uzak"
            )

        # -------------------------------------------------
        # SIKIŞMA PUANI
        # =================================================

        sikisma_puani = (
            sikisma[
                "sikisma_puani"
            ]
        )

        # Sıkışmayı toplam puana ekliyoruz.
        # Ancak tek başına güçlü AL yaptırmıyoruz.

        if sikisma_puani >= 75:

            puan += 18

            nedenler.append(
                "Çok güçlü fiyat sıkışması tespit edildi"
            )

        elif sikisma_puani >= 60:

            puan += 13

            nedenler.append(
                "Güçlü sıkışma tespit edildi"
            )

        elif sikisma_puani >= 45:

            puan += 8

            nedenler.append(
                "Sıkışma oluşuyor"
            )

        # -------------------------------------------------
        # KIRILIM
        # -------------------------------------------------

        if sikisma["kirilim"]:

            puan += 20

            nedenler.append(
                "Sıkışma sonrası hacimli yukarı kırılım"
            )

        # -------------------------------------------------
        # AŞIRI RSI
        # -------------------------------------------------

        if rsi_son > 72:

            puan -= 20

            nedenler.append(
                "RSI aşırı yüksek"
            )

        # =================================================
        # PUAN SINIRI
        # =================================================

        puan = min(
            100,
            max(
                0,
                puan
            )
        )

        # =================================================
        # SİNYAL
        # =================================================

        if rsi_son > 72:

            sinyal = (
                "⚠️ AŞIRI YÜKSELDİ"
            )

        elif (
            puan >= 78
            and (
                hacim_orani >= 1.5
                or sikisma_puani >= 75
            )
        ):

            sinyal = (
                "🔥 GÜÇLÜ AL"
            )

        elif puan >= 55:

            sinyal = (
                "🟢 AL"
            )

        elif (
            puan >= 35
            or sikisma_puani >= 60
        ):

            sinyal = (
                "👀 TAKİBE AL"
            )

        else:

            sinyal = (
                "⏳ BEKLE"
            )

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

        volatilite = float(
            c.pct_change()
            .rolling(14)
            .std()
            .iloc[-1]
        )

        if np.isnan(
            volatilite
        ):

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

        stop = (
            giris_alt
            * (1 - stop_orani)
        )

        # =================================================
        # HEDEF
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
        # RİSK / GETİRİ
        # =================================================

        risk = (
            fiyat - stop
        )

        getiri1 = (
            hedef1 - fiyat
        )

        risk_getiri = (
            getiri1 / risk
            if risk > 0
            else 0
        )

        # =================================================
        # GRAFİK
        # =================================================

        grafik = []

        for tarih, row in (
            data.tail(120).iterrows()
        ):

            try:

                tarih_str = (
                    tarih.strftime(
                        "%Y-%m-%d"
                    )
                )

            except:

                tarih_str = str(
                    tarih
                )

            grafik.append({

                "tarih":
                    tarih_str,

                "fiyat":
                    round(
                        float(
                            row["close"]
                        ),
                        2
                    )
            })

        temiz_hisse = (
            ticker.replace(
                ".IS",
                ""
            )
        )

        # =================================================
        # SONUÇ
        # =================================================

        sonuc = {

            "hisse":
                temiz_hisse,

            "fiyat":
                round(
                    fiyat,
                    2
                ),

            "sinyal":
                sinyal,

            "puan":
                int(puan),

            # -----------------------------
            # SIKIŞMA
            # -----------------------------

            "sikisma_puani":
                sikisma[
                    "sikisma_puani"
                ],

            "sikisma_durumu":
                sikisma[
                    "sikisma_durumu"
                ],

            "kirilim_durumu":
                sikisma[
                    "kirilim_durumu"
                ],

            "bant_genisligi":
                sikisma[
                    "bant_genisligi"
                ],

            "bant_daralma":
                sikisma[
                    "bant_daralma"
                ],

            "volatilite_sikisma":
                sikisma[
                    "volatilite_orani"
                ],

            "hacim_sikisma":
                sikisma[
                    "hacim_sikisma"
                ],

            "bollinger_daralma":
                sikisma[
                    "bollinger_daralma"
                ],

            "ema_sikisma":
                sikisma[
                    "ema_mesafe"
                ],

            "dirence_uzaklik":
                sikisma[
                    "direnc_uzaklik"
                ],

            # -----------------------------
            # TEKNİK
            # -----------------------------

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
                    gunluk,
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

            "nedenler":
                nedenler
                +
                sikisma[
                    "nedenler"
                ],

            "grafik":
                grafik,

            "tradingview":
                temiz_hisse
        }

        return sonuc, None

    except Exception as e:

        return None, str(e)


# =========================================================
# GEÇMİŞ OKU
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


# =========================================================
# GEÇMİŞ KAYDET
# =========================================================

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
        "=== SIKIŞMA DESTEKLİ TARAMA BAŞLADI ==="
    )

    sonuclar = []

    hatalar = []

    toplam = len(
        BIST_HISSELERI
    )

    for i, ticker in enumerate(
        BIST_HISSELERI
    ):

        print(
            f"{i + 1}/{toplam} "
            f"{ticker} taranıyor..."
        )

        try:

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
                        ticker.replace(
                            ".IS",
                            ""
                        ),

                    "hata":
                        hata or
                        "Veri alınamadı"
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

        sikisma = x.get(
            "sikisma_puani",
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
            -puan,
            -sikisma
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
                    x["fiyat"],

                "sikisma_puani":
                    x["sikisma_puani"]

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
        "Başarılı:",
        len(sonuclar)
    )

    print(
        "Hatalı:",
        len(hatalar)
    )

    return jsonify({

        "basarili":
            True,

        "sonuc_sayisi":
            len(sonuclar),

        "sonuclar":
            sonuclar,

        "hatalar":
            hatalar,

        "mesaj":
            "Sıkışma destekli tarama tamamlandı."

    })


# =========================================================
# TEK HİSSE
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

            "basarili":
                False,

            "hata":
                hata or
                "Hisse verisi alınamadı."

        }), 404

    return jsonify({

        "basarili":
            True,

        "hisse":
            sonuc

    })


# =========================================================
# GEÇMİŞ
# =========================================================

@app.route(
    "/api/history"
)
def sinyal_gecmisi():

    return jsonify(
        gecmis_oku()[-200:]
    )


# =========================================================
# HEALTH
# =========================================================

@app.route(
    "/health"
)
def health():

    return jsonify({

        "status":
            "ok",

        "uygulama":
            "BIST Hisse Avcısı V6",

        "hisse_sayisi":
            len(
                BIST_HISSELERI
            ),

        "sikisma_sistemi":
            "aktif"

    })


# =========================================================
# ÇALIŞTIR
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
