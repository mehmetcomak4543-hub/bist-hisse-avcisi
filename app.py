from flask import Flask, render_template, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
import os
import json
import time
import requests
from datetime import datetime

app = Flask(__name__)

HISTORY_FILE = "signal_history.json"

# ============================================================
# BIST HİSSE LİSTESİ
# ============================================================

BIST_LISTE_URL = (
    "https://raw.githubusercontent.com/"
    "ahmeterenodaci/Istanbul-Stock-Exchange--BIST--including-symbols-and-logos/"
    "master/bist.csv"
)

FALLBACK_HISSELER = [
    "AEFES","AGHOL","AKBNK","AKSA","AKSEN","ALARK","ARCLK",
    "ASELS","ASTOR","AYDEM","BIMAS","BRSAN","DOAS","ECILC",
    "EKGYO","ENKAI","EREGL","FROTO","GARAN","GUBRF","HEKTS",
    "ISCTR","KCHOL","KONTR","KOZAA","KOZAL","MGROS","OYAKC",
    "PETKM","PGSUS","SAHOL","SASA","SISE","TCELL","THYAO",
    "TKFEN","TOASO","TSPOR","TUPRS","YKBNK","KLRHO","KLNMA",
    "KAREL","JANTS","BJKAS","GSRAY","TARKM","YEOTK","MIATK",
    "REEDR","ALFAS","CWENE","GESAN","ODAS","ZOREN","ENJSA",
    "SMRTG","GWIND","CANTE","KONKA","LINK","LIDFA","LYDHO",
    "ASGYO","AGYO","AKGRT","AKFGY","ALGYO","AVHOL","BAGFS",
    "BERA","BINHO","BINBT","BNTAS","BRYAT","BUCIM","CIMSA",
    "CLEBI","DOHOL","DGNMO","EGEEN","ENERY","FMIZP","GLYHO",
    "HALKB","ICBCT","INDES","IPEKE","ISDMR","ISFIN","ISGSY",
    "ISMEN","IZMDC","KARSN","KCAER","KERVT","KLGYO","KMPUR",
    "LOGO","MAVI","MEPET","MPARK","NTHOL","NUHCM","OBASE",
    "ORGE","OTKAR","OYAYO","PENTA","POLHO","QUAGR","RALYH",
    "RYSAS","SARKY","SELEC","SISE","SKBNK","SMART","SOKM",
    "TATEN","TATGD","TAVHL","TEZOL","TKNSA","TRCAS","TRGYO",
    "TSKB","TTKOM","TTRAK","TUKAS","ULKER","ULUUN","VAKBN",
    "VAKKO","VESBE","VESTL","YATAS","YIGIT","YYLGD"
]


def bist_hisselerini_getir():

    try:
        cevap = requests.get(
            BIST_LISTE_URL,
            timeout=15
        )

        if cevap.status_code == 200:

            try:
                df = pd.read_csv(
                    pd.io.common.StringIO(
                        cevap.text
                    )
                )

                kolonlar = [
                    str(x).lower()
                    for x in df.columns
                ]

                sembol_kolonu = None

                for i, kolon in enumerate(kolonlar):

                    if (
                        "symbol" in kolon
                        or "sembol" in kolon
                    ):
                        sembol_kolonu = df.columns[i]
                        break

                if sembol_kolonu is not None:

                    hisseler = []

                    for sembol in df[
                        sembol_kolonu
                    ].dropna():

                        sembol = str(
                            sembol
                        ).strip().upper()

                        if (
                            sembol
                            and sembol.isalnum()
                            and 2 <= len(sembol) <= 7
                        ):

                            hisseler.append(
                                sembol
                            )

                    hisseler = sorted(
                        list(
                            set(hisseler)
                        )
                    )

                    if len(hisseler) > 100:

                        print(
                            "BIST listesi alındı:",
                            len(hisseler),
                            "hisse"
                        )

                        return [
                            x + ".IS"
                            for x in hisseler
                        ]

            except Exception as e:

                print(
                    "Liste parse hatası:",
                    e
                )

    except Exception as e:

        print(
            "BIST listesi alınamadı:",
            e
        )

    print(
        "Fallback BIST listesi kullanılıyor:",
        len(FALLBACK_HISSELER)
    )

    return [
        x + ".IS"
        for x in sorted(
            list(
                set(FALLBACK_HISSELER)
            )
        )
    ]


# Uygulama başlarken listeyi al
BIST_HISSELERI = bist_hisselerini_getir()


# ============================================================
# RSI
# ============================================================

def rsi_hesapla(
    series,
    period=14
):

    delta = series.diff()

    gain = delta.clip(
        lower=0
    )

    loss = -delta.clip(
        upper=0
    )

    avg_gain = gain.rolling(
        period
    ).mean()

    avg_loss = loss.rolling(
        period
    ).mean()

    rs = (
        avg_gain
        /
        avg_loss.replace(
            0,
            np.nan
        )
    )

    return 100 - (
        100 /
        (1 + rs)
    )


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

                if (
                    "Close" in df.columns
                    and
                    "Volume" in df.columns
                ):

                    return df, None

        except Exception as e:

            hata = str(e)

        time.sleep(1)

    return None, locals().get(
        "hata",
        "Veri alınamadı"
    )


# ============================================================
# ANALİZ
# ============================================================

def analiz_et(ticker):

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

        # ====================================================
        # TEKNİK GÖSTERGELER
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

        rsi = rsi_hesapla(c)

        volume20 = v.rolling(
            20
        ).mean()

        volume5 = v.rolling(
            5
        ).mean()

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

        high52 = c.rolling(
            252,
            min_periods=60
        ).max()

        low52 = c.rolling(
            252,
            min_periods=60
        ).min()

        # ====================================================
        # FİYAT
        # ====================================================

        fiyat = float(
            c.iloc[-1]
        )

        onceki = float(
            c.iloc[-2]
        )

        gunluk = (
            (
                fiyat /
                onceki -
                1
            ) * 100
            if onceki
            else 0
        )

        rsi_son = float(
            rsi.iloc[-1]
        )

        hacim_ortalama = float(
            volume20.iloc[-1]
        )

        hacim5 = float(
            volume5.iloc[-1]
        )

        son_hacim = float(
            v.iloc[-1]
        )

        hacim_orani = (
            son_hacim /
            hacim_ortalama
            if hacim_ortalama > 0
            else 0
        )

        kisa_hacim_orani = (
            hacim5 /
            hacim_ortalama
            if hacim_ortalama > 0
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
                fiyat /
                zirve -
                1
            ) * 100
            if zirve > 0
            else 0
        )

        # ====================================================
        # SIKIŞMA ANALİZİ
        # ====================================================

        getiriler = c.pct_change()

        volatilite20 = (
            getiriler
            .rolling(20)
            .std()
            .iloc[-1]
        )

        volatilite60 = (
            getiriler
            .rolling(60)
            .std()
            .iloc[-1]
        )

        if pd.isna(
            volatilite20
        ):

            volatilite20 = 0.03

        if pd.isna(
            volatilite60
        ):

            volatilite60 = 0.03

        # Son 20 günlük fiyat aralığı
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
            /
            fiyat
            if fiyat > 0
            else 1
        )

        # Bollinger benzeri bant genişliği
        orta20 = c.rolling(
            20
        ).mean()

        std20 = c.rolling(
            20
        ).std()

        ust_band = (
            orta20 +
            2 * std20
        )

        alt_band = (
            orta20 -
            2 * std20
        )

        bant_genisligi = (
            (
                ust_band.iloc[-1] -
                alt_band.iloc[-1]
            )
            /
            orta20.iloc[-1]
            if orta20.iloc[-1] > 0
            else 1
        )

        # ====================================================
        # SIKIŞMA PUANI 0-100
        # ====================================================

        sikisma_puan = 0
        sikisma_nedenleri = []

        # Düşük volatilite
        if volatilite20 < 0.018:

            sikisma_puan += 25

            sikisma_nedenleri.append(
                "20 günlük volatilite çok düşük"
            )

        elif volatilite20 < 0.025:

            sikisma_puan += 18

            sikisma_nedenleri.append(
                "Volatilite düşük"
            )

        elif volatilite20 < 0.035:

            sikisma_puan += 10

        # Dar fiyat aralığı
        if fiyat_aralik < 0.08:

            sikisma_puan += 25

            sikisma_nedenleri.append(
                "Fiyat son 20 günde dar bantta"
            )

        elif fiyat_aralik < 0.12:

            sikisma_puan += 18

        elif fiyat_aralik < 0.16:

            sikisma_puan += 10

        # Bollinger sıkışması
        if bant_genisligi < 0.10:

            sikisma_puan += 25

            sikisma_nedenleri.append(
                "Bollinger bantları sıkıştı"
            )

        elif bant_genisligi < 0.15:

            sikisma_puan += 18

        elif bant_genisligi < 0.20:

            sikisma_puan += 10

        # Hacim düşüklüğü + son hareket
        if (
            kisa_hacim_orani < 0.80
            and
            hacim_orani < 1.0
        ):

            sikisma_puan += 15

            sikisma_nedenleri.append(
                "Hacim kuruyor"
            )

        # Dirence yakınlık
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
            0 <= direnç_mesafe <= 5
        ):

            sikisma_puan += 10

            sikisma_nedenleri.append(
                "Dirence çok yakın"
            )

        # Puanı sınırla
        sikisma_puan = min(
            100,
            max(
                0,
                int(
                    sikisma_puan
                )
            )
        )

        # ====================================================
        # PATLAMA / KIRILIM HAZIRLIK PUANI
        # ====================================================

        patlama_puan = 0

        patlama_nedenleri = []

        if sikisma_puan >= 70:

            patlama_puan += 30

        elif sikisma_puan >= 55:

            patlama_puan += 20

        elif sikisma_puan >= 40:

            patlama_puan += 10

        # RSI nötr-pozitif
        if 45 <= rsi_son <= 65:

            patlama_puan += 15

            patlama_nedenleri.append(
                "RSI aşırı alımda değil"
            )

        # EMA9 / EMA21
        if e9 > e21:

            patlama_puan += 15

            patlama_nedenleri.append(
                "Kısa trend yukarı dönüyor"
            )

        # Fiyat EMA21 üzerinde
        if fiyat > e21:

            patlama_puan += 10

        # Dirence yakınlık
        if 0 <= direnç_mesafe <= 5:

            patlama_puan += 20

            patlama_nedenleri.append(
                "Kırılmaya yakın direnç"
            )

        # Hacim kıpırdanması
        if (
            kisa_hacim_orani >= 1.15
        ):

            patlama_puan += 10

            patlama_nedenleri.append(
                "Hacimde kıpırdanma başladı"
            )

        patlama_puan = min(
            100,
            int(
                patlama_puan
            )
        )

        # ====================================================
        # NORMAL TEKNİK PUAN
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
            gunluk > 0
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

        # ====================================================
        # SİNYAL
        # ====================================================

        if rsi_son > 72:

            sinyal = "⚠️ AŞIRI YÜKSELDİ"

        elif (
            patlama_puan >= 75
            and
            sikisma_puan >= 65
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
            sikisma_puan >= 75
            and
            patlama_puan >= 50
        ):

            sinyal = "🔒 GÜÇLÜ SIKIŞMA"

        elif puan >= 55:

            sinyal = "🟢 AL"

        elif (
            sikisma_puan >= 60
        ):

            sinyal = "🔒 SIKIŞMA"

        elif puan >= 35:

            sinyal = "👀 TAKİBE AL"

        else:

            sinyal = "⏳ BEKLE"

        # ====================================================
        # GİRİŞ / STOP / HEDEF
        # ====================================================

        gunluk_degisimler = (
            c.pct_change()
            .rolling(14)
            .std()
        )

        volatilite = float(
            gunluk_degisimler.iloc[-1]
        )

        if np.isnan(
            volatilite
        ):

            volatilite = 0.02

        giris_alt = max(
            destek,
            e21
        )

        giris_ust = max(
            giris_alt,
            fiyat
        )

        stop = (
            giris_alt *
            (
                1 -
                max(
                    0.025,
                    min(
                        volatilite * 1.5,
                        0.07
                    )
                )
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

        son_veriler = data.tail(
            120
        )

        for tarih, row in (
            son_veriler.iterrows()
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
                int(
                    max(
                        0,
                        min(
                            100,
                            puan
                        )
                    )
                ),

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

            # YENİ SIKIŞMA VERİLERİ
            "sikisma_puani":
                sikisma_puan,

            "patlama_puani":
                patlama_puan,

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


def gecmis_kaydet(
    veriler
):

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


# ============================================================
# ANA SAYFA
# ============================================================

@app.route("/")
def ana_sayfa():

    return render_template(
        "index.html"
    )


# ============================================================
# TARAMA
# ============================================================

@app.route(
    "/api/scan"
)
def tara():

    sonuclar = []
    hatalar = []

    toplam = len(
        BIST_HISSELERI
    )

    print(
        "BIST taraması başladı:",
        toplam,
        "hisse"
    )

    for sira, ticker in enumerate(
        BIST_HISSELERI,
        start=1
    ):

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
                        hata

                })

            print(
                f"{sira}/{toplam}:",
                ticker
            )

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

    # ========================================================
    # SIKIŞMA + PATLAMA ÖNCELİĞİ
    # ========================================================

    sonuclar.sort(
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
                x.get(
                    "sikisma_puani",
                    0
                ),

            "patlama_puani":
                x.get(
                    "patlama_puani",
                    0
                )

        })

    gecmis_kaydet(
        gecmis
    )

    print(
        "Tarama tamamlandı:",
        len(sonuclar),
        "başarılı /",
        len(hatalar),
        "hatalı"
    )

    return jsonify({

        "basarili":
            True,

        "sonuc_sayisi":
            len(sonuclar),

        "toplam_hisse":
            toplam,

        "sonuclar":
            sonuclar,

        "hatalar":
            hatalar

    })


# ============================================================
# TEK HİSSE
# ============================================================

@app.route(
    "/api/hisse/<ticker>"
)
def hisse_detay(
    ticker
):

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

@app.route(
    "/api/history"
)
def sinyal_gecmisi():

    return jsonify(
        gecmis_oku()[-200:]
    )


# ============================================================
# BIST LİSTE BİLGİSİ
# ============================================================

@app.route(
    "/api/info"
)
def sistem_bilgisi():

    return jsonify({

        "status":
            "ok",

        "uygulama":
            "BIST Hisse Avcısı V7",

        "hisse_sayisi":
            len(
                BIST_HISSELERI
            ),

        "sikisma_sistemi":
            True,

        "patlama_sistemi":
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
            "BIST Hisse Avcısı V7",

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
