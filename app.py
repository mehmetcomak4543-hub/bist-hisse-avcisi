from flask import Flask, render_template, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
import os
import json
import time
from datetime import datetime

app = Flask(__name__)

HISTORY_FILE = "signal_history.json"


# =========================================================
# BIST HİSSE LİSTESİ
# =========================================================

BIST_HISSELERI = [
    "A1CAP.IS","ACSEL.IS","ADEL.IS","ADESE.IS","ADGYO.IS",
    "AEFES.IS","AFYON.IS","AGESA.IS","AGHOL.IS","AGROT.IS",
    "AGYO.IS","AHGAZ.IS","AKBNK.IS","AKCNS.IS","AKENR.IS",
    "AKFGY.IS","AKFYE.IS","AKGRT.IS","AKMGY.IS","AKSA.IS",
    "AKSEN.IS","AKSGY.IS","ALARK.IS","ALBRK.IS","ALCAR.IS",
    "ALCTL.IS","ALFAS.IS","ALGYO.IS","ALKA.IS","ALKIM.IS",
    "ALKLC.IS","ALMAD.IS","ALTNY.IS","ANELE.IS","ANGEN.IS",
    "ANHYT.IS","ANSGR.IS","ARASE.IS","ARCLK.IS","ARDYZ.IS",
    "ARENA.IS","ARMDA.IS","ARSAN.IS","ARTMS.IS","ARZUM.IS",
    "ASELS.IS","ASGYO.IS","ASTOR.IS","ASUZU.IS","ATAGY.IS",
    "ATAKP.IS","ATATP.IS","ATEKS.IS","AVGYO.IS","AVHOL.IS",
    "AVOD.IS","AVPGY.IS","AYCES.IS","AYDEM.IS","AYEN.IS",
    "AYGAZ.IS","AZTEK.IS","BAGFS.IS","BAHKM.IS","BAKAB.IS",
    "BALAT.IS","BANVT.IS","BARMA.IS","BASCM.IS","BASGZ.IS",
    "BAYRK.IS","BEGYO.IS","BERA.IS","BEYAZ.IS","BFREN.IS",
    "BIENY.IS","BIGEN.IS","BIGCH.IS","BIMAS.IS","BINHO.IS",
    "BIOEN.IS","BIZIM.IS","BJKAS.IS","BLCYT.IS","BMSCH.IS",
    "BMSTL.IS","BNTAS.IS","BOBET.IS","BORLS.IS","BORSK.IS",
    "BOSSA.IS","BRISA.IS","BRKO.IS","BRKSN.IS","BRKVY.IS",
    "BRLSM.IS","BRSAN.IS","BRYAT.IS","BSOKE.IS","BTCIM.IS",
    "BUCIM.IS","BURCE.IS","BURVA.IS","BVSAN.IS","BYDNR.IS",
    "CANTE.IS","CASA.IS","CATES.IS","CCOLA.IS","CELHA.IS",
    "CEMAS.IS","CEMTS.IS","CIMSA.IS","CLEBI.IS","CMBTN.IS",
    "CMENT.IS","CONSE.IS","COSMO.IS","CRDFA.IS","CRFSA.IS",
    "CUSAN.IS","CVKMD.IS","CWENE.IS","DAGHL.IS","DAGI.IS",
    "DAPGM.IS","DARDL.IS","DCTTR.IS","DENGE.IS","DERHL.IS",
    "DERIM.IS","DESA.IS","DESPC.IS","DEVA.IS","DGATE.IS",
    "DGNMO.IS","DITAS.IS","DMRGD.IS","DMSAS.IS","DNISI.IS",
    "DOAS.IS","DOBUR.IS","DOCO.IS","DOHOL.IS","DOKTA.IS",
    "DURDO.IS","DYOBY.IS","DZGYO.IS","EBEBK.IS","ECILC.IS",
    "ECZYT.IS","EDATA.IS","EDIP.IS","EFORC.IS","EGEEN.IS",
    "EGEPO.IS","EGGUB.IS","EGPRO.IS","EGSER.IS","EKGYO.IS",
    "EKOS.IS","EKSUN.IS","ELITE.IS","EMKEL.IS","EMNIS.IS",
    "ENDAE.IS","ENERY.IS","ENJSA.IS","ENKAI.IS","ENSRI.IS",
    "ENTRA.IS","EREGL.IS","ERSU.IS","ESCAR.IS","ESCOM.IS",
    "ESEN.IS","ETILR.IS","ETYAT.IS","EUHOL.IS","EUPWR.IS",
    "EUREN.IS","EYGYO.IS","FADE.IS","FENER.IS","FLAP.IS",
    "FMIZP.IS","FONET.IS","FORMT.IS","FORTE.IS","FRIGO.IS",
    "FROTO.IS","GARAN.IS","GEDIK.IS","GEDZA.IS","GENIL.IS",
    "GENTS.IS","GEREL.IS","GESAN.IS","GIPTA.IS","GLBMD.IS",
    "GLCVY.IS","GLRMK.IS","GLYHO.IS","GMTAS.IS","GOKNR.IS",
    "GOLTS.IS","GOODY.IS","GOZDE.IS","GRSEL.IS","GRTHO.IS",
    "GSDDE.IS","GSDHO.IS","GSRAY.IS","GUBRF.IS","GWIND.IS",
    "HATEK.IS","HATSN.IS","HDFGS.IS","HEDEF.IS","HEKTS.IS",
    "HKTM.IS","HLGYO.IS","HOROZ.IS","HRKET.IS","HTTBT.IS",
    "HUBVC.IS","HUNER.IS","ICBCT.IS","ICUGS.IS","IDEAS.IS",
    "IEYHO.IS","IHAAS.IS","IHEVA.IS","IHLAS.IS","IHLGM.IS",
    "IHYAY.IS","IMASM.IS","INDES.IS","INFO.IS","INGRM.IS",
    "INTEK.IS","INVEO.IS","INVES.IS","IPEKE.IS","ISCTR.IS",
    "ISDMR.IS","ISFIN.IS","ISGSY.IS","ISGYO.IS","ISKPL.IS",
    "ISKUR.IS","ISMEN.IS","ISSEN.IS","IZENR.IS","IZFAS.IS",
    "IZMDC.IS","JANTS.IS","KAPLM.IS","KAREL.IS","KARSN.IS",
    "KARTN.IS","KARYE.IS","KATMR.IS","KAYSE.IS","KBORU.IS",
    "KCAER.IS","KCHOL.IS","KENT.IS","KERVT.IS","KFEIN.IS",
    "KGYO.IS","KIMMR.IS","KLGYO.IS","KLKIM.IS","KLMSN.IS",
    "KLRHO.IS","KLSER.IS","KLYPV.IS","KMPUR.IS","KNFRT.IS",
    "KONKA.IS","KONTR.IS","KONYA.IS","KOPOL.IS","KORDS.IS",
    "KOTON.IS","KOZAA.IS","KOZAL.IS","KRDMA.IS","KRDMB.IS",
    "KRDMD.IS","KRGYO.IS","KRONT.IS","KRPLS.IS","KRSTL.IS",
    "KRVGD.IS","KSTUR.IS","KTLEV.IS","KTSKR.IS","KUTPO.IS",
    "KUVVA.IS","KUYAS.IS","KZBGY.IS","LIDER.IS","LINK.IS",
    "LKMNH.IS","LOGO.IS","LRSHO.IS","LUKSK.IS","LYDHO.IS",
    "MAALT.IS","MACKO.IS","MAGEN.IS","MAKIM.IS","MAKTK.IS",
    "MANAS.IS","MARBL.IS","MARTI.IS","MAVI.IS","MEDTR.IS",
    "MEGMT.IS","MEKAG.IS","MEPET.IS","MERCN.IS","MERIT.IS",
    "MERKO.IS","METRO.IS","MGROS.IS","MIATK.IS","MIPAZ.IS",
    "MMCAS.IS","MNDRS.IS","MNDTR.IS","MOBTL.IS","MOGAN.IS",
    "MPARK.IS","MRGYO.IS","MRSHL.IS","MSGYO.IS","MTRKS.IS",
    "MTRYO.IS","MZHLD.IS","NATEN.IS","NETAS.IS","NIBAS.IS",
    "NTGAZ.IS","NTHOL.IS","NUGYO.IS","NUHCM.IS","OBAMS.IS",
    "ODAS.IS","ODINE.IS","OFSYM.IS","ONCSM.IS","ORGE.IS",
    "ORMA.IS","OSMEN.IS","OSTIM.IS","OTKAR.IS","OTTO.IS",
    "OYAKC.IS","OYLUM.IS","OYYAT.IS","OZATD.IS","OZGYO.IS",
    "OZKGY.IS","OZSUB.IS","PAGYO.IS","PAMEL.IS","PAPIL.IS",
    "PARSN.IS","PASEU.IS","PCILT.IS","PEGYO.IS","PEKGY.IS",
    "PENGD.IS","PENTA.IS","PETKM.IS","PETUN.IS","PGSUS.IS",
    "PINSU.IS","PKART.IS","PKENT.IS","PLTUR.IS","PNLSN.IS",
    "PNSUT.IS","POLHO.IS","POLTK.IS","PRDGS.IS","PRKAB.IS",
    "PRKME.IS","PRZMA.IS","PSDTC.IS","PSGYO.IS","QUAGR.IS",
    "RALYH.IS","RAYSG.IS","REEDR.IS","RGYAS.IS","RNPOL.IS",
    "RODRG.IS","ROYAL.IS","RUBNS.IS","RYGYO.IS","RYSAS.IS",
    "SAFKR.IS","SAHOL.IS","SAMAT.IS","SANEL.IS","SANFM.IS",
    "SANKO.IS","SARKY.IS","SASA.IS","SAYAS.IS","SDTTR.IS",
    "SEGYO.IS","SELEC.IS","SELVA.IS","SELGD.IS","SERVE.IS",
    "SISE.IS","SKBNK.IS","SKTAS.IS","SMART.IS","SMRTG.IS",
    "SNGYO.IS","SNICA.IS","SOKE.IS","SOKM.IS","SONME.IS",
    "SRVGY.IS","SUNTK.IS","SURGY.IS","SUWEN.IS","TABGD.IS",
    "TARKM.IS","TATEN.IS","TATGD.IS","TAVHL.IS","TCELL.IS",
    "TDGYO.IS","TEKTU.IS","TERA.IS","TETMT.IS","TEZOL.IS",
    "TGSAS.IS","THYAO.IS","TKFEN.IS","TKNSA.IS","TLMAN.IS",
    "TMPOL.IS","TMSN.IS","TOASO.IS","TRCAS.IS","TRGYO.IS",
    "TRILC.IS","TSGYO.IS","TSKB.IS","TSPOR.IS","TTKOM.IS",
    "TTRAK.IS","TUKAS.IS","TUPRS.IS","TUREX.IS","TURGG.IS",
    "TURSG.IS","UFUK.IS","ULAS.IS","ULKER.IS","ULUSE.IS",
    "ULUUN.IS","UNLU.IS","USAK.IS","VAKBN.IS","VAKFN.IS",
    "VAKKO.IS","VANGD.IS","VERUS.IS","VESBE.IS","VESTL.IS",
    "VKFYO.IS","YAPRK.IS","YATAS.IS","YAYLA.IS","YEOTK.IS",
    "YESIL.IS","YGGYO.IS","YKBNK.IS","YONGA.IS","YUNSA.IS",
    "YYAPI.IS","YYLGD.IS","ZEDUR.IS","ZOREN.IS"
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
# VERİ AL
# =========================================================

def veri_al(ticker):

    hata = "Veri alınamadı"

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

            if df is None or df.empty:
                hata = "Boş veri"
                time.sleep(1)
                continue

            if isinstance(df.columns, pd.MultiIndex):

                df.columns = [
                    x[0] if isinstance(x, tuple) else x
                    for x in df.columns
                ]

            gerekli = ["Close", "Volume"]

            if not all(x in df.columns for x in gerekli):

                hata = (
                    "Close/Volume sütunu bulunamadı: "
                    + str(list(df.columns))
                )

                time.sleep(1)
                continue

            return df, None

        except Exception as e:

            hata = str(e)

            time.sleep(1)

    return None, hata


# =========================================================
# ANALİZ
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

        hacim_ortalama = float(
            volume20.iloc[-1]
        )

        hacim_orani = (
            float(v.iloc[-1]) /
            hacim_ortalama
            if hacim_ortalama > 0
            else 0
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
        # BOLLINGER
        # =================================================

        bb_mid = c.rolling(20).mean()

        bb_std = c.rolling(20).std()

        bb_upper = (
            bb_mid + 2 * bb_std
        )

        bb_lower = (
            bb_mid - 2 * bb_std
        )

        bb_width = (
            (bb_upper - bb_lower)
            / bb_mid
        ) * 100


        # =================================================
        # ATR BENZERİ OYNAKLIK
        # =================================================

        gunluk_volatilite = (
            c.pct_change()
            .rolling(14)
            .std()
        )


        # =================================================
        # SON DEĞERLER
        # =================================================

        fiyat = float(c.iloc[-1])

        onceki = float(c.iloc[-2])

        gunluk = (
            (fiyat / onceki - 1) * 100
            if onceki != 0
            else 0
        )

        rsi_son = float(rsi.iloc[-1])

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

        bb_son = float(
            bb_width.iloc[-1]
        )

        vol_son = float(
            gunluk_volatilite.iloc[-1]
        )

        if np.isnan(bb_son):
            bb_son = 10

        if np.isnan(vol_son):
            vol_son = 0.03


        # =================================================
        # ZİRVE UZAKLIĞI
        # =================================================

        zirveden_uzaklik = (

            (fiyat / zirve - 1) * 100

            if zirve > 0

            else 0
        )


        # =================================================
        # DİRENÇ MESAFESİ
        # =================================================

        direnç_mesafesi = (

            ((direnç - fiyat) / fiyat) * 100

            if fiyat > 0

            else 0
        )


        # =================================================
        # SIKIŞMA SKORU
        # =================================================

        sikisma = 0

        sikisma_nedenleri = []


        # Bollinger daralması
        if bb_son <= 5:
            sikisma += 35
            sikisma_nedenleri.append(
                "Bollinger bantları çok dar"
            )

        elif bb_son <= 7:
            sikisma += 28
            sikisma_nedenleri.append(
                "Bollinger bantları daralıyor"
            )

        elif bb_son <= 10:
            sikisma += 18
            sikisma_nedenleri.append(
                "Bollinger bantları normalin altında"
            )


        # Volatilite düşüklüğü
        if vol_son <= 0.012:

            sikisma += 25

            sikisma_nedenleri.append(
                "Volatilite çok düşük"
            )

        elif vol_son <= 0.018:

            sikisma += 18

            sikisma_nedenleri.append(
                "Volatilite düşük"
            )

        elif vol_son <= 0.025:

            sikisma += 10


        # Fiyatın EMA'lara yaklaşması
        ema_mesafe = (
            abs(fiyat - e21) / fiyat * 100
            if fiyat > 0
            else 0
        )

        if ema_mesafe <= 2:

            sikisma += 15

            sikisma_nedenleri.append(
                "Fiyat EMA21 çevresinde sıkışmış"
            )

        elif ema_mesafe <= 4:

            sikisma += 8


        # Dirence yakınlık
        if 0 < direnç_mesafesi <= 3:

            sikisma += 15

            sikisma_nedenleri.append(
                "Fiyat dirence çok yakın"
            )

        elif 3 < direnç_mesafesi <= 6:

            sikisma += 8


        sikisma = min(
            100,
            int(sikisma)
        )


        # =================================================
        # PATLAMA POTANSİYELİ
        # =================================================

        patlama = 0

        patlama_nedenleri = []


        # Sıkışma
        patlama += int(
            sikisma * 0.40
        )


        # Hacim
        if hacim_orani >= 2:

            patlama += 25

            patlama_nedenleri.append(
                "Hacim patlaması var"
            )

        elif hacim_orani >= 1.5:

            patlama += 18

            patlama_nedenleri.append(
                "Hacim belirgin artıyor"
            )

        elif hacim_orani >= 1.2:

            patlama += 10


        # RSI
        if 50 <= rsi_son <= 65:

            patlama += 12

            patlama_nedenleri.append(
                "RSI yükseliş için uygun bölgede"
            )

        elif 45 <= rsi_son < 50:

            patlama += 7


        # EMA trendi
        if e9 > e21:

            patlama += 8

            patlama_nedenleri.append(
                "Kısa vadeli trend pozitif"
            )

        if e21 > e50:

            patlama += 7


        # Direnç yakınlığı
        if 0 < direnç_mesafesi <= 5:

            patlama += 8

            patlama_nedenleri.append(
                "Direnç bölgesi yakında"
            )


        patlama = min(
            100,
            int(patlama)
        )


        # =================================================
        # TEKNİK PUAN
        # =================================================

        puan = 0

        nedenler = []


        # Hacim
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


        # Fiyat + hacim
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


        # Direnç kırılımı
        if fiyat > direnç:

            puan += 15

            nedenler.append(
                "20 günlük direnç kırıldı"
            )


        # Zirve uzaklığı
        if (
            -65 <= zirveden_uzaklik <= -45
        ):

            puan += 8

            nedenler.append(
                "52 haftalık zirveden uzak"
            )


        # Sıkışma bonusu
        if sikisma >= 75:

            puan += 10

            nedenler.append(
                "Güçlü fiyat sıkışması tespit edildi"
            )

        elif sikisma >= 60:

            puan += 6

            nedenler.append(
                "Fiyat sıkışması mevcut"
            )


        # Aşırı yükseliş
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
        # SIKIŞMA DURUMU
        # =================================================

        if sikisma >= 80:

            sikisma_durumu = "🔥 ÇOK SIKIŞIK"

        elif sikisma >= 65:

            sikisma_durumu = "⚡ SIKIŞIK"

        elif sikisma >= 50:

            sikisma_durumu = "👀 ORTA"

        else:

            sikisma_durumu = "⏳ ZAYIF"


        # =================================================
        # GİRİŞ / STOP / HEDEF
        # =================================================

        giris_alt = max(
            destek,
            e21
        )

        giris_ust = max(
            giris_alt,
            fiyat
        )


        volatilite = vol_son

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

                tarih_text = tarih.strftime(
                    "%Y-%m-%d"
                )

            except:

                tarih_text = str(tarih)


            grafik.append({

                "tarih": tarih_text,

                "fiyat": round(
                    float(row["close"]),
                    2
                )

            })


        # =================================================
        # SONUÇ
        # =================================================

        return {

            "hisse":
                ticker.replace(".IS", ""),

            "fiyat":
                round(fiyat, 2),

            "sinyal":
                sinyal,

            "puan":
                int(puan),

            "rsi":
                round(rsi_son, 1),

            "hacim_orani":
                round(hacim_orani, 2),

            "gunluk_degisim":
                round(gunluk, 2),

            "zirveden_uzaklik":
                round(
                    zirveden_uzaklik,
                    1
                ),

            "ema9":
                round(e9, 2),

            "ema21":
                round(e21, 2),

            "ema50":
                round(e50, 2),

            "ema200":
                round(e200, 2),

            "direnc":
                round(direnç, 2),

            "destek":
                round(destek, 2),

            "zirve52":
                round(zirve, 2),

            "dip52":
                round(dip, 2),

            "giris_alt":
                round(giris_alt, 2),

            "giris_ust":
                round(giris_ust, 2),

            "stop":
                round(stop, 2),

            "hedef1":
                round(hedef1, 2),

            "hedef2":
                round(hedef2, 2),

            "risk_getiri":
                round(
                    risk_getiri,
                    2
                ),

            # Yeni
            "sikisma_skoru":
                sikisma,

            "sikisma_durumu":
                sikisma_durumu,

            "sikisma_nedenleri":
                sikisma_nedenleri,

            "patlama_potansiyeli":
                patlama,

            "patlama_nedenleri":
                patlama_nedenleri,

            "bollinger_genisligi":
                round(bb_son, 2),

            "direnc_mesafesi":
                round(
                    direnç_mesafesi,
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


# =========================================================
# GEÇMİŞ
# =========================================================

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

    except Exception:

        pass


# =========================================================
# ANA SAYFA
# =========================================================

@app.route("/")
def ana_sayfa():

    return render_template(
        "index.html"
    )


# =========================================================
# TÜM HİSSELERİ TARA
# =========================================================

@app.route("/api/scan")
def tara():

    try:

        sonuclar = []

        hatalar = []

        toplam = len(
            BIST_HISSELERI
        )


        for sira, ticker in enumerate(
            BIST_HISSELERI,
            start=1
        ):

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
                            str(hata)

                    })

            except Exception as e:

                hatalar.append({

                    "hisse":
                        ticker.replace(
                            ".IS",
                            ""
                        ),

                    "hata":
                        "ANALIZ HATASI: "
                        + str(e)

                })


            # Render'ı gereksiz yere boğmamak için
            time.sleep(0.15)


        # =================================================
        # SIRALAMA
        # Önce sinyal, sonra patlama, sonra sıkışma
        # =================================================

        def siralama(x):

            sinyal = x.get(
                "sinyal",
                ""
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
                -x.get(
                    "patlama_potansiyeli",
                    0
                ),
                -x.get(
                    "sikisma_skoru",
                    0
                ),
                -x.get(
                    "puan",
                    0
                )
            )


        sonuclar.sort(
            key=siralama
        )


        # =================================================
        # GEÇMİŞ KAYDI
        # =================================================

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

                "puan":
                    x.get(
                        "puan",
                        0
                    ),

                "sikisma":
                    x.get(
                        "sikisma_skoru",
                        0
                    ),

                "patlama":
                    x.get(
                        "patlama_potansiyeli",
                        0
                    )

            })


        gecmis_kaydet(
            gecmis
        )


        return jsonify({

            "basarili":
                True,

            "toplam_hisse":
                toplam,

            "sonuc_sayisi":
                len(sonuclar),

            "hata_sayisi":
                len(hatalar),

            "sonuclar":
                sonuclar,

            "hatalar":
                hatalar

        })


    except Exception as e:

        return jsonify({

            "basarili":
                False,

            "hata":
                "SCAN HATASI: "
                + str(e)

        }), 500


# =========================================================
# TEK HİSSE DETAY
# =========================================================

@app.route("/api/hisse/<ticker>")
def hisse_detay(ticker):

    try:

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


    except Exception as e:

        return jsonify({

            "basarili":
                False,

            "hata":
                str(e)

        }), 500


# =========================================================
# SİNYAL GEÇMİŞİ
# =========================================================

@app.route("/api/history")
def sinyal_gecmisi():

    return jsonify(
        gecmis_oku()[-200:]
    )


# =========================================================
# SAĞLIK KONTROLÜ
# =========================================================

@app.route("/health")
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

        "ozellikler": [

            "Teknik analiz",

            "RSI",

            "EMA",

            "Hacim",

            "Destek/Direnç",

            "Sıkışma Skoru",

            "Patlama Potansiyeli",

            "Risk/Getiri"

        ]

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
