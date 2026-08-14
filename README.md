# BIST Hisse Avcısı V3

Telefon üzerinden kullanılabilen Flask web uygulaması.

## Render
GitHub reposuna dosyaları yükledikten sonra Render > New Web Service ile repo seçilir.

Build command:
`pip install -r requirements.txt`

Start command:
`gunicorn app:app`

## Veri
İlk prototip günlük Yahoo Finance/yfinance verisini kullanır. Bu, lisanslı/resmî BIST veri servisi değildir. Gerçek para ile kullanılmadan önce veri doğruluğu ve kullanım şartları kontrol edilmelidir.

## V3
EMA 9/21/50/200, RSI, hacim/20 günlük ortalama, günlük değişim ve 20 günlük direnç kırılımı ile 0-100 teknik puan üretir.
