# ADR-004: OCR = Tesseract, Interface Arkasında

## Status
Accepted

## Context
OCR motoru seçimi performansı, dil desteğini ve değiştirilebilirliği etkiler.
Adaylar: Tesseract, EasyOCR, PaddleOCR, Azure Computer Vision, Google Cloud Vision.

Tesseract: açık kaynak, offline, Türkçe model mevcut, pytesseract ile Python entegrasyonu kolay.
Cloud tabanlı çözümler: daha yüksek doğruluk, ancak ağ bağımlılığı ve maliyet.

## Decision
V1'de default OCR motoru **Tesseract**'tır.

Tüm OCR çağrıları bir `OcrEngine` arayüzü (abstract base class) üzerinden geçer:

```python
class OcrEngine(Protocol):
    def read(self, frame: np.ndarray, lang: str = "eng") -> OcrResult: ...
```

`TesseractEngine` bu protokolü uygular.

V2'de özel Türkçe model (`tur.traineddata`) swap edilebilir hale getirilir.

## Consequences
- `pytesseract` ve Tesseract binary sistem bağımlılığı olarak belgelenir.
- Tüm OCR testleri `OcrEngine` mock'u üzerinden yazılır; Tesseract binary gerekmez.
- V2 Türkçe model: `TESSDATA_PREFIX` ortam değişkeniyle yapılandırılır.
- Cloud OCR entegrasyonu: yeni bir `OcrEngine` impl olarak eklenir, mevcut kod değişmez.
