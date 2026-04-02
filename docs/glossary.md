# Nexus Agent — Glossary

Bu belge Nexus Agent projesinde kullanılan tüm teknik terimleri tanımlar.
Ekip içi iletişimde ve kod yorumlarında bu terimler tutarlı biçimde kullanılır.

---

## Kaynak Katmanı (Source Layer)

**source_layer**
Dış dünyadan veri okuyan tüm adaptörlerin ortak soyutlaması.
UIA, DOM, File ve Transport alt katmanlarını içerir.

**transport_resolver**
Bir hedef için hangi transport'un (UIA / DOM / File / visual fallback)
kullanılacağına karar veren bileşen. ADR-002 öncelik sırasını uygular.

**uia_adapter**
Windows UI Automation API üzerinden erişilebilirlik ağacını okuyan adaptör.
`pywin32` / `comtypes` kullanır.

**dom_adapter**
Chrome DevTools Protocol (CDP) üzerinden browser DOM'una erişen adaptör.
ADR-011 uyarınca `--remote-debugging-port=9222` gerektirir.

**file_adapter**
Dosya sistemi, Office Open XML ve PDF gibi yapılandırılmış dosya formatlarını
okuyan adaptör.

**visual_fallback**
Structured kaynak başarısız olduğunda devreye giren capture + OCR tabanlı okuma yolu.
ADR-002'de tanımlıdır.

---

## Capture

**capture**
Ekranın ham piksel verisini periyodik olarak yakalayan bileşen.
ADR-010 uyarınca dedicated subprocess olarak çalışır.

**frame**
Tek bir ekran görüntüsünü temsil eden `numpy.ndarray` (uint8, BGR, HxWx3).
ADR-003 uyarınca hot path'te PNG üretilmez.

**stabilization_gate**
Art arda gelen frame'lerin "yeterince farklı" olup olmadığını kontrol eden filtre.
Gereksiz perception döngülerini önler.

**dirty_region**
İki frame arasında piksel değeri değişen bölge. Perception'ı bu bölgeye odaklar.

---

## Perception

**locator**
Bir UI elementinin ekrandaki konumunu veya yapısal kimliğini bulan bileşen.
Bounding box veya XPath/AX tree yolu döndürür.

**reader**
Bir UI elementinin içeriğini (metin, değer, durum) okuyan bileşen.

**matcher**
Bir frame içinde belirli bir görsel şablonu veya elementi arayan bileşen.
Template matching veya özellik tabanlı karşılaştırma kullanır.

**temporal_expert**
Birden fazla frame'e bakarak zaman içindeki değişimi yorumlayan perception bileşeni.
Animasyon tamamlanması, yükleme döngüleri gibi durumları tespit eder.

**spatial_graph**
Ekrandaki UI elementleri arasındaki konumsal ilişkileri (üstünde, yanında, içinde)
temsil eden graf yapısı.

**perception_result**
Tüm perception uzmanlarının çıktısını birleştiren yapılandırılmış veri nesnesi.
`locator`, `reader` ve `matcher` sonuçlarını içerir.

**arbitration**
Birden fazla perception kaynağından gelen çelişkili sonuçları uzlaştıran katman.
`confidence_score` ve `ambiguity_score` kullanır.

**confidence_score**
Bir perception sonucunun doğruluk olasılığını `[0.0, 1.0]` aralığında ifade eden değer.

**ambiguity_score**
Bir perception sonucunun kaç farklı yoruma açık olduğunu `[0.0, 1.0]` aralığında
ifade eden değer. Yüksek değer HITL tetikleyebilir.

---

## Karar (Decision)

**decision**
Agent'ın mevcut `perception_result`'a dayanarak seçtiği bir sonraki eylem.
LLM çıktısından türetilir ve `action_spec`'e dönüştürülür.

**action_spec**
Bir eylemin tüm parametrelerini (hedef, değer, beklenen sonuç) içeren
yapılandırılmış veri nesnesi. `action` katmanının girdisidir.

---

## Aksiyon (Action)

**transport_layer**
Bir `action_spec`'i gerçek sistem çağrısına (UIA, CDP, keyboard/mouse) çeviren katman.

**macroaction**
Birden fazla atomik aksiyonu sıralı veya koşullu olarak zincirleme yapısı.
Örn: "dosyayı aç → içeriği oku → kapat".

**preflight**
Bir aksiyon yürütülmeden önce ön koşulların sağlanıp sağlanmadığını kontrol eden adım.
Başarısız preflight aksiyonu iptal eder ve HITL tetikleyebilir.

---

## Doğrulama (Verification)

**verification**
Bir aksiyonun beklenen sonucu üretip üretmediğini kontrol eden süreç.
Üç stratejisi vardır: visual, semantic, source.

**visual_verification**
Aksiyon sonrası ekranın beklenen görsel duruma uyup uymadığını kontrol eder.
Frame karşılaştırması veya template matching kullanır.

**semantic_verification**
Ekran içeriğinin anlamsal olarak beklenen durumu yansıtıp yansıtmadığını kontrol eder.
OCR + LLM değerlendirmesi kullanır.

**source_verification**
Structured kaynak (UIA/DOM/File) üzerinden beklenen değeri doğrudan sorgular.
En hızlı ve güvenilir doğrulama stratejisi.

**verification_policy**
Hangi doğrulama stratejilerinin hangi sırayla uygulanacağını tanımlayan yapılandırma.

---

## Bellek (Memory)

**fingerprint**
Bir UI durumunu veya sayfa içeriğini kompakt biçimde temsil eden özet.
Daha önce görülen durumları tanımak için kullanılır.

**correction_memory**
Geçmiş HITL düzeltmelerini saklayan yapı. Benzer durumlarda aynı hatanın
tekrarlanmamasını sağlar.

---

## HITL

**hitl**
Human-in-the-Loop. Agent'ın belirsiz veya riskli bir durumda insandan
onay veya yönlendirme istediği mekanizma. ADR-008'de tanımlıdır.

**suspend**
Agent'ın HITL beklerken tüm aksiyonları durdurduğu durum.

**resume**
Kullanıcı onayından sonra agent'ın suspend durumundan çıkması.

---

## Maliyet ve Yapılandırma

**byok**
Bring Your Own Key. Kullanıcının kendi LLM API anahtarlarını sağladığı model.
ADR-006'da tanımlıdır.

**cost_ledger**
Her LLM çağrısının token sayısını ve tahmini maliyetini kaydeden yapı.

**budget_cap**
Günlük veya aylık LLM harcaması için kullanıcı tanımlı üst limit.
Aşıldığında agent suspend moduna geçer.

---

## Test

**golden_scenario**
Belirli bir girdi-çıktı çiftini sabit tutan regresyon test senaryosu.
`tests/golden/` altında saklanır.

**adversarial_test**
Agent'ı kasıtlı olarak zorlamak için tasarlanmış test: bozuk frame, eksik DOM,
yavaş yanıt, vb. `tests/adversarial/` altında saklanır.
