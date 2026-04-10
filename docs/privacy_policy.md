# Nexus Agent — Gizlilik Politikası

**Sürüm:** 1.0  
**Yürürlük tarihi:** 2026-04-10  
**Son güncelleme:** 2026-04-10

---

## 1. Genel Bakış

Nexus Agent, tamamen yerel makinenizde çalışan bir masaüstü otomasyon aracıdır.
Bu politika; hangi verilerin toplandığını, bu verilerin nereye iletildiğini ve
sizi koruyan güvenlik mekanizmalarını açıklar.

---

## 2. Toplanan Veriler

### 2.1 Yerel olarak saklanan veriler

Nexus Agent aşağıdaki verileri yalnızca yerel makinenizde saklar:

| Veri türü | Depolama yeri | Amaç |
|-----------|---------------|-------|
| Görev geçmişi | SQLite (yerel) | Görev takibi |
| API anahtarları | Windows Credential Manager (şifreli) | Bulut sağlayıcı erişimi |
| Ekran görüntüleri (anlık) | RAM (geçici) | Algılama ve otomasyon |
| UIA ağaç verileri | RAM (geçici) | Eleman tanımlama |
| Performans metrikleri | Yerel log dosyaları | Hata ayıklama |
| Onay kayıtları | SQLite (yerel) | KVKK uyumu |

### 2.2 Hassas bölgeler

Hassas olarak işaretlenen ekran bölgeleri (`SensitiveRegion`) **hiçbir zaman**
kaydedilmez, iletilmez veya işlenmez. Bu bölgeler şunları kapsar:

- Parola alanları
- Kredi kartı / ödeme formu alanları
- Kullanıcı tarafından hassas olarak tanımlanan tüm bölgeler

---

## 3. Buluta Gönderilen Veriler

### 3.1 Transport Şeffaflığı

Nexus Agent iki farklı eylem gerçekleştirme yöntemi (transport) kullanır.
Bu iki yöntem arasındaki gizlilik farkı kritik öneme sahiptir:

#### Native Transport (UIA / UIAutomation)

- Windows erişilebilirlik API'si (UIAutomation) aracılığıyla doğrudan eleman etkileşimi
- **Ekran görüntüsü gönderilmez** — bulut sağlayıcıya hiçbir görsel veri iletilmez
- Yalnızca yapısal görev açıklaması buluta gider
- Kullanıcı arayüzü `"veri gönderilmedi"` mesajıyla bilgilendirilir

#### Visual Transport (Mouse / Klavye)

- Ekran görüntüsü alınır ve **maskelenerek** (hassas bölgeler siyaha boyanır) buluta gönderilir
- Gönderilen görsel: orijinal ekranın maskelenmiş kopyası
- Hassas alanlar bu süreçte dahi gönderilmez
- Görev açıklaması buluta gider
- Kullanıcı arayüzü `"ekran görüntüsü gönderildi"` mesajıyla bilgilendirilir

### 3.2 Buluta gönderilen veri özeti

| Veri | Native Transport | Visual Transport |
|------|-----------------|-----------------|
| Maskelenmiş ekran görüntüsü | **HAYIR — hiçbir zaman** | Evet (hassas alanlar maskelenmiş) |
| Orijinal ekran görüntüsü | **HAYIR** | **HAYIR — hiçbir zaman** |
| Hassas bölge içeriği | **HAYIR — hiçbir zaman** | **HAYIR — hiçbir zaman** |
| Görev açıklaması | Evet (kullanıcı girer) | Evet (kullanıcı girer) |
| Sistem bilgisi | Hayır | Hayır |
| Kişisel kimlik verisi | **HAYIR — hiçbir zaman** | **HAYIR — hiçbir zaman** |

### 3.3 Hangi bulut sağlayıcılar kullanılır?

Nexus Agent, yalnızca sizin sağladığınız API anahtarıyla (BYOK — Bring Your Own Key)
aşağıdaki sağlayıcılardan birine bağlanır:

- Anthropic (Claude modelleri)
- OpenAI (GPT modelleri)

Her sağlayıcının kendi gizlilik politikası uygulanır. Nexus Agent, bu sağlayıcılara
gönderilen verileri kontrol etmez; ancak göndermeden önce maskeleme uygular.

---

## 4. Veri Silme

### 4.1 Yerel verileri silme

Tüm yerel verileri silmek için:

```
nexus reset --all
```

Bu komut şunları siler:
- Görev geçmişi veritabanı (`nexus.db`)
- Tüm log dosyaları
- Onay kayıtları
- Önbellek dosyaları

API anahtarlarını silmek için:

```
nexus reset --credentials
```

### 4.2 Bulut sağlayıcı verilerini silme

Bulut sağlayıcıya gönderilen veriler (API çağrıları) sağlayıcının saklama
politikasına tabidir. İlgili sağlayıcının veri silme prosedürlerine başvurun:

- Anthropic: https://privacy.anthropic.com
- OpenAI: https://privacy.openai.com

---

## 5. KVKK Hakları

6698 sayılı Kişisel Verilerin Korunması Kanunu (KVKK) kapsamında aşağıdaki
haklara sahipsiniz:

| Hak | Açıklama |
|-----|----------|
| Bilgi edinme | Hangi verilerinizin işlendiğini öğrenme |
| Erişim | İşlenen verilerinize erişim talep etme |
| Düzeltme | Yanlış verilerin düzeltilmesini isteme |
| Silme | Verilerinizin silinmesini talep etme |
| İtiraz | Veri işlemeye itiraz etme |
| Taşınabilirlik | Verilerinizi yapılandırılmış biçimde alma |

Nexus Agent, kişisel veri işlemeyi mümkün olan en az düzeyde tutar.
Tüm işlemler yerel makinenizde gerçekleşir; kişisel veriler buluta gönderilmez.

KVKK kapsamındaki talepleriniz için: nexus-agent-privacy@example.com

---

## 6. Güvenlik

- API anahtarları: Windows DPAPI ile şifrelenmiş Credential Manager
- Ekran verileri: RAM'de geçici olarak tutulur, diske yazılmaz
- SQLite veritabanı: yalnızca mevcut kullanıcı hesabı erişebilir
- Ağ iletişimi: yalnızca HTTPS (TLS 1.2+)

---

## 7. Politika Güncellemeleri

Bu politika güncellendiğinde:

- Uygulama başlangıcında yeni sürüm gösterilir
- Her sürüm için yeniden onay istenir
- Onay vermezseniz bulut özellikleri devre dışı kalır
- Yerel işlevler her zaman çalışmaya devam eder

---

## 8. İletişim

Gizlilik politikası hakkında sorularınız için:

- GitHub Issues: https://github.com/nexus-agent/nexus-agent/issues
- E-posta: nexus-agent-privacy@example.com

---

*Bu belge Nexus Agent Sürüm 1.0 için geçerlidir.*
