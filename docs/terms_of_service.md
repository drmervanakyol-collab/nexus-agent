# Nexus Agent — Kullanım Koşulları

**Sürüm:** 1.0  
**Yürürlük tarihi:** 2026-04-10  
**Son güncelleme:** 2026-04-10

---

## 1. Kabul

Nexus Agent'ı kurarak veya kullanarak bu Kullanım Koşullarını kabul etmiş
sayılırsınız. Koşulları kabul etmiyorsanız yazılımı kullanmayınız.

---

## 2. Kullanım Koşulları

### 2.1 Lisans

Nexus Agent, açık kaynak lisansı altında sunulmaktadır. Yazılımı:

- Kişisel ve ticari amaçlarla kullanabilirsiniz
- Kopyalayabilir ve dağıtabilirsiniz (lisans koşullarına uygun olarak)
- Kaynak kodunu inceleyebilir ve değiştirebilirsiniz

### 2.2 Kullanıcı Sorumlulukları

Nexus Agent'ı kullanan kişi/kurum:

- Yazılımın kendi makinesinde gerçekleştirdiği **tüm eylemlerden** sorumludur
- Otomatik görevleri başlatmadan önce sonuçları değerlendirmelidir
- Üçüncü taraf hizmetlere yapılan API çağrılarından doğan maliyetlerden sorumludur
- Yerel mevzuat ve üçüncü taraf hizmet koşullarına uyum sağlamalıdır

### 2.3 Hesap ve Kimlik Bilgileri

- Nexus Agent **hesap oluşturmaz**; kullanıcı profili tutmaz
- API anahtarları (BYOK) yerel makinenizde şifreli saklanır
- Anahtarlarınızın güvenliğinden siz sorumlusunuz

---

## 3. Sorumluluk Sınırı

### 3.1 Garanti Reddi

NEXUS AGENT "OLDUĞU GİBİ" SUNULMAKTADIR. AÇIK VEYA ZIMNİ HİÇBİR GARANTİ
VERİLMEMEKTEDİR; TİCARİ ELVERİŞLİLİK, BELİRLİ BİR AMACA UYGUNLUK VEYA
İHLAL ETMEME GARANTİSİ DAHİL ANCAK BUNLARLA SINIRLI OLMAMAK ÜZERE.

### 3.2 Sorumluluk Sınırı

HİÇBİR KOŞULDA NEXUS AGENT GELİŞTİRİCİLERİ VEYA KATKILICILARI;
VERİ KAYBI, SİSTEM HASARI, İŞ KAYBI VEYA DİĞER DOLAYLI, ÖZEL, ARIZI VEYA
SONUÇ NİTELİĞİNDEKİ ZARARLARDAN SORUMLU TUTULAMAZ; BU TÜR ZARARLAR OLASILIĞI
ÖNCEDEn BİLDİRİLMİŞ OLSA DAHİ.

### 3.3 Maksimum Sorumluluk

Uygulanabilir hukukun izin verdiği azami ölçüde, toplam sorumluluk son 12
aylık dönemde yazılım için ödediğiniz tutarla sınırlıdır. Yazılım ücretsiz
kullanılıyorsa bu sınır sıfırdır.

---

## 4. BYOK (Kendi API Anahtarınızı Getirin) Sorumluluğu

### 4.1 API Anahtarı Kullanımı

Nexus Agent, bulut yapay zeka modellerine erişim için kendi API anahtarınızı
kullanır (BYOK — Bring Your Own Key modeli):

- API anahtarları **sizin mülkünüzdür** ve sizin kontrolünüzdedir
- Anahtarınız aracılığıyla yapılan tüm API çağrılarından **siz sorumlusunuz**
- Oluşan maliyetler (token ücretleri) doğrudan ilgili sağlayıcı tarafından
  sizden tahsil edilir
- Nexus Agent hiçbir ödeme bilgisi toplamaz veya işlemez

### 4.2 Anahtar Güvenliği

- API anahtarlarınızı başkalarıyla paylaşmayınız
- Anahtarınızın sızdığından şüpheleniyorsanız derhal sağlayıcı panelinden iptal ediniz
- Nexus Agent, anahtarınızı yalnızca sizin onayladığınız işlemler için kullanır

### 4.3 Maliyet Sınırları

Nexus Agent, yapılandırılabilir maliyet sınırları sunar:
- `max_task_cost_usd`: Görev başına maksimum harcama
- `max_daily_cost_usd`: Günlük maksimum harcama

Bu sınırlar aşıldığında görev otomatik olarak durdurulur. Ancak, sağlayıcı
tarafında oluşmuş ücretler için Nexus Agent sorumlu tutulamaz.

---

## 5. Kabul Edilemez Kullanım

Nexus Agent'ı aşağıdaki amaçlarla kullanmak **kesinlikle yasaktır**:

### 5.1 Yasal Olmayan Faaliyetler

- Herhangi bir yargı bölgesinde yasadışı sayılan eylemler
- Telif hakkı veya fikri mülkiyet ihlali
- Kişisel verilerin izinsiz toplanması veya işlenmesi
- Bilişim sistemlerine yetkisiz erişim (bilgisayar korsanlığı)

### 5.2 Zararlı Faaliyetler

- Zararlı yazılım (malware, ransomware, spyware) oluşturma veya dağıtma
- Spam veya kitlesel istenmeyen mesaj gönderme
- Hizmet engelleme (DDoS) saldırıları
- Kimlik avı (phishing) ve sosyal mühendislik saldırıları

### 5.3 Gizlilik İhlalleri

- Başkalarının ekranlarını izinleri olmadan izleme veya kaydetme
- Başkalarına ait kişisel verileri toplama, işleme veya satma
- Biyometrik veri toplama

### 5.4 Üçüncü Taraf İhlalleri

- Kullandığınız bulut sağlayıcıların (Anthropic, OpenAI vb.) hizmet koşullarının ihlali
- Otomasyon engelleme sistemlerini (CAPTCHA vb.) atlatma girişimleri
- Bot koruması olan sistemleri hedef alma

### 5.5 Kritik Altyapı

- Kritik altyapı sistemleri (elektrik, su, ulaşım, sağlık) üzerinde yetkisiz otomasyon

---

## 6. Değişiklikler

Bu koşullar güncellenebilir. Önemli değişiklikler:

- Uygulama başlangıcında bildirilir
- Her yeni sürüm için yeniden onay istenir
- Önceki versiyonlar `docs/terms_of_service.md` git geçmişinde erişilebilir kalır

---

## 7. Uygulanabilir Hukuk

Bu koşullar Türkiye Cumhuriyeti hukukuna tabidir. Anlaşmazlıklar İstanbul
mahkemelerinde çözülür.

---

## 8. Bölünebilirlik

Bu koşulların herhangi bir maddesinin geçersiz sayılması, geri kalan maddelerin
geçerliliğini etkilemez.

---

## 9. İletişim

Kullanım koşulları hakkında sorularınız için:

- GitHub Issues: https://github.com/nexus-agent/nexus-agent/issues
- E-posta: nexus-agent-legal@example.com

---

*Bu belge Nexus Agent Sürüm 1.0 için geçerlidir.*
