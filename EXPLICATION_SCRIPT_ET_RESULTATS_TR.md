# Pirinçte kuraklık stresinin transkriptomik analizi: metodolojik açıklama ve sonuçların yorumu

## Özet

Bu belge, pirince ait transkriptomik verilere (`n = 18` örnek, yaklaşık `p = 25 000` gen) uygulanan denetimli sınıflandırma hattının akademik bir değerlendirmesini sunar. Amaç, bilgi sızıntısı ve aşırı öğrenme risklerini sınırlandırırken, `Control (CT)` ve `Drought (D)` örneklerini makine öğrenmesi modelleriyle ayırt etmektir.  
Sonuçlar ayırt edici bir sinyalin varlığını göstermektedir; ancak örneklem büyüklüğünün çok düşük olması nedeniyle sağlamlık sınırlıdır. Bu nedenle performanslar ön bulgular ve keşif amaçlı sonuçlar olarak yorumlanmalıdır.

## 1. Bilimsel bağlam ve amaç

Kuraklık stresi, fizyolojik durumların sınıflandırılmasında kullanılabilecek transkriptomik yeniden düzenlenmelere yol açar. Bu çerçevede ele alınan soru şudur:

**Yaprak transkriptomundan su durumu (kontrol vs stres) tahmin edilebilir mi?**

Çalışma, **yüksek boyut - düşük örneklem** (`p >> n`) bağlamında yürütülmüştür; bu durum klasik olarak şu unsurlarla ilişkilidir:

- yüksek aşırı öğrenme riski,
- performans tahminleyicilerinde önemli varyans,
- aday gen imzalarında sınırlı kararlılık.

## 2. Veri ve ön işleme

Veriler başlangıçta `genler x örnekler` biçiminde düzenlenmiş bir TPM tablosundan (`TPM_table.txt`) gelmektedir; daha sonra `örnekler x genler` biçimine transpoze edilmiştir.  
Sınıf etiketleri örnek kimliklerinden türetilmiştir:

- `CT` : kontrol koşulu,
- `D` : kuraklık stresi koşulu.

Gözlenen dağılım dengelidir (9 vs 9).

Stratifiye bir `train/test` bölmesi **her türlü dönüşümden önce** yapılmıştır:

- train : 13 örnek,
- test : 5 örnek.

Varyansı sıfır olan genler eğitim kümesinden çıkarılmış, ardından aynı sütun alt kümesi test kümesine uygulanmıştır.

## 3. Metodolojik strateji

## 3.1 Değişken seçimi

Boyut indirgeme şu yaklaşımla gerçekleştirilmiştir:

- `SelectFromModel`
- `L1` cezalı `LogisticRegression` (`liblinear`)
- seçilen değişken sayısına üst sınır (pratikte en fazla 20 gen)

Bu tercih, seyrekliği destekler ve rastlantısal korelasyonların yakalanma riskini azaltır.

## 3.2 Eğitilen modeller

İki model ailesi karşılaştırılmıştır:

- **Random Forest** (doğrusal olmayan model; değişken önemleri üzerinden yorumlanabilir),
- **MLP** (sinir ağı), standardizasyon ve PCA indirgemesinden sonra uygulanmıştır.

## 3.3 Değerlendirme ve sağlamlık

Birden fazla değerlendirme düzeyi birleştirilmiştir:

- hold-out test skoru (5 örnek),
- stratifiye 3-fold çapraz doğrulama,
- tekrarlı stratifiye 3-fold (10 tekrar),
- LOOCV (Leave-One-Out),
- permütasyon testi (rastgeleye karşı karşılaştırma).

Yaklaşım, bilgi sızıntısını önlemeye yönelik iyi uygulamalarla uyumludur (değişken seçimi ve modelleme CV pipeline'ları içinde kapsüllenmiştir).

## 4. Temel sonuçlar

### Random Forest

- Test F1 : **0.8000**
- CV 3-fold F1 : **0.7667 +/- 0.2055**
- Tekrarlı CV F1 : **0.6617 +/- 0.2004**
- LOOCV F1 : **0.3077 +/- 0.4615**

### MLP

- Test F1 : **0.5714**
- CV 3-fold F1 : **0.6349 +/- 0.0449**
- LOOCV F1 : **0.5385 +/- 0.4985**

### Rastgeleye karşı test (RF)

- gerçek skor (CV) : **0.7667**
- ortalama rastgele skor : **0.4747**
- kazanç : **+0.2920**

Bu sonuçlar sıfır olmayan bir biyolojik sinyale işaret etmektedir; ancak genelleme kapasitesi üzerinde önemli bir belirsizlik vardır.

## 5. Şekillerin yorumu

## 5.1 Ana şekil : `ml_results_final.png`

Bu şekil altı tamamlayıcı görünümü bir araya getirir:

- RF ve MLP karışıklık matrisleri,
- ROC eğrileri,
- RF önemine göre sıralanmış üst genler,
- CV/LOOCV skor dağılımlarının karşılaştırması,
- örneklerin PCA projeksiyonu.

Temel nokta: test üzerindeki gözlenen performanslar (çok küçük örneklem) tek başına değerlendirilmemelidir; LOOCV/CV skorlarıyla birlikte yorumlanması zorunludur.

## 5.2 Öğrenme eğrisi : `learning_curve.png`

Eğri, eğitim ve doğrulama performansı arasında kalıcı bir fark ve doğrulamada geniş aralıklar göstermektedir. Bu profil, yetersiz veri rejimi ve aşırı öğrenme riskiyle uyumludur.

## 5.3 Sentez grafiği : `performance_explicative.png`

Bu şekil, RF ve MLP için şu metrikleri karşılaştırır:

- Test F1,
- CV 3-fold F1,
- Tekrarlı CV F1,
- LOOCV F1.

Hata çubukları tahmini varyansı görselleştirir. Özellikle RF için performansın LOOCV yönünde düşmesi, gerçek performansın temkinli yorumlanmasını destekler.

## 6. Tartışma

## 6.1 Metodolojik olarak güçlü yönler

- train/test ayrımının dönüşümlerden önce yapılması,
- açık CV pipeline'ları (bilgi sızıntısının sınırlandırılması),
- birden fazla doğrulama protokolünün kullanılması,
- rastgeleye karşı istatistiksel test,
- düzenlileştirilmiş ve kardinalite kısıtlı değişken seçimi.

## 6.2 Başlıca sınırlılıklar

- çok düşük örneklem büyüklüğü (`n = 18`),
- çok küçük hold-out test (`n_test = 5`),
- fold'lar arası yüksek değişkenlik,
- seçilen genlerde kısmi kararlılık,
- bağımsız dış doğrulamanın olmaması.

Bu nedenle tanımlanan genler **keşif amaçlı adaylar** olarak nitelendirilmelidir; doğrulanmış biyobelirteçler olarak değil.

## 7. Sonuç

Uygulanan pipeline, prosedürel açıdan titizdir ve hidrik koşullar arasında ayırt edici bir sinyal ortaya koymaktadır. Bununla birlikte, istatistiksel sağlamlık örneklem boyutu nedeniyle sınırlı kalmaktadır.  
Performanslar keşif perspektifinde savunulabilir olsa da, bu aşamada güçlü bir genelleme iddiasını desteklemez.

## 8. Gelecek çalışmalar

Sonuçları bilimsel olarak güçlendirmek için şu adımlar önerilir:

- örneklem boyutunun anlamlı biçimde artırılması (ideal olarak >= 100),
- bağımsız bir kohortta dış doğrulama yapılması,
- kültivarlar arası aktarılabilirliğin açıkça test edilmesi,
- aday genlerin deneysel olarak doğrulanması (ör. qRT-PCR),
- seçim kararlılığı yaklaşımlarıyla tamamlanması (stability selection, bootstrap).

