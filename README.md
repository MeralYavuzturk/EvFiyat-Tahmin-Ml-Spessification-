### 🏠 Machine Learning: Ev Fiyatı Tahminleme Modeli
#### Bu proje, konut özelliklerine (metrekare, oda sayısı vb.) dayanarak ev fiyatlarını tahmin etmek için geliştirilmiş bir Makine Öğrenmesi (ML) modelidir. Proje, regresyon analizi kullanarak veriler arasındaki ilişkileri modeller ve gerçekçi fiyat tahminleri sunar.
### Çalıştırılabilir zip dosyası linki: https://github.com/MeralYavuzturk/EvFiyat-Tahmin-Ml-Spessification-/releases/tag/v.1.0.0

### 📊 Model Performansı ve Metrikler
#### Eğitilen model, başarı oranını ölçmek için aşağıdaki standart regresyon metriklerini kullanmaktadır:

* R-Kare (R-Squared): 0,98 – Modelimiz verideki değişkenliğin %98'ini açıklayabilmektedir (Yüksek doğruluk).

* MAE (Ortalama Mutlak Hata): 10.545,19 – Tahminlerin gerçek fiyatlardan ortalama sapma miktarı.

* RMSE (Ortalama Kare Kök Hata): 11.068,42 – Hataların büyüklüğünü ölçen duyarlı bir metrik.

### 🚀 Özellikler (Features)
* Modelimiz tahmin yaparken şu temel parametreleri kullanır:

* Boyut (m²): Evin kullanım alanı.

* Oda Sayısı: Toplam oda sayısı.

### 🛠️ Teknik Altyapı
* Dil: C#

* Teknoloji: ML.NET (Microsoft'un açık kaynaklı makine öğrenmesi kütüphanesi)

* Model Türü: Regresyon (Regression)

* Platform: .NET Core / .NET 8.0

### 📖 Nasıl Çalışır?
1. Veri Hazırlama: Eğitim verileri modele yüklenir ve özellikler (features) normalize edilir.

2. Eğitim (Training): Algoritma, geçmiş satış verileri üzerinden fiyat ile özellikler arasındaki ilişkiyi öğrenir.

3. Test & Değerlendirme: Modelin tahminleri gerçek verilerle kıyaslanarak başarı metrikleri hesaplanır.

4. Tahmin (Prediction): Kullanıcı yeni bir evin özelliklerini girdiğinde, model öğrenmiş olduğu veriler ışığında bir fiyat tahmini üretir.

#### 💻 Örnek Tahmin Çıktısı
Özellikler: Boyut=110 m², Oda Sayısı=2

Tahmin Edilen Fiyat: 218.101,75 TL
<img width="1457" height="579" alt="image" src="https://github.com/user-attachments/assets/bb9e115c-257c-43b9-bcfe-23026ebf4b4a" />

