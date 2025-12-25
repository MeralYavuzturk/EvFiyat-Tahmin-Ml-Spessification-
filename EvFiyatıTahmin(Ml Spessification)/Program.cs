// 1. GİRİŞ VERİSİ MODELİ (Özellikler ve Etiket)
// Şimdi modelimiz hazır!
using System;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Data;
using System.Linq; // List kullanmak için

// ------------------------------------------------------------------
// 1. VERİ MODELLERİ
// Sınıflar, namespace'in (isim uzayının) içinde, Program sınıfının dışında yer almalıdır.
// ------------------------------------------------------------------

public class EvVerisi
{
    [LoadColumn(0)]
    public float Boyut { get; set; } // Fiyatı tahmin etmek için kullanılan özellikler

    [LoadColumn(1)]
    public float OdaSayisi { get; set; }

    [LoadColumn(2), ColumnName("Label")]
    public float Fiyat { get; set; } // Tahmin etmeye çalıştığımız değer (etiket)
}

public class EvTahmini
{
    [ColumnName("Score")]
    public float TahminEdilenFiyat { get; set; }
}

// ------------------------------------------------------------------
// 2. ANA PROGRAM SINIFI
// ------------------------------------------------------------------
namespace EvFiyatTahmini // İsim Uzayınız (Projenizin Adı)
{
    class Program
    {
        // public static void Main(string[] args) METODU BURADA, 
        // class Program { ... } SÜSLÜ PARANTEZLERİNİN İÇİNDE BAŞLAR
        public static void Main(string[] args)
        {
            // 1. ML.NET bağlamını başlatma
            MLContext mlContext = new MLContext();

            // 2. Eğitim Verisi Oluşturma (Veri Tipleri Düzeltildi: Her sayının sonunda 'f')
            List<EvVerisi> houseDataList = new List<EvVerisi>
            {
                new EvVerisi { Boyut = 80f, OdaSayisi = 1f, Fiyat = 150000f },
                new EvVerisi { Boyut = 100f, OdaSayisi = 2f, Fiyat = 220000f },
                new EvVerisi { Boyut = 120f, OdaSayisi = 3f, Fiyat = 280000f },
                new EvVerisi { Boyut = 95f, OdaSayisi = 2f, Fiyat = 190000f },
                new EvVerisi { Boyut = 160f, OdaSayisi = 4f, Fiyat = 410000f },
                new EvVerisi { Boyut = 70f, OdaSayisi = 1f, Fiyat = 130000f },
                new EvVerisi { Boyut = 130f, OdaSayisi = 3f, Fiyat = 300000f },
                // Eklenen diğer veriler:
                new EvVerisi { Boyut = 110f, OdaSayisi = 2f, Fiyat = 235000f },
                new EvVerisi { Boyut = 140f, OdaSayisi = 3f, Fiyat = 320000f },
                new EvVerisi { Boyut = 180f, OdaSayisi = 4f, Fiyat = 450000f },
            };

            // 3. Veriyi IDataView formatına dönüştürme
            IDataView trainingData = mlContext.Data.LoadFromEnumerable(houseDataList);

            // 4. Veri İşleme ve Eğitim Hattını (Pipeline) Tanımlama
            var pipeline = mlContext.Transforms.Concatenate(
                "Features",
                nameof(EvVerisi.Boyut),
                nameof(EvVerisi.OdaSayisi)
            )
            // Algoritma değiştirildi: Daha az veri için SDCA kullanılıyor.
            //.Append(mlContext.Regression.Trainers.Sdca(labelColumnName: "Label", featureColumnName: "Features"));
            .Append(mlContext.Regression.Trainers.LbfgsPoissonRegression());
            Console.WriteLine("--- Model Eğitimi Başlıyor ---");

            // 5. Modeli Eğitme
            var model = pipeline.Fit(trainingData);

            Console.WriteLine("--- Model Eğitimi Tamamlandı ---");
            Console.WriteLine();
            // Main metodunun içinde, uygun bir yere ekleyin:

            // Test Verisi Oluşturma (Gerçek fiyatları bilinen, eğitimde kullanılmayan veriler)
            List<EvVerisi> testDataList = new List<EvVerisi>
    {
        // Boyut, Oda Sayısı, Gerçek Fiyat
        new EvVerisi { Boyut = 115f, OdaSayisi = 3f, Fiyat = 285000f },
        new EvVerisi { Boyut = 145f, OdaSayisi = 4f, Fiyat = 390000f },
        new EvVerisi { Boyut = 85f, OdaSayisi = 2f, Fiyat = 180000f }
    };

            // Veriyi IDataView formatına dönüştürme
            IDataView testData = mlContext.Data.LoadFromEnumerable(testDataList);
            // Modeli eğittiğiniz satırın hemen altına (var model = pipeline.Fit(trainingData);) ekleyin:

            Console.WriteLine("--- Model Testi Başlıyor (Değerlendirme) ---");

            // Modeli test verileri üzerinde dönüştürerek tahminler yapın
            var predictions = model.Transform(testData);

            // Regresyon görevinde performansı ölçün
            // "Label" (Gerçek Fiyat) ve "Score" (Tahmin Edilen Fiyat) sütunlarını kullanır
            var metrics = mlContext.Regression.Evaluate(predictions, "Label", "Score");

            Console.WriteLine("--- Model Başarı Metrikleri ---");
            Console.WriteLine($"📏 Ortalama Mutlak Hata (MAE): {metrics.MeanAbsoluteError:F2}");
            Console.WriteLine($"📏 Ortalama Kare Kök Hata (RMSE): {metrics.RootMeanSquaredError:F2}");
            Console.WriteLine($"📏 R-Kare (R Squared): {metrics.RSquared:F2}");
            Console.WriteLine("------------------------------------------");
            // 6. Tahmin Motoru Oluşturma
            var predictionEngine = mlContext.Model.CreatePredictionEngine<EvVerisi, EvTahmini>(model);

            // 7. Yeni Bir Tahmin Örneği
            var yeniEv = new EvVerisi()
            {
                Boyut = 110f, // Boyut: 110 m²
                OdaSayisi = 2f // Oda Sayısı: 2
            };

            // 8. Tahmin Yapma
            var tahmin = predictionEngine.Predict(yeniEv);

            // 9. Sonucu Ekrana Yazdırma
            Console.WriteLine($"🔍 Özellikler: Boyut={yeniEv.Boyut} m², Oda Sayısı={yeniEv.OdaSayisi}");
            Console.WriteLine($"💰 Tahmin Edilen Fiyat: {tahmin.TahminEdilenFiyat:C2}");
            Console.WriteLine("------------------------------------------");

            // Konsol penceresinin kapanmaması için bekleme (Zaten F5/Ctrl+F5 kullanıyorsanız gerekmez)
            // Console.ReadKey();

        } // public static void Main(string[] args) METODUNUN BİTİŞ PARANTEZİ
    } // class Program SINIFININ BİTİŞ PARANTEZİ
} // namespace EvFiyatTahmini İSİM UZAYININ BİTİŞ PARANTEZİ