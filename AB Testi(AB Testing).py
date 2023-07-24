# Başlık : AB Testi (AB Testing)

# Başlık : Örneklem (Sampling)

# Temel İstatistik Kavramları

import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.stats.api as sms
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, \
    pearsonr, spearmanr, kendalltau, f_oneway, kruskal
from statsmodels.stats.proportion import proportions_ztest

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 10)
pd.set_option("display.float_format", lambda x: "%.5f" % x)

###################
# Sampling (Örnekleme)
###################

# Bir popülasyon oluşturalım.

populasyon = np.random.randint(0, 80, 10000)

# Popülasyonun ortalamasına bakalım.

populasyon.mean()

np.random.seed(115)

# Örneklem seçelim.

orneklem = np.random.choice(a=populasyon, size=100)

orneklem.mean()

# Diyelim ki 10 tane örneklem çekmek istiyoruz.

np.random.seed(10)
orneklem1 = np.random.choice(a=populasyon, size=100)
orneklem2 = np.random.choice(a=populasyon, size=100)
orneklem3 = np.random.choice(a=populasyon, size=100)
orneklem4 = np.random.choice(a=populasyon, size=100)
orneklem5 = np.random.choice(a=populasyon, size=100)
orneklem6 = np.random.choice(a=populasyon, size=100)
orneklem7 = np.random.choice(a=populasyon, size=100)
orneklem8 = np.random.choice(a=populasyon, size=100)
orneklem9 = np.random.choice(a=populasyon, size=100)
orneklem10 = np.random.choice(a=populasyon, size=100)

# Örneklem ortalamalarının ortalamasını alalım.

(orneklem1.mean() + orneklem2.mean() + orneklem3.mean() + orneklem4.mean() + orneklem5.mean()
 + orneklem6.mean() + orneklem7.mean() + orneklem8.mean() + orneklem9.mean() + orneklem10.mean()) / 10

# Buradan çıkarılacak sonuç: Öreklem sayısı arttıkça, ortalama gerçek kitlenin ortalamasına yakınsar.

# Başlık : Betimsel İstatistikler (Descriptive Statistics)

# Elimizdeki veri setini betimlemeye çalışma çabasıdır. Açıklayıcı istatistik gibi isimlerle de karşımıza çıkar.

df = sns.load_dataset("tips")

# Betimsel istatistikleri getirelim.

df.describe().T


# Başlık : Güven Aralıkları (Confidence Intervals)

# Tanım : Anakütle parametresinin tahmini değerini (istatistik) kapsayabilecek iki sayıdan oluşan bir aralık bulunmasıdır.

# Web sitesinde geçirilen ortalama sürenin güven aralığı nedir sorumuz olsun.
# Elimizdeki bilgiler:
# Ortalama : 180 saniye
# Standart sapma : 40 saniye
# 165 - 170 - 175 - 180 - 185 - 190 - 200
# Açıklama : Aslında elimizde 180 saniye ortalama süremiz mevcut. Peki bu bize yetmiyor mu ? Neden ? Şöyle açıklayalım : Elimizde 180 saniye gibi tek bir değer olmasındansa belli bir ortalama aralığının olması yorumlama açısından daha iyidir. Kullanıcılar 180 saniyenin etrafında geziniyor olabilir ve bu 180 in etrafındaki süreleri daha iyi kapsamak için bir güven aralığı ihtiyacımız oluşur. Örneğin 172-188 aralığına erişmiş olalım. Şu yorumu yapabiliriz : Kullanıcıların web sitesinde geçirdiği süre % 95 güven ile 172-188 saniye aralığındadır.
# Tekrar tekrar örneklem çektiğimizde ortalama sürenin bu aralıkta olacağını bilmiş oluruz.
# Hesaplama Adımları :
# Adım 1 : n, ortalama ve standart sapmayı bul. n = 100, ortalama = 180, standart sapma = 40
# Adım 2 : Güven aralığına karar ver : 95 mi 99 mu ? Z tablo değerini hesapla. (1,96 - 2,57)
# Adım 3 : Yukarıdaki değerleri kullanarak güven aralığını hesapla:
# x +- z * s / sqrt(n) = 180 +- 1.96 * 40 / sqrt(100)
# Açıklama : Sen bu popülasyondan örneklem alacaksın. Başka zaman başka bir örneklem de alabilirsin. Olası alabileceğin 100 örneklemden 95 inin ortalaması 172 - 188 aralığında olacaktır.

# Başlık : Güven Aralıkları Uygulama (Application of Confidence Intervals)

# Tips Veri Setindeki Sayısal Değişkenler için Güven Aralığı Hesabı
df = sns.load_dataset("tips")
df.describe().T

# Ulaşmak istediğimiz hedef şu: Total bill değişkeninin ortalamasının güven aralığı nedir ?

df.head()

# Bir ortalama hesap bilgisi yetmemektedir. Çünkü iyi senaryoda ne kadar ortalama kazanç sağlanabilir ? Kötü senaryoda ortalama ne kadar kazanç sağlanabilir ? gibi senaryoların da göz önünde bulundurulması gerekir.

# Bunun için bir güven aralığı hesabı işlemini gerçekleştirelim.

sms.DescrStatsW(df["total_bill"]).tconfint_mean()

# Bu güven aralığı ne anlama geliyor ?

# Ben 100 defa örneklem çeksem, 100 defa hesabın ortalamasını alsam, bunların 95 inde ortalama hesapladığımız güven aralığı değerlerinin arasında çıkacaktır. ( %5 hata payı )

# Gelecek bahşişler ile ilgili bir değerlendirme yapalım.

sms.DescrStatsW(df["tip"]).tconfint_mean()

# Titanic veri seti için aynı işlemleri gerçekleştirelim.

df = sns.load_dataset("titanic")
df.describe().T
sms.DescrStatsW(df["age"].dropna()).tconfint_mean()
sms.DescrStatsW(df["fare"].dropna()).tconfint_mean()

# Başlık : Korelasyon (Correlation)

# Tanım : Değişkenler arasındaki ilişki, bu ilişkinin yönü ve şiddeti ile ilgili bilgiler sağlayan istatistiksel bir yöntemdir.

# Başlık : Korelasyon Uygulaması (Application of Correlation)

# Bahşiş veri seti :
# total_bill : yemeğin toplam fiyatı (bahşiş ve vergi dahil)
# tip : bahşiş
# sex : ücreti ödeyen kişinin cinsiyeti (0=male, 1=female)
# smoker : grupta sigara içen var mı ? (0=No, 1=Yes)
# day : gün (3=Thur, 4=Fri, 5=Sat, 6=Sun)
# time : ne zaman? (0=Day, 1=Night)
# size : grupta kaç kişi var ?

df = sns.load_dataset("tips")
df.head()

# Verilen bahşişler ile ödenen hesap arasında bir korelasyon var mı araştırması yapacağız.

df["total_bill"] = df["total_bill"] - df["tip"]
df.plot.scatter("tip", "total_bill")
plt.show(block=True)

# Hesaplamasını yapalım.

df["tip"].corr(df["total_bill"])

# Başlık : Hipotez Testleri (Hypothesis Testing)

# Tanım : Bir inanışı, bir savı test etmek için kullanılan istatistiksel yöntemlerdir.

# Hipotez testleri kapsamında grup karşılaştırmalarına odaklanacağız.

# Grup karşılaştırmalarında temel amaç olası farklılıkların şans eseri ortaya çıkıp çıkmadığını göstermeye çalışmaktır.

# Örnek : Mobil uygulamada yapılan arayüz değişikliği sonrasında kullanıcıların uygulamada geçirdikleri günlük ortalama süre arttı mı ?

# Modelleme : Arayüz değişikliği öncesinde yani A grubunda, arayüz değişikliği sonrasında yani B grubunda, kullanıcılar arasında uygulamada geçirilen süre açısından fark yoktur şeklinde hipotez kuruyoruz. Ardından bunu test ediyoruz. Diyelim ki bu iki değişiklik neticesinde mobil uygulama kullanıcılarının bir kısmına eski tasarım, bir kısmına yeni tasarım gösterilsin. Bundan sonra ölçüm yapılsın.
# Tasarım 1 Ortalama : 55 Dakika
# Tasarım 2 Ortalama : 58 Dakika
# 2.tasarımın daha iyi olduğu, web sitesinde geçen süreyi artırdığı iddia edilebilir mi ? Edilemez. Çünkü bir örnek aldık ve bu farklılık şans eseri ortaya çıkmış olabilir. Buradaki farklılığın şans eseri çıkıp çıkmadığını istatistiksel olarak ispat etmek gerekir.

# Başlık : AB Testi ( Bağımsız İki Örneklem T Testi )

# AB testi denildiğinde çok yaygınca ya iki grubun ortalaması, ya da iki gruba ilişkin oranlar hesaplanıyordur.

# Bu bölümde bağımsız iki örneklem t testini yani iki grubu karşılaştırma testini ele alacağız.

# Bağımsız İki Örneklem T Testi iki grup ortalaması arasında karşılaştırma yapmak istenildiğinde kullanılır.

# Burada bahsedilen A grubu deney grubu, B grubu kontrol grubudur.

# Genellikle mobil ya da web uygulamalarında gerçekleştirilen yenilikler ya da deneme yapılması istenilen özelliklerin test edilmesi için kullanılır.

# İki Grup Ortalamasını Karşılaştırma ( Bağımsız İki Örneklem Testi ):

# Hipotezlerimiz mevcuttur.(H0 ve H1)

# H0 yokluk hipotezidir. Sınayacağımız durum bu durumdur. (İki grup ortalaması arasında fark yoktur der.) Bu durumu reddedip reddememeye göre iki grup ortalaması arasında fark olup olmadığını değerlendiririz.

# p value değerine bakarak hipotezlerin sonucunu yorumlayacağız. (p<0.05)

# İlgili fonksiyonları kullandığımızda bu fonksiyonlar bize bir p value değeri veriyor olacak. Bu p value değerlerine bakacağız. Eğer p value < 0.05 ise H0 reddedilir. t istatistiğinde ise th > tt ise H0 reddedilir.(tt = tablo değeri)

# Kullanacak olduğumuz ve bize bazı bilimsel kıyaslamalar ve karara varma imkanları sağlayan yöntemlerin bazı varsayımları vardır:

# Normallik (İki grubun da ayrı ayrı normal dağılmış olması gerekir.)
# Varyans Homojenliği (İki grubun dağılımının benzerliği)

# Çalışmanın başında bir hipotez kuracağız(1.adım)
# Varsayımları inceleyeceğiz(2.adım)
# p valueya bakarak yorum yapacağız.(3.adım)

# Uygulama : AB Testing ( Bağımsız İki Örneklem T Testi )

# Gidiş Yolu :
# 1. Hipotezleri kur.
# 2. Varsayım kontrolü.
#   1 - Normallik varsayımı
#   2 - Varyans homojenliği
# 3. Hipotezin uygulanması.
#   1 - Varsayımlar sağlanıyorsa bağımsız iki örneklem t testi (parametrik test)
#   2 - Varsayımlar sağlanmıyorsa mannwhiteneyu testi (non-parametrik test)
# 4. p-value değerine göre sonuçları yorumla.
# Not:
# Normallik sağlanmıyorsa direkt 2 numara. Varyans homojenliği sağlanmıyorsa 1 numaraya argüman girilir.
# Normallik incelemesi öncesi aykırı değer incelemesi ve düzeltmesi yapmak faydalı olabilir.

# Uygulama 1: Sigara içenler ile içmeyenlerin hesap ortalamaları arasında istatistiki olarak anlamlı fark var mı ?

df = sns.load_dataset("tips")
df.head()

df.groupby("smoker").agg({"total_bill": "mean"})

# Arada gözüktüğü gibi matematiksel olarak fark vardır. Peki bu fark istatistiki olarak da var mıdır ? Yoksa bu fark şans eseri mi oluşmuştur ?

# Başlayalım.

# 1.adım: Hipotezi kur.
# H0: M1 = M2
# H1: M1 != M2

# 2.adım: Varsayım kontrolü yap.

# Normallik Varsayımı
# Varyans Homojenliği

# Normallik Varsayımı: Bir değişkenin dağılımının standart normal dağılıma benzer olup olmadığının hipotez testidir.
# H0: Normal dağılım varsayımı sağlanmaktadır.
# H1: Sağlanmamaktadır.
# shapiro testi bir değişkenin dağılımının normal olup olmadığını test eder.

test_stat, pvalue = shapiro(df.loc[df["smoker"] == "Yes", "total_bill"])
print("Test Stat = %.4f, p-value = %.4f" % (test_stat, pvalue))

# p-value < ise 0.05'ten H0 red.
# p-value < değilse 0.05 H0 reddedilemez.

test_stat, pvalue = shapiro(df.loc[df["smoker"] == "No", "total_bill"])
print("Test Stat = %.4f, p-value = %.4f" % (test_stat, pvalue))

# Bu uygulamamızda iki grup için de normal dağılım varsayımı sağlanmamaktadır. Non-parametrik bir test kullanılmalıdır.

# Varyans Homojenliği Varsayımı

# Buradaki hipotezlerimiz:
# H0: Varyanslar homojendir.
# H1: Varyanslar homojen değildir.
# Buradaki hipotezler varsayım hipotezleridir. Varsayım hipotezlerindeki H0ların bizim tarafımızdan ele alınışı ile yapacak olduğumuz gerçek hipotezin bizim açımızdan ele alınışı farklı olacaktır. Burada H0 ı ilk başta reddettik ama reddedilmemesini istiyoruz. Çünkü: varsayımlar sağlansın istiyoruz. Gerçek hipotez kısmında ise hipotezi reddetmek istiyoruz.

# Varyans homojenliği varsayımını inceleyelim. Bu varsayımı incelemek için levene testi kullanılır.

test_stat, pvalue = levene(df.loc[df["smoker"] == "Yes", "total_bill"],
                           df.loc[df["smoker"] == "No", "total_bill"])
print("Test Stat = %.4f, p-value = %.4f" % (test_stat, pvalue))

# p-value değeri 0.05 ten küçük çıktığı için H0 ı reddederiz. Varyanslar homojen değil.

# 3.adım : Hipotezin Uygulanması

# 1. Varsayımlar sağlanıyorsa bağımsız iki örneklem t testi (parametrik test)
# 2. Varsayımlar sağlanmıyorsa mannwhiteneyu testi (non-parametrik test)

# Sağlanmış senaryosuna göre davranalım. (Öğrenmek için) (Aslında varsaymlar sağlanmadı)

# Varsayımlar sağlanıyorsa(Normallik varsayımı ya da her iki varsayım birden sağlanıyorsa) (Dikkat: Sadece normallik varsayımı sağlanıyorsa ikinci argümana false girilir.) : Bağımsız iki örneklem t testi

test_stat, pvalue = ttest_ind(df.loc[df["smoker"] == "Yes", "total_bill"],
                              df.loc[df["smoker"] == "No", "total_bill"],
                              equal_var=True)
print("Test Stat = %.4f, p-value = %.4f" % (test_stat, pvalue))

# p-value < ise 0.05'ten H0 red.
# p-value < değilse 0.05 H0 reddedilemez.
# p-value bu teste göre 0.05 ten büyük çıktı ve reddedilemedi. Yani anlamlı bir farklılık yok.

# Varsayımlar sağlanmıyorsa mannwhiteneyu testi kullanılır. (non-parametrik test)

test_stat, pvalue = mannwhitneyu(df.loc[df["smoker"] == "Yes", "total_bill"],
                                 df.loc[df["smoker"] == "No", "total_bill"])

print("Test Stat = %.4f, p-value = %.4f" % (test_stat, pvalue))

# mannwhiteneyu ortalama kıyaslama, medyan kıyaslama testidir.
# mannwhiteneyu arasında iki grubun ortalamaları arasında istatistiki olarak anlamlı bir farklılık yoktur.
# Hipotezi kurarken yorumu parantez içinde cümle olarak yaz.

# Uygulama 2 : Titanic Kadın ve Erkek Yolcuların Yaş Ortalamaları Arasında İstatistiksel Olarak Anlamlı Bir Farklılık Var Mıdır ?

df = sns.load_dataset("titanic")
df.head()

df.groupby("sex").agg({"age": "mean"})

# 1. Hipotezleri kur:
# H0: M1 = M2 (Kadın ve erkek yolcuların yaş ortalamaları arasında istatistiki olarak anlamlı bir farklılık yoktur.)
# H1: M1 != M2 (Vardır.)

# 2.Varsayımların incelenmesi:
# H0: Normal dağılım varsayımı sağlanmaktadır.
# H1: sağlanmamaktadır.

test_stat, pvalue = shapiro(df.loc[df["sex"] == "female", "age"].dropna())
print("Test Stat = %.4f, p-value = %.4f" % (test_stat, pvalue))

# H0 bu grup için reddedilir.

test_stat, pvalue = shapiro(df.loc[df["sex"] == "male", "age"].dropna())
print("Test Stat = %.4f, p-value = %.4f" % (test_stat, pvalue))

# H0 bu grup için de reddedilir.

# İki grup için de varsayım sağlanmamaktadır.

# Varyans homojenliğine de bakalım.
# H0: Varyanslar homojemdir.
# H1: Varyanslar homojen değildir.

test_stat, pvalue = levene(df.loc[df["sex"] == "female", "age"].dropna(),
                           df.loc[df["sex"] == "male", "age"].dropna())
print("Test Stat = %.4f, p-value = %.4f" % (test_stat, pvalue))

# Varyanslar homojendir.

# Non-parametrik teste geçelim (mannwhiteneyu):

test_stat, pvalue = mannwhitneyu(df.loc[df["sex"] == "female", "age"].dropna(),
                                 df.loc[df["sex"] == "male", "age"].dropna())
print("Test Stat = %.4f, p-value = %.4f" % (test_stat, pvalue))

# H0 reddedilir. (İstatistiki olarak da fark vardır.)

# Uygulama 3 : Diyabet Hastası Olan ve Olmayanların Yaşlarının Ortalamaları Arasında İstatistiki Olarak Anlamlı Bir Fark Var Mıdır ?

df = pd.read_csv(r"C:\Users\user\PycharmProjects\pythonProject4\diabetes.csv")
df.head()

df.groupby("Outcome").agg({"Age": "mean"})

# 1. Hipotezleri kur.
# H0: M1 = M2
# Diyabet hastası olan ve olmayanların yaşları ortalamaları arasında istatistiki olarak anlamlı bir farklılık yoktur.
# H1: M1 != M2

# 2.Varsayımları incele.

# Normallik Varsayımı(H0: Normal dağılım varsayımı sağlanmaktadır.)
test_stat, pvalue = shapiro(df.loc[df["Outcome"] == 1, "Age"].dropna())
print("Test Stat = %.4f, p-value = %.4f" % (test_stat, pvalue))

test_stat, pvalue = shapiro(df.loc[df["Outcome"] == 0, "Age"].dropna())
print("Test Stat = %.4f, p-value = %.4f" % (test_stat,pvalue))

# H0 < 0.05. Normallik varsayımı her iki grup için de reddedildi. Normallik varsayımı sağlanmadığı için nonparametrik test. (Nonparametrik = Medyanların kıyaslanması olarak karşımıza çıkabilir)

# Hipotez (H0: M1 = M2)
test_stat, pvalue = mannwhitneyu(df.loc[df["Outcome"] == 1, "Age"].dropna(),
                                 df.loc[df["Outcome"] == 0, "Age"].dropna())
print("Test Stat = %.4f, p-value = %.4f" % (test_stat, pvalue))

# H0 reddedildi. Yaş ortalamaları arasında istatistiki olarak anlamlı bir fark vardır.

# Uygulama 4 : İş Problemi : Kursun Büyük Çoğunluğunu İzleyenler ile İzlemeyenlerin Puanları Birbirinden Farklı mı ?

# H0: M1 = M2 (İki grup ortalamaları arasında istatistiki olarak anlamlı bir fark yoktur.)
# H1: M1 != M2 (...vardır.)

df = pd.read_csv(r"C:\Users\user\PycharmProjects\pythonProject4\course_reviews.csv")
df.head()

df[(df["Progress"] > 75)]["Rating"].mean()

df[(df["Progress"] < 25)]["Rating"].mean()

test_stat, pvalue = shapiro(df[(df["Progress"] > 75)]["Rating"])
print("Test Stat = %.4f, p-value = %.4f" % (test_stat,pvalue))

test_stat, pvalue = shapiro(df[(df["Progress"] < 25)]["Rating"])
print("Test Stat = %.4f, p-value = %.4f" % (test_stat,pvalue))

test_stat, pvalue = mannwhitneyu(df[(df["Progress"] > 75)]["Rating"],
                                 df[(df["Progress"] < 25)]["Rating"])
print("Test Stat = %.4f, p-value = %.4f" % (test_stat,pvalue))

# Başlık : İki Grup Oran Karşılaştırma (İki Örneklem Oran Testi)

# İki oran arasında karşılaştırma yapmak için kullanılır.

# Daha öncesinde grup ortalamalarını, grup medyanlarını kıyasladık.

# Bu hesap z hesap istatistiği üzerinden gerçekleştirilir. Ancak biz p value değerine göre karar vermeyi sürdüreceğiz.

# Örnek sayısının 30 dan büyük olma varsayımı vardır.

# Örnek: Kayıt ekranı sadeleştirmesinden sonra çarpan-kaydolan oranı arttı mı ?
# Tasarım 1: 0.30 (Sade ekran) (1000 kişi görüntülemiş 300 ü kayıt olmuş.)
# Tasarım 2: 0.22 (Sade olmayan ekran) (1100 kişi görüntülemiş 250 si kayıt olmuş.)
# Fark var. Ancak bu fark istatistiki olarak var mı ? Bunu inceleyeceğiz.
# Uygulamaya geçelim.

# H0: Yeni tasarımın dönüşüm oranı ile eski tasarımın dönüşüm oranı arasında istatistiki olarak anlamlı bir farklılık yoktur. (p1 = p2)
# H1: ... vardır.(p1 != p2)

# Bu oran testini yapmak için kullanacak olduğumuz metod şunu bekler:
basari_sayisi = np.array([300, 250])
gozlem_sayisi = np.array([1000, 1100])

# Kıyaslamamızı yapalım.

proportions_ztest(count=basari_sayisi, nobs=gozlem_sayisi)

# 0.05 ten küçük olduğu için istatistiki olarak anlamlı bir farklılık vardır.

# Uygulama 2 : Kadın ve Erkeklerin Hayatta Kalma Oranları Arasında İstatistiki Olarak Anlamlı Bir Farklılık Var Mıdır ?

# H0: p1 = p2 ( p1 - p2 = 0 olarak da ifade edilebilir. )
# Kadın ve erkeklerin hayatta kalma oranları arasında istatistiki olarak anlamlı bir farklılık yoktur.

# H1: p1 != p2
# ... vardır.

df = sns.load_dataset("titanic")
df.head()

df.loc[df["sex"] == "female", "survived"].mean()

df.loc[df["sex"] == "male", "survived"].mean()

female_succ_count = df.loc[df["sex"] == "female", "survived"].sum()
male_succ_count = df.loc[df["sex"] == "male", "survived"].sum()

test_stat, pvalue = proportions_ztest(count=[female_succ_count, male_succ_count],
                                      nobs=[df.loc[df["sex"] == "female", "survived"].shape[0],
                                            df.loc[df["sex"] == "male", "survived"].shape[0]])
print("Test Stat = %.4f, p-value = %.4f" % (test_stat, pvalue))

# H0 reddedilir. Fark vardır.

# Başlık : İkiden Fazla Grup Ortalama Karşılaştırma (ANOVA - Analysis of Variance)

# Hipotezlerimiz :
# H0: M1 = M2 = M3
# H1: Eşit değillerdir (en az biri farklıdır.)

# ANOVA (Analysis of Variance)

df = sns.load_dataset("tips")
df.head()

# Varsayımlarımız aynı. (Normallik, Varyans Homojenliği)

# Problem : Günler bazından ödenen hesap ortalamaları arasında farklılık

df.groupby("day")["total_bill"].mean()

# 1. Hipotezi kuralım.
# H0: M1 = M2 = M3 = M4
# Grup ortalamaları arasında fark yoktur.
# H1: ... fark vardır.

# 2. Varsayım kontrolü.
# Normallik varsayımı
# Varyans homojenliği varsayımı

# Varsayım sağlanıyorsa one way anova (tek yönlü anova)
# Varsayım sağlanmıyorsa kruskal

# H0: Normal dağılım varsayımı sağlanmaktadır.
for group in list(df["day"].unique()):
    pvalue = shapiro(df.loc[df["day"] == group, "total_bill"])[1]
    print(group, "p-value: %.4f" % pvalue)

# Normal dağılım varsayımı sağlanmamaktadır.

# H0: Varyans homojenliği varsayımı sağlanmaktadır.

test_stat, pvalue = levene(df.loc[df["day"] == "Sun", "total_bill"],
                           df.loc[df["day"] == "Sat", "total_bill"],
                           df.loc[df["day"] == "Thur", "total_bill"],
                           df.loc[df["day"] == "Fri", "total_bill"])
print("Test Stat = %.4f, p-value = %.4f" % (test_stat, pvalue))

# H0 reddedilemez. (Varyans açısından)

# 3. Hipotez Testi ve P-Value Yorumu

df.groupby("day").agg({"total_bill": ["mean", "median"]})

# Dikkat : Anova ile yapılan genel karşılaştırma ile özelde yapılan ikili karşılaştırmalar aynı değildir. Arada farklılıklar olabilir.

# H0 : Grup ortalamaları arasında istatistiki olarak anlamlı bir fark yoktur.

# Varsayımın sağlandığı ortamda parametrik anova testi kullanılır:
f_oneway(df.loc[df["day"] == "Thur", "total_bill"],
         df.loc[df["day"] == "Fri", "total_bill"],
         df.loc[df["day"] == "Sat", "total_bill"],
         df.loc[df["day"] == "Sun", "total_bill"])

# H0 reddedilir.

# Varsayımlar sağlanmamıştı. Dolayısıyla nonparametrik anova testi kullanacağız.
kruskal(df.loc[df["day"] == "Thur", "total_bill"],
        df.loc[df["day"] == "Fri", "total_bill"],
        df.loc[df["day"] == "Sat", "total_bill"],
        df.loc[df["day"] == "Sun", "total_bill"])

# H0 reddedilir. (Anlamlı farklılık vardır.)

# Yeni problem : Fark kimden kaynaklanıyor ?
# Birden çok yöntem var. Biz statsmodeldeki çoklu karşılaştırmayı kullanacağız.

from statsmodels.stats.multicomp import MultiComparison
comparison = MultiComparison(df["total_bill"], df["day"])
tukey = comparison.tukeyhsd(0.05)
tukey.summary()
print(tukey.summary())
# İkili olarak bakıldığında farklılık bulunamaz. Ne yapılabilir ?
# 1- Alfa değeri değiştirilebilir.
# 2- Fark yokmuş muamelesi yapılabilir. Çünkü bütün gruba anova açısından f testiyle bakmakla, grup içi ve gruplar arası değişkenliği değerlendirmekle, ikili karşılaştırmalara geçildiğinde bir fark değerlendirme işlemi birbirinden farklıdır.