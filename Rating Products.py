# Başlık : Ölçüm Problemleri

# Bir alıcının satın almasını en çok etkileyen kavramlardan birisi social prooftur. Bu noktada ürünle ilgili topluluğun kanaati çok önemli bir noktada durmaktadır.

# The Wisdom of Crowds: Topluluğun bilgeliğine olan inanç gibi yorumlanabilecek olan bu kavram topluluğun üründen memnun kalmış olmasının bizim de o üründen memnun kalacağımız hissiyatını bizde uyandıran kavramdır.

# Başlık : Ürün Puanlama

# Olası faktörleri göz önünde bulundurarak ağırlıklı ürün puanlama işlemi gerçekleştireceğiz.

# Başlık : Average

# Bu bölüme amacımız bir ürüne verilen puanlar üzerinden çeşitli değerlendirmeler yaparak en doğru puanın nasıl hesaplanabileceği dair bir uygulama yapmak olacaktır.

# - Average
# - Time-Based Weighted Average
# - User-Based Weighted Average
# - Weighted Rating

###############################
#Uygulama : Kullanıcı ve Zaman Ağırlıklı Kurs Puanı Hesaplama
###############################

import pandas as pd
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", 500)
pd.set_option("display.expand_frame_repr", False)
pd.set_option("display.float_format", lambda x: "%.5f" % x)

# (50+ Saat) Python A-Z: Veri Bilimi ve Machine Learning
# Puan: 4.8 (4.764925)
# Toplam Puan: 4611
# Puan Yüzdeleri: 75, 20, 4, 1, <1
# Yaklaşık Sayısal Karşılıkları: 3458, 922, 184, 46, 6

df = pd.read_csv(r"C:\Users\user\PycharmProjects\pythonProject3\course_reviews.csv")
df.head()
df.shape

# Rating Dağılımı
df["Rating"].value_counts()

# Sorulan Soru Sayısı Dağılımı
df["Questions Asked"].value_counts()

# Sorulan Soru Sayısı Dağılımına Göre Rating Ortalamaları

df.groupby("Questions Asked").agg({"Questions Asked": "count",
                                   "Rating": "mean"})

df.head()

########################
# Average
########################

# Ortalama Puan
df["Rating"].mean()

# Sadece üstteki şekilde hesaplama yaparsak ürünle ilgili memnuniyet trendini kaçırabiliriz.

# Ne yaparsak güncel trendi ortalamaya daha iyi yansıtabiliriz ? -> Time-Based Weighted Average

# Başlık : Puan Zamanlarına Göre Ağırlıklı Ortalama (Time-Based Weighted Average)

df.head()
df.info()

# Zaman bazlı bir işlem yapacağımız için Timestamp değişkenini dönüştürmemiz gerekmektedir.

df["Timestamp"] = pd.to_datetime(df["Timestamp"])

# Yapılan puanlamaları gün cinsinden ifade etmemiz gerekmektedir. Elimizde puanların hangi tarihlerde verildiği bilgisi var. Şimdi öyle bir işlem yapalım ki bugünün tarihi diye bir tarih belirleyelim ardından yorumların ve puanlamaların yapıldığı tarihleri bu tarihten çıkartalım. Veri seti eski olduğu için tarihlere takılmayacağız.

current_date = pd.to_datetime("2021-02-10 0:0:0")

df["days"] = (current_date - df["Timestamp"]).dt.days

df[df["days"] <= 30].count()

df.loc[df["days"] <= 30, "Rating"].mean()

df.loc[(df["days"] > 30) & (df["days"] <= 90), "Rating"].mean()

df.loc[(df["days"] > 90) & (df["days"] <= 180), "Rating"].mean()

df.loc[(df["days"] > 180), "Rating"].mean()

# Yukarıda yaptığımız farklı zaman aralıkları için olan farklı rating ortalamaları hesabına farklı ağırlıklar vererek zamanın etkisini yansıtabiliriz.

df.loc[df["days"] <= 30, "Rating"].mean() * 28/100 + \
df.loc[(df["days"] > 30) & (df["days"] <= 90), "Rating"].mean() * 26/100 + \
df.loc[(df["days"] > 90) & (df["days"] <= 180), "Rating"].mean() * 24/100 + \
df.loc[df["days"] > 180, "Rating"].mean() * 22/100

# Şimdi bu duruma bir fonksiyon yazalım.

def time_based_weighted_average(dataframe, w1=28, w2=26, w3=24, w4=22):
    return dataframe.loc[dataframe["days"] <= 30, "Rating"].mean() * w1 / 100 + \
           dataframe.loc[(dataframe["days"] > 30) & (dataframe["days"] <= 90), "Rating"].mean() * w2 / 100 + \
           dataframe.loc[(dataframe["days"] > 90) & (dataframe["days"] <= 180), "Rating"].mean() * w3 / 100 + \
           dataframe.loc[dataframe["days"] > 180, "Rating"].mean() * w4 / 100

time_based_weighted_average(df)

time_based_weighted_average(df, 30, 26, 22, 22)

# Soru: Herkesin Verdiği Puanın Ağırlığı Aynı mı Olmalı ?

# Başlık: Kullanıcı Temelli Ağırlıklı Ortalama

####################################################
# User-Based Weighted Average
####################################################

# Örneğin bir e ticaret sitesinde yüzlerce ürün alıp yorum yapmış bir kişinin verdiği puanın ağırlığı ile ilk kez üyelik oluşturup yorum yapmış bir kişinin verdiği puanın ağırlığı aynı değildir.

df.head()

# Kurstaki ilerleme durumuna göre bir ağırlıklandırma yapalım.

df.groupby("Progress").agg({"Rating": "mean"})

df.loc[df["Progress"] <= 10, "Rating"].mean() * 22 / 100 + \
    df.loc[(df["Progress"] > 10) & (df["Progress"] <= 45), "Rating"].mean() * 24 / 100 + \
    df.loc[(df["Progress"] > 45) & (df["Progress"] <= 75), "Rating"].mean() * 26 / 100 + \
    df.loc[(df["Progress"] > 75), "Rating"].mean() * 28 / 100

# Fonksiyonumuzu yazalım.

def user_based_weighted_average(dataframe, w1=22, w2=24, w3=26, w4=28):
    return dataframe.loc[dataframe["Progress"] <= 10, "Rating"].mean() * w1 / 100 + \
           dataframe.loc[(dataframe["Progress"] > 10) & (dataframe["Progress"] <= 45), "Rating"].mean() * w2 / 100 + \
           dataframe.loc[(dataframe["Progress"] > 45) & (dataframe["Progress"] <= 75), "Rating"].mean() * w3 / 100 + \
           dataframe.loc[dataframe["Progress"] > 75, "Rating"].mean() * w4 / 100

user_based_weighted_average(df)

user_based_weighted_average(df, 20, 24, 26, 30)

# Başlık : Ağırlıklı Derecelendirme

# Weighted Rating

# Yaptığımız tüm işlemleri bir araya getireceğiz. Time-based ve user-basedlerin ağırlıklı ortalamalarını alma işlemi gerçekleştireceğiz.

def course_weighted_rating(dataframe, time_w=50, user_w=50):
    return time_based_weighted_average(dataframe) * time_w / 100 + user_based_weighted_average(dataframe) * user_w / 100

course_weighted_rating(df)

course_weighted_rating(df, time_w=40, user_w=60)
