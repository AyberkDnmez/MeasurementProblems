# Başlık : Ürün Sıralama

# Sıralama konusu sadece ürünlerde değil birçok konuda geçerlidir.

# Ürünlerin özelinde oluşmuş olan bilgilere göre bu ürünleri nasıl sıralayacağımıza dair yaklaşım geliştireceğiz.

# Başlık : Derecelendirmeye Göre Sıralama

#########################################
# Sorting Products
#########################################

#########################################
# Uygulama : Kurs Sıralama
#########################################

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.expand_frame_repr", False)
pd.set_option("display.float_format", lambda x: "%.5f" % x)

df = pd.read_csv(r"C:\Users\user\PycharmProjects\pythonProject3\product_sorting.csv")
print(df.shape)
df.head(10)

# Sorting by Rating

df.sort_values("rating", ascending=False).head(20)

# Bazı göz ardı edilemeyecek durumları kapsamadığından dolay sadece ratinge göre sıralama yapılmamalıdır. (Satın alma sayısı, yorum sayısı gibi değişkenler de göz önünde bulundurulmalıdır.)

# Başlık : Yorum ve Satın Alma Sayısına Göre Sıralama

# Sorting by Comment Count or Purchase Count

df.sort_values("purchase_count", ascending=False).head(20)

df.sort_values("commment_count", ascending=False).head(20)

# Başlık : Derecelendirme, Satın Alma, Yoruma Göre Sıralama

###########################################################
# Sorting by Rating, Comment and Purchase
###########################################################

# MinMaxScaler standartlaştırma işlemi için kullanılmaktadır.

# Bu üç faktörü aynı anda göz önünde bulundurmaya çalışacağız.

# Hepsinin ölçeği birbirinden farklı olduğu için bunların hepisini aynı ölçekten ifade etmemiz gerekir. Ratingler 1 ila 5 arasındaki sayılardan oluşuyor. Diğer değişkenleri de bu ölçeğe göre şekillendirelim.

df["purchase_count_scaled"] = MinMaxScaler(feature_range=(1, 5)). \
    fit(df[["purchase_count"]]). \
    transform(df[["purchase_count"]])

df.describe().T

df["comment_count_scaled"] = MinMaxScaler(feature_range=(1, 5)). \
    fit(df[["commment_count"]]). \
    transform(df[["commment_count"]])

df.head()

# Skorları hesaplayalım.

(df["comment_count_scaled"] * 32 / 100 +
 df["purchase_count_scaled"] * 26 / 100 +
 df["rating"] * 42 / 100)

# Fonksiyon yazalım.

def weighted_sorting_score(dataframe, w1=32, w2=26, w3=42):
    return(dataframe["comment_count_scaled"] * w1 / 100 +
           dataframe["purchase_count_scaled"] * w2 / 100 +
           dataframe["rating"] * w3 / 100)

df["weighted_sorting_score"] = weighted_sorting_score(df)

df.sort_values("weighted_sorting_score", ascending=False).head(20)

# Sadece Veri Bilimi ismini içeren kursları getirmek için:

df[df["course_name"].str.contains("Veri Bilimi")].sort_values("weighted_sorting_score", ascending=False).head(20)

# Başlık : Bayes Ortalama Derecelendirme Puanı

###############################
# Bayesian Average Rating Score
###############################

# Sorting Products with 5 Star Rated
# Sorting Products According to Distribution of 5 Star Rating

# Ratingleri Daha Farklı Açılardan Hassaslaştırabilir Miyiz ?, Sadece Ratinge Odaklanarak Bir Sıralama Yapabilir Miyiz ?

# Bayesian Average Rating score'u ürün hakkında ortalama bir skor değeri verdiğinden ötürü tek başına sıralama için kullanabiliriz hatta bunu ürün puanı olarak da değerlendirebiliriz.

# Bu skoru hesaplarken odak noktamız verilen puanların (5, 4, 3, 2, 1) dağılımı olacaktır. Bu skor puanların dağılım bilgisini kullanarak ağırlıklı bir şekilde olasılıksal bir ortalama hesabı yapar.

import math
import scipy.stats as st

def bayesian_average_rating(n, confidence=0.95):
    if sum(n) == 0:
        return 0
    K = len(n)
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    N = sum(n)
    first_part = 0.0
    second_part = 0.0
    for k, n_k in enumerate(n):
        first_part += (k + 1) * (n[k] + 1) / (N + K)
        second_part += (k + 1) * (k + 1) * (n[k] + 1) / (N + K)
    score = first_part - z * math.sqrt((second_part - first_part * first_part) / (N + K + 1))
    return score

df.head()

df["bar_score"] = df.apply(lambda x: bayesian_average_rating(x[["1_point",
                                                                "2_point",
                                                                "3_point",
                                                                "4_point",
                                                                "5_point"]]), axis=1)

df.sort_values("weighted_sorting_score", ascending=False).head(20)
df.sort_values("bar_score", ascending=False).head(20)

df[df["course_name"].index.isin([5, 1])].sort_values("bar_score", ascending=False)

# Bayesian Ortalama Derecelendirme Puanı da yukarıda yaptığımız kıyasta da görüleceği üzere tek başına yeterli olmayabilir. Bu yüzden karma(hybrid) sıralama yapmamız gereklidir.

# Başlık : Karma Sıralama (Hybrid Sorting)

# Hybrid Sorting : BAR Score + Diğer Faktörler(Rating, Comment and Purchase)

# Bayesian Average Rating Score ürünleri puanlamada da kullanılabilir bir yöntemdir. Ürünlerin puanlarını olduğundan daha az gösterdiği için kullanıp kullanılmaması tartışmalıdır. Ne yapılabilir ? Average, Time-Based Weighted Average, User-Based Weighted Average, Weighted Rating ve Bayesian Average Rating Score birlikte kullanılabilir.

def hybrid_sorting_score(dataframe, bar_w=60, wss_w=40):
    bar_score = dataframe.apply(lambda x: bayesian_average_rating(x[["1_point",
                                                                     "2_point",
                                                                     "3_point",
                                                                     "4_point",
                                                                     "5_point"]]), axis=1)
    wss_score = weighted_sorting_score(dataframe)

    return bar_score * bar_w / 100 + wss_score * wss_w / 100

df["hybrid_sorting_score"] = hybrid_sorting_score(df)

df.sort_values("hybrid_sorting_score", ascending=False).head(20)

df[df["course_name"].str.contains("Veri Bilimi")].sort_values("hybrid_sorting_score", ascending=False).head(20)

# Başlık : IMDB Film Puanlama ve Sıralama (IMDB Movie Scoring and Sorting)

import pandas as pd
import math
import scipy.stats as st
pd.set_option("display.max_columns", None)
pd.set_option("display.expand_frame_repr", False)
pd.set_option("display.float_format", lambda x: "%.5f" % x)

df = pd.read_csv("C:/Users/user/PycharmProjects/pythonProject3/movies_metadata.csv",
                 low_memory=False) #DtypeWarning kapamak için

df = df[["title", "vote_average", "vote_count"]]

df.head()
df.shape

# Vote Average'a Göre Hesaplama

df.sort_values("vote_average", ascending=False).head(20)

# Sadece vote average'a göre sıralama yapmak doğru bir sıralama oluşturmayacaktır. Average'ı 10 olan filmlerin çok az sayıda oylandığı gözlemlenmektedir. Vote_counta bir filtre koymayı deneyelim.

df["vote_count"].describe([0.10, 0.25, 0.50, 0.70, 0.80, 0.90, 0.95, 0.99])

df[df["vote_count"] > 400].sort_values("vote_average", ascending=False).head(20)

# Vote_countları 1 ile 10 arasına göre ölçeklendirelim.

from sklearn.preprocessing import MinMaxScaler

df["vote_count_score"] = MinMaxScaler(feature_range=(1, 10)). \
    fit(df[["vote_count"]]). \
    transform(df[["vote_count"]])

df["average_count_score"] = df["vote_average"] * df["vote_count_score"]

df.sort_values("average_count_score", ascending=False).head(20)

# Başlık : IMDB Ağırlıklı Derecelendirme (IMDB Weighted Rating)

# weighted_rating = (v/(v+M) * r) + (M/(v+M) * C)

# r = vote_average
# v = vote_count
# M = minimum votes required to be listed in the Top 250
# C = the mean vote across the whole report (currently 7.0)

M = 2500
C = df["vote_average"].mean()

def weighted_rating(r, v, M, C):
    return(v / (v + M) * r) + (M / (v+M) * C)

# Deadpool filmi için skor hesabı yapalım.

df.sort_values("average_count_score", ascending=False).head(20)

weighted_rating(7.40000, 11444.00000, M, C)

# Inception için skor hesabı yapalım.

weighted_rating(8.10000, 14075.00000, M, C)

# Esaretin bedeli için skor hesabı yapalım.

weighted_rating(8.5000, 8358.00000, M, C)

# Tüm filmlere uygulayıp sıralama yapalım.

df["weighted_rating"] = weighted_rating(df["vote_average"],
                                        df["vote_count"], M, C)

df.sort_values("weighted_rating", ascending=False).head(20)

# Başlık : Bayes Ortalama Derecelendirme Puanı(Bar Score)

# Bayesian Average Rating Score

# İlk 5 i not edelim.

# 12481                                    The Dark Knight
# 314                             The Shawshank Redemption
# 2843                                          Fight Club
# 15480                                          Inception
# 292                                         Pulp Fiction

bayesian_average_rating(([34733, 4355, 4704, 6561, 13515, 26183, 87368, 273082, 600260, 1295351]))

df = pd.read_csv("C:/Users/user/PycharmProjects/pythonProject3/imdb_ratings.csv")
df = df.iloc[0:, 1:]

df["bar_score"] = df.apply(lambda x: bayesian_average_rating(x[["one", "two", "three", "four", "five",
                                                                "six", "seven", "eight", "nine", "ten"]]), axis=1)

df.sort_values("bar_score", ascending=False).head(20)

weighted_rating(7.40, 12000, M, C)
weighted_rating(8.10, 14075, M, C)
weighted_rating(8.50, 8358, M, C)