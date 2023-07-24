# Görev 1

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
pd.set_option("display.max_columns", None)
pd.set_option("display.expand_frame_repr", False)
pd.set_option("display.float_format", lambda x: "%.5f" % x)

amazon = pd.read_csv(r"C:\Users\user\PycharmProjects\pythonProject4\amazon_review.csv")
amazon.head(20)
amazon.shape

# Ürünün Ortalama Puanını Hesaplayalım.

amazon["overall"].mean()

amazon["day_diff_scaled"] = MinMaxScaler(feature_range=(1, 5)). \
    fit(amazon[["day_diff"]]). \
    transform(amazon[["day_diff"]])

amazon

# Tarihe göre ağırlıklandırma skalası: day_diff_scaled 1-2: 35 , 2-3: 30 , 3-4: 25, 4-5: 10

# Önce ayrı ayrı ortalama hesabı yapalım.

amazon.loc[(amazon["day_diff_scaled"] >= 1) & (amazon["day_diff_scaled"] < 2), "overall"].mean()

amazon.loc[(amazon["day_diff_scaled"] >= 2) & (amazon["day_diff_scaled"] < 3), "overall"].mean()

amazon.loc[(amazon["day_diff_scaled"] >= 3) & (amazon["day_diff_scaled"] < 4), "overall"].mean()

amazon.loc[(amazon["day_diff_scaled"] >= 4) & (amazon["day_diff_scaled"] < 5), "overall"].mean()

# Ağırlıklandırmaları yapacağımız fonksiyonu yazalım.

def weighted_rating_average(dataframe, w1=40, w2=35, w3=15, w4=10):
    return dataframe.loc[(dataframe["day_diff_scaled"] >= 1) & (dataframe["day_diff_scaled"] < 2), "overall"].mean() * w1/100 + \
           dataframe.loc[(dataframe["day_diff_scaled"] >= 2) & (dataframe["day_diff_scaled"] < 3), "overall"].mean() * w2/100 + \
           dataframe.loc[(dataframe["day_diff_scaled"] >= 3) & (dataframe["day_diff_scaled"] < 4), "overall"].mean() * w3/100 + \
           dataframe.loc[(dataframe["day_diff_scaled"] >= 4) & (dataframe["day_diff_scaled"] < 5), "overall"].mean() * w4/100

weighted_rating_average(amazon)

amazon["overall"].mean()

# Görev 2

amazon.head()

# helpful_no'yu üretelim

amazon["helpful_no"] = amazon["total_vote"] - amazon["helpful_yes"]

amazon.head(50)

# score_pos_neg_diff fonksiyonunu tanımlayalım.

def score_pos_neg_diff(up, down):
    return up-down

amazon["score_pos_neg_diff"] = amazon.apply(lambda x: score_pos_neg_diff(x["helpful_yes"], x["helpful_no"]), axis=1)

# score_average_rating fonksiyonunu tanımlayalım.

def score_average_rating(up, down):
    if up + down == 0:
        return 0
    return (up/(up + down))

amazon["score_average_rating"] = amazon.apply(lambda x: score_average_rating(x["helpful_yes"], x["helpful_no"]), axis=1)

# wilson_lower_bound fonksiyonunu tanımlayalım.

def wilson_lower_bound(up, down, confidence=0.95):
    """
    Wilson Lower Bound Score hesapla

    -Bernoulli parametresi p için hesaplanacak güven aralığının alt sınırı WLB skoru olarak kabul edilir.
    -Hesaplanacak skor ürün sıralaması için kullanılır.
    -Not:
    Eğer skorlar 1-5 arasındaysa 1-3 negatif, 4-5 pozitif olarak işaretlenir ve bernoulli'ye uygun hale getirilebilir.
    Bu beraberinde bazı problemleri de getirir. Bu sebeple bayesian average rating yapmak gerekir.

    Parameters
    ----------
    up: int
        up count
    down: int
        down count

    confidence: float
        confidence

    Returns
    -------
    wilson score: float

    """
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n )

amazon["wilson_lower_bound_score"] = amazon.apply(lambda x: wilson_lower_bound(x["helpful_yes"], x["helpful_no"]), axis=1)

amazon.head(100)

amazon.sort_values("wilson_lower_bound_score", ascending=False).head(20)
