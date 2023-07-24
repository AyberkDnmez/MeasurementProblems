import pandas as pd
import math
import scipy.stats as st

pd.set_option("display.max_columns", None)
pd.set_option("display.expand_frame_repr", False)
pd.set_option("display.float_format", lambda x: '%.5f' % x)

def score_up_down_diff(up, down):
    return up - down

def score_average_rating(up, down):
    if up + down == 0:
        return 0
    return up / (up + down)

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

wilson_lower_bound(600, 400)
wilson_lower_bound(5500, 4500)

wilson_lower_bound(2, 0)
wilson_lower_bound(100, 1)

# Başlık : Örnek Uygulama (Case Study)

#####################################
# Case Study
#####################################

up = [15, 70, 14, 4, 2, 5, 8, 37, 21, 52, 28, 147, 61, 30, 23, 40, 37, 61, 54, 18, 12, 68]
down = [0, 2, 2, 2, 15, 2, 6, 5, 23, 8, 12, 2, 1, 1, 5, 1, 2, 6, 2, 0, 2, 2]
comments = pd.DataFrame({"up": up, "down": down})

# Bu zamana kadar uyguladığımız tüm skor yöntemlerini ele alalım.

comments["score_pos_neg_diff"] = comments.apply(lambda x: score_up_down_diff(x["up"],
                                                                             x["down"]), axis=1)
comments["score_average_rating"] = comments.apply(lambda x: score_average_rating(x["up"],
                                                                                 x["down"]), axis=1)
comments["wilson_lower_bound"] = comments.apply(lambda x: wilson_lower_bound(x["up"],
                                                                             x["down"]), axis=1)

# Wilson lower bound skoruna göre sıralama işlemi yapalım.

comments.sort_values("wilson_lower_bound", ascending=False)
