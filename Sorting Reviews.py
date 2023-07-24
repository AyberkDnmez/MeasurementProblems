# Başlık : Yorum Sıralama

# Mantık eldeki yorumları en doğru olacak şekilde sıralamaktır.

# SORTING REVIEWS

import pandas as pd
import math
import scipy.stats as st

pd.set_option("display.max_columns", None)
pd.set_option("display.expand_frame_repr", False)
pd.set_option("display.float_format", lambda x: '%.5f' % x)

# Up-Down Diff Score = (up ratings) - (down ratings)

# Review 1: 600 up 400 down total 1000
# Review 2: 5500 up 4500 down total 10000

def score_up_down_diff(up, down):
    return up - down

# Review 1 Score:
score_up_down_diff(600, 400)

# Review 2 Score:
score_up_down_diff(5500, 4500)

# Bu yöntemin kullanılmasının sakıncası şudur: Yazdığımız fonksiyona göre Review 2nin daha üstte çıkması gerekir. Ancak yüzdelik olarak incelendiğinde review 1 yüzde 60 review 2 yüzde 55 olumlu yorum almıştır. Bu da yorumların nasıl sıralanması gerektiğiyle alakalı olarak karışıklığa sebebiyet vermektedir.

# Başlık : Ortalama Puanı(Average Rating)

def score_average_rating(up, down):
    if up + down == 0:
        return 0
    return up / (up + down)

score_average_rating(600, 400)
score_average_rating(5500, 4500)

# Review 1: 2 up 0 down total 2
# Review 2: 100 up 1 down total 101

score_average_rating(2, 0)
score_average_rating(100, 1)

# Bu metot ikinci örnekte frekans sayısının önemini kaçırmıştır.

# Başlık : Wilson Alt Sınır Puanı (Wilson Lower Bound Score)

# İkili etkileşimin olduğu tüm alanlarda skorlama yapma imkanı sunar.

# WLB Score yöntemi Bernoulli parametresi p için bir güven aralığı hesaplar ve bu güven aralığının alt sınırını WLB Score olarak kabul eder.

# Bernoulli bir olasılık dağılımıdır. İkili olayların olasılığını hesaplamak için kullanılır. Örneğin: Bir yazı-tura olayının yazı gelmesi olasılığını vb hesaplar.

# Bizim durumumuzda bir olayın gerçekleşme olasılığı diye ifade ettiğimiz olay up olayıdır. (Up vs. down) Yani up/bütün olay dediğimizde buradaki orana ilişkin bize bir güven aralığı verir.

# Peki neden buna ihtiyaç duyarız ?

# Elimizde müşterilerle ilgili bütün etkileşimler bulunmamaktadır. Örneğin bir kişi bir ürünle alakalı yorum yaptı ve bu yoruma beğenildi beğenilmedi biçiminde bazı yorumlar geldi. Ancak gelebilecek tüm yorumları bilmiyoruz. Bütün veri elimizde yok. Fakat elimizde bir örneklem var. Örneğin 600 like, 400 dislike gibi. Dolayısıyla varolanların içerisinden bir up oranım var. Burada öyle bir genelleme yapmak istiyorum ki bilimsel olsun, bunu tüm kitleye yansıtabileyim ve bana güvenilir bir referans noktası versin. Dolayısıyla bu problemi bir olasılık problemi olarak ele aldığımızda ve bu ilgilendiğimiz olayı ifade eden oran değeri üzerinden bir güven aralığı hesapladığımızda elimizde çok değerli şöyle bir bilgi olacak. Örneğin:

# 600 - 400
# 0.6
# 0.5 0.7. Bu 0.6 için bir güven aralığı hesapladığımızda örneğin şunu diyor oluruz: İstatistiksel olarak şunu diyebiliyorum, 100 kullanıcıdan 95 i bu yorumla ilgili bir etkileşim sağladığında yüzde 5 yanılma payım olmakla birlikte bu yorumun up oranı 0.5 ile 0.7 arasında olacaktır. Alt skoru (0.5) skor olarak belirliyoruz.
