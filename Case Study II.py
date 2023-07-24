# İki bidding yöntemi var. Maximum bidding vs. Average bidding
# Maximum bidding = Mevcut kullanılan bidding türü
# Average bidding = Yeni tanıtılan bidding türü
# Müşterilerden birisi olan şirket bu yeni özelliği test etmeye karar veriyor ve average biddingin maximum biddingten daha fazla dönüşüm getirip getirmediğini anlamak için A/B testi yapmak istiyor.
# Bizden beklenen 1 aydır süregelen bu A/B testinin sonuçlarını analiz etmemiz.
# Şirket için nihai başarı ölçütü Purchase. Bu yüzden istatistiksel testlerde bu metriğe odaklanılmalı.

# Görev 1 : Veriyi hazırlama ve analiz etme.

# İlk olarak belli başlı import işlemleri gerçekleştirmemiz gerekir.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import shapiro, levene, ttest_ind

pd.set_option("display.max_columns", None)
pd.set_option("display.expand_frame_repr", False)
pd.set_option("display.float_format", lambda x: "%.5f" % x)

# Adım 1 : ab_testing_data.xlsx adlı kontrol ve test grubu verilerinden oluşan veri setini okutunuz. Kontrol ve test grubu verilerini ayrı değişkenlere atayınız.

df_control = pd.read_excel(r"C:\Users\user\PycharmProjects\pythonProject4\ab_testing.xlsx", sheet_name="Control Group")

df_test = pd.read_excel(r"C:\Users\user\PycharmProjects\pythonProject4\ab_testing.xlsx", sheet_name="Test Group")

# Gerektiğinde veri setlerinin ham hallerine dönüş yapabilmek için kopyalarını oluşturup işlemlere kopyaları üstünden devam edeceğiz.

dfc = df_control.copy()
dft = df_test.copy()

# Adım 2: Kontrol ve test grubu verilerini analiz ediniz.

def analyze(dataframe, head=10):
    print("##### SHAPE #####")
    print(dataframe.shape)
    print("##### HEAD #####")
    print(dataframe.head(10))
    print("##### DESCRIPTIVE STATISTICS #####")
    print(dataframe.describe().T)
    print("##### DATA TYPES #####")
    print(dataframe.dtypes)
    print("##### TAIL #####")
    print(dataframe.tail())
    print("##### NA #####")
    print(dataframe.isnull().sum)
    print("##### QUANTILES #####")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

analyze(dfc)
analyze(dft)

# Adım 3: Analiz işleminden sonra concatmetodunu kullanarak kontrol ve test grubu verilerini birleştiriniz.

dfc["group"] = "control"
dft["group"] = "test"
dfct = pd.concat([dfc, dft], axis=0, ignore_index=False)
dfct.head()

# Görev 2: A/B Testinin Hipotezinin Tanımlanması

# Adım 1: Hipotezi tanımlayınız.

# H0: M1 = M2 (Average bidding uygulanan grubun ortalama satın alma sayısı ile maximum bidding uygulanan grubun ortalama satın alma sayısı arasında istatistiki olarak anlamlı bir farklılık yoktur.)
# H1: M1 != M2 (...vardır.)

# Adım 2: Kontrol ve test grubu için purchase(kazanç) ortalamalarını analiz ediniz.

dfct.groupby("group").agg({"Purchase" : "mean"})

# Görev 3: Hipotez Testinin Gerçekleştirilmesi

# Adım 1: Hipotez testi yapılmadan önce varsayım kontrollerini yapınız.
# Bunlar Normallik Varsayımıve Varyans Homojenliğidir. Kontrol ve test grubunun normallik varsayımına uyup uymadığını Purchase değişkeni üzerinden ayrı ayrı test ediniz.

# Normallik Varsayımı :
# H0: Normal dağılım varsayımı sağlanmaktadır.
# H1: Normal dağılım varsayımı sağlanmamaktadır.
# p < 0.05 H0 RED , p > 0.05 H0 REDDEDİLEMEZ.Test sonucuna göre normallik varsayımı kontrol ve test grupları için sağlanıyor mu ? Elde edilen p-value değerlerini yorumlayınız.

# Control grubu için:
test_stat, pvalue = shapiro(dfct.loc[dfct["group"] == "control", "Purchase"])
print("Test Stat = %.4f, p_value = %.4f" % (test_stat, pvalue))

# Test grubu için:
test_stat, pvalue = shapiro(dfct.loc[dfct["group"] == "test", "Purchase"])
print("Test Stat = %.4f, p_value = %.4f" % (test_stat, pvalue))

# Varyans Homojenliği :
# H0: Varyanslar homojendir.
# H1: Varyanslar homojen Değildir.
# p < 0.05 H0 RED , p > 0.05 H0 REDDEDİLEMEZ. Kontrol ve test grubu için varyanshomojenliğinin sağlanıp sağlanmadığını Purchase değişkeni üzerinden test ediniz. Test sonucuna göre normallik varsayımı sağlanıyor mu? Elde edilen p-value değerlerini yorumlayınız.

test_stat, pvalue = levene(dfct.loc[dfct["group"] == "control", "Purchase"],
                           dfct.loc[dfct["group"] == "test", "Purchase"])

print("Test Stat = %.4f, p-value = %.4f" % (test_stat, pvalue))

# Adım 2: Normallik Varsayımı ve Varyans Homojenliği sonuçlarına göre uygun testi seçiniz. (t testi)

# Adım 3: Test sonucunda elde edilen p_valuedeğerini göz önünde bulundurarak kontrol ve test grubu satın alma ortalamaları arasında istatistiki olarak anlamlı bir fark olup olmadığını yorumlayınız.Görev 3:  Hipotez Testinin Gerçekleştirilmesi


test_stat, pvalue = ttest_ind(dfct.loc[dfct["group"] == "control", "Purchase"],
                              dfct.loc[dfct["group"] == "test", "Purchase"])
print("Test Stat = %.4f, p-value = %.4f" % (test_stat, pvalue))

# H0 reddedilemez. İstatistiki olarak anlamlı bir farklılık yoktur.
