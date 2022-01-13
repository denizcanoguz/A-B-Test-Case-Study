import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.stats.api as sms
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, pearsonr, spearmanr, kendalltau, \
    f_oneway, kruskal
from statsmodels.stats.proportion import proportions_ztest
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
# çıktının tek bir satırda olmasını sağlar.
pd.set_option('display.expand_frame_repr', False)

#veri okuma
control_df_ = pd.read_excel("datasets/ab_testing.xlsx",sheet_name="Control Group")
control_df = control_df_.copy()
test_df_ = pd.read_excel("datasets/ab_testing.xlsx",sheet_name="Test Group")
test_df = test_df_.copy()

control_df.head()
test_df.head()

# E.D.A
control_df.describe().T.round(1)

# TOPLAM GÖRÜNTÜLENME SAYISI : 4.068.458.0
control_df["Impression"].sum().round(1)
# TOPLAM TIK SAYISI : 204.026.3
control_df["Click"].sum().round(1)
# TOPLAM SATIN ALIM SAYISI : 22.035.8
control_df["Purchase"].sum().round(1)
# TOPLAM KAZANÇ : 76.342.7
control_df["Earning"].sum().round(1)

test_df.describe().T.round(1)

# TOPLAM GÖRÜNTÜLENME SAYISI : 4.820.496.5
test_df["Impression"].sum().round(1)
# TOPLAM TIK SAYISI : 158.702.0
test_df["Click"].sum().round(1)
# TOPLAM SATIN ALIM SAYISI : 23.284.2
test_df["Purchase"].sum().round(1)
# TOPLAM KAZANÇ : 100.595.6
test_df["Earning"].sum().round(1)



######################################################
# AB Testing (İki Örneklem Oran Testi)
######################################################

# H0: M1=M2  ---> MAX BIDDING İLE AVERAGE BIDDING ARASINDA CLICK PER PURCHASE BAZINDA ANLAMLI BİR FARK YOKTUR.
# H1: M1!=M2 ---> MAX BIDDING İLE AVERAGE BIDDING  ARASINDA CLICK PER PURCHASE BAZINDA ANLAMLI BİR FARK VARDIR.
# örnk
# basari_sayisi = np.array([300, 250])
# gozlem_sayilari = np.array([1000, 1100])
# proportions_ztest(count=basari_sayisi, nobs=gozlem_sayilari)


# CONTROL GROUP, tıklanma başına satın alma:
purchase_per_click_control = (control_df["Purchase"] / control_df["Click"]).sum()
# TEST GROUP, tıklanma başına satın alma:
purchase_per_click_test = (test_df["Purchase"] / test_df["Click"]).sum()

test_stat, pvalue = proportions_ztest(count=[purchase_per_click_control, purchase_per_click_test],
                                      nobs=[control_df.shape[0],test_df.shape[0]])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# Test Stat = -0.5298, p-value = 0.5962


### İki örneklem Oran testi ile İş problemimizi inceledik.
# Veri setinden tıklanma başına satın alma oranlarını türettik
# iki grup arasındaki tıklanma başına satın alma oranlarına göre oran testi uyguladık

# Hipotez:
# H0: MAX BIDDING ve AVERAGE BIDDING arasında purchase_per_click bazında anlamlı bir fark yoktur.
# H1: ...vardır.
# Test Stat = -0.5298, p-value = 0.5962

# sonuç p value değerimiz alpha (0.05) hata payı değerinden büyük olduğu için H0 ı red edemeyiz.
# MAX BIDDING ve AVERAGE BIDDING arasında purchase_per_click bazında anlamlı bir fark yoktur.

