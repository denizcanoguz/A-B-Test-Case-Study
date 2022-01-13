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
######################################################
# AB Testing (Bağımsız İki Örneklem T Testi)
######################################################

# İki grup ortalaması arasında karşılaştırma yapılmak istenildiğinde kullanılır.

# 1. Varsayım Kontrolü
#   - 1. Normallik Varsayımı
#   - 2. Varyans Homojenliği
# 2. Hipotezin Uygulanması
#   - 1. Varsayımlar sağlanıyorsa bağımsız iki örneklem t testi (parametrik test)
#   - 2. Varsayımlar sağlanmıyorsa mannwhitneyu testi (non-parametrik test)
# Not:
# - Normallik sağlanmıyorsa direk 2 numara. Varyans homojenliği sağlanmıyorsa 1 numaraya arguman girilir.
# - Normallik incelemesi öncesi aykırı değer incelemesi ve düzeltmesi yapmak faydalı olabilir.

##############################################################################################################

# AB Test Project
# --------------------------------------------------------------------------------------------------

# İŞ PROBLEMİ
# --------------------------------------------------------------------------------------------------
# Facebook kısa süre önce mevcut maximum bidding adı verilen teklif
# verme türüne alternatif olarak yeni bir teklif türü olan average bidding’i
# tanıttı.
# Müşterilerimizden biri   olan  bombabomba.com, bu yeni özelliği test
# etmeye karar verdi ve averagebidding’in, maximumbidding’den daha
# fazla dönüşüm getirip getirmediğini anlamak için bir A/B testi yapmak
# istiyor.
# --------------------------------------------------------------------------------------------------

# VERİ SETİ HİKAYESİ
# --------------------------------------------------------------------------------------------------
# bombabomba.com’un web site bilgilerini içeren bu veri setinde kullanıcıların
# gördükleri ve tıkladıkları reklam sayıları gibi bilgilerin yanı sıra buradan gelen
# kazanç bilgileri yer almaktadır.
# Kontrol ve Test grubu olmak üzere iki ayrı veri seti vardır.
# --------------------------------------------------------------------------------------------------

# DEĞİŞKENLER
# --------------------------------------------------------------------------------------------------
# Impression – Reklam görüntüleme sayısı

# Click – Tıklama
# Görüntülenen reklama tıklanma sayısını belirtir.

# Purchase – Satın alım
# Tıklanan reklamlar sonrası satın alınan ürün sayısını belirtir.

# Earning – Kazanç
# Satın alınan ürünler sonrası elde edilen kazanç
# --------------------------------------------------------------------------------------------------



# VERİ SETİMİZİ GETİRELİM :
# --------------------------------------------------------------------------------------------------

control_df_ = pd.read_excel("datasets/ab_testing.xlsx",sheet_name="Control Group")
control_df = control_df_.copy()
test_df_ = pd.read_excel("datasets/ab_testing.xlsx",sheet_name="Test Group")
test_df = test_df_.copy()

control_df.head()
#       Impression        Click    Purchase      Earning
# 0   82529.459271  6090.077317  665.211255  2311.277143
# 1   98050.451926  3382.861786  315.084895  1742.806855
# 2   82696.023549  4167.965750  458.083738  1797.827447
# 3  109914.400398  4910.882240  487.090773  1696.229178
# 4  108457.762630  5987.655811  441.034050  1543.720179
test_df.head()
#      Impression        Click    Purchase      Earning
# 0  120103.503796  3216.547958  702.160346  1939.611243
# 1  134775.943363  3635.082422  834.054286  2929.405820
# 2  107806.620788  3057.143560  422.934258  2526.244877
# 3  116445.275526  4650.473911  429.033535  2281.428574
# 4  145082.516838  5201.387724  749.860442  2781.697521

# EDA AŞAMASI ;
# --------------------------------------------------------------------------------------------------

control_df.describe().T.round(1)
#             count      mean      std      min      25%      50%       75%       max
# Impression   40.0  101711.4  20302.2  45475.9  85726.7  99790.7  115212.8  147539.3
# Click        40.0    5100.7   1330.0   2189.8   4124.3   5001.2    5923.8    7959.1
# Purchase     40.0     550.9    134.1    267.0    470.1    531.2     638.0     801.8
# Earning      40.0    1908.6    302.9   1254.0   1685.8   1975.2    2119.8    2497.3

# 40 GÜN İÇERİSİNDE Kİ;

# TOPLAM GÖRÜNTÜLENME SAYISI : 4.068.458.0
control_df["Impression"].sum().round(1)
# TOPLAM TIK SAYISI : 204.026.3
control_df["Click"].sum().round(1)
# TOPLAM SATIN ALIM SAYISI : 22.035.8
control_df["Purchase"].sum().round(1)
# TOPLAM KAZANÇ : 76.342.7
control_df["Earning"].sum().round(1)

# YUKARIDA Kİ BİLGİLER MAX BIDDING UYGULAMASI SÜRECİNDE Kİ VERİLERİN BİLGİLERİDİR.


test_df.describe().T.round(1)
#             count      mean      std      min       25%       50%       75%       max
# Impression   40.0  120512.4  18807.4  79033.8  112692.0  119291.3  132050.6  158605.9
# Click        40.0    3967.5    923.1   1836.6    3376.8    3931.4    4660.5    6019.7
# Purchase     40.0     582.1    161.2    311.6     444.6     551.4     699.9     889.9
# Earning      40.0    2514.9    282.7   1939.6    2280.5    2544.7    2761.5    3171.5

# 40 GÜN İÇERİSİNDE Kİ;

# TOPLAM GÖRÜNTÜLENME SAYISI : 4.820.496.5
test_df["Impression"].sum().round(1)
# TOPLAM TIK SAYISI : 158.702.0
test_df["Click"].sum().round(1)
# TOPLAM SATIN ALIM SAYISI : 23.284.2
test_df["Purchase"].sum().round(1)
# TOPLAM KAZANÇ : 100.595.6
test_df["Earning"].sum().round(1)

# YUKARIDA Kİ BİLGİLER AVERAGE BIDDING UYGULAMASI SÜRECİNDE Kİ VERİLERİN BİLGİLERİDİR.





# YAPILAN ANALİZE GÖRE AB TESTİ Mİ --  ORTALAMA SATIN ALMA "Purchase" ÜZERİNE TANIMLAMAYA KARAR VERİYORUM.

# PROJE GÖREVLERİ
# --------------------------------------------------------------------------------------------------

# GÖREV 1:
# A/B testinin hipotezini tanımlayınız.
# --------------------------------------------------------------------------------------------------


# H0: M1=M2  ---> MAX BIDDING İLE AVERAGE BIDDING ARASINDA PURCHASE BAZINDA ANLAMLI BİR FARK YOKTUR.
# H1: M1!=M2 ---> MAX BIDDING İLE AVERAGE BIDDING ARASINDA PURCHASE BAZINDA ANLAMLI BİR FARK VARDIR.

# YUKARIDA HİPOTEZLERİMİZİ TANIMLADIK.

#  ÖNCE VARSAYIM KONTROLLERİMİZİ YAPALIM.
# --------------------------------------------------------------------------------------------------


# Normallik Varsayımı
# Varyans Homojenliği

# İLK OLARAK NORMALLİK VARSAYIMI:
# --------------------------------------------------------------------------------------------------

# H0: Normal dağılım varsayımı sağlanmaktadır.
# H1:  Normal dağılım varsayımı sağlanmamaktadır.

test_stat, pvalue = shapiro(control_df["Purchase"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# Test Stat = 0.9773, p-value = 0.5891
#  P VALUE DEĞERİMİZ ALPHADAN BÜYÜK H0 red edilemez. NORMALLİK VARSAYIMI SAĞLANIR.

test_stat, pvalue = shapiro(test_df["Purchase"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# Test Stat = 0.9589, p-value = 0.1541
#   P VALUE DEĞERİMİZ ALPHADAN BÜYÜK H0 red edilemez. NORMALLİK VARSAYIMI SAĞLANIR.

# NORMALLİK VARSAYIMLARI İKİ GRUP İÇİNDE SAĞLANDI ŞİMDİ HOMOJENLİK VARSAYIMLARINA GEÇELİM.

# VARYANS HOMOJENLİGİ VARSAYIMI:
# --------------------------------------------------------------------------------------------------

# H0: Varyanslar Homojendir
# H1: Varyanslar Homojen Değildir

test_stat, pvalue = levene(control_df["Purchase"],test_df["Purchase"])
print('Test Stats = %.4f, p-value = %.4f' % (test_stat, pvalue))

# Test Stats = 2.6393, p-value = 0.1083
#  P VALUE DEĞERİMİZ ALPHADAN BÜYÜK H0 red edilemez. HOMOJENLİK VARSAYIMI SAĞLANIR.

# ŞİMDİ UYGULAMAYA GEÇİCEZ.
# --------------------------------------------------------------------------------------------------
# her iki varsayım sağlandığı için bağımsız iki örneklem t testi (parametrik test) uygulayacağız.
test_stat, pvalue = ttest_ind(control_df["Purchase"],test_df["Purchase"],equal_var=True) # equal_var = false olur ise welct testi uygular
print('Test Stats = %.4f, p-value = %.4f' % (test_stat, pvalue))
# Test Stats = -0.9416, p-value = 0.3493

# H0: M1=M2  ---> MAX BIDDING İLE AVERAGE BIDDING PURCHASE BAZINDA ARASINDA ANLAMLI BİR FARK YOKTUR.
# H1: M1!=M2 ---> MAX BIDDING İLE AVERAGE BIDDING PURCHASE BAZINDA ARASINDA ANLAMLI BİR FARK VARDIR.
#  P VALUE DEĞERİMİZ ALPHADAN BÜYÜK H0 red edilemez.


# GÖREV 2:
# Çıkan test sonuçlarının istatistiksel olarak
# anlamlı olup olmadığını yorumlayınız.
# --------------------------------------------------------------------------------------------------
# H0: M1=M2  ---> MAX BIDDING İLE AVERAGE BIDDING  ARASINDA PURCHASE BAZINDA ANLAMLI BİR FARK YOKTUR.
# H1: M1!=M2 ---> MAX BIDDING İLE AVERAGE BIDDING  ARASINDA PURCHASE BAZINDA ANLAMLI BİR FARK VARDIR.
# Test Stats = -0.9416, p-value = 0.3493
# P VALUE (0.3493) DEĞERİ (0.05) ALPHA' DAN BÜYÜK OLDUĞU İÇİN H0: M1=M2 HİPOTEZİ RED EDİLEMEZ.
# MAX BIDDING İLE AVERAGE BIDDING ARASINDA PURCHASE BAZINDA İSTATİKSEL ANLAMDA OLUMLU BİR FARK YOKTUR.



# GÖREV 3:
# Hangi testleri kullandınız?
# Sebeplerini belirtiniz.
# --------------------------------------------------------------------------------------------------

# Varsayım Kontrollerimizi yaptık ve Normallik varsayımı ile Homojenlik Varsayımı sağlanıyor.
# Varsayımlarımız sağlandığı için bağımsız iki örneklem t testi (parametrik test) uygulaması gerçekleştirdik.



# GÖREV 4:
# Görev 2’de verdiğiniz cevaba göre, müşteriye
# tavsiyeniz nedir?
# --------------------------------------------------------------------------------------------------

# EĞER MAX BIDDING VE AVERAGE BIDDING ARASINDA PURCHASE BAZINDA ANLAMLI BİR FARK YOK İSE NE YAPILMALI;

# MEVSİMSELLİK ÖRÜNTÜLERİ İLE ÖRTÜŞEN YA DA ÖRTÜŞMEYEN BİR DÖNEMDE Mİ TEST YAPILDI ?
# --> Eğer cevap örtüşen bir dönem de ise bir işleme gerek yoktur.

# --> Eğer cevap örtüşmeyen bir dönem de ise bu test aşamalarını uygun dönem de tekrarlamak sağlıklı
#  bir analiz için gerekli olabilir.

# İÇİNDE BARINDIRDIĞI HAFTA İÇİ VE HAFTA SONU SAYISI NEDİR BAKILABİLİR
# --> Eğer cevap eşit sayıda  bir dönem de ise bir işleme gerek yoktur.

# --> Eğer cevap eşit sayıda  bir dönem de değil ise bu test aşamalarını uygun dönem de tekrarlamak sağlıklı
#  bir analiz için gerekli olabilir.

# GRAFİKLER İNCELENDİ Mİ?
# --> YENİ DURUM İÇİN GRAFİKLER DE SON GÜNLERDE BİR ÖRÜNTÜ YAKALANMIŞ OLABİLİR EĞRİSİ İNCELENİP ,
# BÖYLE BİR ÖRÜNTÜ YAKALANDIYSA TEST SÜRESİNİ UZATMAK TAVSİYE EDİLEBİLİR.


