#!/usr/bin/env python
# coding: utf-8

# # 1. Import library dan data

# In[1]:


# 'Pandas' is used for data manipulation and analysis
import pandas as pd
# 'Numpy' is used for mathematical operations on large, multi-dimensional arrays and matrices
import numpy as np

# 'Matplotlib' is a data visualization library for 2D and 3D plots, built on numpy
import matplotlib.pyplot as plt
# 'Seaborn' is based on matplotlib; used for plotting statistical graphics
import seaborn as sns
# 'scipy' is for advanced math problems
import scipy.stats as scp

from sklearn.preprocessing import LabelEncoder, Normalizer, StandardScaler


# In[2]:


titanic3 = pd.read_csv('data_titanic3.csv')


# In[3]:


print("data titanic: ", titanic3.shape)


# In[4]:


titanic3.info()


# VARIABLE DESCRIPTIONS:
# pclass          Passenger Class
#                 (1 = 1st; 2 = 2nd; 3 = 3rd)
# survival        Survival
#                 (0 = No; 1 = Yes)
# name            Name
# sex             Sex
# age             Age
# sibsp           Number of Siblings/Spouses Aboard
# parch           Number of Parents/Children Aboard
# ticket          Ticket Number
# fare            Passenger Fare
# cabin           Cabin
# embarked        Port of Embarkation
#                 (C = Cherbourg; Q = Queenstown; S = Southampton)
# boat            Lifeboat
# body            Body Identification Number
# home.dest       Home/Destination
# 
# SPECIAL NOTES:
# Pclass is a proxy for socio-economic status (SES)
#  1st ~ Upper; 2nd ~ Middle; 3rd ~ Lower

# In[5]:


titanic3.head()


# # 2. Basic summary statistic

# In[6]:


titanic3.describe()


# In[7]:


display(titanic3.describe(include=np.number).transpose())
display(titanic3.describe(include=np.object).transpose())


# In[8]:


def countplot(column):
    return sns.countplot(x= column, data= titanic3)


# In[9]:


countplot('pclass')


# In[10]:


countplot('survived')


# In[11]:


countplot('embarked')


# In[12]:


countplot('sex')


# In[16]:


def compute_freq_chi2(x,y):
    freqtab = pd.crosstab(x,y)
    print("Frequency table")
    print("============================")
    print(freqtab)
    print("============================")
    chi2,pval,dof,expected = scp.chi2_contingency(freqtab)
    print("ChiSquare test statistic: ",chi2)
    print("p-value: ",pval)
    return


# In[19]:


compute_freq_chi2(titanic3.survived,titanic3.sex)


# In[ ]:


sns.boxplot(x="embarked", y="fare", hue="survived", data=titanic3);


# # 3. Data Preprocessing

# In[23]:


#Cek duplikasi data

titanic3.duplicated(keep=False).sum()


# In[24]:


#Cek missing value

def cek_null(df):
    col_na = df.isnull().sum().sort_values(ascending=False)
    percent = col_na / len(df)
    
    missing_data = pd.concat([col_na, percent], axis=1, keys=['Total', 'Percent'])
    print(missing_data[missing_data['Total'] > 0])


# In[25]:


cek_null(titanic3)


# In[32]:


#drop atribut yang tidak penting

titanic3_cleaned = titanic3.drop(['name', 'ticket', 'body', 'cabin','home.dest'], axis=1)
titanic3_cleaned.head()


# In[33]:


cek_null(titanic3_cleaned)


# In[ ]:


titanic3_cleaned['boat'] = titanic3_cleaned['boat'].fillna('None')


# In[36]:


titanic3_cleaned.head()


# In[37]:


cek_null(titanic3_cleaned)


# In[38]:


col = ["age"]
for c in col:
    median = titanic3_cleaned[~titanic3_cleaned.isna()].median()[0]
    titanic3_cleaned[c] = titanic3_cleaned[c].fillna(median)


# In[39]:


cek_null(titanic3_cleaned)


# In[40]:


titanic3_cleaned["embarked"] = titanic3_cleaned["embarked"].fillna('C')


# In[41]:


titanic3_cleaned[titanic3_cleaned['fare'].isnull()]


# In[42]:


#drop data atau baris yg ada missing value pada atributnya

titanic3_cleaned.dropna(inplace=True)


# In[43]:


#jumlah baris berubah dari 1309 menjadi 1308 dan jumlah fitur dari 14 menjadi 9

titanic3_cleaned.shape


# In[44]:


#remove data outlier pada fitur fare data dengan fare > 500
idx = titanic3_cleaned[titanic3_cleaned["fare"] > 500].index
titanic3_cleaned = titanic3_cleaned.drop(idx, axis=0)


# In[45]:


#jumlah baris berubah dari 1308 menjadi 1304 karena hasil remove outlier 

titanic3_cleaned.shape


# In[46]:


#dataset setelah preprocessing

titanic3_cleaned.head()


# In[47]:


titanic3_cleaned.info()


# In[48]:


#melakukan label encoding data kategorikal yang masih string ke numeric value

col = titanic3_cleaned.select_dtypes(include=["object"]).columns

for c in col:
    if len(titanic3_cleaned[c].value_counts()) <= 28:
        le = LabelEncoder() 
        le.fit(list(titanic3_cleaned[c].values)) 
        titanic3_cleaned[c] = le.transform(list(titanic3_cleaned[c].values))


# In[49]:


titanic3_cleaned.head()


# In[52]:


titanic3_cleaned.info()


# In[53]:


#memisahkan kolom label kelas dengan kolom atribut

X = titanic3_cleaned.drop('survived', axis=1)
Y = titanic3_cleaned['survived']


# In[56]:


X.shape


# In[55]:


Y.shape


# In[65]:


#Normalisasi

sc = StandardScaler()
X_scaled = sc.fit_transform(X)


# In[57]:


#reduksi dimensi menggunakan PCA
#PCA hanya mereduksi dimensi, tidak mengurangi feature

from sklearn.decomposition import PCA as sklearnPCA

pca = sklearnPCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.scatter(X_pca[:, 0], X_pca[:, 1], marker='o', c=Y)
plt.show()


# In[59]:


#terjadi transformasi data jika menggunakan PCA (Principal Component Analysis)

X_pca[:3]


# In[60]:


#feature selection (rank) filter-based 
#data tdk berubah hanya berkurang jumlah fitur menjadi 2 fitur

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif

X_selector = SelectKBest(mutual_info_classif, k=2).fit_transform(X, Y)


# In[61]:


X_selector[:3]


# In[62]:


#klasifikasi menggunakan Decision Tree pada data sebelum direduksi dimensi ada 8 fitur

from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, stratify=Y, random_state=0)
clf_DT = tree.DecisionTreeClassifier(max_depth=2)
clf_DT = clf_DT.fit(X_train, Y_train)

hasil_testing_DT = clf_DT.predict(X_test)
accuracy_DT = accuracy_score(Y_test, hasil_testing_DT)
print('Accuracy:',accuracy_DT)


# In[63]:


#klasifikasi menggunakan Decision Tree pada data setelah direduksi dimensi PCA 2 fitur

from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score

X_train, X_test, Y_train, Y_test = train_test_split(X_pca, Y, test_size=.2, stratify=Y, random_state=0)
clf_DT = tree.DecisionTreeClassifier(max_depth=2)
clf_DT = clf_DT.fit(X_train, Y_train)

hasil_testing_DT = clf_DT.predict(X_test)
accuracy_DT = accuracy_score(Y_test, hasil_testing_DT)
print('Accuracy:',accuracy_DT)


# In[64]:


#klasifikasi menggunakan Decision Tree pada data setelah di seleksi fitur menjadi 2 fitur

from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score

X_train, X_test, Y_train, Y_test = train_test_split(X_selector, Y, test_size=.2, stratify=Y, random_state=0)
clf_DT = tree.DecisionTreeClassifier(max_depth=3)
clf_DT = clf_DT.fit(X_train, Y_train)

hasil_testing_DT = clf_DT.predict(X_test)
accuracy_DT = accuracy_score(Y_test, hasil_testing_DT)
print('Accuracy:',accuracy_DT)


# In[ ]:




