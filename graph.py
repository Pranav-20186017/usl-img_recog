import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC 
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import f1_score
from sklearn.metrics import jaccard_score
import matplotlib.pyplot as plt
df = pd.read_csv("Dataset.csv")
y = df['Label']
df = df.drop(['Label'], axis=1)
components = list(range(1,321))
explained_variance = list()
for _ in components:
    pca = PCA(n_components = _)
    print("Fitting PCA for " + str(_) + " principal components")
    print("-----------------------------------------------------")
    pca.fit_transform(df.values)
    cumsum = np.max(np.cumsum(pca.explained_variance_ratio_))
    explained_variance.append(cumsum)
plt.plot(components, explained_variance)
plt.show()
