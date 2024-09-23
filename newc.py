# importer les modules necessaires
import matplotlib  # visualisation de données
import matplotlib.pyplot as plt
import pandas as pd  # offre des structures de donnnées et des outils pour la manipulation et l'analyse de données
import numpy as np
import seaborn as sns  # visualisation de données permet de creer des graphiques statiques facilement
import sklearn
import imblearn  # 'imbalance learn' concue pour traiter des pbs de desequilibrages dans les ensembles de données
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler  # pour faire la mise a l'echelle
from sklearn.ensemble import RandomForestClassifier  # afin d'effectuer la selection d'attributs
from sklearn.feature_selection import RFE  # RFE utilisé pour la selection des caracteristiques
import itertools  # manpiulation des iterables
import cProfile  # details sur le temps d'execution
from collections import defaultdict
from sklearn.svm import SVC  # Support Vector Classification
from sklearn.neighbors import KNeighborsClassifier  # K nearest neighbor classifier
from sklearn import metrics  # évaluation des modeles
from sklearn.metrics import confusion_matrix
from sklearn import tree  # foret aleatoire
from sklearn.model_selection import cross_val_score  # score de cross validation
# Ignorer les warnings
import warnings

warnings.filterwarnings('ignore')

# Reglages
pd.set_option('display.max_columns', None)  # toutes les colonnes du dataframe seront affichées lorqu'il est imrpimé
# np.set_printoptions(threshold=sys.maxsize)  # afficher le tableau Numpy en entier
np.set_printoptions(precision=3)  # precision lors de l'affichage des nb flottants (3 chifres apres la virgule)
sns.set(style="darkgrid")  # modifier couleur grille
# modifier la taille de la police des etiquettes de graduation
# plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 7
np.random.seed(10000)

print("pandas : {0}".format(pd.__version__))
print("numpy : {0}".format(np.__version__))
print("matplotlib : {0}".format(matplotlib.__version__))
print("seaborn : {0}".format(sns.__version__))
print("sklearn : {0}".format(sklearn.__version__))
print("imblearn : {0}".format(imblearn.__version__))

# CHARGER LES DONNéES:
train = pd.read_csv('UKM_IDS20Train.csv')  # importer le dataset d'entrainement
train = train.iloc[:, :-1]  # Supprimer la colonne Class binary

test = pd.read_csv('UKM_IDS20test.csv')  # importer le dataset de test
test = test.iloc[:, :-1]  # Supprimer la colonne Class binary

# afficher les Datasets print(train.head(10))

# PRéTRAITEMENT DES DONNéES

# MAPPING :Application du mappage sur notre dataset (ceci va resulter à l'ajout d'une colonne a la fin de notre dataset 'attack_class')
'''on a 7 attaques qu'on peut classifier sous 4 grandes classes d'attques: DoS(denial of service), ARP poisoning, 
Scans,Exploits '''
mapping = {'Normal': 'Normal', 'TCP flood': 'DoS', 'UDP data flood': 'DoS', 'Mass HTTP requests': 'DoS',
           'ARP poisoning': 'ARP poisoning',
           'Metasploit exploits': 'Exploits', 'BeEF HTTP exploits': 'Exploits', 'Port scanning': 'Scans'}

train['attack_class'] = train['Class name'].apply(lambda v: mapping[v])
test['attack_class'] = test['Class'].apply(lambda v: mapping[v])

# Supprimer la colonne "Class name"/"class" de nos deux datasets ( train et test)

train.drop(['Class name'], axis=1, inplace=True)
test.drop(['Class'], axis=1, inplace=True)
print(train.head)
"""
trnspt_train = train[['trnspt']].apply(lambda x: x.value_counts())
print(trnspt_train)
"""
# PRINCIPE DE MISE à L'ECHELLE:
# Le principe de la mise a l'echelle est la modification des valeur des caracteristiques en ajustant la moyenne et l'ecart type afin de rendre les differentes
# caracteristiques comparables et minimiser lerus influence sur le modele


# MISE à L'éCHELLE
scaler = StandardScaler()

# extraire les attributs numériques et les Scale afin d'avoir une moyenne de 0 et une variance unie
cols = train.select_dtypes(include=['float64', 'int64']).columns
sc_train = scaler.fit_transform(train.select_dtypes(include=['float64', 'int64']))
sc_test = scaler.fit_transform(test.select_dtypes(include=['float64', 'int64']))

# mettre a jour le dataset
sc_traindf = pd.DataFrame(sc_train, columns=cols)
sc_testdf = pd.DataFrame(sc_test, columns=cols)

# ENCODAGE DES VALEURS CATEGORIELLES (en valeurs numériques)

# encodage de attack_class
attack_map = {'Normal': 0, 'DoS': 1, 'Scans': 1, 'Exploits': 1, 'ARP poisoning': 1}
train['attack_class'] = train['attack_class'].apply(lambda v: attack_map[v])
test['attack_class'] = test['attack_class'].apply(lambda v: attack_map[v])

# creation de variables cibles encodées
cat_Ytrain = train[['attack_class']].copy()
cat_Ytest = test[['attack_class']].copy()

# séparation de la colonne cible des données encodées
enctrain = train.drop(['attack_class'], axis=1)
enctest = test.drop(['attack_class'], axis=1)

refclasscol = enctrain.columns
X = sc_traindf

# changer les dimensions de la colonne cible de l'ensemble d'entrainement (en un tableau 1D)
c, r = cat_Ytrain.values.shape
y = cat_Ytrain.values.reshape(c, 1)

# changer les dimensions de la colonne cible de l'ensemble de test (en un tableau 1D)
c, r = cat_Ytest.values.shape
y_test = cat_Ytest.values.reshape(c, 1)

# FEATURE SELECTION
# selection des 12 attributs les plus importants

rfc = RandomForestClassifier(random_state=172)
rfe = RFE(rfc, n_features_to_select=12)
rfe = rfe.fit(X, y)

feature_map = [(i, v) for i, v in itertools.zip_longest(rfe.get_support(), refclasscol)]
selected_features = [v for i, v in feature_map if i == True]

print("feature map:\n", feature_map)
print("selected_features\n", selected_features)


selected_features = ['dur', 'srvs', 'flag_sign', 'src_pkts', 'dst_pkts', 'no_lnkd', 'src_byts', 'src_avg_byts',
                     'strt_t', 'end_t', 'host_dst _count', 'avg_t_sent']

selected_features= ['dur', 'srvs', 'flag_sign', 'avg_t_got', 'dst_pkts', 'no_lnkd', 'src_byts', 'src_avg_byts',
                     'strt_t', 'end_t', 'host_dst _count', 'avg_t_sent']
# PARTITION DU DATASET

newcol = list(train.columns)

# ajouter une dimension a notre vecteur cible
new_y = y[:, np.newaxis]
# creation du dataframe d'entrainement depuis les données surechantillonées

res_arr = np.concatenate((X, y), axis=1)  # np.concatenate((X_res, new_y_res), axis=1)
res_df = pd.DataFrame(res_arr, columns=newcol)
# create test dataframe
reftest = pd.concat([sc_testdf, cat_Ytest], axis=1)
reftest['attack_class'] = reftest['attack_class'].astype(np.float64)

X_train = res_df[selected_features]
X_test = reftest[selected_features]

# préparation des etiquettes cibles
y_train = res_df[['attack_class']].copy()
c, r = y_train.values.shape
Y_train = y_train.values.reshape(c, 1)

max_depth_values = list(range(1, 31))
accuracy_values = []

y_test = reftest[['attack_class']].copy()
c, r = y_test.values.shape
Y_test = y_test.values.reshape(c, 1)



# entrainer le modele SVM
SVC_Classifier = SVC(random_state=0)
SVC_Classifier.fit(X_train, Y_train)


# entrainer le modele KNN
KNN_Classifier = KNeighborsClassifier( n_neighbors=3)
KNN_Classifier.fit(X_train, Y_train)


# entrainer un modele d'arbre de décision
DTC_Classifier = tree.DecisionTreeClassifier(random_state=0,max_depth=7)
DTC_Classifier.fit(X_train, Y_train)



classifiers = {SVC_Classifier: 'SVC', KNN_Classifier: 'KNN', DTC_Classifier: 'DT'}
for i in classifiers:
    scores = cross_val_score(i, X_train, Y_train, cv=5)
    accuracy = metrics.accuracy_score(Y_train, i.predict(X_train))
    confusion_matrix = metrics.confusion_matrix(Y_train, i.predict(X_train))
    classification = metrics.classification_report(Y_train, i.predict(X_train))

    print()
    print('============================== ', classifiers[i],
          ' Classifier Model Train Results ==============================')
    print()
    print("scores ( cross validation):" "\n", scores)
    print()

    print("Model Accuracy:" "\n", accuracy)
    print()
    print("Confusion matrix:" "\n", confusion_matrix)
    print()
    print("Classification report:" "\n", classification)
    print()

    # TESTE DU MODELE

    accuracy = metrics.accuracy_score(Y_test, i.predict(X_test))
    confusion_matrix = metrics.confusion_matrix(Y_test, i.predict(X_test))
    classification = metrics.classification_report(Y_test, i.predict(X_test))

    print()

    print('============================== ', classifiers[i],
          ' Classifier Model Test Results ==============================')
    print()
    print("Model Accuracy:" "\n", accuracy)
    print()
    print("Confusion matrix:" "\n", confusion_matrix)
    print()
    print("Classification report:" "\n", classification) 
    print()