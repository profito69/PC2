# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 23:49:53 2023

@author: USUARIO
"""

import streamlit as st 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split


from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, f1_score,precision_score,recall_score,auc
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import  ConfusionMatrixDisplay
# plot_confusion_matrix, confusion_matrix,
from sklearn.preprocessing import LabelBinarizer
from mlxtend.plotting import plot_learning_curves

st.title('Proyecto completo')

st.write("""
## Explorando diferentes clasificadores con diferentes dataset
¿Cual de ellos es el mejor?
""")

dataset_name = st.sidebar.selectbox(
    'Seleccione el dataset',
    ('Iris', 'Breast Cancer', 'Wine')
)

st.write(f"## {dataset_name} Dataset")

classifier_name = st.sidebar.radio(
    'Seleccione el modelo',
    ['KNN', 'Random Forest']
)

def get_dataset(name):
    data = None
    if name == 'Iris':
        data = datasets.load_iris()
    elif name == 'Wine':
        data = datasets.load_wine()
    else:
        data = datasets.load_breast_cancer()
    X = data.data
    y = data.target
    return X, y, data

X, y, data = get_dataset(dataset_name)
st.write('Estructura del dataset:', X.shape)
st.write('Número de clases:', len(np.unique(y)))

def add_parameter(clf_name):
    params = dict()
    if clf_name == 'KNN':
        K = st.sidebar.slider('K', 1, 15)
        params['K'] = K
    else:
        max_depth = st.sidebar.slider('max_depth', 2, 15)
        params['max_depth'] = max_depth
        n_estimators = st.sidebar.number_input('n_estimators', 1, 100, step=1)
        #n_estimators = st.sidebar.slider('n_estimators', 1, 100)
        params['n_estimators'] = n_estimators
    return params

params = add_parameter(classifier_name)

def get_classifier(clf_name, params):
    clf = None
    if clf_name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=params['K'])
    else:
        clf = RandomForestClassifier(n_estimators=params['n_estimators'], 
            max_depth=params['max_depth'], random_state=1234)
    return clf

clf = get_classifier(classifier_name, params)
#### CLASSIFICATION ####

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

y_pred = clf.fit(X_train, y_train).predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
Precision_Score = precision_score(y_test,y_pred, average='macro')
Recall_Score = recall_score(y_test,y_pred, average='macro')
F1_Score = f1_score(y_test,y_pred,average='macro')

st.write(f'Classifier = {classifier_name}')
dat = [['Accuracy ',accuracy ], ['Precision_Score ', Precision_Score ], 
        ['Recall_Score', Recall_Score],['F1_Scoree', F1_Score] ]
  
df1 = pd.DataFrame(dat, columns=['Medida', 'valor'])
st.write(df1.style.background_gradient(cmap='GnBu').bar(subset=['valor'], color='#716807'))


class_names= pd.Series(y).astype('category').cat.categories

y_prob = clf.fit(X_train, y_train).predict_proba(X_test)
n_classes = len(np.unique(y))

            

def plot_metrics(metrics_list):
    if 'Confusion Matrix' in metrics_list:
        st.subheader("Confusion Matrix")
        fig_cm, ax_cm = plt.subplots()
        ConfusionMatrixDisplay.from_estimator(
            clf, X_test, y_test,
            display_labels=class_names,
            cmap=plt.cm.Blues,
            values_format="0.3g",
            ax=ax_cm
        )
        st.pyplot(fig_cm)
               
    if 'ROC Curve' in metrics_list:
        st.subheader("ROC Curve")

        if n_classes < 3:
            fig_roc, ax_roc = plt.subplots(dpi=80)
            m_auc = roc_auc_score(y_test, y_prob[:, 1])
            lr_fpr, lr_tpr, _ = roc_curve(y_test, y_prob[:, 1])
            ax_roc.plot([0, 1], [0, 1], linestyle='--')
            ax_roc.plot(lr_fpr, lr_tpr, color="navy", label=f'AUC = {m_auc:.3f}')
            ax_roc.set_title('Curve ROC - Model')
            ax_roc.set_xlabel('False Positive Rate')
            ax_roc.set_ylabel('True Positive Rate')
            ax_roc.legend(loc="lower right")
            st.pyplot(fig_roc)

        else:
            label_binarizer = LabelBinarizer().fit(y_train)
            y_onehot_test = label_binarizer.transform(y_test)
        
            fpr, tpr, roc_auc = dict(), dict(), dict()
        
            # Graficar One-vs-Rest para cada clase
            fig_all_roc, ax_all = plt.subplots(dpi=80)
        
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_onehot_test[:, i], y_prob[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
                ax_all.plot(
                    fpr[i], tpr[i], 
                    label=f'Class {i} ROC curve (AUC = {roc_auc[i]:.3f})',
                    linestyle='--'
                )
        
            # Cálculo del AUC macro
            fpr_grid = np.linspace(0.0, 1.0, 1000)
            mean_tpr = np.zeros_like(fpr_grid)
        
            for i in range(n_classes):
                mean_tpr += np.interp(fpr_grid, fpr[i], tpr[i])

            mean_tpr /= n_classes
            fpr["macro"] = fpr_grid
            tpr["macro"] = mean_tpr
            roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

            # Curva macro resaltada (línea sólida)
            ax_all.plot(
                fpr["macro"], tpr["macro"],
                label=f"Macro-average ROC curve (AUC = {roc_auc['macro']:.3f})",
                linewidth=3, color="deepskyblue", alpha=0.9, linestyle='-'
            )

            # Cálculo del AUC micro
            fpr_micro, tpr_micro, _ = roc_curve(y_onehot_test.ravel(), y_prob.ravel())
            roc_auc["micro"] = auc(fpr_micro, tpr_micro)

            # Curva micro resaltada (línea sólida)
            ax_all.plot(
                fpr_micro, tpr_micro,
                label=f"Micro-average ROC curve (AUC = {roc_auc['micro']:.3f})",
                linewidth=3, color="deeppink", alpha=0.9, linestyle='-'
            )

            # Línea diagonal
            ax_all.plot([0, 1], [0, 1], linestyle="-", color="gray")
            ax_all.set_xlabel("False Positive Rate")
            ax_all.set_ylabel("True Positive Rate")
            ax_all.set_title("ROC Curve (One-vs-Rest for multiclass classification)")
            ax_all.legend(loc="lower right")
            st.pyplot(fig_all_roc)
 
    if 'Bias variance' in metrics_list:
        st.subheader("Bias variance") 
        fig = plt.figure()
        plot_learning_curves(X_train, y_train, X_test, y_test, clf, style="fast", test_marker='o')
        plt.title('Evaluation Bias Variance Trade-off Model')
        plt.ylabel("Error", fontsize=15)
        st.pyplot(fig)


        
metrics = st.sidebar.multiselect("What metrics to plot?",('Confusion Matrix', 'ROC Curve' ,"Bias variance"))
plot_metrics(metrics)
      

if st.sidebar.checkbox("Mostrar la data", False):
    st.subheader("Descripción del conjuto de datos (Classification)")
    st.write(data)
    

if dataset_name == 'Iris':
    st.header('INGRESE LOS VALORES QUE QUIERE PREDECIR')
    P1 = st.number_input("sepal length (cm):")
    P2 = st.number_input("sepal width (cm):")
    P3 = st.number_input("petal length (cm):")
    P4 = st.number_input("petal width (cm):")

elif dataset_name == 'Wine':
    st.header('INGRESE LOS VALORES QUE QUIERE PREDECIR')
    P1 = st.number_input("alcohol:")
    P2 = st.number_input("malic_acid:")
    P3 = st.number_input("ash:")
    P4 = st.number_input("alcalinity_of_ash:")
    P5 = st.number_input("magnesium:")
    P6 = st.number_input("total_phenols:")
    P7 = st.number_input("flavanoids:")
    P8 = st.number_input("nonflavanoid_phenols:")
    P9 = st.number_input("proanthocyanins:")
    P10 = st.number_input("color_intensity:")
    P11 = st.number_input("hue:")
    P12 = st.number_input("od280/od315_of_diluted_wines:")
    P13 = st.number_input("proline:")

else:
    st.header('INGRESE LOS VALORES QUE QUIERE PREDECIR')
    P1 = st.number_input("mean radius:")
    P2 = st.number_input("mean texture:")
    P3 = st.number_input("mean perimeter:")
    P4 = st.number_input("mean area:")
    P5 = st.number_input("mean smoothness:")
    P6 = st.number_input("mean compactness:")
    P7 = st.number_input("mean concavity:")
    P8 = st.number_input("mean concave points:")
    P9 = st.number_input("mean symmetry:")
    P10 = st.number_input("mean fractal dimension:")
    P11 = st.number_input("radius error:")
    P12 = st.number_input("texture error:")
    P13 = st.number_input("perimeter error:")
    P14 = st.number_input("area error:")
    P15 = st.number_input("smoothness error:")
    P16 = st.number_input("compactness error:")
    P17 = st.number_input("concavity error:")
    P18 = st.number_input("concave points error:")
    P19 = st.number_input("symmetry error:")
    P20 = st.number_input("fractal dimension error:")
    P21 = st.number_input("worst radius:")
    P22 = st.number_input("worst texture:")
    P23 = st.number_input("worst perimeter:")
    P24 = st.number_input("worst area:")
    P25 = st.number_input("worst smoothness:")
    P26 = st.number_input("worst compactness:")
    P27 = st.number_input("worst concavity:")
    P28 = st.number_input("worst concave points:")
    P29 = st.number_input("worst symmetry:")
    P30 = st.number_input("worst fractal dimension:")
         

def get_dataset_predict(name):
    data = None
    if name == 'Iris':
        data = np.array([[P1,P2,P3,P4]])
    elif name == 'Wine':
        data = np.array([[P1,P2,P3,P4,P5,P6,P7,P8,P9,P10,P11,P12,P13]])
    else:
        data = np.array([[P1,P2,P3,P4,P5,P6,P7,P8,P9,P10,P11,P12,P13,P14,P15,
                          P16,P17,P18,P19,P20,P21,P22,P23,P24,P25,P26,P27,P28,
                          P29,P30]])
    return data

data_X = get_dataset_predict(dataset_name)

if st.button("Predicción :"): 

   y_pred_fin = clf.fit(X_train, y_train).predict(data_X)
   y_prob_fin = clf.fit(X_train, y_train).predict_proba(data_X)
   
   if dataset_name == 'Iris':
       if y_pred_fin == 0 :
           st.write('Probabilidades:', y_prob_fin,'SETOSA')
       elif y_pred_fin == 1 :
           st.write('Probabilidades:', y_prob_fin,'VERSICOLOR')
       else:
           st.write('Probabilidades:', y_prob_fin,'VIRGINICA')
   elif dataset_name == 'Wine':
       if y_pred_fin == 0 :
           st.write('Probabilidades:', y_prob_fin,'class_0')
       elif y_pred_fin == 1 :
           st.write('Probabilidades:', y_prob_fin,'class_1')
       else:
           st.write('Probabilidades:', y_prob_fin,'class_2')
   else:
       if y_pred_fin == 0 :
            st.write('Probabilidades:', y_prob_fin,'malignant')
       else:
           st.write('Probabilidades:', y_prob_fin,'benign')
           