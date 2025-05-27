import streamlit as st 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve, auc, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelBinarizer

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
        params['n_estimators'] = n_estimators
    return params

params = add_parameter(classifier_name)

def get_classifier(clf_name, params):
    if clf_name == 'KNN':
        return KNeighborsClassifier(n_neighbors=params['K'])
    else:
        return RandomForestClassifier(n_estimators=params['n_estimators'], 
            max_depth=params['max_depth'], random_state=1234)

clf = get_classifier(classifier_name, params)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

y_pred = clf.fit(X_train, y_train).predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
Precision_Score = precision_score(y_test, y_pred, average='macro')
Recall_Score = recall_score(y_test, y_pred, average='macro')
F1_Score = f1_score(y_test, y_pred, average='macro')

st.write(f'Classifier = {classifier_name}')
df1 = pd.DataFrame([
    ['Accuracy', accuracy],
    ['Precision Score', Precision_Score],
    ['Recall Score', Recall_Score],
    ['F1 Score', F1_Score]
], columns=['Medida', 'valor'])
st.write(df1.style.background_gradient(cmap='GnBu').bar(subset=['valor'], color='#716807'))

class_names = pd.Series(y).astype('category').cat.categories
y_prob = clf.fit(X_train, y_train).predict_proba(X_test)
n_classes = len(np.unique(y))

def plot_metrics(metrics_list):
    if 'Confusion Matrix' in metrics_list:
        st.subheader("Confusion Matrix")
        fig_cm, ax_cm = plt.subplots()
        ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test,
                                              display_labels=class_names,
                                              cmap=plt.cm.Blues,
                                              values_format="0.3g",
                                              ax=ax_cm)
        st.pyplot(fig_cm)
               
    if 'ROC Curve' in metrics_list:
        st.subheader("ROC Curve")

        if n_classes < 3:
            fig_roc, ax_roc = plt.subplots(dpi=80)
            m_auc = roc_auc_score(y_test, y_prob[:, 1])
            fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
            ax_roc.plot([0, 1], [0, 1], linestyle='--')
            ax_roc.plot(fpr, tpr, label=f'AUC = {m_auc:.3f}')
            ax_roc.set_title('ROC Curve')
            ax_roc.set_xlabel('False Positive Rate')
            ax_roc.set_ylabel('True Positive Rate')
            ax_roc.legend(loc="lower right")
            st.pyplot(fig_roc)
        else:
            label_binarizer = LabelBinarizer().fit(y_train)
            y_onehot_test = label_binarizer.transform(y_test)
            fpr, tpr, roc_auc = dict(), dict(), dict()
            fig_all_roc, ax_all = plt.subplots(dpi=80)

            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_onehot_test[:, i], y_prob[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
                ax_all.plot(fpr[i], tpr[i], linestyle='--',
                            label=f'Class {i} AUC = {roc_auc[i]:.3f}')
            fpr_grid = np.linspace(0.0, 1.0, 1000)
            mean_tpr = np.zeros_like(fpr_grid)
            for i in range(n_classes):
                mean_tpr += np.interp(fpr_grid, fpr[i], tpr[i])
            mean_tpr /= n_classes
            fpr["macro"] = fpr_grid
            tpr["macro"] = mean_tpr
            roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

            ax_all.plot(fpr["macro"], tpr["macro"],
                        label=f"Macro-average AUC = {roc_auc['macro']:.3f}",
                        linewidth=3, color="deepskyblue")
            fpr_micro, tpr_micro, _ = roc_curve(y_onehot_test.ravel(), y_prob.ravel())
            roc_auc["micro"] = auc(fpr_micro, tpr_micro)
            ax_all.plot(fpr_micro, tpr_micro,
                        label=f"Micro-average AUC = {roc_auc['micro']:.3f}",
                        linewidth=3, color="deeppink")
            ax_all.plot([0, 1], [0, 1], linestyle="-", color="gray")
            ax_all.set_xlabel("False Positive Rate")
            ax_all.set_ylabel("True Positive Rate")
            ax_all.set_title("ROC Curve - Multiclase")
            ax_all.legend(loc="lower right")
            st.pyplot(fig_all_roc)

    if 'Bias variance' in metrics_list:
        st.subheader("Bias-Variance Tradeoff: Curvas de aprendizaje personalizadas")

        train_sizes, train_scores, val_scores = learning_curve(
            clf, X, y, cv=5,
            scoring='accuracy',
            train_sizes=np.linspace(0.1, 1.0, 10),
            n_jobs=-1
        )

        train_error = 1 - np.mean(train_scores, axis=1)
        val_error = 1 - np.mean(val_scores, axis=1)

        fig, ax = plt.subplots()
        ax.plot(train_sizes, train_error, 'o-', label="Training error")
        ax.plot(train_sizes, val_error, 'o-', label="Validation error")
        ax.set_title('Learning Curves')
        ax.set_xlabel('Training examples')
        ax.set_ylabel('Error')
        ax.legend(loc='best')
        st.pyplot(fig)

metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', "Bias variance"))
plot_metrics(metrics)

if st.sidebar.checkbox("Mostrar la data", False):
    st.subheader("Descripción del conjuto de datos (Classification)")
    st.write(data)

# Entradas por dataset
def get_input_features(dataset_name):
    if dataset_name == 'Iris':
        return [st.number_input(label) for label in [
            "sepal length (cm):", "sepal width (cm):", "petal length (cm):", "petal width (cm):"
        ]]
    elif dataset_name == 'Wine':
        labels = [
            "alcohol:", "malic_acid:", "ash:", "alcalinity_of_ash:", "magnesium:", "total_phenols:",
            "flavanoids:", "nonflavanoid_phenols:", "proanthocyanins:", "color_intensity:", "hue:",
            "od280/od315_of_diluted_wines:", "proline:"
        ]
        return [st.number_input(label) for label in labels]
    else:
        labels = [ 
            "mean radius:", "mean texture:", "mean perimeter:", "mean area:", "mean smoothness:",
            "mean compactness:", "mean concavity:", "mean concave points:", "mean symmetry:",
            "mean fractal dimension:", "radius error:", "texture error:", "perimeter error:",
            "area error:", "smoothness error:", "compactness error:", "concavity error:",
            "concave points error:", "symmetry error:", "fractal dimension error:", "worst radius:",
            "worst texture:", "worst perimeter:", "worst area:", "worst smoothness:",
            "worst compactness:", "worst concavity:", "worst concave points:", "worst symmetry:",
            "worst fractal dimension:"
        ]
        return [st.number_input(label) for label in labels]

st.header('INGRESE LOS VALORES QUE QUIERE PREDECIR')
user_input = get_input_features(dataset_name)
input_array = np.array([user_input])

if st.button("Predicción :"): 
    prediction = clf.fit(X_train, y_train).predict(input_array)
    prob = clf.predict_proba(input_array)

    if dataset_name == 'Iris':
        clases = ['SETOSA', 'VERSICOLOR', 'VIRGINICA']
    elif dataset_name == 'Wine':
        clases = ['class_0', 'class_1', 'class_2']
    else:
        clases = ['malignant', 'benign']
    
    pred_clase = clases[prediction[0]]
    st.write('Probabilidades:', prob)
    st.success(f'Predicción: {pred_clase}')
