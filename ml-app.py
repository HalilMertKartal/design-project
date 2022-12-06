import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb

from sklearn import preprocessing, metrics
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error,\
 f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.decomposition import TruncatedSVD


st.set_page_config(page_title="Design Decision Classification", layout="wide")
st.set_option('deprecation.showPyplotGlobalUse', False)

st.write(
    """
    # Design Decision Clasifier
    In this implementation, different models are used for classifying the design decisions.
    Try adjusting the parameters and select the model you desire. Note that\
    hyperparameter optimization will be done in the background for the selected model.
    """
)

def create_model():
    # Read the csv
    df = pd.read_csv("dataset/design_decisions_v1.csv")
    X = df["text"].values.astype('U')
    y = df["label"].values

    st.subheader("**1. Dataset summary**")
    st.markdown("**1.1 Tabular view of the data**")
    st.table(df.head())

    st.markdown("**1.2 Data split shapes**")

    st.write("Training set")
    st.info(X.shape)
    st.write("Test set")
    st.info(y.shape)

    st.markdown("**1.3 Variable Details**")

    st.write("Variable X")
    st.info(df["text"].name)
    st.write("Variable Y")
    st.info(df["label"].name)

    
    with st.sidebar.header("1. Set Parameters"):
        split_size = st.sidebar.slider("Data split ratio (% for training set)", 50, 90, 85)
        random_seed = st.sidebar.slider("Random state for the dataset", 1, 1000, 100)
    
    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X,
    y, random_state=random_seed, train_size=split_size*0.01)

    # Define the steps for the knn pipeline
    steps_for_knn = [
        ("tfidf", TfidfVectorizer()),
        ("knn", KNeighborsClassifier())
        ]

    # Define the steps for the svc pipeline
    steps_for_svc = [
        ("tfidf", TfidfVectorizer()),
        ("pca", TruncatedSVD()),
        ("SupVM", SVC(kernel="rbf"))
        ]
    
    # Create the parameter space for knn
    parameters_knn = {"knn__n_neighbors": np.arange(3, 13, 2),
                'knn__weights': ['uniform', 'distance'],
                'knn__leaf_size': np.arange(2, 4)}

    # Create the parameter space for pca
    parameters_pca = {"pca__n_components": [4],
                'SupVM__C': np.linspace(0.2, 0.8, num=7),
                'SupVM__gamma': np.arange(20, 80, 20)}

    # Build pipelines
    pipeline_knn = Pipeline(steps_for_knn)
    pipeline_svc = Pipeline(steps_for_svc)

    # GridSearchCV definitions
    cv_knn = GridSearchCV(pipeline_knn, param_grid=parameters_knn)
    cv_svc = GridSearchCV(pipeline_svc, param_grid=parameters_pca)

    

    model_names_arr = ["K Nearest Neighbors Classifier",
    "C-Support Vector Classifier", "Random Forest Classifier", "xgboost", "lightgbm"]

    with st.sidebar.header("2. Select a model"):
        st.sidebar.write("A pipeline using optimized hyperparameters\
        and with cross-validation will be created according to your selection.\
        This process could take some time with respect to the selected model.")
        model_name = st.sidebar.radio("label",label_visibility="collapsed", 
        options=(model_names_arr[0], model_names_arr[1], model_names_arr[2],
         model_names_arr[3], model_names_arr[4]))

    model = None
    best_params = None

    # Fit the models
    if(model_name == model_names_arr[0]):
        model = cv_knn
        # Fit the model
        model.fit(X_train, y_train)
        # Metrics
        best_score = model.best_score_
        best_params = model.best_params_
        y_pred_train = model.best_estimator_.predict(X_train)
        y_pred_test = model.best_estimator_.predict(X_test)

        # Test scores
        acc_score_test = accuracy_score(y_test, y_pred_test)
        f1_test = f1_score(y_test, y_pred_test)

        # Train scores
        acc_score_train = accuracy_score(y_train, y_pred_train)
        f1_train = f1_score(y_train, y_pred_train)

    elif(model_name == model_names_arr[1]):
        model = cv_svc
        # Fit the model
        model.fit(X_train, y_train)
        # Metrics
        best_score = model.best_score_
        best_params = model.best_params_
        y_pred_train = model.best_estimator_.predict(X_train)
        y_pred_test = model.best_estimator_.predict(X_test)

        # Test scores
        acc_score_test = accuracy_score(y_test, y_pred_test)
        f1_test = f1_score(y_test, y_pred_test)

        # Train scores
        acc_score_train = accuracy_score(y_train, y_pred_train)
        f1_train = f1_score(y_train, y_pred_train)

    elif(model_name == model_names_arr[2]):
        # Random Forest

        vectorizer = TfidfVectorizer()

        X_train = vectorizer.fit_transform(X_train)
        X_test = vectorizer.transform(X_test)

        n_estimators = [5,20,50,100] # number of trees in the random forest
        max_features = ['auto', 'sqrt'] # number of features in consideration at every split
        max_depth = [int(x) for x in np.linspace(10, 120, num = 12)] # maximum number of levels allowed in each decision tree
        min_samples_split = [2, 6, 10] # minimum sample number to split a node
        min_samples_leaf = [1, 3, 4] # minimum sample number that can be stored in a leaf node
        bootstrap = [True, False] # method used to sample data points

        random_grid = {
            'n_estimators': n_estimators,
            'max_features': max_features,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'bootstrap': bootstrap
        }

        clf=RandomForestClassifier(n_estimators=10)
        rf_random = RandomizedSearchCV(estimator = clf,param_distributions = random_grid,
                    n_iter = 10, cv = 5, verbose=2, random_state=35, n_jobs = -1)
        rf_random.fit(X_train, y_train)
        y_pred_test = rf_random.predict(X_test)
        y_pred_train = rf_random.predict(X_train)

        # Test scores
        acc_score_test = accuracy_score(y_test, y_pred_test)
        f1_test = f1_score(y_test, y_pred_test)

        # Train scores
        acc_score_train = accuracy_score(y_train, y_pred_train)
        f1_train = f1_score(y_train, y_pred_train)

    elif(model_name == model_names_arr[3]):
        vectorizer = TfidfVectorizer()

        X_train = vectorizer.fit_transform(X_train)
        X_test = vectorizer.transform(X_test)
        model = xgb.XGBClassifier()

        model.fit(X_train, y_train, eval_metric='rmse')
        y_pred_test = model.predict(X_test)
        y_pred_train = model.predict(X_train)
        
        # Test scores
        acc_score_test = accuracy_score(y_test, y_pred_test)
        f1_test = f1_score(y_test, y_pred_test)

        # Train scores
        acc_score_train = accuracy_score(y_train, y_pred_train)
        f1_train = f1_score(y_train, y_pred_train)
    
    elif(model_name == model_names_arr[4]):
        vectorizer = TfidfVectorizer()

        X_train = vectorizer.fit_transform(X_train)
        X_test = vectorizer.transform(X_test)

        model = lgb.LGBMClassifier()
        model.fit(X_train, y_train)
        y_pred_test = model.predict(X_test)
        y_pred_train = model.predict(X_train)

        # Test scores
        acc_score_test = accuracy_score(y_test, y_pred_test)
        f1_test = f1_score(y_test, y_pred_test)

        # Train scores
        acc_score_train = accuracy_score(y_train, y_pred_train)
        f1_train = f1_score(y_train, y_pred_train)
    


    # Confusion matrix
    cf = confusion_matrix(y_test, y_pred_test)
    disp = ConfusionMatrixDisplay(cf)

    res_text = "2 Model Performance For %s" %(model_name)
    st.subheader(res_text)

    st.markdown("**2.1 Training Set**")

    # st.write("Mean cross-validated score of the best_estimator")
    # st.info(round(best_score, 2))
    # st.write("R^2 score")
    # st.info(round(r2_train, 2))
    # st.write("Mean Squarred Error (MSE)")
    # st.info(round(mse_train, 2))
    st.write("Accuracy score")
    st.info(round(acc_score_train, 2))
    st.write("F1 score")
    st.info(round(f1_train, 2))

    
    st.markdown("**2.2 Test Set**")
    # st.write("R^2 score")
    # st.info(r2_test)
    # st.write("Mean Squarred Error (MSE)")
    # st.info(mse_test)
    st.write("Accuracy score")
    st.info(round(acc_score_test, 2))
    st.write("F1 score")
    st.info(round(f1_test, 2))

    st.markdown("**2.3 Confusion Matrix**")
    disp.plot()
    st.pyplot()

    st.subheader("**3. Parameters For The Model**")
    st.markdown("**3.1 Selected Train Test Split**")
    str_ = "%s : %s" %(split_size, 100-split_size)
    st.info(str_)

    st.markdown("**3.2 Selected Random State**")
    st.info(random_seed)

    if (best_params is not None):
        st.markdown("**3.3 Best Parameters Found**")
        st.info(best_params)


if __name__ == "__main__":
    create_model()