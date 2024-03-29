import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import json
import re

from enum import Enum
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
from SPARQLWrapper import SPARQLWrapper, JSON


st.set_page_config(page_title="Design Project", layout="wide")
st.set_option('deprecation.showPyplotGlobalUse', False)

st.write(
    """
    # Design Decision Classifier, NLP Keyword Extractor and Design Solution Recommendation Maker
    In this part, different models are used for classifying the design decisions.
    Try adjusting the parameters and select the model you desire. Note that\
    hyperparameter optimization will be done in the background for the selected model.
    """
)
st.write("Halil Mert KARTAL, 21946314")
st.write("Mert Kuğ, 21827646")

def create_model():
    # Read the csv
    df = pd.read_csv("dataset/design_decisions_v1.csv")
    X = df["text"].values.astype('U')
    y = df["label"].values

    st.subheader("**1. Dataset summary**")
    st.markdown("**1.1 Tabular view of the data (first 50)**")
    df.rename(columns={"Unnamed: 0":"index"})
    for col in df.columns:
        print(col)
    st.table(df.head(50))

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
        split_size = st.sidebar.slider("Data split ratio (% for training set)", 50, 90, 80)
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
    cv_knn = GridSearchCV(pipeline_knn, param_grid=parameters_knn, n_jobs=5)
    cv_svc = GridSearchCV(pipeline_svc, param_grid=parameters_pca, n_jobs=5)

    # Model names are enumerated
    class Model(Enum):
        KNN = "K Nearest Neighbors Classifier"
        CSV = "C-Support Vector Classifier"
        RFC = "Random Forest Classifier"
        XGB = "xgboost"
        GBM = "lightgbm"


    with st.sidebar.header("2. Select a model"):
        st.sidebar.write("A pipeline using optimized hyperparameters\
        and with cross-validation will be created according to your selection.\
        This process could take some time with respect to the selected model.")
        model_name = st.sidebar.radio("label",label_visibility="collapsed", 
        options=(Model.KNN.value, Model.CSV.value, Model.RFC.value, Model.XGB.value, Model.GBM.value))

    model = None
    best_params = None
    vectorizer = TfidfVectorizer()

    # Fit the models
    if(model_name == Model.KNN.value):
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

    elif(model_name == Model.CSV.value):
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

    elif(model_name == Model.RFC.value):
        # Random Forest


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
        model = RandomizedSearchCV(estimator = clf,param_distributions = random_grid,
                    n_iter = 10, cv = 5, verbose=2, random_state=35, n_jobs = -1)
        model.fit(X_train, y_train)
        y_pred_test = model.predict(X_test)
        y_pred_train = model.predict(X_train)

        # Test scores
        acc_score_test = accuracy_score(y_test, y_pred_test)
        f1_test = f1_score(y_test, y_pred_test)

        # Train scores
        acc_score_train = accuracy_score(y_train, y_pred_train)
        f1_train = f1_score(y_train, y_pred_train)

    elif(model_name == Model.XGB.value):
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
    
    elif(model_name == Model.GBM.value):
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

    
    st.markdown("**2.1 Metrics**")
    # st.write("R^2 score")
    # st.info(r2_test)
    # st.write("Mean Squarred Error (MSE)")
    # st.info(mse_test)
    st.write("Accuracy score")
    st.info(round(acc_score_test, 2))
    st.write("F1 score")
    st.info(round(f1_test, 2))

    st.markdown("**2.2 Confusion Matrix**")
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


    # Add text box to get user input and classify it
    st.subheader("**4. Test the model with custom input**")
    new_input = st.text_input(label="You can test the classifier by giving custom input below:", placeholder="Enter here")

    pred = -1
    result_to_show = ""

    # When user types something
    if new_input != "":
        # Convert and put into an array
        new_input = [new_input]

        # Two models have pipelines which have vectorizers so we don't have to vectorize the input seperately.
        if(model_name == Model.KNN.value or model_name == Model.CSV.value):
            
            # Predict
            pred = model.best_estimator_.predict(new_input)[0]
        
        # Other three models doesn't have pipelines which have vectorizers so we have to vectorize the input seperately.
        else:
            new_input = vectorizer.transform(new_input)

            # Predict
            pred = model.predict(new_input)[0]
        
        result_to_show = "Not a Design Decision" if pred == 0 else "Design Decision"
        st.markdown("**Prediction**")
        st.info(result_to_show)

    # Read allEntities.json and create a table to show.
    allEntities = ""
    f = open("allEntities_v2.json")
    for line in f:
        allEntities += line
    entitiesDict = json.loads(allEntities)
    # Convert values from array to string
    for key in entitiesDict:
        newVal = ""
        for i in entitiesDict[key]:
            newVal += i
            newVal += " "
        entitiesDict[key] = newVal
    
    entitiesDf = pd.DataFrame.from_dict([entitiesDict])
    entitiesDf = entitiesDf.rename(index = {0: 'Design Decision'})

    st.subheader("**5. Extraction of the entities and quality attributes of design decisions**")
    st.markdown("**In this part, Natural Language Processing techniques are used to extract corresponding synonyms\
                and representing keywords of each quality attribute. It can be seen at the table below (first 100)**")
    st.table(entitiesDf.T.head(100))

    st.subheader("**6. Recommandations by SPARQL queries using the entities**")
    st.markdown("**In this part, you can try the recommendation maker by selecting an example entity below. Execution of \
                the SPARQL query may take some time.**")

    wordToQuery = st.selectbox(
    'Please select an entity below',
    ('algorithm', 'audit', 'authentication', 'cache', 'calculator', 'database', 'debugging', 'encryption',
      'filesystem', 'image editor', 'kernel', 'linear algebra', 'maven', 'metadata', 'parsing', 'python', 'rpc', 'server', 
      'ssh', 'usenet', 'vanilla', 'web application', 'yarn'))
    
    sparql = SPARQLWrapper("http://dbpedia.org/sparql")

    sparql.setReturnFormat(JSON)
    sparql.setQuery(
    """
    PREFIX      owl: <http://www.w3.org/2002/07/owl#>
    PREFIX      xsd: <http://www.w3.org/2001/XMLSchema#>
    PREFIX     rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX      rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX     foaf: <http://xmlns.com/foaf/0.1/>
    PREFIX       dc: <http://purl.org/dc/elements/1.1/>
    PREFIX      res: <http://dbpedia.org/resource/>
    PREFIX dbpedia2: <http://dbpedia.org/property/>
    PREFIX  dbpedia: <http://dbpedia.org/>
    PREFIX     skos: <http://www.w3.org/2004/02/skos/core#>
    PREFIX dbc:	<http://dbpedia.org/resource/Category:>
    PREFIX dbo:	<http://dbpedia.org/ontology/>
    PREFIX dbp:	<http://dbpedia.org/property/>
    PREFIX ns: <http://example.org/namespace/>
    PREFIX dct: <http://purl.org/dc/terms/>


    SELECT DISTINCT ?software WHERE {
    {
        ?software dbo:genre ?genre ;
                rdf:type dbo:Software .
        ?genre rdfs:label ?genreLabel .
        FILTER (lcase(str(?genreLabel)) = lcase("%s"))
    }
    UNION
    {
        ?software dct:subject ?concept ;
                rdf:type dbo:Software .
        ?concept rdfs:label ?conceptLabel .
        FILTER (lcase(str(?conceptLabel)) = lcase("%s"))
    }
    

    }
    """
    %(wordToQuery, wordToQuery)
    
    )

    resultArr = []
    try:
        ret = sparql.queryAndConvert()

        for r in ret["results"]["bindings"]:
            for i in r:
                for j in r[i]:
                    if j == "value":
                        result = re.search(r'/([^/]+)$', r[i][j]).group(1)
                        resultArr.append(result)
    except Exception as e:
        print(e)
        
    for i in resultArr:
        st.info(i, icon="✅")
    if len(resultArr) == 0:
        st.info("Could not find anything about '{0}'".format(wordToQuery))

    # Read allEntities.json and create a table to show.
    allRecommendations = ""
    f = open("allRecommendations.json")
    for line in f:
        allRecommendations += line
    recommendationsDict = json.loads(allRecommendations)
    # Convert values from array to string
    for key in recommendationsDict:
        newVal = ""
        for i in recommendationsDict[key]:
            if i == None:
                i = "-"
            newVal += i
            newVal += " "
        recommendationsDict[key] = newVal
    
    recommendationsDf = pd.DataFrame.from_dict([recommendationsDict])
    #recommendationsDict = recommendationsDict.rename(index = {0: 'Design Decision'})
    
    st.subheader("**7. All recommendations done by the extracted entities**")
    st.markdown("**In this part, you can see some of the recommendations (first 200) for all entries if it was found.\
                 A '-' was placed if nothing found about the specific entry.**")
    st.table(recommendationsDf.T.head(200))

if __name__ == "__main__":
    create_model()
