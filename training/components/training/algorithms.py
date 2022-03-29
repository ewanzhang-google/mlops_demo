from kfp.v2.dsl import (
    component,
    Input,
    Output,
    Dataset,
    Metrics,
    Model,
    ClassificationMetrics,
    Markdown
)

from typing import NamedTuple

@component(
    packages_to_install=["pandas==1.3.5", "numpy==1.21.5", "scikit-learn==0.24.0", "fsspec==2021.11.1", "gcsfs==2021.11.1"], 
    base_image='python:3.9',
    #output_component_file='train.yaml',
    #target_image='gcr.io/feature-strore-mars21/sklearn-pipeline/train')
)
def svm_comp( 
    train_set: Input[Dataset], 
    test_set: Input[Dataset],
    label: str,
    model: Output[Model],
    metrics_class: Output[ClassificationMetrics],
    metrics_params: Output[Metrics],
    report: Output[Markdown]
)  -> NamedTuple(
    'ModelPathOut',
    [
      ('path', str)
    ]):
    
    
    model.uri =  model.uri+".pkl"
    
    import pandas as pd
    import numpy as np
    
    import os, pathlib
    
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import roc_curve
    from sklearn.metrics import classification_report
    from sklearn.metrics import f1_score
    from sklearn.model_selection import cross_val_score
    
    from sklearn.pipeline import make_pipeline, Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
    
    from sklearn.svm import SVC
    
    from typing import List

    from collections import namedtuple
    import pickle
    
    ### Load data ###
    train_df = pd.read_csv(train_set.uri)
    train_df_label = train_df[label]
    train_df = train_df.drop(columns=[label])
    
    test_df = pd.read_csv(test_set.uri)
    test_df_label = test_df[label]
    test_df = test_df.drop(columns=[label])

    dict_types = train_df.dtypes.to_dict()
    CATEGORICAL_FEATURES = []
    BINARY_FEATURES = []
    NUMERIC_FEATURES = []
    BINARY_FEATURES_IDX = []
    NUMERIC_FEATURES_IDX = []
    CATEGORICAL_FEATURES_IDX = []

    indexer = 0
    for k,v in dict_types.items():
        if v==np.object_:
            CATEGORICAL_FEATURES.append(k)
            CATEGORICAL_FEATURES_IDX.append(indexer)
        elif v==np.bool_:
            BINARY_FEATURES.append(k)
            BINARY_FEATURES_IDX.append(indexer)
        else:
            NUMERIC_FEATURES.append(k)
            NUMERIC_FEATURES_IDX.append(indexer)

        indexer+=1

    ALL_COLUMNS = BINARY_FEATURES+NUMERIC_FEATURES+CATEGORICAL_FEATURES

    #BINARY_FEATURES_IDX = list(range(0,len(BINARY_FEATURES)))
    #NUMERIC_FEATURES_IDX = list(range(len(BINARY_FEATURES), len(BINARY_FEATURES)+len(NUMERIC_FEATURES)))
    #CATEGORICAL_FEATURES_IDX = list(range(len(BINARY_FEATURES+NUMERIC_FEATURES), len(ALL_COLUMNS)))

  
    preprocessor = ColumnTransformer(
        transformers=[
            ('bin', OrdinalEncoder(), BINARY_FEATURES_IDX),
            ('num', StandardScaler(), NUMERIC_FEATURES_IDX),
            ('cat', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL_FEATURES_IDX)], n_jobs=-1)

    
    model_params = {"kernel":"linear", "C":2, "class_weight":None, "probability": True}
    
    model_obj = SVC()
    model_obj.set_params(**model_params)

    clf = Pipeline(steps=[ ('preprocessor', preprocessor),
                          ('classifier', model_obj)])
    
    
    #clf = pipeline_builder(model_params, BINARY_FEATURES_IDX, NUMERIC_FEATURES_IDX, CATEGORICAL_FEATURES_IDX)
    score = cross_val_score(clf, train_df, train_df_label, cv=10, n_jobs=-1).mean()
    clf.fit(train_df, train_df_label)
    
    
    pred = clf.predict(test_df)
    metrics_class.log_confusion_matrix(["0", "1"], confusion_matrix(test_df_label, pred).tolist())
    
    
    pred_prob = clf.predict_proba(test_df)
    fpr, tpr, thresholds = roc_curve(y_true=test_df_label, y_score=pred_prob[:,1], pos_label=test_df_label[0])
    metrics_class.log_roc_curve(fpr, tpr, thresholds)
    
    with open(report.path, 'w') as f:
        f.write(classification_report(test_df_label,pred))

    metrics_params.log_metric("svm_f1_test_score", (f1_score(test_df_label, pred, pos_label=test_df_label[0])))
    metrics_params.log_metric("svm_cross_val_score", (score))
    
    #pathlib.Path(model.path).mkdir(parents=True, exist_ok=True)
    with open(model.path, 'wb') as handle:
        pickle.dump(clf, handle)
    
                            
    output = namedtuple('ModelPathOut',
        ['path'])
    
    return output(model.uri.strip('/model.pkl'))


@component(
    packages_to_install=["pandas==1.3.5", "numpy==1.21.5", "scikit-learn==0.24.0", "fsspec==2021.11.1", "gcsfs==2021.11.1"], 
    base_image='python:3.9',
    #output_component_file='train.yaml',
    #target_image='gcr.io/feature-strore-mars21/sklearn-pipeline/train')
)
def bt_comp( 
    train_set: Input[Dataset], 
    test_set: Input[Dataset],
    label: str,
    model: Output[Model],
    metrics_class: Output[ClassificationMetrics],
    metrics_params: Output[Metrics],
    report: Output[Markdown]
)  -> NamedTuple(
    'ModelPathOut',
    [
      ('path', str)
    ]):
    
    
    model.uri =  model.uri+".pkl"
    
    import pandas as pd
    import numpy as np
    
    import os, pathlib
    
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import roc_curve
    from sklearn.metrics import classification_report
    from sklearn.metrics import f1_score
    from sklearn.model_selection import cross_val_score
    
    from sklearn.pipeline import make_pipeline, Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
    
    from sklearn import ensemble
    
    from typing import List

    from collections import namedtuple
    import pickle
    
    ### Load data ###
    train_df = pd.read_csv(train_set.uri)
    train_df_label = train_df[label]
    train_df = train_df.drop(columns=[label])
    
    test_df = pd.read_csv(test_set.uri)
    test_df_label = test_df[label]
    test_df = test_df.drop(columns=[label])
    
    dict_types = train_df.dtypes.to_dict()
    CATEGORICAL_FEATURES = []
    BINARY_FEATURES = []
    NUMERIC_FEATURES = []
    
    BINARY_FEATURES_IDX = []
    NUMERIC_FEATURES_IDX = []
    CATEGORICAL_FEATURES_IDX = []
    
    indexer = 0
    for k,v in dict_types.items():
        if v==np.object_:
            CATEGORICAL_FEATURES.append(k)
            CATEGORICAL_FEATURES_IDX.append(indexer)
        elif v==np.bool_:
            BINARY_FEATURES.append(k)
            BINARY_FEATURES_IDX.append(indexer)
        else:
            NUMERIC_FEATURES.append(k)
            NUMERIC_FEATURES_IDX.append(indexer)
            
        indexer+=1
        
    ALL_COLUMNS = BINARY_FEATURES+NUMERIC_FEATURES+CATEGORICAL_FEATURES

    #BINARY_FEATURES_IDX = list(range(0,len(BINARY_FEATURES)))
    #NUMERIC_FEATURES_IDX = list(range(len(BINARY_FEATURES), len(BINARY_FEATURES)+len(NUMERIC_FEATURES)))
    #CATEGORICAL_FEATURES_IDX = list(range(len(BINARY_FEATURES+NUMERIC_FEATURES), len(ALL_COLUMNS)))
 
    preprocessor = ColumnTransformer(
        transformers=[
            ('bin', OrdinalEncoder(), BINARY_FEATURES_IDX),
            ('num', StandardScaler(), NUMERIC_FEATURES_IDX),
            ('cat', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL_FEATURES_IDX)], n_jobs=-1)

    
    model_params = {"kernel":"linear", "C":2, "class_weight":None}
    model_params = {
        "n_estimators": 1200,
        "max_depth": 3,
        "subsample": 0.5,
        "learning_rate": 0.01,
        "min_samples_leaf": 1,
        "random_state": 3,
    }
    model_obj = ensemble.GradientBoostingClassifier(**model_params)

    clf = Pipeline(steps=[ ('preprocessor', preprocessor),
                          ('classifier', model_obj)])
    
    
    #clf = pipeline_builder(model_params, BINARY_FEATURES_IDX, NUMERIC_FEATURES_IDX, CATEGORICAL_FEATURES_IDX)
    score = cross_val_score(clf, train_df, train_df_label, cv=10, n_jobs=-1).mean()
    clf.fit(train_df, train_df_label)
    

    pred = clf.predict(test_df)
    metrics_class.log_confusion_matrix(["0", "1"], confusion_matrix(test_df_label, pred).tolist())
    
    
    pred_prob = clf.predict_proba(test_df)
    fpr, tpr, thresholds = roc_curve(y_true=test_df_label, y_score=pred_prob[:,1], pos_label=test_df_label[0])
    metrics_class.log_roc_curve(fpr, tpr, thresholds)
    
    with open(report.path, 'w') as f:
        f.write(classification_report(test_df_label,pred))


    metrics_params.log_metric("bt_f1_test_score", (f1_score(test_df_label, pred, pos_label=test_df_label[0])))
    metrics_params.log_metric("bt_cross_val_score", (score))
    
    #pathlib.Path(model.path).mkdir(parents=True, exist_ok=True)
    #joblib.dump(, model.path)
    with open(model.path, 'wb') as handle:
        pickle.dump(clf, handle)
   
                            
    output = namedtuple('ModelPathOut',
        ['path'])
    
    return output(model.uri.strip('/model.pkl'))