# some_file.py
import sys, os, pathlib
from kfp.v2.dsl import (
    Artifact, 
    Dataset, 
    Model, 
    Output, 
    HTML, 
    ClassificationMetrics, 
    Metrics, 
    Markdown)
from unittest.mock import Mock, MagicMock, patch
from unittest import mock
import pytest

sys.path.insert(1, os.path.dirname(os.path.dirname(pathlib.Path(__file__).absolute())))

bq_uri= "test_project.propensity_model.propensity_airline_data"

artifact_path = os.path.dirname(pathlib.Path(__file__).absolute())+"/artifacts"
dataset_path = os.path.dirname(pathlib.Path(__file__).absolute())+"/test_data"


bq_export_data_uri = "gs://propensity_model/pipeline/tests/bq_export/exported_dataset"


dataset = Mock(spec=Dataset, uri = dataset_path)

train_set = MagicMock(spec=Dataset, uri = dataset_path+"/train.csv")

validation_set = Mock(spec=Dataset, uri = dataset_path+"/validation.csv")

test_set = Mock(spec=Dataset, uri = dataset_path+"/test.csv")

model = Mock(spec=Model, uri = artifact_path+"/model", path = artifact_path+"/model.pkl")

metrics_class = Mock(spec=ClassificationMetrics)

metrics_params = Mock(spec=Metrics)

report = Mock(spec=Markdown, path = artifact_path+"/report")


#pathlib.Path(model.path).mkdir(parents=True, exist_ok=True)


@patch("google.cloud.bigquery.Client.extract_table")
def test_bq_export(extract_table):
    from components.bq_export import bq_export_comp
    import dask.dataframe as dd
    
    bq_export_comp.python_func(
        bq_uri, 
        Dataset(uri=bq_export_data_uri))
    
    extract_table.assert_called()
    
      
def test_gen_stats():
    
    from components.stats import gen_stats_comp
    import tensorflow_data_validation as tfdv
    
    #dataset = Dataset(uri=bq_export_data_uri)
    
    html = Mock(spec=HTML)
    html.path = artifact_path+"/html_artifact.html"
    
    arf = Mock(spec=Artifact)
    arf.path = artifact_path+"/stats_artifact/af.proto"
    
    print(html.path)
    print(arf.path)
    
    gen_stats_comp.python_func(
        dataset,
        arf,
        html)
    
    import dask.dataframe as dd
    df = dd.read_csv(dataset_path+"/data_*.csv")
    
    assert tfdv.load_statistics(arf.path).datasets[0].num_examples == len(df.index)
    
def test_fail_exception_data_split_comp_1():
    from components.data_split import data_split_comp
    
    with pytest.raises(ValueError, match=r".*should add up to 1.*"):
        data_split_comp.python_func(
            dataset=dataset,
            train_set=train_set,
            validation_set=validation_set,
            test_set=test_set,
            train_data_size=0.1,
            validation_data_size=0.2,
            test_data_size=0.4)
    
    
def test_fail_exception_data_split_comp_2():
    from components.data_split import data_split_comp
    
    with pytest.raises(ValueError, match=r".*should add up to 1.*"):
        data_split_comp.python_func(
            dataset=dataset,
            train_set=train_set,
            validation_set=validation_set,
            test_set=test_set,
            train_data_size=0.9,
            validation_data_size=0.2,
            test_data_size=0.4)
    
def test_data_split_comp():
    from components.data_split import data_split_comp
    
    
    train_data_size=0.8
    validation_data_size=0.0
    test_data_size=0.2
    
    data_split_comp.python_func(
        dataset=dataset,
        train_set=train_set,
        validation_set=validation_set,
        test_set=test_set,
        train_data_size=train_data_size,
        validation_data_size=validation_data_size,
        test_data_size=test_data_size)

    import dask.dataframe as dd
    df = dd.read_csv(dataset.uri+"/data_*.csv")
    df_train = dd.read_csv(train_set.uri)
    df_validation = dd.read_csv(validation_set.uri)
    df_test = dd.read_csv(test_set.uri)
    
    assert len(df.index)*train_data_size == len(df_train)
    assert len(df.index)*validation_data_size == len(df_validation)
    assert len(df.index)*test_data_size == len(df_test)
    
def test_svm_training():
    from components.training import svm_comp
    
    svm_comp.python_func(
        train_set=train_set,
        test_set=test_set,
        label='label',
        model=model,
        metrics_class=metrics_class,
        metrics_params=metrics_params,
        report=report)
    
    
def test_bt_training():
    from components.training import bt_comp
    
    bt_comp.python_func(
        train_set=train_set,
        test_set=test_set,
        label='label',
        model=model,
        metrics_class=metrics_class,
        metrics_params=metrics_params,
        report=report)