from pytest_mock import MockerFixture
import pytest
import pickle
import pathlib, os
from kfp.v2.dsl import (
    Artifact,
    Dataset,
    Model,
    Output,
    HTML,
    ClassificationMetrics,
    Metrics,
    Markdown)


artifact_path = os.path.dirname(pathlib.Path(__file__).absolute())+"/artifacts"
dataset_path = os.path.dirname(pathlib.Path(__file__).absolute())+"/test_data"

mock = MockerFixture(config=None)
train_set = mock.MagicMock(spec=Dataset, uri = dataset_path+"/train.csv")
validation_set = mock.Mock(spec=Dataset, uri = dataset_path+"/validate.csv")
test_set = mock.Mock(spec=Dataset, uri = dataset_path+"/test.csv")
model = mock.Mock(spec=Model, uri = artifact_path+"/model", path = artifact_path+"/model.pkl")
metrics_class = mock.Mock(spec=ClassificationMetrics)
metrics_params = mock.Mock(spec=Metrics)
report =mock.Mock(spec=Markdown, path = artifact_path+"/report")

@pytest.mark.unit
def test_svm_training():
    from training.components.training.algorithms import  svm_comp

    svm_comp.python_func(
        train_set=train_set,
        test_set=test_set,
        label='label',
        model=model,
        metrics_class=metrics_class,
        metrics_params=metrics_params,
        report=report)

    with open(model.path,"rb") as f:
        clf = pickle.load(f)
        predictions = clf.predict([['2276-YDAVZ','Green',0,False,False,3,True,'Medium','Typica','washed','Mexico',False,False,False,'cold','large',True,'christina dusing',75,270]])
        assert(predictions[0] in ['Arabica', 'Robusta'])


@pytest.mark.unit
def test_bt_training():
    from training.components.training.algorithms import bt_comp
    bt_comp.python_func(
        train_set=train_set,
        test_set=test_set,
        label='label',
        model=model,
        metrics_class=metrics_class,
        metrics_params=metrics_params,
        report=report)

    with open(model.path,"rb") as f:
        clf = pickle.load(f)
        predictions = clf.predict([['2276-YDAVZ','Green',0,False,False,3,True,'Medium','Typica','washed','Mexico',False,False,False,'cold','large',True,'christina dusing',75,270]])
        assert(predictions[0] in ['Arabica', 'Robusta'])