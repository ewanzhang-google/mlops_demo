import os
import pytest, pathlib
from pytest_mock import MockerFixture
from kfp.v2.dsl import Dataset, HTML, Artifact
from training.components.data.stats import gen_stats_comp

dataset_path = os.path.dirname(pathlib.Path(__file__).absolute())+"/test_data"
artifact_path = os.path.dirname(pathlib.Path(__file__).absolute())+"/artifacts"

@pytest.mark.unit
def test_bq_export_unit(mocker: MockerFixture):

    dataset = mocker.MagicMock(spec=Dataset, uri = dataset_path)
    html_stats = mocker.Mock(spec=HTML, path = artifact_path+"/html_artifact.html")
    json_stats = mocker.Mock(spec=Artifact, path = artifact_path+"/json_artifact.json")

    gen_stats_comp.python_func(dataset, html_stats, json_stats)

    #TODO: check that files are generated and that are valid json and html