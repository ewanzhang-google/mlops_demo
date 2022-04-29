import os
import pytest
from pytest_mock import MockerFixture
from unittest.mock import patch

from training.components.data.bq_export import bq_export_comp

# TODO: move into a config file
project_id = "ewans-demo-project"
dataset = "mlops_demo"
table = "coffee_beans_data"
bq_uri="{}.{}.{}".format(project_id, dataset, table)

gcs_uri="gs://{}/dev-tests".format(project_id)
sa = "vertex-pipeline-sa@{}.iam.gserviceaccount.com".format(project_id)
pipeline_project = project_id
pipeline_bucket = "gs://{}/test-pipeline".format(project_id)
pipeline_location = 'us-central1'

"""
@pytest.fixture
def et():
    return MockerFixture.patch("google.cloud.bigquery.Client.extract_table")
"""

@pytest.mark.unit
def test_bq_export_unit(mocker: MockerFixture):
    from kfp.v2.dsl import Dataset
    from google.cloud.bigquery import Client

    client_mock = mocker.MagicMock(
        spec=Client
    )

    mock_client = mocker.patch("google.cloud.bigquery.Client", return_value=client_mock)

    bq_export_comp.python_func(
        bq_uri=bq_uri,
        project=pipeline_project,
        location=pipeline_location,
        exported_dataset=Dataset(uri=gcs_uri))

    bq_project_id, bq_dataset_id, bq_table_id = bq_uri.split('.')

    mock_client.assert_called_once_with(project=bq_project_id)


@pytest.mark.inte
def test_bq_export_int(mocker: MockerFixture):
    from kfp.v2.dsl import Dataset

    dataset = mocker.Mock(spec=Dataset, uri = gcs_uri)
    bq_export_comp.python_func(
        bq_uri,
        project=pipeline_project,
        location='us-central1',
        exported_dataset=dataset)


@pytest.mark.inte
#@pytest.mark.skipif('RUN_ENV' in os.environ and os.environ['RUN_ENV']=='test', reason="This is integration test and takes time -using only while developing")
def test_pipeline_using_component_e2e():

    from google.cloud.aiplatform.pipeline_jobs import PipelineJob
    from kfp.v2 import compiler, dsl
    import inspect

    from datetime import datetime
    timestamp = str(int(datetime.now().timestamp()))

    pipeline_path = os.path.join(os.path.dirname(__file__), './artifacts/{}.json'.format(inspect.currentframe().f_code.co_name))

    @dsl.pipeline(
        name='test-bq-export-comp',
        description='testing pipeline for exporting data from BQ to CSV'
    )
    def pipeline(
            bq_uri: str,
            project: str,
            location: str,
            gcs_uri: str
    ):
        export_features_from_bq_search_op = bq_export_comp(
            bq_uri,
            project,
            location
        )
        export_features_from_bq_search_op.set_display_name("export_data_bq")

    compiler.Compiler().compile(
        pipeline_func=pipeline,
        package_path=pipeline_path)

    pl = PipelineJob(display_name= "test-bq-export",
                     job_id= None,
                     template_path= pipeline_path,
                     pipeline_root= pipeline_bucket,
                     project=pipeline_project,
                     labels= {"env":"test"},
                     location=pipeline_location,
                     enable_caching=False,
                     parameter_values={
                         'bq_uri': bq_uri,
                         'project': pipeline_project,
                         'location': 'us-central1',
                         'gcs_uri': gcs_uri})

    print(pl.run(sync=True, service_account=sa) )
