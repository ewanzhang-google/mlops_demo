import pytest, os
from pytest_mock import MockerFixture
import training.pipelines.training_pipeline as pl
import inspect

# TODO: move into a config file
project_id = "ewans-demo-project"
dataset = "vertex_ai"
table = "coffee_beans_data"
bq_uri="{}.{}.{}".format(project_id, dataset, table)

gcs_uri="gs://{}/dev-tests".format(project_id)
sa = "vertex-pipelines-sa@{}.iam.gserviceaccount.com".format(project_id)
pipeline_project = project_id
pipeline_name = "test-pipeline-coffee"
pipeline_bucket = "gs://{}/test-pipeline".format(project_id)
location = 'europe-west4'
pipeline_params = {'project_id': project_id,
                  "bq_uri":bq_uri,
                   "label":"Label",
                   "assets_prefix":"test-pipeline",
                   "location":location}


@pytest.mark.unit
def test_compile_training_pipeline():

    pipeline_path = os.path.join(os.path.dirname(__file__), './artefacts/{}.json'.format(inspect.currentframe().f_code.co_name))

    pl.compile_pipeline(pipeline_path, pipeline_name)

    import json
    with open(pipeline_path, encoding='utf-8') as fh:
        data = json.load(fh)

    assert(isinstance(data, dict))



@pytest.mark.inte
def test_run_training_pipeline():

    from datetime import datetime
    timestamp = str(int(datetime.now().timestamp()))

    pipeline_path = os.path.join(os.path.dirname(__file__), './artefacts/{}.json'.format(inspect.currentframe().f_code.co_name))

    pl.compile_pipeline(pipeline_path, pipeline_name)
    pl.run_pipeline(pipeline_name = pipeline_name,
                    run_name = None,
                    project_id= pipeline_project,
                    pl_root = gcs_uri,
                    pl_path = pipeline_path,
                    location= location,
                    pipeline_params= pipeline_params,
                    service_account=sa,
                    enable_caching=False)





