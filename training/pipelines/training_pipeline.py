import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import kfp.v2.dsl as dsl
from kfp.v2 import compiler
from google_cloud_pipeline_components import aiplatform as gcc_aip

from training.components.data.bq_export import bq_export_comp
from training.components.data.stats import gen_stats_comp, validate_stats_comp
from training.components.data.data_split import data_split_comp
from training.components.training.algorithms import svm_comp, bt_comp
from training.components.training.blessed_model import blessed_model_comp

from google.cloud.aiplatform.pipeline_jobs import PipelineJob



def compile_pipeline(pipeline_filename, pipeline_name):

    @dsl.pipeline(
        name=pipeline_name,
    )
    def model_training_pipeline(
            project_id: str,
            bq_uri: str,
            label: str,
            assets_prefix: str,
            location: str
    ):



        #    bigquery_query_op(
        #        query='SELECT * FROM `{}` LIMIT 10'.format(bq_uri),
        #        project_id='datauki-demo-airline',
        #       dataset_id='propensity_model',
        #        table_id='propensity_airline_data_exp',
        #        dataset_location='europe-west4',
        #        job_config='')


        bq_export_op = bq_export_comp(bq_uri= bq_uri,
                                      project=project_id,
                                      location=location).set_display_name('Export Data from BQ')

        gen_stats_op = gen_stats_comp(
            dataset = bq_export_op.outputs['exported_dataset']
        ).set_display_name('Generate Statistics')

        validate_stats_op = validate_stats_comp(
            json_artifact=gen_stats_op.outputs['json_artifact']
        ).set_display_name('Validate Statistics')

        data_split_op = data_split_comp(
            dataset= bq_export_op.outputs['exported_dataset'],
            train_data_size=0.8,
            validation_data_size=0.0,
            test_data_size=0.2).set_display_name('Split Dataset').after(validate_stats_op)

        bt_op = bt_comp(
            data_split_op.outputs['train_set'],
            data_split_op.outputs['test_set'],
            label='label'
        ).set_display_name('Train Gradient Boosting Model')

        svm_op = svm_comp(
            data_split_op.outputs['train_set'],
            data_split_op.outputs['test_set'],
            label='label'
        ).set_display_name('Train SVM Model')
        svm_op.set_cpu_limit('4')
        svm_op.set_memory_limit('14Gi')
        #svm_op.add_node_selector_constraint('cloud.google.com/gke-accelerator', 'nvidia-tesla-k80')
        #svm_op.set_gpu_limit(1)
        #svm_op.set_display_name("123")




        blessed_model_op = blessed_model_comp(
            bt_op.outputs['metrics_params'],
            svm_op.outputs['metrics_params']
        ).set_display_name('Select Best Model')



        with dsl.Condition(blessed_model_op.output == 1,
                           name="deploy Gradient Boosting model"):


            ### Use predefined component to upload model
            model_upload_op = gcc_aip.ModelUploadOp(
                project=project_id,
                display_name=assets_prefix,
                location=location,
                artifact_uri=bt_op.outputs["path"],
                serving_container_image_uri='europe-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.0-24:latest')\
                .set_display_name('Upload Gradient Boosting Model')\
                .after(bt_op)

            ### Create endpoint
            endpoint_create_op = gcc_aip.EndpointCreateOp(
                project=project_id,
                display_name=assets_prefix,
                location=location
            ).set_display_name('Create Endpoint')

            custom_model_deploy_op = gcc_aip.ModelDeployOp(
                endpoint=endpoint_create_op.outputs["endpoint"],
                model=model_upload_op.outputs["model"],
                dedicated_resources_machine_type="n1-standard-4",
                dedicated_resources_min_replica_count=1,
                traffic_split={"0": 100}
            ).set_display_name('Deploy Gradient Boosting Model to Endpoint')

        with dsl.Condition(blessed_model_op.output == 2,
                           name="deploy SVM blessed model"):

            ### Use predefined component to upload model
            model_upload_op = gcc_aip.ModelUploadOp(
                project=project_id,
                display_name=assets_prefix,
                artifact_uri=svm_op.outputs["path"],
                serving_container_image_uri='europe-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.0-24:latest')\
                .set_display_name('Upload SVM Model')

            ### Create endpoint

            endpoint_create_op = gcc_aip.EndpointCreateOp(
                project=project_id,
                display_name=assets_prefix,
                location=location
            ).set_display_name('Create Endpoint')

            custom_model_deploy_op = gcc_aip.ModelDeployOp(
                endpoint=endpoint_create_op.outputs["endpoint"],
                model=model_upload_op.outputs["model"],
                dedicated_resources_machine_type="n1-standard-4",
                dedicated_resources_min_replica_count=1,
                traffic_split={"0": 100}
            ).set_display_name('Deploy SVM Model to Endpoint')


    
    compiler.Compiler().compile(
        pipeline_func=model_training_pipeline,
        package_path=pipeline_filename
    )
    
    
def run_pipeline(pipeline_name, run_name, project_id, pl_root, pl_path, location, pipeline_params, service_account, enable_caching=False):
    pl = PipelineJob(
        enable_caching=enable_caching,
        display_name = pipeline_name,
        job_id = run_name,
        pipeline_root=pl_root,
        template_path = pl_path,
        project = project_id,
        location = location,
        parameter_values = pipeline_params)
    print("--- CCC -- running as :",service_account)
    status = pl.run(sync=True, service_account=service_account)
    
    if(pl.has_failed):
        exit(1)

#bigquery_query_op = comp.load_component_from_file('components/bq.yaml')


