import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import kfp.components as comp
import kfp.v2 as kfp
import kfp.v2.dsl as dsl
from kfp.v2 import compiler
from kfp.v2.dsl import (
    component,
    Output,
    HTML
)

from google_cloud_pipeline_components import aiplatform as gcc_aip
import pipelines.training_pipeline as pl 

from components.bq_export import bq_export_comp
from components.stats import gen_stats_comp
from components.data_split import data_split_comp
from components.training import svm_comp, bt_comp
from components.blessed_model import blessed_model_comp

from google.cloud.aiplatform.pipeline_jobs import PipelineJob



def compile_pipeline(pipeline_filename):
        
    pipeline_function = pl.model_training_pipeline
    
    compiler.Compiler().compile(
        pipeline_func=pipeline_function,
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

    status = pl.run(sync=True, service_account=service_account)
    
    if(pl.has_failed):
        exit(1)

#bigquery_query_op = comp.load_component_from_file('components/bq.yaml')

from kfp.dsl.types import GCPProjectID
@dsl.pipeline(
  name='airline-propensity-pipeline',
  description='pipeline training propensity model',
)
def model_training_pipeline(
    project_id: str,
    bq_uri: str,
    lable: str="label"
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
    

    bq_export_op = bq_export_comp(bq_uri= bq_uri)
    
    gen_stats_op = gen_stats_comp(
        dataset = bq_export_op.outputs['exported_dataset']
    )
    
    data_split_op = data_split_comp(
        dataset= bq_export_op.outputs['exported_dataset'],
        train_data_size=0.8,
        validation_data_size=0.0,
        test_data_size=0.2)
    
    bt_op = bt_comp(
        data_split_op.outputs['train_set'],
        data_split_op.outputs['test_set'],
        label='label'
    )
    
    svm_op = svm_comp(
        data_split_op.outputs['train_set'],
        data_split_op.outputs['test_set'],
        label='label'
    )
    svm_op.set_cpu_limit('4')
    svm_op.set_memory_limit('14Gi')
    #svm_op.add_node_selector_constraint('cloud.google.com/gke-accelerator', 'nvidia-tesla-k80')
    #svm_op.set_gpu_limit(1)
    #svm_op.set_display_name("123")
    
    
    
    
    blessed_model_op = blessed_model_comp(
        bt_op.outputs['metrics_params'],
        svm_op.outputs['metrics_params']
    )
    
  
    
    with dsl.Condition(blessed_model_op.output == 1,
                        name="deploy bt blessed model"):
            
    
        ### Use predefined component to upload model
        model_upload_op = gcc_aip.ModelUploadOp(
        project=project_id,
        display_name=assets_prefix,
        location=location,
        artifact_uri=bt_op.outputs["path"],
        serving_container_image_uri='europe-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.0-24:latest').after(bt_op)
            
        ### Create endpoint
        endpoint_create_op = gcc_aip.EndpointCreateOp(
            project=project_id,
            display_name=assets_prefix,
            location=location
        ) 
        
        custom_model_deploy_op = gcc_aip.ModelDeployOp(
           endpoint=endpoint_create_op.outputs["endpoint"],
            model=model_upload_op.outputs["model"],
            dedicated_resources_machine_type="n1-standard-4",
            dedicated_resources_min_replica_count=1,
            traffic_split={"0": 100}
        )  

    with dsl.Condition(blessed_model_op.output == 2,
                        name="deploy svm blessed model"):
      
        ### Use predefined component to upload model
        model_upload_op = gcc_aip.ModelUploadOp(
        project=project_id,
        display_name=assets_prefix,
        artifact_uri=svm_op.outputs["path"],
        serving_container_image_uri='europe-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.0-24:latest')
            
        ### Create endpoint
        
        endpoint_create_op = gcc_aip.EndpointCreateOp(
            project=project_id,
            display_name=assets_prefix,
            location=location
        ) 
        
        custom_model_deploy_op = gcc_aip.ModelDeployOp(
           endpoint=endpoint_create_op.outputs["endpoint"],
            model=model_upload_op.outputs["model"],
            dedicated_resources_machine_type="n1-standard-4",
            dedicated_resources_min_replica_count=1,
            traffic_split={"0": 100}
        )
        
    
