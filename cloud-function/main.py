import base64

from google.cloud.aiplatform.pipeline_jobs import PipelineJob


def trigger(event, context):
    """Triggered from a message on a Cloud Pub/Sub topic.
    Args:
         event (dict): Event payload.
         context (google.cloud.functions.Context): Metadata for the event.
    """
    
    # get gcs path for pub/sub msg
    # read file from gcs
    # get configs from file and use the below
    
    
    #pubsub_message = base64.b64decode(event['data']).decode('utf-8')
    #print(pubsub_message)
    pl = PipelineJob(
        enable_caching=False,
        display_name = "airline-propensity-pipeline",
        job_id = "sr-"+"".join(e for e in str(context.timestamp) if e.isalnum()).lower(),
        pipeline_root="gs://propensity_model/pipeline",
        template_path = "gs://propensity_model/cloud_build_pipeline/airlines-mlops/dcf26a9/training_pipeline.json",
        project = "datauki-demo-airline",
        location = "europe-west4",
        parameter_values = {"project_id": "datauki-demo-airline", "bq_uri": "{}.{}.{}".format("datauki-demo-airline","propensity_model","propensity_airline_data"), "assets_prefix": "prop-cf", "location": "europe-west4" })

    pl.run(sync=False, service_account="vertex-pipelines-sa@datauki-demo-airline.iam.gserviceaccount.com")
   
