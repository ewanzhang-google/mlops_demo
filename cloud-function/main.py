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
    BUCKET = "_BUCKET_NAME_HERE_"
    PROJECT_ID = "_PROJECT_ID_HERE_"
    BQ_DATA_URI = "PROJECT.DATASET.TABLE"
    LOCATION = "europe-west4"
    
    #pubsub_message = base64.b64decode(event['data']).decode('utf-8')
    #print(pubsub_message)
    pl = PipelineJob(
        enable_caching=False,
        display_name = "pipeline_name",
        job_id = "sr-"+"".join(e for e in str(context.timestamp) if e.isalnum()).lower(),
        pipeline_root="gs://{}/pipeline".format(BUCKET),
        template_path = "gs://{}/cloud_build_pipeline/airlines-mlops/dcf26a9/training_pipeline.json".format(BUCKET),
        project = PROJECT_ID,
        location = LOCATION,
        parameter_values = {"project_id": "datauki-demo-airline", "bq_uri": "{}".format(BQ_DATA_URI), "assets_prefix": "prop-cf", "location": LOCATION })

    pl.run(sync=False, service_account="vertex-pipelines-sa@{}.iam.gserviceaccount.com".format(PROJECT_ID))
   
