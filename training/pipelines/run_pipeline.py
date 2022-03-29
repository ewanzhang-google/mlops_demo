from argparse import ArgumentParser
import json
from training.pipelines import training_pipeline

    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-pn", "--pipeline_name",
                        dest="pipeline_name",
                        default="$REPO_NAME",
                        help="")
    
    parser.add_argument("-rn", "--run_name",
                        dest="run_name",
                        default="$SHORT_SHA",
                        help="")


    parser.add_argument("-pid", "--project_id",
                        dest="project_id",
                        default="$PROJECT_ID",
                        help="")
    
       
    parser.add_argument("-ruri", "--pipeline_root_uri",
                        dest="pipeline_root_uri",
                        required=True,
                        help="")
    
       
    parser.add_argument("-uri", "--pipeline_gs_path",
                        dest="pipeline_gs_path",
                        required=True,
                        help="")
       
    parser.add_argument("-l", "--location",
                        dest="location",
                        default="us-central1",
                        required=True,
                        help="")
       
    parser.add_argument("-pp", "--pipeline_params",
                        dest="pipeline_params",
                        default="{}",
                        required=True,
                        help="json string of pipelines params")
       
    parser.add_argument("-sa", "--service_account",
                        dest="service_account",
                        required=True,
                        help="")

    
    args = parser.parse_args()

    training_pipeline.run_pipeline(
        args.pipeline_name,
        args.run_name,
        args.project_id,
        args.pipeline_root_uri,
        args.pipeline_gs_path,
        args.location,
        json.loads(args.pipeline_params),
        args.service_account)