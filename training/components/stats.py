from kfp.v2.dsl import (
    component,
    Input,
    Output,
    Dataset,
    Artifact,
    HTML,
)

@component(
    packages_to_install=["tensorflow-data-validation==1.5.0", 
                         "apache-beam==2.35.0"
                        ]
)
def gen_stats_comp(
    dataset: Input[Dataset],
    stats_artifact: Output[Artifact],
    html_artifact: Output[HTML]
) -> None:
    """
    import tensorflow_data_validation as tfdv
    from apache_beam.options.pipeline_options import PipelineOptions

    train_stats = tfdv.generate_statistics_from_csv(
        data_location= dataset.uri+"/data_*.csv", 
        output_path= stats_artifact.path,
        pipeline_options = PipelineOptions(
            runner='DirectRunner',
            project='datauki-demo-airline',
            job_name='gen-stats-job',
            temp_location='gs://propensity_model/df/temp',
            region='europe-west4')
    )
    
    html_content = tfdv.utils.display_util.get_statistics_html(train_stats)
    
    
    print("in Stats")
    print(html_artifact.path)
    print(stats_artifact.path)
    with open(html_artifact.path, 'w') as f:
        f.write(html_content)
    """
    return
  
