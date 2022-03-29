from kfp.v2.dsl import (
    component,
    Input,
    Output,
    Dataset,
    Artifact,
    HTML,
)


@component(
    packages_to_install=["pandas_profiling==3.1.0",
                         "dask[dataframe]==2021.12.0",
                         "gcsfs==2021.11.1"
                         ]
)
def gen_stats_comp(
        dataset: Input[Dataset],
        html_artifact: Output[HTML],
        json_artifact: Output[Artifact]
) -> None:

    from pandas_profiling import ProfileReport

    import dask.dataframe as dd
    df = dd.read_csv(dataset.uri+"/data_*.csv")
    df = df.compute()

    profile = ProfileReport(df, title="Pandas Profiling Report", explorative=True)

    print(html_artifact.path)
    profile.to_file(html_artifact.path)

    with open(json_artifact.path, 'w') as f:
        f.write(profile.to_json())

    return



@component(
    packages_to_install=["pandas_profiling==3.1.0",
                         "dask[dataframe]==2021.12.0",
                         "gcsfs==2021.11.1"
                         ]
)
def validate_stats_comp(
        json_artifact: Input[Artifact]
) -> None:

    import json

    with open(json_artifact.path, 'r') as f:
        data=f.read()

    # parse file
    obj = json.loads(data)
    print(obj['alerts'])

    return
  
