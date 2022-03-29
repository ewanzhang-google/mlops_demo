from kfp.v2.dsl import (
    component,
    Input,
    Output,
    Dataset,
    Artifact,
    HTML,
)

@component(
    packages_to_install=[
        "dask[dataframe]==2021.12.0",
        "gcsfs==2021.11.1"]
)
def data_split_comp(
    dataset: Input[Dataset],
    train_set: Output[Dataset],
    validation_set: Output[Dataset],
    test_set: Output[Dataset],
    train_data_size: float  = 0.8,
    validation_data_size: float = 0.1,
    test_data_size: float = 0.1,
    seed: int = 0
) -> None:
    
    if (train_data_size+validation_data_size+test_data_size!=1):
        raise ValueError('Train, Validation and Test data splits should add up to 1. Training:{}, Validation:{}, Test:{}'.format(train_data_size, validation_data_size, test_data_size)) 
    
    import numpy as np 
    import dask.dataframe as dd
    df = dd.read_csv(dataset.uri+"/data_*.csv")
    df = df.compute()
    
    np.random.seed(seed)
    perm = np.random.permutation(df.index)
    m = len(df.index)
    train_end = int(train_data_size * m)
    validate_end = int(validation_data_size * m) + train_end
    
    train = df.iloc[perm[:train_end]]
    
    validate = df.iloc[perm[train_end:validate_end]]
    test = df.iloc[perm[validate_end:]]
    
    train_set.uri = train_set.uri
    validation_set.uri = validation_set.uri
    test_set.uri = test_set.uri
    
    train.to_csv(train_set.uri, index=False)
    validate.to_csv(validation_set.uri, index=False)
    test.to_csv(test_set.uri, index=False)
    
