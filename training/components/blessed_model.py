from kfp.v2.dsl import (
    component,
    Input,
    Metrics,
)

@component()
def blessed_model_comp(
    model_1: Input[Metrics],
    model_2: Input[Metrics]
) -> int:

    if model_1.metadata['bt_f1_test_score']>model_2.metadata['svm_f1_test_score']:
        return 1
    else:
        return 2