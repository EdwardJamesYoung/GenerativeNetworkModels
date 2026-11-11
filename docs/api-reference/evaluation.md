# gnm.evaluation

::: gnm.evaluation
    options:
      members: false
      show_root_heading: true
      show_root_full_path: false

## Evaluation 

::: gnm.evaluation.EvaluationCriterion
    options:
        members:
            - __init__
            - __str__
            - __call__
            - _pre_call
            - _evaluate


## Criterion types

::: gnm.evaluation.BinaryEvaluationCriterion
    options:
        members:
            - _pre_call

::: gnm.evaluation.WeightedEvaluationCriterion
    options:
        members:
            - _pre_call

::: gnm.evaluation.KSCriterion
    options:
        members:
            - _get_graph_statistics

::: gnm.evaluation.CorrelationCriterion
    options:
        members:
            - __init__
            - _get_graph_statistics 

## Composite Criteria

::: gnm.evaluation.CompositeCriterion
    options: 
        members:
            - __init__

::: gnm.evaluation.MaxCriteria
    options:
        members: []

::: gnm.evaluation.MeanCriteria
    options:
        members: []

::: gnm.evaluation.WeightedSumCriteria
    options:
        members: []

### Binary KS Criteria

::: gnm.evaluation.DegreeKS
    options:
        members: []

::: gnm.evaluation.ClusteringKS
    options:
        members: []

::: gnm.evaluation.BetweennessKS
    options:
        members: []

::: gnm.evaluation.EdgeLengthKS
    options:
        members:
            - __init__

### Weighted KS Criteria

::: gnm.evaluation.WeightedNodeStrengthKS
    options:
        members:
            - __init__

::: gnm.evaluation.WeightedBetweennessKS
    options:
        members:
            - __init__

::: gnm.evaluation.WeightedClusteringKS
    options:
        members: []

### Binary Correlation Criteria

::: gnm.evaluation.DegreeCorrelation
    options:
        members: 
            - __init__
    
::: gnm.evaluation.BinaryClusteringCorrelation
    options:
        members: 
            - __init__
    
::: gnm.evaluation.BinaryBetweennessCorrelation
    options:
        members:
            - __init__
