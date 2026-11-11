# User guide

Use this guide after you [install the package](../getting-started.md).
It shows the main workflow and points to a working example for each task.

## Typical workflow

1. Load a distance matrix and one or more observed networks.
2. Select a [generative rule](../api-reference/generative-rules.md).
3. Define binary or weighted parameters with a [`SweepConfig`][gnm.fitting.SweepConfig].
4. Select [evaluation criteria](../api-reference/evaluation.md).
5. Run [`perform_sweep`][gnm.fitting.perform_sweep].
6. Compare the experiments, then save the results that you need.

Read [Understanding GNMs](../understanding-gnms/index.md) for the model concepts.
Use the [API reference](../api-reference/index.md) for class and function details.

## Choose an example

| Example | Use it when |
| --- | --- |
| [Parameter sweep](../examples/sweep_example.ipynb) | You want a complete binary and weighted fitting workflow. Start here. |
| [Sparse and coarse sweep](../examples/sparse_and_coarse_sweep_example.ipynb) | You want to search a broad parameter range, then refine the best region. |
| [Heterochronicity sweep](../examples/heterochronicity_example.ipynb) | You want to test how developmental timing changes binary network growth. |
| [Weighted sweep](../examples/weighted_sweep.ipynb) | You want to compare weighted models across `alpha` and `omega` values. |
| [Weighted increments](../examples/weighted_increments.ipynb) | You want to test distance regularisation during weight optimisation. |
| [Save and query experiments](../examples/experiment_saving_example.ipynb) | You want to store sweep results and select runs by parameter value. |
| [Weights & Biases logging](../examples/example_wandb_run.ipynb) | You want to track sweep parameters and results in Weights & Biases. |
| [Graph metrics](../examples/get_graph_metrics_example.ipynb) | You want to calculate batched graph metrics or compare metric runtimes with BCT. |
| [Performance benchmark](../examples/eval_bct_gnm_performance.py) | You want to collect runtime data across network sizes and connection counts. |
| [Performance plots](../examples/graph_model_performance.ipynb) | You want to plot runtime data from the GNM and BCT benchmarks. |

See the [examples overview](../examples/index.md) for run instructions.

## Package behaviour

The package validates matrix values, shapes, and types before it runs a model.
It also uses small interfaces for rules, evaluations, and optimisation criteria.
Read [Design philosophy](design-philosophy.md) to learn what these choices mean for your code.
