# Example scripts

The examples show how to configure, run, evaluate, and analyse models.
Most model examples use data from `gnm.defaults`.

## Choose an example

| Example | Use it when |
| --- | --- |
| [Parameter sweep](sweep_example.ipynb) | You want a complete binary and weighted fitting workflow. Start here. |
| [Sparse and coarse sweep](sparse_and_coarse_sweep_example.ipynb) | You want to search a broad parameter range, then refine the best region. |
| [Heterochronicity sweep](heterochronicity_example.ipynb) | You want to test how developmental timing changes binary network growth. |
| [Weighted sweep](weighted_sweep.ipynb) | You want to compare weighted models across `alpha` and `omega` values. |
| [Weighted increments](weighted_increments.ipynb) | You want to test distance regularisation during weight optimisation. |
| [Save and query experiments](experiment_saving_example.ipynb) | You want to store sweep results and select runs by parameter value. |
| [Weights & Biases logging](example_wandb_run.ipynb) | You want to track sweep parameters and results in Weights & Biases. |
| [Graph metrics](get_graph_metrics_example.ipynb) | You want to calculate batched graph metrics or compare metric runtimes with BCT. |
| [Performance benchmark](eval_bct_gnm_performance.py) | You want to collect runtime data across network sizes and connection counts. |
| [Performance plots](graph_model_performance.ipynb) | You want to plot runtime data from the GNM and BCT benchmarks. |

## Run an example

1. [Install the package](../getting-started.md).
2. Open a notebook from `docs/examples` in Jupyter.
3. Run its cells in order.

The performance plots require CSV files from the benchmark script.
The Weights & Biases example also requires an account and login.
