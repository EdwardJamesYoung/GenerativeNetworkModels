# Weighted Generative Network Model Toolbox

[Read the full documentation](https://generative-network-models-toolbox.readthedocs.io/en/latest/)

## User guide

Use this guide after you [install the package](docs/getting-started.md). It shows
the main workflow and points to a working example for each task.

### Typical workflow

1. Load a distance matrix and one or more observed networks.
2. Select a [generative rule](docs/api-reference/generative-rules.md).
3. Define binary or weighted parameters with a `SweepConfig`.
4. Select [evaluation criteria](docs/api-reference/evaluation.md).
5. Run `perform_sweep`.
6. Compare the experiments, then save the results that you need.

Read [Understanding GNMs](docs/understanding-gnms/index.md) for the model
concepts. Use the [API reference](docs/api-reference/index.md) for class and
function details.

### Choose an example

| Example | Use it when |
| --- | --- |
| [Parameter sweep](docs/examples/sweep_example.ipynb) | You want a complete binary and weighted fitting workflow. Start here. |
| [Sparse and coarse sweep](docs/examples/sparse_and_coarse_sweep_example.ipynb) | You want to search a broad parameter range, then refine the best region. |
| [Heterochronicity sweep](docs/examples/heterochronicity_example.ipynb) | You want to test how developmental timing changes binary network growth. |
| [Weighted sweep](docs/examples/weighted_sweep.ipynb) | You want to compare weighted models across `alpha` and `omega` values. |
| [Weighted increments](docs/examples/weighted_increments.ipynb) | You want to test distance regularisation during weight optimisation. |
| [Save and query experiments](docs/examples/experiment_saving_example.ipynb) | You want to store sweep results and select runs by parameter value. |
| [Weights & Biases logging](docs/examples/example_wandb_run.ipynb) | You want to track sweep parameters and results in Weights & Biases. |
| [Graph metrics](docs/examples/get_graph_metrics_example.ipynb) | You want to calculate batched graph metrics or compare metric runtimes with BCT. |
| [Performance benchmark](docs/examples/eval_bct_gnm_performance.py) | You want to collect runtime data across network sizes and connection counts. |
| [Performance plots](docs/examples/graph_model_performance.ipynb) | You want to plot runtime data from the GNM and BCT benchmarks. |

See the [examples overview](docs/examples/index.md) for run instructions.

### Package behaviour

The package validates matrix values, shapes, and types before it runs a model.
It also uses small interfaces for rules, evaluations, and optimisation criteria.
The design choices behind this behaviour are described below.

## Design philosophy

The package uses validation, modular interfaces, and dataclasses. These choices
make model configurations explicit and reusable.

### Validation

The package checks inputs at runtime and reports invalid data before the model
starts.

- `binary_checks` requires symmetric matrices with a zero diagonal and binary
  values.
- `weighted_checks` requires symmetric matrices with a zero diagonal and
  non-negative values.
- Jaxtyping checks tensor shapes and dimensions.
- Typeguard checks function argument types.

These checks help you find input errors early.

### Modular interfaces

Each extension point has a common interface, so you can add a component without
changing the model workflow.

- Extend `GenerativeRule` to define a new wiring rule.
- Extend `EvaluationCriterion` to define a new fit measure.
- Extend `OptimisationCriterion` to define a new weighted objective.

Use the [parameter sweep example](docs/examples/sweep_example.ipynb) to see these
components together. Use the
[weighted increments example](docs/examples/weighted_increments.ipynb) to
compare weighted objectives.

### Dataclasses

Dataclasses keep related settings and results in one object.

- `BinaryGenerativeParameters` stores binary model settings.
- `WeightedGenerativeParameters` stores weighted model settings.
- `SweepConfig` defines a parameter sweep.
- `EvaluationResults` stores model scores.
- `RunHistory` stores network changes during a run.
- `Experiment` groups the configuration, model, history, and scores.

Use the
[experiment saving example](docs/examples/experiment_saving_example.ipynb) when
you need to store or query these objects.
