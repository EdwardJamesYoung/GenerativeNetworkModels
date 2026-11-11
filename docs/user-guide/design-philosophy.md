# Design philosophy

The package uses validation, modular interfaces, and dataclasses.
These choices make model configurations explicit and reusable.

## Validation

The package checks inputs at runtime.
It reports invalid data before the model starts.

- [`binary_checks`][gnm.utils.binary_checks] requires symmetric matrices with a zero diagonal and binary values.
- [`weighted_checks`][gnm.utils.weighted_checks] requires symmetric matrices with a zero diagonal and non-negative values.
- Jaxtyping checks tensor shapes and dimensions.
- Typeguard checks function argument types.

These checks help you find input errors early.

## Modular interfaces

Each extension point has a common interface.
You can add a component without changing the model workflow.

- Extend [`GenerativeRule`][gnm.generative_rules.GenerativeRule] to define a new wiring rule.
- Extend [`EvaluationCriterion`][gnm.evaluation.EvaluationCriterion] to define a new fit measure.
- Extend [`OptimisationCriterion`][gnm.weight_criteria.OptimisationCriterion] to define a new weighted objective.

Use the [parameter sweep example](../examples/sweep_example.ipynb) to see these components together.
Use the [weighted increments example](../examples/weighted_increments.ipynb) to compare weighted objectives.

## Dataclasses

Dataclasses keep related settings and results in one object.

- [`BinaryGenerativeParameters`][gnm.BinaryGenerativeParameters] stores binary model settings.
- [`WeightedGenerativeParameters`][gnm.WeightedGenerativeParameters] stores weighted model settings.
- [`SweepConfig`][gnm.fitting.SweepConfig] defines a parameter sweep.
- [`EvaluationResults`][gnm.fitting.EvaluationResults] stores model scores.
- [`RunHistory`][gnm.fitting.RunHistory] stores network changes during a run.
- [`Experiment`][gnm.fitting.Experiment] groups the configuration, model, history, and scores.

Use the [experiment saving example](../examples/experiment_saving_example.ipynb) when you need to store or query these objects.
