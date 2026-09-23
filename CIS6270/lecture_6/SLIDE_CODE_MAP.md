# Lecture 6 slide-to-code map

The updated deck adds spoken equation readings before intuition; see [Equation readings](EQUATION_READINGS.md). These links use stable native slide identifiers. Each code slide connects to the same functions used by the training runner.

| Code walkthrough | Slide | Implementation |
| --- | --- | --- |
| The map takes the state and both times as inputs | [Open](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u038_b0) | `common.finite_map`, `continuous.diagonal_loss`, `continuous.semigroup_loss`, `numerical_examples.train_scalar` |
| The two losses connect local motion to a longer move | [Open](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u039_b0) | `common.finite_map`, `continuous.diagonal_loss`, `continuous.semigroup_loss`, `numerical_examples.train_scalar` |
| A JVP computes that correction without forming a Jacobian | [Open](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u051_b0) | `continuous.meanflow_loss` |
| Softmax constrains the prediction before the map is applied | [Open](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u080_b0) | `categorical.categorical_map`, `categorical.composition_target`, `categorical.probability_kl` |
| The target combines probabilities from two different states | [Open](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u081_b0) | `categorical.categorical_map`, `categorical.composition_target`, `categorical.probability_kl` |
| We can differentiate through the posterior samples | [Open](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u122_b0) | `posterior.posterior_samples`, `posterior.posterior_value` |
| The stop-gradient placement is essential in this fine-tuning loss | [Open](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u139_b0) | `posterior.fine_tune_surrogate`, `posterior.train_reward_drift` |
| Keep the count, noise, and clock updates together | [Open](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u168_b0) | `expanding.bounded_counts`, `expanding.insert_tokens`, `expanding.local_clock` |
| Sampling and combining two coefficients is only a few lines | [Open](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u198_b0) | `stochastic.sample_coefficients`, `stochastic.chen_two`, `stochastic.train_stochastic` |
| The consistency update compares those two routes | [Open](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u199_b0) | `stochastic.sample_coefficients`, `stochastic.chen_two`, `stochastic.train_stochastic` |

## Method sections

| Lecture section | First slide | Code |
| --- | --- | --- |
| From velocities to finite motion | [Open](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u001_b0) | [continuous.py](continuous.py) |
| Learning the identities | [Open](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u025_b0) | [continuous.py](continuous.py) |
| Related finite-step formulations | [Open](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u041_b0) | [continuous.py](continuous.py) |
| MeanFlow and the direction of time | [Open](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u047_b0) | [continuous.py](continuous.py) |
| Flow maps in learned latent spaces | [Open](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u054_b0) | [posterior.py](posterior.py) |
| Flow Map Language Models | [Open](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u060_b0) | [categorical.py](categorical.py) |
| Categorical Flow Maps | [Open](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u083_b0) | [categorical.py](categorical.py) |
| Discrete Flow Maps and logit consistency | [Open](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u093_b0) | [categorical.py](categorical.py) |
| Diamond Maps and posterior lookahead | [Open](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u103_b0) | [posterior.py](posterior.py) |
| Meta Flow Maps | [Open](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u125_b0) | [posterior.py](posterior.py) |
| Expanding Flow Maps | [Open](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u143_b0) | [expanding.py](expanding.py) |
| Strong Stochastic Flow Maps | [Open](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u173_b0) | [stochastic.py](stochastic.py) |
| Connecting the formulations | [Open](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u206_b0) | [stochastic.py](stochastic.py) |

The exponential-flow code slides use the exact scalar drift `b(x)=x`; the generative runners use learned fields on a Gaussian mixture. The scalar training experiment remains in `numerical_examples.py`. The SSFM consistency code slide corresponds to `--ssfm-target paper`; the default follows the released code orientation, described in `SOURCES.md`.
