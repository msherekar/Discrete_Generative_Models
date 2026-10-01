# Lecture 7 slide-to-code map

Native links use stable IDs, so additional reading reveals preserve these destinations. Each displayed equation is followed by a literal reading and then its intuition. [All equation readings](EQUATION_READINGS.md).

| Code walkthrough | Slides | Implementation |
| --- | --- | --- |
| Log-space updates remain stable for large costs | [115–117](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u020_b0) | `ot_sbm_examples.sinkhorn, transport_example` |
| We can build this bridge with a few matrix operations | [184–186](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u032_b0) | `ot_sbm_examples.finite_bridge_example` |
| The rollout gives us a reverse-regression target | [266–268](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u046_b0) | `learned_bridges.dsb` |
| We can form both targets in a single training batch | [323–325](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u057_b0) | `learned_bridges.sf2m, sf2m_targets` |
| We can compute this small jump bridge exactly | [360–362](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u064_b0) | `discrete_learning.fit_ddsbm; ot_sbm_examples.ctmc_bridge_example` |
| We can enumerate the paths and watch IMF converge | [400–402](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u072_b0) | `discrete_learning.fit_csbm; ot_sbm_examples.discrete_imf_example` |
| Weighted denoising learns from completed candidates | [467–469](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u085_b0) | `learned_bridges.tr2d2` |
| Joint refinement combines energy and mass losses | [524–526](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u096_b0) | `learned_bridges.branch` |
| The update uses detached trajectory weights | [605–607](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u111_b0) | `learned_bridges.entangled` |

## Method sections

| Section | Native slide | Implementation |
| --- | --- | --- |
| Choosing an optimal transport | [2](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u001_b0) | `ot_sbm_examples` |
| Transport along a probability path | [71](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u012_b0) | `ot_sbm_examples` |
| Computing transport with Sinkhorn | [85](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u014_b0) | `ot_sbm_examples.sinkhorn, transport_example` |
| From a coupling to a Schrödinger bridge | [121](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u023_b0) | `ot_sbm_examples.finite_bridge_example` |
| Continuous bridges and stochastic control | [187](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u033_b0) | `ot_sbm_examples.finite_bridge_example` |
| Diffusion Schrödinger Bridge | [242](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u042_b0) | `learned_bridges.dsb` |
| Diffusion Schrödinger Bridge Matching | [272](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u048_b0) | `learned_bridges.dsbm` |
| Simulation-free score and flow matching | [306](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u054_b0) | `learned_bridges.sf2m, sf2m_targets` |
| Bridges on discrete spaces | [329](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u059_b0) | `discrete_learning.fit_ddsbm; ot_sbm_examples.ctmc_bridge_example` |
| Categorical Schrödinger Bridge Matching | [383](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u069_b0) | `discrete_learning.fit_csbm; ot_sbm_examples.discrete_imf_example` |
| Reward-guided discrete bridges with TR2-D2 | [412](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u076_b0) | `learned_bridges.tr2d2` |
| Branching Schrödinger Bridge Matching | [476](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u088_b0) | `learned_bridges.branch` |
| Entangled Schrödinger Bridge Matching | [536](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u100_b0) | `learned_bridges.entangled` |
| Putting the formulations together | [617](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u115_b0) | `ot_sbm_examples.finite_bridge_example` |
| Course wrap-up | [630](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_end01_b0) | `ot_sbm_examples.finite_bridge_example` |
