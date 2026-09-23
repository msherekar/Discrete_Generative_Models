# Lecture 7 · Short exercises and answers

These extensions use the same numerical examples as the lecture.

1. **Check the transport certificate.** With a=(0.6,0.4), b=(0.3,0.7), and C=[[1,9],[1,1]], verify that f=(0,−8), g=(1,9) is dual-feasible. Which entries of the optimal plan can carry mass?

   **Answer.** fᵢ+gⱼ≤Cᵢⱼ entrywise, with equality at (0,0), (0,1), (1,1). The remaining entry is strictly slack. The plan [[0.3,0.3],[0,0.4]] has cost 3.4, equal to a·f+b·g. Equality of primal and dual values certifies optimality.

2. **Perform one Sinkhorn cycle.** Set ε=2, K=exp(−C/ε), and v=(1,1). Compute u=a/(Kv), then v=b/(Kᵀu). Which constraint holds after the column update?

   **Answer.** u≈(0.971440,0.329744) and v≈(0.380128,3.320813). The resulting plan has column sums (0.3,0.7), while its row sums are approximately (0.259812,0.740188). The next row scaling corrects the row constraint and can disturb the columns again.

3. **Recover a path bridge.** For A=[[0.8,0.2],[0.3,0.7]], a=(0.5,0.5), b=(0.2,0.8), and two transitions, use g=(1,6). Compute h₂, h₁, h₀ and the transformed matrices.

   **Answer.** h₂=(1,6), h₁=(2,4.5), h₀=(2.5,3.75). Q₀=[[0.64,0.36],[0.16,0.84]] and Q₁=[[0.4,0.6],[1/15,14/15]]. The intermediate marginal is (0.4,0.6), followed by (0.2,0.8). A compatible initial potential is f=(0.4,4/15).

4. **Separate endpoints from paths.** In `discrete_imf_example`, print the two endpoint marginals before the first projection. Why is the initial path KL still positive?

   **Answer.** The initialization already uses a coupling with the correct marginals. Its conditional reference bridges can retain history after the endpoints are mixed. The Markov and reciprocal projections correct path dependence while preserving the endpoints.

5. **Check the score correction.** Substitute the displayed conditional velocity and score into vᶜ+εsᶜ/2. What happens if the factor 1/2 in the velocity is omitted?

   **Answer.** The correct expression simplifies to (x₁−x)/(1−t), the Brownian conditional forward drift. Omitting the factor changes the conditional probability flow and breaks this identity. The executable check uses multiple times, endpoints, and noise values.

6. **Change the Gaussian reference noise.** Compute c=(√(ε²+4)−ε)/2 for ε=0.1, 1, 10. What are the limiting couplings?

   **Answer.** c≈0.951249, 0.618034, 0.099020. As ε→0, c→1 and the equal-variance Gaussian coupling approaches the deterministic OT relation X₁=X₀+2. As ε→∞, c→0 and endpoint dependence weakens.

7. **Check a reward tilt.** Starting from (0.5,0.3,0.2), use reward log(1,2,4) and α=1. Does the same global tilt preserve every possible initial distribution?

   **Answer.** The terminal law is (0.5,0.6,0.8)/1.9. A deterministic initial state is preserved. A random initial law generally changes by h₀(x)/Z; preserving it instead requires normalization conditional on X₀. A separate terminal constraint requires the full Schrödinger system.

8. **Inspect search weighting.** Compare the actual recorded search proposal with the model’s unmasking likelihood. Why can substituting the latter change the objective?

   **Answer.** Search selection depends on node values and visit counts as well as policy predictions. Its sampling probability therefore differs from an ordinary model rollout. Importance weights require the law that actually produced the samples. The toy implementation records a full-support selection law and retains all rollouts; heuristic curation would need an additional correction to retain the exact identity.

9. **Check mass conservation.** Differentiate w=(1−t,0.6t,0.4t). Can small sampled mass penalties guarantee nonnegative weights at every time?

   **Answer.** g=(−1,0.6,0.4), so the total derivative is zero and every analytic weight is nonnegative on [0,1]. A soft loss evaluated at sampled times provides only an approximate constraint. Inspect the small residuals in the learned example and compare with the exact construction.

10. **Test an oversized cone step.** Use d=(3,4) and b=(2.24,−0.68). What is the largest bias-only step justified by the squared-distance argument?

    **Answer.** The bound is 2d·b/||b||²=8/5.48≈1.45985. At Δt=0.1, squared distance falls from 25 to 24.2548. A larger step than the bound can increase distance even though d·b>0. Independent noise adds a positive covariance contribution to expected squared distance.

11. **Read a benchmark carefully.** Does a better terminal FID prove a better path-space SB approximation? Does a higher target-hit percentage prove lower physical transition energy?

    **Answer.** Neither implication holds. Terminal FID ignores intermediate dependence and the reference path law. Target-hit probability and transition energy measure different properties. The lecture retains these distinctions in the CSBM, BranchSBM, and EntangledSBM results.
