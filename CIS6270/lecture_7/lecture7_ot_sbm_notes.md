# CIS 6270 · Lecture 7 · Teaching notes

[Native lecture](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit)



| Section | Slides |
|---|---|
| Choosing an optimal transport | 2–70 |
| Transport along a probability path | 71–84 |
| Computing transport with Sinkhorn | 85–120 |
| From a coupling to a Schrödinger bridge | 121–186 |
| Continuous bridges and stochastic control | 187–241 |
| Diffusion Schrödinger Bridge | 242–271 |
| Diffusion Schrödinger Bridge Matching | 272–305 |
| Simulation-free score and flow matching | 306–328 |
| Bridges on discrete spaces | 329–382 |
| Categorical Schrödinger Bridge Matching | 383–411 |
| Reward-guided discrete bridges with TR2-D2 | 412–475 |
| Branching Schrödinger Bridge Matching | 476–535 |
| Entangled Schrödinger Bridge Matching | 536–616 |
| Putting the formulations together | 617–629 |
| Course wrap-up | 630–651 |


## Slides 2–4 · Last time, we learned to move across an entire interval

[Completed slide](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u001_b2)



Flow maps let us learn a finite transition from one time to another. We also saw that continuous paths, discrete jumps, and stochastic paths can connect the same endpoint distributions.

Today, we can ask which connection we should learn. We will first assign a cost to moving probability, then ask how much we should change a reference stochastic process.







Primary source: [Computational Optimal Transport — Peyré and Cuturi](https://arxiv.org/abs/1803.00567v4)




## Slides 5–7 · The same endpoints can require different amounts of motion

[Completed slide](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u002_b2)





Both pairings move all the source mass to the target. The crossing pairing sends samples farther, so it has a larger squared-distance cost.



Course example. Two equally weighted points at 0 and 2 map to points at 1 and 3. Ordered pairing costs 1. Crossing pairing costs 5. These are expected squared distances.



Primary source: [Computational Optimal Transport — Peyré and Cuturi](https://arxiv.org/abs/1803.00567v4)




## Slides 8–14 · A transport map chooses a destination for every starting point

[Completed slide](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u003_b4)



Applying the map T to a source sample must produce the target distribution.



$$
T_{\#}\mu=\nu\quad\Longleftrightarrow\quad X\sim\mu\ \Rightarrow\ T(X)\sim\nu
$$



Spoken equation reading: The pushforward condition says that applying T to a random sample with law μ produces a sample with law ν; the map must transform the entire source distribution into the target distribution.



Monge chooses the admissible map with the smallest average transport cost c.



$$
\inf_{T:\,T_{\#}\mu=\nu}\ \int c(x,T(x))\,\mu(dx)
$$



Spoken equation reading: The Monge objective chooses, among maps that send μ to ν, the smallest average of the movement cost c from each source point x to its assigned destination T(x).





The function c tells us which moves are expensive. Choosing squared distance favors destinations close to their starting points.



μ is the source probability measure. ν is the target. T is a deterministic map. Monge posed the transport problem in 1781. A map may fail to exist when source atoms must split.



Primary source: [Computational Optimal Transport — Peyré and Cuturi](https://arxiv.org/abs/1803.00567v4)




## Slides 15–21 · A single source atom may need to split its probability

[Completed slide](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u004_b4)



Our source has masses 0.6 and 0.4, while the target needs masses 0.3 and 0.7.



$$
a=(0.6,0.4),\qquad b=(0.3,0.7)
$$



Spoken equation reading: The source probability vector a assigns masses 0.6 and 0.4 to its two states, while the target vector b requires masses 0.3 and 0.7.



A coupling assigns a nonnegative mass πᵢⱼ to each source and target pair.



$$
\Pi(a,b)=\{\pi\geq0:\ \pi\mathbf1=a,\ \pi^{\mathsf T}\mathbf1=b\}
$$



Spoken equation reading: The set Π(a,b) contains nonnegative matrices π whose row sums equal a and whose column sums equal b; each matrix entry specifies how much probability travels between one source and one target.





A deterministic map sends the entire mass 0.6 to one destination. A coupling can split it, which makes these endpoint constraints feasible.



The mass of any destination under a deterministic map must be a subset sum of 0.6 and 0.4. Neither 0.3 nor 0.7 is such a sum. Kantorovich permits probabilistic assignments. Rows correspond to source states throughout the lecture.



Primary source: [Computational Optimal Transport — Peyré and Cuturi](https://arxiv.org/abs/1803.00567v4)




## Slides 22–28 · We can now minimize the cost over all valid couplings

[Completed slide](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u005_b4)



Kantorovich minimizes the expected cost under the joint distribution π.



$$
\operatorname{OT}_{c}(\mu,\nu)=\inf_{\pi\in\Pi(\mu,\nu)}\int c(x,y)\,\pi(dx,dy)
$$



Spoken equation reading: The Kantorovich cost is the smallest average movement cost c(x,y) over joint distributions π with source marginal μ and target marginal ν.



For finite supports, the integral becomes a sum over entries of the cost matrix C.



$$
\min_{\pi\in\Pi(a,b)}\langle C,\pi\rangle,\qquad C_{ij}=\lVert x_i-y_j\rVert^2
$$



Spoken equation reading: For finite states, the objective sums each transported mass π times its corresponding matrix cost C, where C is the squared distance between the source and target locations.





The row constraints spend exactly the available source mass. The column constraints deliver exactly the required target mass.



Cᵢⱼ is the cost of transporting one unit from xᵢ to yⱼ. Every feasible matrix has total mass one. On a finite support, this is a linear program over a nonempty compact polytope, so an optimum exists.



Primary source: [Computational Optimal Transport — Peyré and Cuturi](https://arxiv.org/abs/1803.00567v4)




## Slides 29–35 · The constraints reduce our example to one unknown

[Completed slide](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u006_b4)



For source locations 0 and 2 and target locations 1 and 3, squared distance gives C.



$$
C=\begin{pmatrix}1&9\\1&1\end{pmatrix}
$$



Spoken equation reading: The cost matrix assigns cost one to three possible moves and cost nine to the move from the first source at zero to the second target at three.



Let r be the mass sent from the first source to the first target.



$$
\pi(r)=\begin{pmatrix}r&0.6-r\\0.3-r&0.1+r\end{pmatrix},\qquad 0\leq r\leq0.3
$$



Spoken equation reading: Choosing the first entry r fixes the other three transported masses through the marginal constraints; nonnegativity restricts r to the interval from zero to 0.3.





Once we choose r, the marginal constraints determine every other entry. Nonnegativity leaves only the interval from zero to 0.3.



Check each row and column explicitly. Row one is 0.6, row two 0.4, column one 0.3, column two 0.7. The bounds follow from all four entries being nonnegative.



Primary source: [Computational Optimal Transport — Peyré and Cuturi](https://arxiv.org/abs/1803.00567v4)




## Slides 36–42 · Increasing r removes the most expensive move

[Completed slide](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u007_b4)



Multiplying each transported mass by its cost gives a linear function of r.



$$
\langle C,\pi(r)\rangle=r+9(0.6-r)+(0.3-r)+(0.1+r)=5.8-8r
$$



Spoken equation reading: The transport cost is the sum of four mass-times-cost terms, which simplifies to 5.8 minus eight times r; increasing r therefore decreases the objective.



The largest feasible r gives the optimal coupling and total cost.



$$
r^*=0.3,\qquad \pi^*=\begin{pmatrix}0.3&0.3\\0&0.4\end{pmatrix},\qquad \operatorname{OT}=3.4
$$



Spoken equation reading: Setting r to its largest feasible value, 0.3, gives masses 0.3, 0.3, zero, and 0.4 in the coupling matrix and yields total transport cost 3.4.





Moving more mass along the cheap diagonal reduces the expensive first-to-second transfer. The target still needs 0.7, so some expensive transport remains.



This is a complete proof for the example because the objective decreases everywhere on the feasible interval. The companion linear program independently verifies cost 3.4.



Primary source: [Computational Optimal Transport — Peyré and Cuturi](https://arxiv.org/abs/1803.00567v4)




## Slides 43–49 · A second optimization can certify that this cost is minimal

[Completed slide](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u008_b4)



Introduce source prices f and target prices g in the Lagrangian.



$$
\mathcal L(\pi,f,g)=a^{\mathsf T}f+b^{\mathsf T}g+\sum_{ij}\pi_{ij}(C_{ij}-f_i-g_j)
$$



Spoken equation reading: The Lagrangian adds the source and target masses weighted by prices f and g, then sums each transported mass times its cost minus the two endpoint prices.



Minimizing over nonnegative π leaves the dual feasibility condition.



$$
\max_{f,g}\ a^{\mathsf T}f+b^{\mathsf T}g\quad\text{subject to}\quad f_i+g_j\leq C_{ij}
$$



Spoken equation reading: The dual maximizes the total mass-weighted source and target prices, subject to the condition that the two prices on every possible move sum to at most that move’s cost.





Every feasible pair of prices supplies a lower bound on the cost of every feasible transport. Matching that bound proves optimality.



If Cᵢⱼ−fᵢ−gⱼ is negative, sending πᵢⱼ to infinity makes the unconstrained Lagrangian infimum negative infinity. Otherwise its infimum is aᵀf+bᵀg. Finite-dimensional linear-program duality gives equality of optimal values. Complementary slackness says positive transported mass uses tight price constraints.



Primary source: [Computational Optimal Transport — Peyré and Cuturi](https://arxiv.org/abs/1803.00567v4)




## Slides 50–56 · Our example has an exact primal and dual certificate

[Completed slide](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u009_b4)



The source and target prices below satisfy every cost inequality.



$$
f=(0,-8),\quad g=(1,9),\quad f\mathbf1^{\mathsf T}+\mathbf1g^{\mathsf T}=\begin{pmatrix}1&9\\-7&1\end{pmatrix}\leq C
$$



Spoken equation reading: The chosen source prices are zero and minus eight, and the target prices are one and nine; their pairwise sums are no greater than the corresponding entries of C.



Their lower bound equals the cost of the coupling we already found.



$$
a^{\mathsf T}f+b^{\mathsf T}g=-3.2+0.3+6.3=3.4=\langle C,\pi^*\rangle
$$



Spoken equation reading: Weighting these prices by a and b gives 3.4, exactly the cost of the proposed coupling, so the feasible dual lower bound meets the feasible primal upper bound.





No feasible coupling can cost less than 3.4, and our coupling attains 3.4. The zero gap proves that it is optimal.



This proves weak duality numerically and certifies the solution without trusting the optimizer. The only strict price inequality is the unused second-to-first edge.



Primary source: [Computational Optimal Transport — Peyré and Cuturi](https://arxiv.org/abs/1803.00567v4)




## Slides 57–63 · Distance costs give the Wasserstein distances

[Completed slide](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u010_b4)



The p-Wasserstein distance takes the pth root of the minimum pth-power distance cost.



$$
W_p(\mu,\nu)=\left[\inf_{\pi\in\Pi(\mu,\nu)}\int\lVert x-y\rVert^p\,d\pi\right]^{1/p}
$$



Spoken equation reading: The p-Wasserstein distance minimizes the expected pth power of the distance between coupled samples, then takes the pth root to return to the units of distance.



On the real line, matching equal quantiles gives the optimal cost.



$$
W_p^p(\mu,\nu)=\int_0^1\left|F_\mu^{-1}(u)-F_\nu^{-1}(u)\right|^pdu
$$



Spoken equation reading: On the real line, the pth power of the Wasserstein distance averages the pth power of the gap between source and target quantiles at the same probability level u.





Wasserstein distance measures how far probability must move. Equal-quantile matching gives a direct one-dimensional algorithm.



Assume p≥1 and finite pth moments. F⁻¹ is the generalized quantile function. In our discrete example W₂=√3.4≈1.8439. For N(0,1) and N(2,1), equal quantiles differ by two and W₂²=4.



Primary source: [Computational Optimal Transport — Peyré and Cuturi](https://arxiv.org/abs/1803.00567v4)




## Slides 64–70 · The one-dimensional rule follows by uncrossing pairs

[Completed slide](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u011_b4)



For ordered points x₁<x₂ and y₁<y₂, compare crossing and ordered assignments.



$$
(x_1-y_2)^2+(x_2-y_1)^2-(x_1-y_1)^2-(x_2-y_2)^2=2(x_2-x_1)(y_2-y_1)>0
$$



Spoken equation reading: The cost of the two crossed assignments minus the cost of the ordered assignments equals twice the source gap times the target gap, which is positive when both pairs are strictly ordered.



Moving any shared positive mass δ from crossing edges to ordered edges lowers the cost.



$$
\Delta\operatorname{cost}=-2\delta(x_2-x_1)(y_2-y_1)<0
$$



Spoken equation reading: Moving a positive mass δ from crossed to ordered assignments changes the cost by minus twice δ times the two positive gaps, so the change is strictly negative.





An optimal one-dimensional squared-distance plan cannot retain crossing mass. Sorting and matching quantiles enforces this ordering.



The exchange preserves all marginals. For absolutely continuous μ on Euclidean space, Brenier’s theorem extends the squared-distance result to T=∇φ for a convex φ, under finite second moments. The scalar uncrossing argument proves the one-dimensional ordering, not the full multidimensional theorem.



Primary source: [Computational Optimal Transport — Peyré and Cuturi](https://arxiv.org/abs/1803.00567v4)




## Slides 71–77 · We can describe the same transport through a velocity field

[Completed slide](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u012_b4)



Benamou and Brenier minimize the kinetic action over densities ρₜ and velocities vₜ.



$$
W_2^2(\mu,\nu)=\inf_{\rho,v}\int_0^1\!\int\lVert v_t(x)\rVert^2\rho_t(x)\,dx\,dt
$$



Spoken equation reading: The squared Wasserstein distance minimizes, over density and velocity paths, the time integral of squared speed averaged under the current density ρ.



The continuity equation moves probability while preserving the required endpoints.



$$
\partial_t\rho_t+\nabla\!\cdot(\rho_tv_t)=0,\qquad \rho_0=\mu,\quad\rho_1=\nu
$$



Spoken equation reading: The continuity equation says that the time change in density plus the divergence of probability flux ρv is zero, while the initial and final densities are constrained to μ and ν.





This returns to the velocity fields from flow matching. Optimal transport now selects the field by minimizing its total kinetic action.



The formula holds in the weak measure-valued formulation for finite second moments. The display uses density notation for readability. Our W₂² convention has no factor one half. A one-half kinetic convention yields W₂²/2.



Primary source: [A computational fluid mechanics solution to the Monge–Kantorovich mass transfer problem — Benamou and Brenier](https://doi.org/10.1007/s002110050002)




## Slides 78–84 · Straight constant-speed motion attains the dynamic optimum

[Completed slide](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u013_b4)



For each path, Cauchy-Schwarz bounds displacement by its integrated squared speed.



$$
\lVert X_1-X_0\rVert^2=\left\lVert\int_0^1\dot X_t\,dt\right\rVert^2\leq\int_0^1\lVert\dot X_t\rVert^2dt
$$



Spoken equation reading: The squared endpoint displacement equals the squared integral of velocity and is bounded above by the integral of squared speed over the unit time interval.



An optimal endpoint coupling achieves equality with linear interpolation.



$$
(X_0,X_1)\sim\pi^*,\qquad X_t=(1-t)X_0+tX_1,\quad\dot X_t=X_1-X_0
$$



Spoken equation reading: Sampling endpoints from the optimal coupling and linearly interpolating between them gives a constant velocity equal to the endpoint difference, attaining equality in the speed bound.





Averaging the first inequality gives the lower bound. Interpolating an optimal coupling at constant speed attains it.



A rigorous measure-valued proof uses the superposition principle and the conditional mean velocity. For quadratic optimal couplings, displacement interpolation attains the bound. For a general nonoptimal coupling, conditional averaging can reduce action and changes the induced path coupling. The N(0,1) to N(2,1) example has vₜ=2 and action 4.



Primary source: [A computational fluid mechanics solution to the Monge–Kantorovich mass transfer problem — Benamou and Brenier](https://doi.org/10.1007/s002110050002)




## Slides 85–87 · Large transport matrices make the linear program expensive

[Completed slide](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u014_b2)



A batch with n source samples and m target samples has n times m possible assignments. We need a practical way to find a coupling while preserving both marginals.

Adding an entropy term gives a smooth positive coupling and a particularly simple alternating algorithm.







Primary source: [Sinkhorn Distances — Cuturi](https://arxiv.org/abs/1306.0895v1)




## Slides 88–90 · Sinkhorn uses repeated scaling to satisfy both marginals

[Completed slide](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u015_b2)





Cuturi’s formulation made entropy-regularized transport practical through matrix scaling. We can derive those updates directly from the objective.



Original paper header. NeurIPS 2013



Primary source: [Sinkhorn Distances — Cuturi](https://arxiv.org/abs/1306.0895v1)




## Slides 91–97 · Entropy spreads mass across feasible assignments

[Completed slide](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u016_b4)



Let ε>0 control the entropy penalty added to the transport cost.



$$
\min_{\pi\in\Pi(a,b)}\ \langle C,\pi\rangle+\varepsilon\sum_{ij}\pi_{ij}(\log\pi_{ij}-1)
$$



Spoken equation reading: The entropic objective adds the transport cost to ε times the sum of π times log π minus π, and minimizes that total over couplings with the required marginals.



Stationarity of the Lagrangian gives the positive kernel K and its two scaling vectors.



$$
\pi_{ij}=e^{f_i/\varepsilon}e^{-C_{ij}/\varepsilon}e^{g_j/\varepsilon}=u_iK_{ij}v_j
$$



Spoken equation reading: Every optimal coupling entry factors into a source scaling u, a kernel entry K equal to the exponential of negative cost divided by ε, and a target scaling v.





Low-cost pairs have larger kernel entries. The positive vectors u and v then adjust those preferences to meet the endpoint masses.



Differentiate with respect to πᵢⱼ to obtain Cᵢⱼ+ε log πᵢⱼ−fᵢ−gⱼ=0. Strict convexity of z log z gives uniqueness on a convex feasible set. Using KL(π∥a⊗b) changes this objective only by constants when marginals are fixed. Kᵢⱼ=exp(−Cᵢⱼ/ε).



Primary source: [Sinkhorn Distances — Cuturi](https://arxiv.org/abs/1306.0895v1)




## Slides 98–104 · Each marginal constraint gives one scaling update

[Completed slide](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u017_b4)



Matching the row sums gives u by dividing a by K times v.



$$
\pi\mathbf1=u\odot(Kv)=a\quad\Longrightarrow\quad u\leftarrow a\oslash(Kv)
$$



Spoken equation reading: The row sums of the scaled kernel are u multiplied elementwise by Kv; matching them to a gives the update that divides each source mass by the corresponding entry of Kv.



Matching the column sums then gives v by dividing b by K transpose times u.



$$
\pi^{\mathsf T}\mathbf1=v\odot(K^{\mathsf T}u)=b\quad\Longrightarrow\quad v\leftarrow b\oslash(K^{\mathsf T}u)
$$



Spoken equation reading: The column sums are v multiplied elementwise by the transpose of K times u; matching them to b gives the update that divides each target mass by that corresponding column quantity.





One update fixes the rows and the next fixes the columns. Repeating them reduces the remaining mismatch until both constraints hold.



⊙ and ⊘ mean entrywise multiplication and division. Start with v=1. Positive K and positive compatible marginals ensure convergence of the scaled coupling. The vectors have a gauge freedom, since multiplying u by a constant and dividing v by it preserves π.



Primary source: [Sinkhorn Distances — Cuturi](https://arxiv.org/abs/1306.0895v1)




## Slides 105–111 · We can calculate the first Sinkhorn iteration by hand

[Completed slide](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u018_b4)



For ε=2, exponentiating the negative half-cost gives the kernel.



$$
K=\begin{pmatrix}0.60653&0.01111\\0.60653&0.60653\end{pmatrix},\qquad v^{(0)}=(1,1)
$$



Spoken equation reading: With ε equal to two, each kernel entry is the exponential of negative half its cost, and the initial target scaling vector v is set to one in both coordinates.



Row scaling gives u, then column scaling gives the next v.



$$
u^{(1)}\approx(0.97144,0.32974),\qquad v^{(1)}\approx(0.38013,3.32081)
$$



Spoken equation reading: The first row normalization gives u approximately equal to (0.97144, 0.32974), and the following column normalization gives v approximately equal to (0.38013, 3.32081).





The costly first-to-second edge begins with very little mass. The large second-column scaling compensates because the target needs 0.7 there.



Worked values use the unrounded exponential kernel. Compute u₁=0.6/(0.60653+0.011109) and u₂=0.4/(2×0.60653). Then compute each column denominator from Kᵀu. The companion output supplies full precision and the next transport matrix.



Primary source: [Sinkhorn Distances — Cuturi](https://arxiv.org/abs/1306.0895v1)




## Slides 112–114 · The repeated scalings settle on both target marginals

[Completed slide](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u019_b2)





The marginal error falls as the updates alternate. The final coupling retains a small amount of mass on the edge that unregularized OT left unused.



Course calculation with a=(0.6,0.4), b=(0.3,0.7), C=[[1,9],[1,1]], ε=2. The plot reports actual row and column residuals from the included log-domain implementation.



Primary source: [Sinkhorn Distances — Cuturi](https://arxiv.org/abs/1306.0895v1)




## Slides 115–117 · Log-space updates remain stable for large costs

[Completed slide](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u020_b2)



Let’s implement Sinkhorn in log space so small ε does not underflow the kernel.

```python

log_K = -C / epsilon
log_v = torch.zeros_like(b)
for _ in range(n_iters):
    log_u = a.log() - torch.logsumexp(
        log_K + log_v[None, :], dim=1)
    log_v = b.log() - torch.logsumexp(
        log_K + log_u[:, None], dim=0)
pi = (log_u[:, None] + log_K + log_v).exp()

```



These are exactly the two scaling equations written with log-sum-exp. We still verify the final row and column sums.



The standalone implementation uses NumPy and SciPy logsumexp and accepts positive marginal support. The displayed PyTorch expression is equivalent. Dividing ε changes the cost-entropy tradeoff and can worsen numerical conditioning.



Primary source: [Sinkhorn Distances — Cuturi](https://arxiv.org/abs/1306.0895v1)




## Slides 118–120 · The kernel can describe which transitions are naturally likely

[Completed slide](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u021_b2)



So far, we built the kernel from a transport cost. We can also begin with the transition probabilities of an existing stochastic process.

Then the question becomes how little we must change that process to reach the endpoint distributions we want.







Primary source: [Sinkhorn Distances — Cuturi](https://arxiv.org/abs/1306.0895v1)




## Slides 121–127 · A path measure assigns probabilities to entire trajectories

[Completed slide](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u023_b4)



A Markov path probability multiplies its initial law and transition kernels.



$$
R(x_{0:N})=r_0(x_0)\prod_{k=0}^{N-1}K_k(x_k,x_{k+1})
$$



Spoken equation reading: The reference probability of a complete Markov trajectory equals the probability of its initial state multiplied by the transition probability for every consecutive pair of states.



Relative entropy averages the log likelihood ratio of two path measures.



$$
\operatorname{KL}(P\Vert R)=\mathbb E_P\!\left[\log\frac{dP}{dR}\right]
$$



Spoken equation reading: The path-space KL is the expectation under P of the logarithm of P’s density relative to R, so trajectories are weighted according to P when comparing the two path laws.





Two processes can reach the same final distribution while assigning very different probabilities to the routes taken along the way.



x₀:ₙ is a complete state sequence. R is the reference path law, r₀ its initial law, and Kₖ its transition probabilities. dP/dR is the Radon-Nikodym likelihood ratio. Require P≪R for a finite KL. In a finite path space it is simply P(path)/R(path).



Primary source: [Foundations of Schrödinger Bridges for Generative Modeling — Tang](https://arxiv.org/abs/2603.18992v1)




## Slides 128–134 · The bridge changes the reference as little as possible

[Completed slide](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u024_b4)



Fix both endpoint marginals and minimize relative entropy on path space.



$$
P^*=\arg\min_{P:\,P_0=\mu,\,P_1=\nu}\operatorname{KL}(P\Vert R)
$$



Spoken equation reading: The Schrödinger bridge P star minimizes KL relative to the reference path law R over all path laws whose initial marginal is μ and whose terminal marginal is ν.



The KL chain rule separates the endpoint cost from the conditional path cost.



$$
\operatorname{KL}(P\Vert R)=\operatorname{KL}(\pi\Vert R_{01})+\mathbb E_{\pi}\operatorname{KL}(P^{xy}\Vert R^{xy})
$$



Spoken equation reading: The total path KL equals the KL between endpoint couplings plus the average, over the chosen endpoint coupling π, of the KL between paths conditioned on those endpoints.





We can change how often each endpoint pair occurs while retaining the reference trajectories conditioned on that pair.



Pᵡʸ is the conditional path law given X₀=x and X₁=y. R₀₁ is the joint endpoint law under the reference. Factor dP/dR into dπ/dR₀₁ times dPᵡʸ/dRᵡʸ, take the logarithm, and integrate. This proves the decomposition. Existence requires feasible endpoint constraints compatible with reference support and finite entropy. See Léonard 2014 as well as the guide.



Primary source: [Foundations of Schrödinger Bridges for Generative Modeling — Tang](https://arxiv.org/abs/2603.18992v1)




## Slides 135–141 · The static bridge selects the optimal endpoint coupling

[Completed slide](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u025_b4)



Nonnegativity of conditional KL makes the reference conditional bridges optimal.



$$
\pi^*=\arg\min_{\pi\in\Pi(\mu,\nu)}\operatorname{KL}(\pi\Vert R_{01}),\qquad P^*(d\omega)=\int R^{xy}(d\omega)\,\pi^*(dx,dy)
$$



Spoken equation reading: The optimal endpoint coupling minimizes KL relative to the reference endpoint law, and the optimal path law mixes the reference’s endpoint-conditioned bridges using that coupling.



The optimal coupling rescales the reference endpoint law by two positive potentials.



$$
\pi^*(dx,dy)=f(x)g(y)R_{01}(dx,dy)
$$



Spoken equation reading: The optimal joint endpoint law equals the reference joint law multiplied by a source potential f(x) and a terminal potential g(y), which adjust the two marginals.





This is the same scaling structure that appeared in Sinkhorn. The reference now also specifies the paths between the paired endpoints.



f and g are Schrödinger scaling potentials, unique up to reciprocal scalar multiplication under positivity assumptions. The path KL is strictly convex on its convex feasible set, yielding a unique finite-entropy bridge. Its conditional bridges equal Rᵡʸ almost surely. Historical sources include Schrödinger 1931, Fortet 1940, Beurling 1960 and Jamison 1974, discussed in Léonard’s survey.



Primary source: [Foundations of Schrödinger Bridges for Generative Modeling — Tang](https://arxiv.org/abs/2603.18992v1)




## Slides 142–148 · A Brownian reference recovers entropy-regularized transport

[Completed slide](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u026_b4)



Brownian motion with noise variance ε has a Gaussian endpoint transition density.



$$
K_{01}(x,y)=(2\pi\varepsilon)^{-d/2}\exp\!\left[-\frac{\lVert y-x\rVert^2}{2\varepsilon}\right]
$$



Spoken equation reading: The Brownian transition density is an isotropic Gaussian centered at x with covariance ε times the identity, so reaching y incurs the exponential of negative squared displacement divided by twice ε.



Substituting this density into the static KL gives squared-distance transport plus entropy.



$$
\varepsilon\operatorname{KL}(\pi\Vert R_{01})=\tfrac12\mathbb E_\pi\lVert X_1-X_0\rVert^2-\varepsilon H(\pi)+\text{constant}
$$



Spoken equation reading: For fixed endpoint marginals, ε times the endpoint KL equals half the expected squared displacement minus ε times the coupling entropy, plus a constant independent of the coupling.





Brownian motion makes long moves less likely. The bridge balances that distance penalty against the entropy of its endpoint assignments.



Assume densities and fixed first marginal with finite required entropy terms. H(π)=−∫π log π is differential entropy relative to Lebesgue measure. The constant includes the fixed source density and Gaussian normalization. Our earlier cost was squared distance without one half, so its regularization parameter is 2ε. Léonard describes rigorous small-noise limits. Simply changing a generic reference does not always produce Euclidean OT.



Primary source: [Foundations of Schrödinger Bridges for Generative Modeling — Tang](https://arxiv.org/abs/2603.18992v1)




## Slides 149–155 · A two-state chain gives a bridge we can compute exactly

[Completed slide](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u027_b4)



Let R use the same transition matrix A at two consecutive steps.



$$
a=(0.5,0.5),\quad b=(0.2,0.8),\quad A=\begin{pmatrix}0.8&0.2\\0.3&0.7\end{pmatrix}
$$



Spoken equation reading: The reference starts with equal probabilities, the desired terminal probabilities are 0.2 and 0.8, and each row of A gives one step’s destination probabilities from its corresponding source state.



The reference joint endpoint probabilities equal the initial masses times A squared.



$$
R_{02}=\operatorname{diag}(a)A^2=\begin{pmatrix}0.35&0.15\\0.225&0.275\end{pmatrix}
$$



Spoken equation reading: Multiplying the two-step transition matrix A squared by the initial masses gives the reference joint endpoint probabilities 0.35, 0.15, 0.225, and 0.275.





The reference ends at probabilities 0.575 and 0.425. The bridge must reach 0.2 and 0.8 while retaining the source probabilities 0.5 and 0.5.



The discrete time index is 0,1,2 in this example, so its terminal time is 2. Multiply A by itself to obtain [[0.7,0.3],[0.45,0.55]], then multiply each row by 0.5. This avoids conflating a transition matrix with a joint coupling.



Primary source: [Foundations of Schrödinger Bridges for Generative Modeling — Tang](https://arxiv.org/abs/2603.18992v1)




## Slides 156–162 · Sinkhorn rescales the endpoint probabilities of the chain

[Completed slide](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u028_b4)



Scaling the reference endpoint matrix to a and b gives π star.



$$
\pi^*=\begin{pmatrix}0.14&0.36\\0.06&0.44\end{pmatrix}
$$



Spoken equation reading: The optimal endpoint coupling assigns joint probabilities 0.14 and 0.36 from the first source and 0.06 and 0.44 from the second, matching both prescribed marginals.



One convenient choice of endpoint potentials gives the required scaling.



$$
f=(0.4,\,4/15),\qquad g=(1,6),\qquad \pi^*_{ij}=f_i(R_{02})_{ij}g_j
$$



Spoken equation reading: Multiplying each reference endpoint probability by its source factor f and destination factor g gives the optimal coupling, with g favoring the second destination by a factor of six.





The target potential favors state two by a factor of six. The source potential compensates so the initial distribution remains unchanged.



Verify all four entries explicitly. The potentials returned by a numerical solver can differ by a scalar gauge. Here fᵢ=1/(A²g)ᵢ because the reference starts from the prescribed source law.



Primary source: [Foundations of Schrödinger Bridges for Generative Modeling — Tang](https://arxiv.org/abs/2603.18992v1)




## Slides 163–169 · The terminal potential can be propagated backward

[Completed slide](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u029_b4)



Let hₖ be the expected terminal potential given the current state.



$$
h_k(x)=\mathbb E_R[g(X_N)\mid X_k=x],\qquad h_k=A_kh_{k+1},\quad h_N=g
$$



Spoken equation reading: The potential h at state x and step k is the reference expectation of the final potential g given that current state; it is computed backward by multiplying the next potential by A.



Doob’s transform uses the ratio of future and current potentials to modify each transition.



$$
Q_k(x,y)=A_k(x,y)\frac{h_{k+1}(y)}{h_k(x)}
$$



Spoken equation reading: The transformed transition Q from x to y equals the reference transition A multiplied by the next potential at y divided by the current potential at x.





A move becomes more likely when it leads to states from which the desired endpoint is easier to reach.



hₖ is a positive backward potential. In the two-step example h₂=(1,6), h₁=(2,4.5), h₀=(2.5,3.75). This construction needs the endpoint potentials from both constraints. Choosing g arbitrarily generally reaches a different target.



Primary source: [Foundations of Schrödinger Bridges for Generative Modeling — Tang](https://arxiv.org/abs/2603.18992v1)




## Slides 170–176 · The backward recursion normalizes every transition

[Completed slide](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u030_b4)



Summing the transformed transitions gives one by the recursion for h.



$$
\sum_yQ_k(x,y)=\frac{\sum_yA_k(x,y)h_{k+1}(y)}{h_k(x)}=1
$$



Spoken equation reading: Summing Q over all destinations divides the reference-weighted sum of next-step potentials by the current potential; the backward recursion makes this ratio exactly one.



Multiplying transitions cancels all intermediate potentials.



$$
\prod_k\frac{Q_k(x_k,x_{k+1})}{A_k(x_k,x_{k+1})}=\frac{g(x_N)}{h_0(x_0)}
$$



Spoken equation reading: Multiplying the transformed-to-reference transition ratios along a path cancels every intermediate potential, leaving only the final potential g divided by the initial future potential h.





The local transition changes produce exactly an endpoint reweighting of the whole path. This proves that the transformed chain retains the reference conditional bridges.



With initial law a and reference initial law a, f=1/h₀, so the likelihood ratio is f(X₀)g(Xₙ). If initial laws differ, multiply by a(X₀)/r₀(X₀). Positivity of h is required on the retained support.



Primary source: [Foundations of Schrödinger Bridges for Generative Modeling — Tang](https://arxiv.org/abs/2603.18992v1)




## Slides 177–183 · The transformed chain reaches the requested endpoint exactly

[Completed slide](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u031_b4)



Substituting the potentials gives the two transition matrices.



$$
Q_0=\begin{pmatrix}0.64&0.36\\0.16&0.84\end{pmatrix},\qquad Q_1=\begin{pmatrix}0.4&0.6\\1/15&14/15\end{pmatrix}
$$



Spoken equation reading: The first transformed transition uses rows (0.64, 0.36) and (0.16, 0.84), while the second uses rows (0.4, 0.6) and (one fifteenth, fourteen fifteenths).



Successive probability updates give the intermediate and terminal distributions.



$$
(0.5,0.5)Q_0=(0.4,0.6),\qquad (0.4,0.6)Q_1=(0.2,0.8)
$$



Spoken equation reading: Multiplying the initial probability vector by the first transition gives (0.4, 0.6), and multiplying again by the second gives the required terminal probabilities (0.2, 0.8).





The transitions increasingly favor state two as the deadline approaches. Both endpoints agree exactly with the constraints.



For example Q₀(0,1)=0.2×4.5/2.5=0.36 and Q₁(0,1)=0.2×6/2=0.6. Enumerating all eight paths gives KL(P*∥R)=0.3143842895, identical to KL(π*∥R₀₂).



Primary source: [Foundations of Schrödinger Bridges for Generative Modeling — Tang](https://arxiv.org/abs/2603.18992v1)




## Slides 184–186 · We can build this bridge with a few matrix operations

[Completed slide](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u032_b2)



Let’s use the Sinkhorn solution to construct the backward potentials and the two transition matrices.

```python

joint = a[:, None] * np.linalg.matrix_power(A, N)
pi, f, g, _ = sinkhorn(a, b, np.log(joint))
h = [np.linalg.matrix_power(A, N-k) @ g
     for k in range(N+1)]
Q = [A * h[k+1][None, :] / h[k][:, None]
     for k in range(N)]
p = a.copy()
for q in Q: p = p @ q

```



The final vector p must equal b. Checking it confirms that the local transitions realize the endpoint scaling.



Run finite_bridge_example in ot_sbm_examples.py. It also enumerates the path law and checks the KL chain rule numerically. Here N=2. The exact matrix-exponential analogue for continuous-time chains appears later.



Primary source: [Foundations of Schrödinger Bridges for Generative Modeling — Tang](https://arxiv.org/abs/2603.18992v1)




## Slides 187–193 · For a diffusion, we can change the drift and keep the noise

[Completed slide](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u033_b4)



The reference drift b and noise scale √ε define the original stochastic paths.



$$
R:\ dX_t=b_t(X_t)dt+\sqrt\varepsilon\,dW_t
$$



Spoken equation reading: Under the reference law R, the state increment equals drift b times the time increment plus a Brownian increment scaled by the square root of ε.



A control u changes the drift and defines the path measure P superscript u.



$$
P^u:\ dX_t=[b_t(X_t)+u_t(X_t)]dt+\sqrt\varepsilon\,dW_t^u
$$



Spoken equation reading: Under the controlled law, the increment adds control u to the reference drift b while retaining the same noise scale; the resulting trajectories define the new path law P.





The control changes the likelihood of trajectories while preserving their local noise scale. This lets us compare the path measures through a likelihood ratio.



W and Wᵘ are Brownian motions under their respective measures. Both laws use the same initial law here. Assume adapted controls, well-posed dynamics and sufficient exponential integrability, for example Novikov’s condition. Different nondegenerate diffusion covariances can make continuous path laws singular.



Primary source: [Foundations of Schrödinger Bridges for Generative Modeling — Tang](https://arxiv.org/abs/2603.18992v1)




## Slides 194–200 · Girsanov turns drift changes into a path likelihood ratio

[Completed slide](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u034_b4)



Under the controlled law, the log ratio contains a martingale and control energy.



$$
\log\frac{dP^u}{dR}=\frac1{\sqrt\varepsilon}\int_0^1u_t\cdot dW_t^u+\frac1{2\varepsilon}\int_0^1\lVert u_t\rVert^2dt
$$



Spoken equation reading: With matching initial laws, the controlled-to-reference log likelihood ratio is a noise-scaled stochastic integral of u plus one over twice ε times the integrated squared control.



Taking the controlled expectation removes the martingale term.



$$
\operatorname{KL}(P^u\Vert R)=\frac1{2\varepsilon}\mathbb E_{P^u}\int_0^1\lVert u_t\rVert^2dt
$$



Spoken equation reading: The KL from the controlled process to the reference equals one over twice ε times the controlled expectation of integrated squared control, because the martingale term has zero expectation.





Relative entropy becomes the average energy needed to change the reference drift. The diffusion bridge is therefore a stochastic control problem with endpoint constraints.



Starting under R, log(dPᵘ/dR)=∫u/√ε·dW−(1/2ε)∫||u||²dt. Substitute dW=dWᵘ+u/√ε dt to obtain the displayed positive-energy expression. Its stochastic integral has zero expectation under the stated integrability assumptions. Add KL(P₀∥R₀) if the initial laws differ.



Primary source: [Foundations of Schrödinger Bridges for Generative Modeling — Tang](https://arxiv.org/abs/2603.18992v1)




## Slides 201–207 · The backward potential determines the continuous control

[Completed slide](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u035_b4)



The conditional expectation h satisfies the backward Kolmogorov equation.



$$
h_t(x)=\mathbb E_R[g(X_1)\mid X_t=x],\qquad \partial_th+b\cdot\nabla h+\tfrac\varepsilon2\Delta h=0
$$



Spoken equation reading: The future potential h averages the terminal potential under the reference conditioned on the current state, and its time derivative plus reference drift and diffusion terms sums to zero.



The continuous Doob transform adds the gradient of log h to the drift.



$$
u_t^*(x)=\varepsilon\nabla\log h_t(x),\qquad dX_t=[b_t+\varepsilon\nabla\log h_t](X_t)dt+\sqrt\varepsilon\,dW_t
$$



Spoken equation reading: The optimal added control is ε times the spatial gradient of log h, so the bridge drift is the reference drift plus this control and the Brownian noise scale is unchanged.





The backward potential measures future endpoint preference. Its spatial gradient gives the direction that increases that preference most rapidly.



Apply Itô’s formula to hₜ(Xₜ). The backward PDE removes its drift, proving the conditional-expectation representation. The transformed generator is h⁻¹L(hϕ)−h⁻¹ϕLh=Lϕ+ε∇log h·∇ϕ. For fixed two-endpoint SB, g must solve the Schrödinger system together with f.



Primary source: [Foundations of Schrödinger Bridges for Generative Modeling — Tang](https://arxiv.org/abs/2603.18992v1)




## Slides 208–214 · Taking the logarithm gives an optimal-control equation

[Completed slide](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u036_b4)



Bellman’s equation minimizes the local control cost plus the change in value V.



$$
\partial_tV+b\cdot\nabla V+\tfrac\varepsilon2\Delta V+\min_u\{u\cdot\nabla V+\tfrac12\lVert u\rVert^2\}=0
$$



Spoken equation reading: Bellman’s equation adds the time, reference-drift, and diffusion changes in value V to the minimum over controls of control–value alignment plus half the squared control norm.



Completing the square gives u star and the Hopf-Cole transform linearizes the equation.



$$
u^*=-\nabla V,\qquad V=-\varepsilon\log h\quad\Longrightarrow\quad\partial_th+b\cdot\nabla h+\tfrac\varepsilon2\Delta h=0
$$



Spoken equation reading: The minimizing control is minus the gradient of V, and substituting V equal to minus ε times log h converts the nonlinear control equation into the linear backward equation for h.





The nonlinear value equation and the linear backward potential equation describe the same optimal control.



Complete the square ||u+∇V||²/2−||∇V||²/2. For V=−ε log h, ∇V=−ε∇h/h and ΔV=−εΔh/h+ε||∇h||²/h². The gradient-squared terms cancel after substitution. V₁=−ε log g is a terminal potential, with g calibrated to both endpoints for an exact bridge.



Primary source: [Foundations of Schrödinger Bridges for Generative Modeling — Tang](https://arxiv.org/abs/2603.18992v1)




## Slides 215–221 · Two potentials recover every intermediate marginal

[Completed slide](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u037_b4)



Propagate the initial scaling forward and the terminal scaling backward.



$$
\widehat h_t(y)=\int f(x)r_0(x)K_{0t}(x,y)dx,\qquad h_t(y)=\int K_{t1}(y,z)g(z)dz
$$



Spoken equation reading: The forward potential averages the initial density weighted by f through the reference transition to time t, while the backward potential averages g through the transition from t to the endpoint.



Their product gives the bridge density at time t.



$$
\rho_t(y)=\widehat h_t(y)h_t(y),\qquad \rho_0=\mu,\quad\rho_1=\nu
$$



Spoken equation reading: The bridge density at an intermediate state is the product of its forward and backward potentials, with the two potentials chosen so that the endpoint densities equal μ and ν.





One potential carries probability from the source and the other carries information from the target. Their product gives the density along the bridge.



The forward potential solves ∂ₜĥ=L* ĥ, while h solves ∂ₜh=−Lh. Differentiate their product and use the product rule to verify Fokker-Planck with drift b+ε∇log h. In finite spaces, replace the integrals by sums.



Primary source: [Foundations of Schrödinger Bridges for Generative Modeling — Tang](https://arxiv.org/abs/2603.18992v1)




## Slides 222–228 · A Brownian bridge gives an explicit training path

[Completed slide](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u038_b4)



Given both endpoints, the intermediate state is a Gaussian around the straight line.



$$
X_t\mid x_0,x_1\sim\mathcal N((1-t)x_0+tx_1,\ \varepsilon t(1-t)I)
$$



Spoken equation reading: Conditioned on both endpoints, the Brownian bridge state at an interior time t is Gaussian with the linearly interpolated mean and covariance ε times t times one minus t.



The conditional drift points toward the endpoint, scaled by the remaining time.



$$
\beta_t(x\mid x_1)=\frac{x_1-x}{1-t}
$$



Spoken equation reading: The conditional Brownian bridge drift equals the remaining displacement from x to the terminal point divided by the remaining time one minus t.





The conditional paths fluctuate between fixed endpoints. Their variance vanishes at both ends, while the drift tightens the terminal constraint.



For Brownian transition Kₜ₁(x,x₁), ε∇ₓlog K=(x₁−x)/(1−t). A single sampled intermediate state needs one Gaussian draw. Independent Gaussian draws at several times do not constitute one coherent Brownian bridge trajectory. Train away from singular endpoints or use appropriate loss weights.



Primary source: [Foundations of Schrödinger Bridges for Generative Modeling — Tang](https://arxiv.org/abs/2603.18992v1)




## Slides 229–235 · Bridge noise changes the Gaussian interior

[Completed slide](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u039_b4)



For unit-variance Gaussian endpoints and ε=1, the covariance has a closed form.



$$
\mu=\mathcal N(0,1),\quad\nu=\mathcal N(2,1),\qquad c^*=\frac{\sqrt{\varepsilon^2+4}-\varepsilon}{2}\approx0.61803
$$



Spoken equation reading: For unit-variance Gaussian endpoints with means zero and two, the optimal endpoint covariance is half of the square root of ε squared plus four minus ε, approximately 0.61803 when ε is one.



At the midpoint, conditional interpolation and Brownian variance add together.



$$
\mathbb E[X_{1/2}]=1,\qquad \operatorname{Var}(X_{1/2})=\tfrac12+\tfrac12c^*+\tfrac\varepsilon4\approx1.05902
$$



Spoken equation reading: The midpoint mean is one, while its variance combines one half from the endpoint variances, one half of their covariance, and ε divided by four from the conditional Brownian bridge.





The endpoints keep variance one, while the stochastic bridge becomes slightly broader in the middle. Zero-noise optimal transport keeps variance one throughout.



For Gaussian coupling covariance c, minimize (2−2c)/(2ε)−(1/2)log(1−c²) up to constants. Differentiation gives c²+εc−1=0, and the positive feasible root gives c*. For general endpoint standard deviations σ₀,σ₁, c*=(√(ε²+4σ₀²σ₁²)−ε)/2. The mean displacement contributes a fixed cost.



Primary source: [Foundations of Schrödinger Bridges for Generative Modeling — Tang](https://arxiv.org/abs/2603.18992v1)




## Slides 236–238 · The same endpoints can have different noisy paths

[Completed slide](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u040_b2)





All curves start and end at the prescribed Gaussian distributions. Increasing the reference noise broadens the intermediate bridge and reduces endpoint correlation.



Course calculation from the analytic Gaussian bridge. The plot compares ε=0, 0.2, 1 and 3. The ε=0 curve is displacement interpolation. The inset shows midpoint variance and endpoint covariance, not learned benchmark results.



Primary source: [Foundations of Schrödinger Bridges for Generative Modeling — Tang](https://arxiv.org/abs/2603.18992v1)




## Slides 239–241 · The remaining task is to learn these bridges from samples

[Completed slide](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u041_b2)



The small examples gave us transition matrices or analytic Gaussian formulas. For images, graphs, and high-dimensional observations, the required potentials and conditional expectations are much harder to compute.

We can now use the bridge structure to build training algorithms. We will first alternate endpoint fitting, then learn Markovian projections of conditional bridges.







Primary source: [Foundations of Schrödinger Bridges for Generative Modeling — Tang](https://arxiv.org/abs/2603.18992v1)




## Slides 242–244 · DSB alternates between the two endpoints

[Completed slide](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u042_b2)





De Bortoli, Thornton, Heng, and Doucet use learned diffusion time reversals to approximate iterative proportional fitting on path space.



Original paper header. NeurIPS 2021



Primary source: [Diffusion Schrödinger Bridge with Applications to Score-Based Generative Modeling](https://arxiv.org/abs/2106.01357v5)




## Slides 245–251 · Iterative proportional fitting corrects one endpoint at a time

[Completed slide](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u043_b4)



A terminal projection replaces the terminal marginal while retaining the conditional paths.



$$
P^{2n+1}(d\omega)=\frac{\nu(X_1)}{P^{2n}_1(X_1)}P^{2n}(d\omega)
$$



Spoken equation reading: The terminal projection multiplies each current path probability by the desired terminal density ν divided by the current terminal density, evaluated at that path’s endpoint.



An initial projection then restores the source marginal.



$$
P^{2n+2}(d\omega)=\frac{\mu(X_0)}{P^{2n+1}_0(X_0)}P^{2n+1}(d\omega)
$$



Spoken equation reading: The next projection multiplies each path probability by the desired initial density μ divided by the current initial density, evaluated at the starting point, to restore the source marginal.





These are the path-space equivalents of alternating column and row scaling. Each correction can disturb the other endpoint, so we repeat both.



Ratios denote Radon-Nikodym derivatives when densities are unavailable. Initialize P⁰=R. The KL chain rule proves each projection, since the constrained marginal is fixed and the conditional KL is minimized at zero. Under positivity and finite entropy assumptions, exact IPF converges. Neural and time-discretization errors need separate control.



Primary source: [Diffusion Schrödinger Bridge with Applications to Score-Based Generative Modeling](https://arxiv.org/abs/2106.01357v5)




## Slides 252–258 · Time reversal makes each endpoint correction useful

[Completed slide](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u044_b4)



Bayes’ rule gives the reverse transition of a discrete-time reference chain.



$$
K_k^{\leftarrow}(y,x)=\frac{p_k(x)K_k(x,y)}{p_{k+1}(y)}
$$



Spoken equation reading: The reverse transition from y to x equals the forward joint probability of x then y divided by the marginal probability of y at the next step.



For a diffusion, reverse time s=1−t changes the drift and adds its density score.



$$
dY_s=[-b_{1-s}(Y_s)+\varepsilon\nabla\log p_{1-s}(Y_s)]ds+\sqrt\varepsilon\,d\overline W_s
$$



Spoken equation reading: In reverse time, the diffusion drift is minus the forward drift plus ε times the forward density score, while the Brownian noise scale remains the square root of ε.





Once we learn the reverse dynamics, we can start them from the desired terminal distribution. Repeating in both directions implements the endpoint projections.



Yₛ=X₁₋ₛ uses increasing reverse time s. The displayed sign is for this convention. The diffusion coefficient is constant and isotropic. Resetting the reverse initial marginal realizes the terminal projection because the conditional reverse paths stay unchanged.



Primary source: [Diffusion Schrödinger Bridge with Applications to Score-Based Generative Modeling](https://arxiv.org/abs/2106.01357v5)




## Slides 259–265 · DSB learns each reverse step by regression on simulated pairs

[Completed slide](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u045_b4)



Let Fₖ be the current forward mean map, with a Gaussian transition around it.



$$
X_{k+1}\sim\mathcal N(F_k(X_k),\,2\gamma_{k+1}I)
$$



Spoken equation reading: The next state is sampled from a Gaussian whose mean is the current forward map F applied to the present state and whose covariance is twice the step parameter γ times the identity.



The paper regresses a backward mean map B onto a target formed from the current F.



$$
\min_B\ \mathbb E\left\lVert B(X_{k+1})-\left[X_{k+1}+F_k(X_k)-F_k(X_{k+1})\right]\right\rVert^2
$$



Spoken equation reading: The backward regression minimizes the expected squared difference between B at the next state and a target formed by that next state plus F at the previous state minus F at the next state.





The current forward process supplies its own regression pairs. After learning the backward process, the next training round runs in the opposite direction.



De Bortoli et al., Proposition 3 and Algorithm 1. γ is the numerical time increment in their noise convention √2. The reverse regression target uses the same current F in both evaluations. The forward counterpart replaces F with B and reverses the roles of Xₖ and Xₖ₊₁. Approximate Gaussian reverse kernels become accurate in the small-step limit under regularity.



Primary source: [Diffusion Schrödinger Bridge with Applications to Score-Based Generative Modeling](https://arxiv.org/abs/2106.01357v5)




## Slides 266–268 · The rollout gives us a reverse-regression target

[Completed slide](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u046_b2)



Let’s calculate a DSB backward target from a cached forward pair.

```python

with torch.no_grad():
    x_next = F(x, k) + (2 * dt)**0.5 * noise
    target = x_next + F(x, k) - F(x_next, k)
prediction = B(x_next, k + 1)
loss = (prediction - target).square().mean()
optimizer.zero_grad()
loss.backward()
optimizer.step()

```



We freeze the rollout and its target while updating the backward network. The next half-iteration repeats the calculation in the other direction.



The companion neural script includes an affine Gaussian DSB example using this regression target, separate forward and backward mean maps, and both endpoint resets. It is a small computation illustrating Algorithm 1, not a reproduction of the image benchmark.



Primary source: [Diffusion Schrödinger Bridge with Applications to Score-Based Generative Modeling](https://arxiv.org/abs/2106.01357v5)




## Slides 269–271 · Repeated fitting improves the generated distribution

[Completed slide](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u047_b2)





The paper reports improved image generation across DSB iterations at a fixed small number of diffusion steps. Each additional fitting round also requires fresh simulation and training.



Original De Bortoli et al. Figure 6 and accompanying MNIST experiment. Keep the sampling step counts shown in the original figure. These are reported paper results, separate from the course Gaussian demonstration.



Primary source: [Diffusion Schrödinger Bridge with Applications to Score-Based Generative Modeling](https://arxiv.org/abs/2106.01357v5)




## Slides 272–274 · We can learn from conditional bridge samples

[Completed slide](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u048_b2)





Shi, De Bortoli, Campbell, and Doucet alternate Markovian and reciprocal projections to learn the Schrödinger bridge.



Original paper header. NeurIPS 2023



Primary source: [Diffusion Schrödinger Bridge Matching](https://arxiv.org/abs/2303.16852v3)




## Slides 275–281 · An endpoint mixture can retain hidden history

[Completed slide](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u049_b4)



An endpoint coupling π and reference conditional paths define a reciprocal mixture Λ.



$$
\Lambda=\int R^{xy}\,\pi(dx,dy)
$$



Spoken equation reading: The reciprocal mixture Λ first draws an endpoint pair from π and then draws an entire reference bridge conditioned on that pair.



A Markov law depends on the present state; the mixture can retain its start.



$$
P(X_{t+\Delta}\mid X_{[0,t]})=P(X_{t+\Delta}\mid X_t)\quad\text{for a Markov law}
$$



Spoken equation reading: For a Markov law, conditioning the future on the complete past gives the same distribution as conditioning only on the current state; the earlier history supplies no additional information.





After averaging over endpoint pairs, knowing the starting point can still change the likely destination. This is why a mixture of Markov bridges can lose the Markov property.



A reciprocal mixture retains the reference path law conditioned on both endpoints. It need not be Markov. Two pairs can pass through the same intermediate region while retaining different endpoint preferences. The SB is Markov and belongs to the reciprocal class under the standard positivity assumptions.



Primary source: [Diffusion Schrödinger Bridge Matching](https://arxiv.org/abs/2303.16852v3)




## Slides 282–288 · Markov projection averages the conditional drift

[Completed slide](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u050_b4)



Average the Brownian bridge drift over pairs that pass through the current state.



$$
v_t^*(x)=\mathbb E_\Lambda\!\left[\frac{X_1-X_t}{1-t}\,\middle|\,X_t=x\right]
$$



Spoken equation reading: The projected drift at x is the conditional average, over mixture trajectories passing through x, of the displacement to their terminal states divided by the remaining time.



Squared-error regression learns this conditional expectation.



$$
\min_\theta\ \mathbb E_{t,\Lambda}\left\lVert v_\theta(t,X_t)-\frac{X_1-X_t}{1-t}\right\rVert^2
$$



Spoken equation reading: The drift-matching loss averages squared error between the learned drift at the current state and the endpoint-conditioned Brownian bridge drift, over times and trajectories from Λ.





The projected process keeps every one-time marginal of the mixture, including both endpoints. Its joint endpoint coupling can change.



For square-integrable target Y, E||v(X)−Y||²=E||v(X)−E[Y|X]||²+E||Y−E[Y|X]||². This proves the minimizer. Substituting the conditional drift into the weak Fokker-Planck equation proves marginal preservation. A common positive time weight, such as inverse diffusion variance, leaves the pointwise minimizer unchanged.



Primary source: [Diffusion Schrödinger Bridge Matching](https://arxiv.org/abs/2303.16852v3)




## Slides 289–295 · Reciprocal projection restores the reference bridges

[Completed slide](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u051_b4)



Keep the endpoint coupling and replace the process’s conditional bridges.



$$
\mathcal R(M)=\int R^{xy}\,M_{01}(dx,dy)
$$



Spoken equation reading: The reciprocal projection retains M’s joint endpoint distribution and replaces every endpoint-conditioned path law with the corresponding reference bridge.



Iterative Markovian fitting alternates the two projections.



$$
M^{n+1}=\mathcal M(\Lambda^n),\qquad \Lambda^{n+1}=\mathcal R(M^{n+1})
$$



Spoken equation reading: One fitting cycle first takes the Markov projection of the current reciprocal mixture, then takes the reciprocal projection of that resulting Markov process.





The Markovian step removes the extra dependence on history. The reciprocal step restores the reference bridges. Their common solution is the Schrödinger bridge.



Initialize Λ⁰ with the desired marginals and a coupling such as μ⊗ν. Exact Markovian projection preserves all time marginals. Exact reciprocal projection preserves the endpoint coupling. Both endpoint marginals therefore persist through exact IMF. In practice DSBM alternates forward and backward fitting to limit marginal drift from approximation.



Primary source: [Diffusion Schrödinger Bridge Matching](https://arxiv.org/abs/2303.16852v3)




## Slides 296–302 · A KL identity explains why exact projection makes progress

[Completed slide](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u052_b4)



The KL identity separates the correction from the remaining distance to the SB.



$$
\operatorname{KL}(P^n\Vert P^*)=\operatorname{KL}(P^n\Vert P^{n+1})+\operatorname{KL}(P^{n+1}\Vert P^*)
$$



Spoken equation reading: Under the exact projection theorem’s assumptions, the current KL to the bridge splits into the KL paid by the projection plus the new KL remaining to the bridge.



The correction is nonnegative, so the distance to the bridge cannot increase.



$$
\operatorname{KL}(P^{n+1}\Vert P^*)\leq\operatorname{KL}(P^n\Vert P^*)
$$



Spoken equation reading: Because the projection’s KL contribution is nonnegative, the next exact iterate’s KL to the bridge is no larger than the current iterate’s KL.





The exact projections successively remove incompatible path structure. The theorem applies to the ideal projection sequence.



Shi et al., Lemma 6, Proposition 7 and Theorem 8. For Markovian projection, use the orthogonal conditional-expectation decomposition of drift energy. For reciprocal projection, use the endpoint KL chain rule. Apply each with the SB, which belongs to both classes. Uniqueness of their intersection plus the paper’s compactness and integrability assumptions establishes convergence. This identity does not by itself prove convergence for arbitrary neural networks.



Primary source: [Diffusion Schrödinger Bridge Matching](https://arxiv.org/abs/2303.16852v3)




## Slides 303–305 · The Gaussian test also checks endpoint dependence

[Completed slide](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u053_b2)





At dimension 50, the reported mean marginal KL is 8.75×10⁻³ for DSBM-IPF and 32.8×10⁻³ for DSB. The covariance plots also test whether the learned endpoint dependence matches the bridge.



Original Figure 3 and Table 3 from Shi et al. Table 3 reports average KL(Pₜ∥Pₜ^SB), multiplied by 10⁻³. DSBM-IMF is 9.76±1.67 at d=50. These are marginal KL estimates, not full path-space KL. The 2D benchmark also shows that OT-CFM can outperform DSBM when exact minibatch OT is inexpensive.



Primary source: [Diffusion Schrödinger Bridge Matching](https://arxiv.org/abs/2303.16852v3)




## Slides 306–308 · An entropic coupling makes training simulation-free

[Completed slide](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u054_b2)





Tong and colleagues combine entropic OT, Brownian bridge samples, flow matching, and score matching in [SF]²M.



Original paper header. AISTATS 2024



Primary source: [Simulation-Free Schrödinger Bridges via Score and Flow Matching](https://arxiv.org/abs/2307.03672v3)




## Slides 309–315 · The conditional Gaussian gives both a velocity and a score

[Completed slide](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u055_b4)



Differentiate the mean mₜ and standard deviation σₜ of the Gaussian interpolant.



$$
m_t=(1-t)x_0+tx_1,\quad\sigma_t^2=\varepsilon t(1-t),\quad v_t^c=x_1-x_0+\frac{1-2t}{2t(1-t)}(x-m_t)
$$



Spoken equation reading: The conditional velocity combines endpoint displacement with a correction proportional to deviation from the interpolated mean, while the Gaussian variance is ε times t times one minus t.



Differentiating the Gaussian log density gives the conditional score.



$$
s_t^c(x)=\nabla_x\log p_t(x\mid x_0,x_1)=-\frac{x-m_t}{\varepsilon t(1-t)}
$$



Spoken equation reading: The conditional score is the gradient of the Gaussian log density, equal to the negative displacement from the interpolated mean divided by the conditional variance.





One sampled endpoint pair and one Gaussian draw provide both regression targets. The factor one half in the velocity comes from differentiating the standard deviation.



Differentiate x=mₜ+σₜz with fixed z to obtain vᶜ=ṁₜ+(σ̇ₜ/σₜ)(x−mₜ). Here σ̇/σ=(1−2t)/(2t(1−t)). This also verifies vᶜ+εsᶜ/2=(x₁−x)/(1−t), the conditional forward drift. Endpoint pairs must follow the SB entropic coupling to recover that SB. Minibatch EOT introduces coupling approximation.



Primary source: [Simulation-Free Schrödinger Bridges via Score and Flow Matching](https://arxiv.org/abs/2307.03672v3)




## Slides 316–322 · The learned velocity and score recover either an ODE or an SDE

[Completed slide](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u056_b4)



Match both conditional targets with a positive score-loss weight λₜ.



$$
\mathcal L=\mathbb E\!\left[\lVert v_\theta-v_t^c\rVert^2+\lambda_t^2\lVert s_\theta-s_t^c\rVert^2\right]
$$



Spoken equation reading: The loss averages the squared velocity-prediction error plus the squared score-prediction error multiplied by λ squared, with both targets supplied by the conditional Gaussian bridge.



Adding half the diffusion variance times the score recovers the SDE drift.



$$
dX_t=\left[v_\theta(t,X_t)+\tfrac\varepsilon2s_\theta(t,X_t)\right]dt+\sqrt\varepsilon\,dW_t
$$



Spoken equation reading: The sampling SDE uses learned velocity plus ε over two times the learned score as its drift, and adds Brownian noise with scale equal to the square root of ε.





The velocity generates the marginal probability flow. The score compensates for the probability spreading caused by diffusion.



The conditional-score identity follows by differentiating the marginal mixture density. The conditional-velocity identity follows from the weak continuity equation. With exact fields, changing the sampling diffusion and its score correction preserves one-time marginals, but changes the path law. Only the original reference noise recovers the specified SB path measure.



Primary source: [Simulation-Free Schrödinger Bridges via Score and Flow Matching](https://arxiv.org/abs/2307.03672v3)




## Slides 323–325 · We can form both targets in a single training batch

[Completed slide](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u057_b2)



Let’s draw coupled endpoints, add Brownian bridge noise, and build the two regression targets.

```python

m = (1-t)*x0 + t*x1
sigma = (epsilon*t*(1-t)).sqrt()
x = m + sigma*z
v_target = x1-x0 + (1-2*t)/(2*t*(1-t))*(x-m)
s_target = -(x-m)/(epsilon*t*(1-t))
v, s = network(t, x).chunk(2, dim=-1)
loss = ((v-v_target)**2 +
        (sigma*(s-s_target))**2).mean()

```



The Gaussian demonstration uses the exact entropic endpoint coupling, so we can compare the learned fields with analytic answers.



The standalone neural example trains a small PyTorch network and reports held-out velocity and score errors plus sampled endpoint mean and variance. Its time interval excludes the singular endpoints during training. λₜ=σₜ in the displayed loss.



Primary source: [Simulation-Free Schrödinger Bridges via Score and Flow Matching](https://arxiv.org/abs/2307.03672v3)




## Slides 326–328 · The method also learns from high-dimensional data

[Completed slide](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u058_b2)





The paper evaluates Gaussian bridge recovery and single-cell distributions. Its experiments separate endpoint generation from the accuracy of intermediate distributions.



Original Tong et al. Table 3 is displayed. Table 4 supplies the additional single-cell comparison discussed here. Table 3 compares KL against an analytic Gaussian SB. Table 4 compares held-out intermediate-distribution W₁ on single-cell datasets. Training avoids simulation of learned SDEs, while sampling the trained SDE still requires numerical integration.



Primary source: [Simulation-Free Schrödinger Bridges via Score and Flow Matching](https://arxiv.org/abs/2307.03672v3)




## Slides 329–331 · The bridge can move through a discrete space

[Completed slide](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u059_b2)



Our two-state bridge used a fixed sequence of times. We can also let a categorical state jump at a random time.

We now need to distinguish a transition probability from a transition rate. That distinction changes both the sampler and its training loss.







Primary source: [Discrete Diffusion Schrödinger Bridge Matching for Graph Transformation](https://arxiv.org/abs/2410.01500v2)




## Slides 332–338 · A rate describes the probability of a jump over a short interval

[Completed slide](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u060_b4)



For distinct states x and y, the off-diagonal generator entry is a nonnegative rate.



$$
\Pr(X_{t+dt}=y\mid X_t=x)=G_t(x,y)\,dt+o(dt),\qquad y\ne x
$$



Spoken equation reading: Over a sufficiently small interval, the probability of jumping from x to a different state y is the rate G from x to y times the interval, plus a smaller-order remainder.



The diagonal records the total rate of leaving, so each generator row sums to zero.



$$
G_t(x,x)=-\sum_{y\ne x}G_t(x,y),\qquad \dot p_t=p_tG_t
$$



Spoken equation reading: The diagonal generator entry is minus the sum of all outgoing rates, and the row probability vector evolves by multiplying itself by the generator.





A transition matrix has rows summing to one. A generator has rows summing to zero. For a constant generator, the finite-time transition is Kₛₜ=exp((t−s)G).



We use row vectors for categorical distributions. The no-jump probability is 1−λₜ(x)dt+o(dt), with λₜ(x)=Σᵧ≠ₓGₜ(x,y). Positivity and row conservation imply that the matrix exponential is stochastic. Euler transitions require dt small enough that every diagonal stays nonnegative.



Primary source: [Discrete Diffusion Schrödinger Bridge Matching for Graph Transformation](https://arxiv.org/abs/2410.01500v2)




## Slides 339–345 · Changing rates also changes the waiting times

[Completed slide](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u061_b4)



The log ratio sums jump terms and integrates the difference in escape rates.



$$
\log\frac{dP^Q}{dR}=\log\frac{p_0(X_0)}{r_0(X_0)}+\sum_{\tau:\,\mathrm{jump}}\log\frac{Q_\tau(X_{\tau^-},X_\tau)}{G_\tau(X_{\tau^-},X_\tau)}-\int_0^1(\lambda_t^Q-\lambda_t^G)(X_t)\,dt
$$



Spoken equation reading: The path log ratio adds the initial-density log ratio and the log rate ratio at each realized jump, then subtracts the time integral of the difference in total escape rates.



Taking its expectation gives a local control cost for each possible transition.



$$
\operatorname{KL}(P^Q\Vert R)=\operatorname{KL}(p_0\Vert r_0)+\mathbb E_{P^Q}\!\int_0^1\sum_{y\ne X_t}\!\left[Q\log\frac QG-Q+G\right](X_t,y)\,dt
$$



Spoken equation reading: The controlled path KL equals the initial KL plus the controlled expectation of the time integral, summed over destinations, of Q log(Q/G) minus Q plus G.





The discrete analogue of quadratic drift energy is a rate-relative-entropy cost. Omitting waiting times would optimize a different path law.



Derive the jump sum from the product of jump intensities and the integrated survival factors. The compensator identity EΣjump f=E∫ΣᵧQₜ(Xₜ,y)f dt converts the sum to the integral. Require Q=0 wherever G=0; otherwise the controlled law can leave the support of R and have infinite KL.



Primary source: [Discrete Diffusion Schrödinger Bridge Matching for Graph Transformation](https://arxiv.org/abs/2410.01500v2)




## Slides 346–352 · Conditioning a jump process also produces a Doob transform

[Completed slide](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u062_b4)



A positive future potential solves a backward equation on the state graph.



$$
h_t=K_{t1}g,\qquad \partial_t h_t=-G_t h_t
$$



Spoken equation reading: The future potential h is the terminal potential propagated backward through the reference transition kernel, and its time derivative is minus the generator applied to h.



The bridge multiplies each off-diagonal rate by the ratio of future potentials.



$$
Q_t(x,y)=G_t(x,y)\frac{h_t(y)}{h_t(x)},\quad y\ne x;\qquad Q_t(x,x)=-\sum_{y\ne x}Q_t(x,y)
$$



Spoken equation reading: Each off-diagonal bridge rate equals the reference rate times the destination potential divided by the source potential at the same time; the diagonal is reset to minus the outgoing-rate sum.





A state with a larger future potential becomes easier to enter. The diagonal changes at the same time, which adjusts how long the process waits.



Expand K^hₜ,ₜ₊dt(x,y)=Kₜ,ₜ₊dt(x,y)hₜ₊dt(y)/hₜ(x) to first order. The backward equation cancels the row-sum error. A conditional bridge to z uses hₜ(x)=Kₜ₁(x,z). When this is zero, the conditioned bridge is only defined on its reachable support.



Primary source: [Discrete Diffusion Schrödinger Bridge Matching for Graph Transformation](https://arxiv.org/abs/2410.01500v2)




## Slides 353–359 · Our two-state example now evolves in continuous time

[Completed slide](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u063_b4)



Choose a generator and solve the endpoint scaling problem using its matrix exponential.



$$
G=\begin{pmatrix}-2&2\\1&-1\end{pmatrix},\quad a=(0.5,0.5),\quad b=(0.2,0.8),\quad K_{01}=e^G
$$



Spoken equation reading: The generator has jump rates two from state zero and one from state one, and its matrix exponential gives the unit-time reference transition used to match a to b.



At the midpoint, the future-potential ratio changes both jump rates.



$$
h_{1/2}=(0.91138,1.05216),\quad Q_{1/2}(0,1)=2.30894,\quad Q_{1/2}(1,0)=0.86620
$$



Spoken equation reading: At the midpoint, the potential values approximately 0.91138 and 1.05216 reweight the reference rates into 2.30894 toward state one and 0.86620 toward state zero.





The resulting midpoint marginal is approximately (0.33937, 0.66063), and the terminal marginal is exactly (0.2, 0.8) up to numerical precision.



The overall scale of h is arbitrary. The companion obtains its scale from Sinkhorn potentials for diag(a)exp(G). The endpoint coupling is [[0.10884716,0.39115284],[0.09115284,0.40884716]]. Compare the two midpoint controlled rates with the reference rates 2 and 1.



Primary source: [Discrete Diffusion Schrödinger Bridge Matching for Graph Transformation](https://arxiv.org/abs/2410.01500v2)




## Slides 360–362 · We can compute this small jump bridge exactly

[Completed slide](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u064_b2)



We can solve the endpoint problem and evaluate the transformed generator at any time.

```python

K = scipy.linalg.expm(G)
pi, f, g, history = sinkhorn(a, b, np.log(a[:,None]*K))
h = scipy.linalg.expm((1-t)*G) @ g
Q = G * h[None,:] / h[:,None]
np.fill_diagonal(Q, 0.0)
np.fill_diagonal(Q, -Q.sum(axis=1))
assert np.allclose(Q.sum(axis=1), 0)

```



The companion verifies the terminal constraint independently from the analytic transition matrix. Larger state spaces require structure or approximation.



See ctmc_bridge_example in ot_sbm_examples.py. The scaling function returns f and g for the supplied positive reference matrix. Matrix exponentiation is exact to floating-point numerical accuracy, unlike a coarse Euler simulation.



Primary source: [Discrete Diffusion Schrödinger Bridge Matching for Graph Transformation](https://arxiv.org/abs/2410.01500v2)




## Slides 363–365 · DDSBM learns bridges for continuous-time jumps

[Completed slide](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u065_b2)





Kim and colleagues derive iterative Markovian fitting for discrete diffusion and apply it to graph transformations.



Original paper header. ICLR 2025



Primary source: [Discrete Diffusion Schrödinger Bridge Matching for Graph Transformation](https://arxiv.org/abs/2410.01500v2)




## Slides 366–372 · The conditional bridge supplies a rate target for DDSBM

[Completed slide](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u066_b4)



Conditioning on terminal state z reweights the reference jump rate.



$$
G_t^z(x,y)=G_t(x,y)\frac{K_{t1}(y,z)}{K_{t1}(x,z)},\qquad y\ne x
$$



Spoken equation reading: Conditioning on endpoint z multiplies the reference rate from x to y by the reference probability of reaching z from y divided by the probability of reaching z from x.



Average this rate over destinations consistent with the current state.



$$
Q_t^*(x,y)=\mathbb E_\Lambda[G_t^{X_1}(x,y)\mid X_t=x]
$$



Spoken equation reading: The projected rate averages those endpoint-conditioned reference rates over the terminal states consistent with the mixture’s current state x.





This is the same conditional-expectation principle we used for Brownian bridge drifts. The learned quantity is now a collection of nonnegative jump rates.



Kim et al., Markovian projection and discrete bridge matching formulation. Substitute the conditional rates into the weak master equation to prove marginal preservation. The endpoint-coupling update is still required; fitting once from an arbitrary coupling yields a bridge mixture projection rather than necessarily the SB.



Primary source: [Discrete Diffusion Schrödinger Bridge Matching for Graph Transformation](https://arxiv.org/abs/2410.01500v2)




## Slides 373–379 · Rate matching penalizes excessive jumping

[Completed slide](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u067_b4)



Match each conditional rate r with a positive predicted rate q.



$$
\mathcal L=\mathbb E\int_0^1\sum_{y\ne X_t}\left[r_t^y\log\frac{r_t^y}{q_\theta^y}-r_t^y+q_\theta^y\right]dt
$$



Spoken equation reading: The rate loss averages over paths the time integral, summed over possible destinations, of target rate r times log(r/q), minus r, plus the predicted positive rate q.



Differentiating with respect to a positive predicted rate gives its conditional mean.



$$
\frac{\partial}{\partial q}\mathbb E[r\log(r/q)-r+q\mid X_t=x]=1-\frac{\mathbb E[r\mid X_t=x]}q=0
$$



Spoken equation reading: Differentiating the conditional expected rate loss with respect to q gives one minus the conditional mean target rate divided by q, which vanishes when q equals that conditional mean.





The logarithmic term rewards jumps supported by the conditional bridges. The linear term prevents the model from increasing every rate without limit.



The divergence is nonnegative by z log z−z+1≥0. Its second derivative is E[r|x]/q², giving the positive unique minimizer when E[r|x]>0. Boundary cases have q*=0. Exact alternating Markovian and reciprocal projections converge under the assumptions in Kim et al.; learned rates and numerical simulation introduce approximation.



Primary source: [Discrete Diffusion Schrödinger Bridge Matching for Graph Transformation](https://arxiv.org/abs/2410.01500v2)




## Slides 380–382 · The graph benchmark checks preservation and fit

[Completed slide](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u068_b2)





On ZINC, DDSBM improves FCD from 1.046 to 0.833 compared with single-pass DBM, while validity rises from 87.6% to 94.8%. Some latent baselines have higher validity and lower uniqueness.



Original Table 1 from Kim et al. FCD compares generated and target distributions; NLL measures changes under the chosen graph reference. These metrics answer different questions. The bridge is optimal relative to that reference and constraints, not relative to every chemical property. No biological sequence design is needed for the course implementation.



Primary source: [Discrete Diffusion Schrödinger Bridge Matching for Graph Transformation](https://arxiv.org/abs/2410.01500v2)




## Slides 383–385 · Categorical SBM learns a finite chain of transitions

[Completed slide](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u069_b2)





Categorical Schrödinger Bridge Matching develops discrete-time IMF on categorical spaces and learns transitions that generalize to new samples.



Original paper header. ICML 2025



Primary source: [Categorical Schrödinger Bridge Matching](https://arxiv.org/abs/2502.01416v2)




## Slides 386–392 · Markov projection learns adjacent conditionals

[Completed slide](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u070_b4)



Project a path distribution Λ onto the Markov chain defined by its adjacent conditionals.



$$
\mathcal M(\Lambda)(x_{0:N})=\Lambda_0(x_0)\prod_{k=0}^{N-1}\Lambda(x_{k+1}\mid x_k)
$$



Spoken equation reading: The Markov projection assigns a path its original initial probability multiplied by the original mixture’s adjacent conditional probabilities at every step.



A categorical cross-entropy learns each of those conditional distributions.



$$
\min_\theta\ \mathbb E_\Lambda\left[-\sum_{k=0}^{N-1}\log q_\theta(X_{k+1}\mid X_k,k)\right]
$$



Spoken equation reading: The categorical objective minimizes the expectation under Λ of the sum of negative log probabilities that the learned transitions assign to the observed next states.





The projection preserves every adjacent pair marginal, and therefore every one-time marginal. It can change the dependence between distant times.



Expand KL(Λ∥q₀∏qₖ) into a constant plus independent cross-entropies. Each term is HΛ(Xₖ₊₁|Xₖ)+E KL(Λ(·|Xₖ)∥qₖ(·|Xₖ)), proving the minimizer. Starting from Λ₀ and inducting over k verifies preservation of adjacent marginals. This is stronger than the continuous-time statement about only one-time marginals.



Primary source: [Categorical Schrödinger Bridge Matching](https://arxiv.org/abs/2502.01416v2)




## Slides 393–399 · The reciprocal step restores conditional paths

[Completed slide](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u071_b4)



Normalize the path product by the reference endpoint transition probability.



$$
R(x_{1:N-1}\mid x_0,x_N)=\frac{\prod_{k=0}^{N-1}K_k(x_k,x_{k+1})}{K_{0N}(x_0,x_N)}
$$



Spoken equation reading: The probability of the intermediate reference path conditioned on its endpoints equals the product of its one-step transitions divided by the total reference transition probability between those endpoints.



Keep the fitted endpoint coupling and restore the reference conditional paths.



$$
\Lambda^{n+1}(x_{0:N})=M^n_{0N}(x_0,x_N)R(x_{1:N-1}\mid x_0,x_N)
$$



Spoken equation reading: The next reciprocal mixture multiplies the fitted joint endpoint probability by the reference probability of the intermediate path conditioned on that same endpoint pair.





The two endpoint marginals survive both projections. Repeated fitting searches for a process that is simultaneously Markov and in the reference reciprocal class.



Finite positive kernels and compatible marginals give the finite-state characterization and D-IMF convergence in the CSBM paper. The common solution factors as f(x₀)g(xN)R and is unique. With only one transition, every path is already an endpoint pair, so this intersection argument needs the nontrivial interior-time conditions in the theorem.



Primary source: [Categorical Schrödinger Bridge Matching](https://arxiv.org/abs/2502.01416v2)




## Slides 400–402 · We can enumerate the paths and watch IMF converge

[Completed slide](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u072_b2)



For two states and three transitions, only sixteen paths are possible.

```python

paths = list(itertools.product(range(2), repeat=4))
q = independent_endpoints_times_reference_bridges(paths)
for iteration in range(80):
    transitions = adjacent_conditionals(q, paths)
    markov = path_probabilities(a, transitions, paths)
    endpoint = endpoint_joint(markov, paths)
    q = endpoint_weights(endpoint, paths) * reference_bridge
assert np.abs(q-exact_sb).sum() < 1e-10

```



Every quantity is explicitly normalized. This lets us compare the complete path law with the Sinkhorn solution, instead of checking only its endpoints.



Executable equivalent in discrete_imf_example. Helper names on this slide identify the operations; the companion spells them out with array indices and normalization checks. The default run reaches L1 error 1.4×10⁻¹³ after 80 iterations. It also records path KL at every projection.



Primary source: [Categorical Schrödinger Bridge Matching](https://arxiv.org/abs/2502.01416v2)




## Slides 403–405 · Correct endpoints can still have incorrect paths

[Completed slide](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u073_b2)





Both endpoints are already correct before fitting. The path KL continues to decrease as the intermediate dependence approaches the exact Schrödinger bridge.



Course computation, two states and three transitions, reference K=[[0.8,0.2],[0.3,0.7]], source (0.5,0.5), target (0.2,0.8). All sixteen path probabilities are enumerated. This plot measures full path KL, unlike a plot of only terminal error.



Primary source: [Categorical Schrödinger Bridge Matching](https://arxiv.org/abs/2502.01416v2)




## Slides 406–408 · Categorical latents also allow the method to translate images

[Completed slide](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u074_b2)





The paper reports FID 10.60 for low-stochasticity CSBM, compared with 16.86 for ASBM and 24.06 for DSBM. CSBM uses 100 time steps and a categorical image representation.



Original Figure 4 and Table 2. The authors explicitly note the 100-step CSBM versus 3-step ASBM comparison. The VQ representation, reference noise parameters, and numerical step budgets differ. The results support this experimental system; they do not isolate categorical state space as the sole cause of the improvement.



Primary source: [Categorical Schrödinger Bridge Matching](https://arxiv.org/abs/2502.01416v2)




## Slides 409–411 · So far, the destination distribution has been supplied to us

[Completed slide](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u075_b2)



For a new generation task, we may have a pretrained model and a reward function instead of samples from the desired destination.

We can ask which terminal distribution follows from balancing that reward against changes to the pretrained path law.







Primary source: [Categorical Schrödinger Bridge Matching](https://arxiv.org/abs/2502.01416v2)




## Slides 412–414 · TR2-D2 combines trajectory-aware fine-tuning with tree search

[Completed slide](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u076_b2)





Tang, Zhu, Tao, and Chatterjee use a pretrained discrete diffusion process as the reference, then combine reward-guided search with weighted denoising.



Original paper header. From our lab | arXiv 2025



Primary source: [TR2-D2: Tree Search Guided Trajectory-Aware Fine-Tuning for Discrete Diffusion](https://arxiv.org/abs/2509.25171v1)




## Slides 415–421 · A reward expresses a preference over trajectories

[Completed slide](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u077_b4)



Let R be the pretrained path law and α control the cost of changing it.



$$
\max_{P\ll R}\ \mathbb E_P[r(X_1)]-\alpha\operatorname{KL}(P\Vert R),\qquad \alpha>0
$$



Spoken equation reading: The objective maximizes expected terminal reward under P minus α times the path KL from P to the pretrained reference R, restricting P to trajectories supported by R.



Exponentiating the terminal reward gives the optimal tilted law.



$$
P^*(d\omega)=\frac{e^{r(X_1)/\alpha}}Z R(d\omega),\qquad Z=\mathbb E_R[e^{r(X_1)/\alpha}]
$$



Spoken equation reading: The optimal path law multiplies each reference trajectory’s probability by the exponential of its terminal reward divided by α, then divides by the reference expectation Z of that multiplier.





The reference contains the entire denoising dynamics. For a deterministic fully masked initial state, this tilt also preserves the initial distribution.



TR2-D2 Equations 3–4. Assume Z is finite and nonzero. The displayed unconstrained optimizer also solves the fixed-initial problem when X₀ is deterministic. A base terminal distribution alone does not specify a path-space SB; R must include the transitions and initial law.



Primary source: [TR2-D2: Tree Search Guided Trajectory-Aware Fine-Tuning for Discrete Diffusion](https://arxiv.org/abs/2509.25171v1)




## Slides 422–428 · The gap to the optimum is another KL divergence

[Completed slide](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u078_b4)



Insert log(dP*/dR)=r(X₁)/α−log Z into the relative entropy.



$$
\mathbb E_P[r]-\alpha\operatorname{KL}(P\Vert R)=\alpha\log Z-\alpha\operatorname{KL}(P\Vert P^*)
$$



Spoken equation reading: Expected reward minus the reference KL penalty equals α times log Z minus α times KL to the optimally tilted path law, so the remaining gap is itself a KL penalty.



Its terminal law is a reward-weighted version of the pretrained terminal distribution.



$$
p_1^*(x)=\frac{p_1^R(x)e^{r(x)/\alpha}}Z
$$



Spoken equation reading: The optimal terminal density is the reference terminal density multiplied by the exponential reward factor and divided by the same normalizer Z.





KL is nonnegative and vanishes at P=P*. Because the tilt depends only on the endpoint, it also preserves the reference paths conditioned on the endpoints.



This proves both optimality and the SB interpretation. With the deterministic initial root and terminal law p₁*, the KL chain rule says that retaining R’s conditional bridges minimizes path KL. The target is induced by the reward and temperature; it is not an independently specified empirical marginal.



Primary source: [TR2-D2: Tree Search Guided Trajectory-Aware Fine-Tuning for Discrete Diffusion](https://arxiv.org/abs/2509.25171v1)




## Slides 429–435 · Three outcomes reveal the reward–reference tradeoff

[Completed slide](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u079_b4)



Suppose the base probabilities are (0.5,0.3,0.2) and the reward multipliers are (1,2,4).



$$
p^R=(0.5,0.3,0.2),\quad r=\log(1,2,4),\quad \alpha=1,\quad Z=1.9
$$



Spoken equation reading: The reference assigns probabilities 0.5, 0.3, and 0.2, the exponentiated rewards multiply them by one, two, and four, and their resulting total mass is Z equal to 1.9.



Multiply first, then normalize to obtain the desired terminal probabilities.



$$
p^*=\frac{(0.5,0.6,0.8)}{1.9}=(0.26316,0.31579,0.42105)
$$



Spoken equation reading: Multiplying the reference masses by their reward factors gives 0.5, 0.6, and 0.8; dividing by their sum 1.9 gives the normalized target probabilities shown.





The third outcome becomes more likely, while the original probabilities still matter. Decreasing α sharpens the reward preference.



Course example uses abstract outcomes A, B, C. If pᴿ(x)=0, finite terminal reweighting cannot create that outcome. At α→∞ the tilt returns to the base distribution. At α→0 it concentrates on reward maximizers within the reference support. These limits assume bounded rewards in the finite example.



Primary source: [TR2-D2: Tree Search Guided Trajectory-Aware Fine-Tuning for Discrete Diffusion](https://arxiv.org/abs/2509.25171v1)




## Slides 436–442 · A random fixed source needs conditional normalization

[Completed slide](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u080_b4)



A global terminal tilt changes a random initial law by its future reward potential.



$$
P_0^*(x)=R_0(x)\frac{h_0(x)}Z,\qquad h_0(x)=\mathbb E_R[e^{r(X_1)/\alpha}\mid X_0=x]
$$



Spoken equation reading: A global terminal reward tilt multiplies each initial-state probability by its expected future reward multiplier h, then divides by the global normalizer Z.



Keeping that initial law fixed instead requires conditional normalization.



$$
\frac{d\widehat P}{dR}(\omega)=\frac{e^{r(X_1)/\alpha}}{h_0(X_0)}
$$



Spoken equation reading: To preserve the reference initial law instead, each path’s reward multiplier is divided by the future-reward normalizer conditioned on that path’s own starting state.





This distinction disappears at a deterministic masked root. With a random source, the desired endpoint constraints determine which normalization is correct.



For a fixed R₀, condition the reward-minus-KL optimization separately on X₀=x. Each conditional optimum has normalizer h₀(x), and integrating with R₀ gives the second formula. If a separate terminal marginal is also fixed, solve for both Schrödinger potentials. This avoids equating every terminal reward tilt with every two-marginal bridge problem.



Primary source: [TR2-D2: Tree Search Guided Trajectory-Aware Fine-Tuning for Discrete Diffusion](https://arxiv.org/abs/2509.25171v1)




## Slides 443–449 · Off-policy samples estimate weighted denoising

[Completed slide](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u081_b4)



For trajectories sampled from a proposal Pᵛ, use the target-to-proposal likelihood ratio.



$$
w(\omega)=\frac{dP^*}{dP^v}(\omega)\propto e^{r(X_1)/\alpha}\frac{dR}{dP^v}(\omega)
$$



Spoken equation reading: The importance weight is the target path density divided by the proposal path density, proportional to the reward multiplier times the reference-to-proposal path likelihood ratio.



The target expectation becomes a weighted expectation under the proposal.



$$
\mathbb E_{P^*}[\mathcal L_{\rm denoise}(\theta;X_1)]=\mathbb E_{P^v}[w(\omega)\mathcal L_{\rm denoise}(\theta;X_1)]
$$



Spoken equation reading: The expected denoising loss under the target law equals the proposal expectation of that same loss multiplied by its normalized path importance weight.





The weights account for both reward and how the trajectory was sampled. The proposal must cover the target support.



TR2-D2 weighted denoising cross-entropy, Equation 7. An exact normalized ratio gives an unbiased expectation identity. Replacing an unknown normalizer with a batch sum gives self-normalized importance sampling, which is consistent under suitable assumptions but biased at finite batch size. Monitor effective sample size and weight concentration.



Primary source: [TR2-D2: Tree Search Guided Trajectory-Aware Fine-Tuning for Discrete Diffusion](https://arxiv.org/abs/2509.25171v1)




## Slides 450–456 · Matched masking schedules simplify the weight

[Completed slide](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u082_b4)



The paper accumulates log probabilities along the unmasking trajectory.



$$
\log\widetilde w=\frac{r(X_1)}\alpha+\sum_{\ell\in\mathrm{unmasking\ events}}\log\frac{p_{\rm pre}(x_\ell\mid x_{<\ell})}{p_v(x_\ell\mid x_{<\ell})}
$$



Spoken equation reading: The log weight adds terminal reward divided by α to the sum, over unmasking events, of log pretrained-token probability divided by proposal-token probability under the visible context.



Normalizing in log space avoids overflow and gives the weighted training loss.



$$
\overline w_i=\frac{e^{\log\widetilde w_i}}{\sum_j e^{\log\widetilde w_j}},\qquad \widehat{\mathcal L}_{\rm WDCE}=\sum_i\overline w_i\,\mathcal L_{\rm denoise}(\theta;x_i)
$$



Spoken equation reading: Exponentiating and normalizing the log weights gives batch weights that sum to one, and the WDCE estimate sums each example’s denoising loss multiplied by its batch weight.





The visible context x<ℓ means previously unmasked positions, which need not follow left-to-right order. General jump processes also require the waiting-time terms derived earlier.



TR2-D2 Equation 8. Cancellation of clock and masking terms relies on the proposal and reference sharing the relevant masking schedule. Weights are detached during the supervised denoising update. The companion uses an explicitly known finite proposal to check the importance identity before illustrating full-support tree-search training.



Primary source: [TR2-D2: Tree Search Guided Trajectory-Aware Fine-Tuning for Discrete Diffusion](https://arxiv.org/abs/2509.25171v1)




## Slides 457–459 · Tree search builds a buffer for repeated training

[Completed slide](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u083_b2)





Selection balances reward and exploration. Expansion and rollout produce completed candidates, whose rewards are backed up through the tree. The resulting buffer is repeatedly remasked for training.



Original TR2-D2 Figure 1 and Algorithms 2 and 5. A node is a partially unmasked sequence; children reveal additional content. The method separates search from policy updates, allowing reuse of high-reward candidates. Search changes the sampling distribution, so exact proposal weighting and heuristic search curation must be distinguished.



Primary source: [TR2-D2: Tree Search Guided Trajectory-Aware Fine-Tuning for Discrete Diffusion](https://arxiv.org/abs/2509.25171v1)




## Slides 460–466 · The search score balances reward and exploration

[Completed slide](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u084_b4)



The selection score adds an exploration bonus to the accumulated reward.



$$
U(s,a)=\frac{R(s,a)}{M\,N(s,a)}+c\,p_\theta(a\mid s)\frac{\sqrt{N(s)}}{1+N(s,a)}
$$



Spoken equation reading: The search score adds accumulated reward divided by M times the action visit count to an exploration bonus scaled by c, the model’s action prior, and the parent and action visit counts.



Effective sample size measures how many paths contribute to the weighted update.



$$
N_{\rm eff}=\frac{(\sum_i w_i)^2}{\sum_i w_i^2}=\frac1{\sum_i\overline w_i^2}
$$



Spoken equation reading: Effective sample size is the squared sum of unnormalized weights divided by their squared-weight sum, or equivalently the reciprocal of the squared normalized-weight sum.





High reward encourages reuse, while the bonus encourages exploration. A large buffer can still produce a small effective sample size if a few trajectories dominate.



TR2-D2 Equation 9; R is cumulative reward, M the paper’s normalization factor, N visits, c exploration strength. Unvisited children require an initialization convention. The paper selects among high-scoring candidates with its search policy. A simple base/current token ratio does not automatically correct the additional selection law of a curated MCTS buffer. The companion labels this approximation.



Primary source: [TR2-D2: Tree Search Guided Trajectory-Aware Fine-Tuning for Discrete Diffusion](https://arxiv.org/abs/2509.25171v1)




## Slides 467–469 · Weighted denoising learns from completed candidates

[Completed slide](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u085_b2)



Freeze the sampled trajectories and their weights before remasking the completed sequences.

```python

log_weight = reward/alpha + log_base - log_proposal
weight = torch.softmax(log_weight, dim=0).detach()
masked, mask = remask(sequences)
logits = model(masked)
ce = token_cross_entropy(logits, sequences)
per_sequence = (ce*mask).sum(-1)/mask.sum(-1).clamp_min(1)
loss = (weight*per_sequence).sum()
loss.backward()

```



The companion includes a small abstract-token search and training example, plus an exact finite importance-sampling calculation that verifies the underlying identity.



The slide uses one remasking realization for clarity. The paper repeats masked versions R times and uses its denoising schedule weights. A masked-token predictor sees the entire visible context. Course code reports reward, base-relative KL, and effective sample size; it does not claim to reproduce paper-scale biological evaluations.



Primary source: [TR2-D2: Tree Search Guided Trajectory-Aware Fine-Tuning for Discrete Diffusion](https://arxiv.org/abs/2509.25171v1)




## Slides 470–472 · Weakening the reference penalty changes the tradeoff

[Completed slide](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u086_b2)





Table 1 reports higher predicted activity at α=0.001 than at α=0.1, alongside lower 3-mer correlation and lower base log-likelihood. Reward gains and reference preservation should be read together.



Original Table 1, 640 sequences and three seeds as reported. TR2-D2 α=0.1: activity 6.56±0.02, ATAC 86.9±1.18, 3-mer correlation 0.925±0.002, base log-likelihood −259.4±0.2. α=0.001: 9.74±0.01, 99.9±0.01, 0.548±0.001, −271.8±0.1. These are predictive benchmark measurements. A Pareto archive retains nondominated visited candidates; it does not certify discovery of the global Pareto frontier.



Primary source: [TR2-D2: Tree Search Guided Trajectory-Aware Fine-Tuning for Discrete Diffusion](https://arxiv.org/abs/2509.25171v1)




## Slides 473–475 · The reference can also describe a branching population

[Completed slide](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u087_b2)



A multimodal target can already be handled by an ordinary Schrödinger bridge.

If we also want explicit branch identities and changing branch masses, we need to describe how probability is allocated among those branches.







Primary source: [TR2-D2: Tree Search Guided Trajectory-Aware Fine-Tuning for Discrete Diffusion](https://arxiv.org/abs/2509.25171v1)




## Slides 476–478 · BranchSBM introduces explicit branches with a shared origin

[Completed slide](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u088_b2)





BranchSBM learns branch trajectories, velocity fields, and growth rates so that one source population can reach several weighted terminal populations.



Original paper header. From our lab | ICLR 2026



Primary source: [Branched Schrödinger Bridge Matching](https://arxiv.org/abs/2506.09007v2)




## Slides 479–481 · Four stages learn paths, velocities, and masses

[Completed slide](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u089_b2)





We first learn endpoint-conditioned interpolants, then fit velocity fields to those paths. A growth model allocates branch masses before a joint refinement stage.



Original BranchSBM Figure 1, arXiv 2506.09007v2. Branch labels make the allocation explicit. A conventional SB can already have multimodal marginals; the additional structure here is the branch-specific representation and mass dynamics.



Primary source: [Branched Schrödinger Bridge Matching](https://arxiv.org/abs/2506.09007v2)




## Slides 482–488 · The interpolant preserves each branch’s endpoints

[Completed slide](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u090_b4)



The correction vanishes at both endpoints because it is multiplied by t(1−t).



$$
I_t^k(x_0,x_1^k)=(1-t)x_0+tx_1^k+t(1-t)\phi_\eta^k(t,x_0,x_1^k)
$$



Spoken equation reading: The branch interpolant adds linear motion from the source to that branch’s endpoint to a learned correction multiplied by t times one minus t, which makes the correction zero at both ends.



Differentiating all time-dependent terms gives the conditional velocity target.



$$
\partial_t I_t^k=x_1^k-x_0+(1-2t)\phi_\eta^k+t(1-t)\partial_t\phi_\eta^k
$$



Spoken equation reading: The interpolant’s time derivative equals endpoint displacement plus the derivative of the correction’s time factor and the correction network’s own time derivative, by the product rule.





The network changes the route between a source sample and a branch-specific destination. It cannot move the prescribed endpoints through this correction.



BranchSBM conditional interpolant formulation. The endpoints are fixed when taking the time derivative. Forgetting the (1−2t)φ term gives the wrong velocity. The companion computes the derivative with automatic differentiation and compares it with the analytic expression for its small interpolant.



Primary source: [Branched Schrödinger Bridge Matching](https://arxiv.org/abs/2506.09007v2)




## Slides 489–495 · A state cost can guide the route between endpoints

[Completed slide](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u091_b4)



The interpolant is trained with kinetic energy and a state-dependent path cost.



$$
\mathcal L_{\rm traj}=\sum_k\mathbb E\int_0^1\left[\tfrac12\lVert\partial_t I_t^k\rVert^2+V_t(I_t^k)\right]dt
$$



Spoken equation reading: The trajectory loss sums over branches the expected time integral of half the squared interpolant speed plus the state-dependent cost V evaluated along that interpolant.



A separate velocity model then matches the fitted conditional derivative.



$$
\mathcal L_{\rm flow}=\sum_k\mathbb E\left\lVert u_\theta^k(t,I_t^k)-\partial_t I_t^k\right\rVert^2
$$



Spoken equation reading: The flow loss sums over branches the expected squared difference between the learned velocity evaluated on the interpolant and the interpolant’s time derivative.





The first stage chooses conditional routes under the supplied endpoint pairing. The second stage gives a velocity that can be evaluated from the current state alone.



The state cost can encode constraints or a data-manifold preference. The conditional-expectation minimizer is E[∂ₜIᵏ|Iᵏ=x]. Jensen’s inequality bounds its mean kinetic energy by the conditional energy. This does not by itself prove optimality over all endpoint couplings; choosing or optimizing that coupling is a separate part of the transport problem.



Primary source: [Branched Schrödinger Bridge Matching](https://arxiv.org/abs/2506.09007v2)




## Slides 496–502 · Branch weights record how much mass follows each trajectory

[Completed slide](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u092_b4)



The primary branch begins with unit mass and the secondary branches with zero.



$$
w_0^0=1,\quad w_0^k=0\ (k>0),\qquad w_t^k=w_0^k+\int_0^t g_s^k\,ds
$$



Spoken equation reading: The primary branch starts with weight one and every secondary branch starts with zero, and each later branch weight equals its initial weight plus the integrated growth rate.



A weighted mixture reconstructs the total population at each time.



$$
\mu_t=\sum_k\mathbb E\!\left[w_t^k\,\delta_{X_t^k}\right],\qquad w_t^k\geq0
$$



Spoken equation reading: The population measure is the sum over branches of their expected point masses at their current states, each multiplied by a nonnegative branch weight.





The state trajectory says where a branch goes. Its weight says how much of the population it carries.



Weights and growth can depend on the sampled source trajectory. δ is a unit point mass. Nonnegative weights are needed to obtain a positive measure. In the mass-conserving setting, the total weights sum to one; generalized unbalanced models can instead prescribe a varying total mass.



Primary source: [Branched Schrödinger Bridge Matching](https://arxiv.org/abs/2506.09007v2)




## Slides 503–509 · The weighted continuity equation includes growth

[Completed slide](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u093_b4)



Differentiate a weighted test function along a branch trajectory.



$$
\frac d{dt}\mathbb E[w_t^k f(X_t^k)]=\mathbb E[g_t^k f(X_t^k)+w_t^k\nabla f(X_t^k)\cdot u_t^k]
$$



Spoken equation reading: The time derivative of a branch’s weighted test-function expectation has one term from growth changing its weight and another from velocity changing the test function along its motion.



Choosing the constant function f=1 gives the conservation condition.



$$
\frac d{dt}\mu_t(\mathcal X)=\sum_k\mathbb E[g_t^k],\qquad \sum_k g_t^k=0\ \Longrightarrow\ \mu_t(\mathcal X)=1
$$



Spoken equation reading: The total population mass changes at the sum of expected branch growth rates; starting from mass one, growth rates summing to zero conserve total mass one.





Transport changes where the mass is located. Growth changes how much mass is assigned to a branch. Their sum can still conserve the total population.



This is the weak form of ∂ₜμᵏ+div(uᵏμᵏ)=sᵏ, where the signed source measure is sᵏ=E[gᵏδXᵏ]. It is not generally gᵏμᵏ unless a relative growth rate is defined. The pointwise condition Σgᵏ=0 is sufficient for total conservation; it need not cancel the sources pointwise in space when branches occupy different locations.



Primary source: [Branched Schrödinger Bridge Matching](https://arxiv.org/abs/2506.09007v2)




## Slides 510–516 · We can allocate sixty percent to one destination

[Completed slide](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u094_b4)



Choose linear branch weights so that the source mass is gradually reassigned.



$$
(w_t^0,w_t^1,w_t^2)=(1-t,\,0.6t,\,0.4t),\qquad (g_t^0,g_t^1,g_t^2)=(-1,0.6,0.4)
$$



Spoken equation reading: The primary branch loses weight at rate one while the other two gain weight at rates 0.6 and 0.4, producing weights one minus t, 0.6t, and 0.4t.



At the midpoint and endpoint, every weight is nonnegative and the total stays one.



$$
w_{1/2}=(0.5,0.3,0.2),\qquad w_1=(0,0.6,0.4),\qquad \sum_k w_t^k=1
$$



Spoken equation reading: The branch weights are (0.5, 0.3, 0.2) halfway through and (zero, 0.6, 0.4) at the endpoint, while their sum remains one at every time.





This allocation exactly satisfies the mass constraints. Learning is needed when the branch destinations, geometry, and growth functions are more complicated.



Course example. The primary branch loses mass while the two secondary branches gain it. These are trajectory weights, so a shared origin can launch different conditional routes even when the secondary weights start at zero. The companion checks nonnegativity, terminal proportions, and conservation on a time grid.



Primary source: [Branched Schrödinger Bridge Matching](https://arxiv.org/abs/2506.09007v2)




## Slides 517–523 · Growth losses encourage valid branch masses

[Completed slide](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u095_b4)



The terminal matching term compares each final branch weight with its target.



$$
\mathcal L_{\rm match}=\sum_k\mathbb E[(w_1^k-w_{\rm target}^k)^2]
$$



Spoken equation reading: The terminal matching loss sums, across branches, the expected squared difference between the branch’s final weight and its prescribed target weight.



A separate penalty discourages incorrect total mass and negative branch weights.



$$
\mathcal L_{\rm mass}=\mathbb E\int_0^1\left[(\sum_k w_t^k-w_{\rm total}(t))^2+\sum_k\max(0,-w_t^k)\right]dt
$$



Spoken equation reading: The mass penalty integrates the squared error in total branch weight plus the sum of negative-weight violations, then averages this quantity over the sampled trajectories.





These are training penalties. Small finite loss provides an approximate constraint, while the analytic allocation in our example satisfies the constraints exactly.



BranchSBM growth and mass loss formulations. Quadrature approximates the time integral. A mass penalty evaluated at sampled times is not a guarantee of positivity at every time. Hard parameterizations or sufficiently controlled integration can enforce stronger guarantees when needed.



Primary source: [Branched Schrödinger Bridge Matching](https://arxiv.org/abs/2506.09007v2)




## Slides 524–526 · Joint refinement combines energy and mass losses

[Completed slide](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u096_b2)



After fitting interpolants, velocities, and growth, we can refine the branch system together.

```python

energy = 0.5*velocity.square().sum(-1) + state_cost
weighted_energy = (weights*energy).sum(-1).mean()
terminal = (final_weights-target_weights).square().mean()
mass = (weights.sum(-1)-1).square().mean()
negative = (-weights).clamp_min(0).mean()
loss = weighted_energy + lam_match*terminal
loss = loss + lam_mass*(mass+negative)
loss.backward()

```



The companion runs all four stages on a small branching example. Its diagnostics include terminal mass error, total mass error, and velocity-fit error.



The equation uses normalized units and a chosen state cost. The paper’s fourth stage combines weighted path energy with the relevant fitting and mass penalties. Course code includes a velocity-fit regularizer during refinement so that reducing energy does not simply detach the velocity from its interpolant.



Primary source: [Branched Schrödinger Bridge Matching](https://arxiv.org/abs/2506.09007v2)




## Slides 527–529 · Explicit branching improves the split-population fit

[Completed slide](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u097_b2)





On the LiDAR experiment, reported W₁ falls from 0.975 to 0.239 compared with the single-branch baseline. The held-out cell-differentiation endpoint W₁ falls from 0.940 to 0.210.



Original BranchSBM Table 1 and Figure 3, and Table 3. LiDAR uses 100 Euler steps and five runs. Hematopoiesis intermediate W₁ is 0.582 versus 0.366, and terminal W₁ is 0.940 versus 0.210. These are benchmark comparisons with the specified single-branch implementation; multimodality is not impossible for a general SB.



Primary source: [Branched Schrödinger Bridge Matching](https://arxiv.org/abs/2506.09007v2)




## Slides 530–532 · Intermediate-time accuracy remains a separate test

[Completed slide](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u098_b2)





In the pancreatic benchmark, BranchSBM has lower W₁ at some times and higher W₁ at others. Endpoint accuracy alone does not establish the accuracy of the inferred intermediate dynamics.



Original BranchSBM Table 2. Against DeepRUOT, BranchSBM improves t=2 (7.4643 versus 8.0773) and t=7 (6.8702 versus 7.8346), while t=1 is worse (11.9774 versus 8.0447), as are t=3 through t=6. Training information and objectives differ. The reported uncertainties and zero-rounded standard deviations should be kept as printed.



Primary source: [Branched Schrödinger Bridge Matching](https://arxiv.org/abs/2506.09007v2)




## Slides 533–535 · Branch identity describes the population’s outcomes

[Completed slide](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u099_b2)



We may also need to model components whose motions influence one another throughout the trajectory.

That brings us to an interacting reference process and a coupled control field.







Primary source: [Branched Schrödinger Bridge Matching](https://arxiv.org/abs/2506.09007v2)




## Slides 536–538 · EntangledSBM learns a coupled control field

[Completed slide](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u100_b2)





EntangledSBM combines an interacting stochastic reference, a geometry-constrained bias, and cross-entropy learning from weighted trajectories.



Original paper header. From our lab | arXiv 2025



Primary source: [Entangled Schrödinger Bridge Matching](https://arxiv.org/abs/2511.07406v1)




## Slides 539–541 · Each control can depend on the whole configuration

[Completed slide](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u101_b2)





The model uses the joint state to produce component-specific controls. Interactions in the reference and interactions in the learned control both influence the resulting trajectory.



Original EntangledSBM Figure 1, arXiv 2511.07406v1. The state may contain positions and velocities of multiple particles or samples. Entanglement here refers to learned coupling of the components, not a claim of quantum entanglement.



Primary source: [Entangled Schrödinger Bridge Matching](https://arxiv.org/abs/2511.07406v1)




## Slides 542–548 · The underdamped reference evolves position and velocity

[Completed slide](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u102_b4)



Position changes according to the current velocity.



$$
dr_i=v_i\,dt,\qquad \sigma_i=\sqrt{2\gamma k_BT/m_i}
$$



Spoken equation reading: A particle’s position increment is its velocity times the time increment, and its thermal noise scale is the square root of twice friction times thermal energy divided by mass.



Forces, friction, learned bias, and thermal noise determine the velocity increment.



$$
dv_i=[m_i^{-1}(-\nabla_iU(R)+b_i(R,V))-\gamma v_i]dt+\sigma_i\,dW_i
$$



Spoken equation reading: The velocity increment combines potential force and learned force bias divided by mass, subtracts friction proportional to velocity, and adds a Brownian increment scaled by the particle’s thermal noise.





The potential U couples the reference dynamics. The bias b can depend on the full configuration, so each component’s control can respond to the others.



EntangledSBM underdamped formulation. R here denotes the position configuration, while the earlier italic R denoted a reference path law; use context to distinguish them. mᵢ is mass, γ friction, kBT thermal energy. Brownian increments are independent unless a correlated noise model is specified. Degeneracy in position noise is compatible with Girsanov when control is applied in the noisy velocity directions.



Primary source: [Entangled Schrödinger Bridge Matching](https://arxiv.org/abs/2511.07406v1)




## Slides 549–555 · The control cost must use the units of the reference noise

[Completed slide](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u103_b4)



Write the drift change as σu so that u is measured relative to the noise covariance.



$$
dZ_t=[a(Z_t)+\sigma(Z_t)u_\theta(Z_t)]dt+\sigma(Z_t)dW_t
$$



Spoken equation reading: The controlled state increment uses reference drift a plus noise matrix σ times a dimensionless control u, and the same matrix σ scales the Brownian increment.



For the velocity equation above, a force bias corresponds to a noise-scaled control.



$$
u_i=\frac{b_i}{\sqrt{2\gamma k_BT\,m_i}},\qquad \operatorname{KL}(P^u\Vert R)=\frac12\mathbb E_{P^u}\int_0^1\sum_i\lVert u_i\rVert^2dt
$$



Spoken equation reading: The force bias divided by its mass-and-noise scale gives control u; with matching initial laws, path KL is half the controlled expectation of the time-integrated sum of squared controls.





A quadratic penalty on the raw force has the correct path-KL meaning only after the mass and noise scales are accounted for.



Assume the same initial law and the usual absolute-continuity and integrability conditions. σᵢ=√(2γkBT/mᵢ) in velocity; the drift addition bᵢ/mᵢ equals σᵢuᵢ. The energy in physical units is Σᵢ∫||bᵢ||²/(4γkBT mᵢ)dt. With a general covariance use the corresponding whitened norm; unreachable control directions can give singular measures.



Primary source: [Entangled Schrödinger Bridge Matching](https://arxiv.org/abs/2511.07406v1)




## Slides 556–562 · The bias has aligned and perpendicular components

[Completed slide](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u104_b4)



Normalize the target gradient and project a free vector into its perpendicular space.



$$
s_i=\nabla_i\log\pi_B,\quad \widehat s_i=s_i/\lVert s_i\rVert,\quad b_i=\operatorname{softplus}(\alpha_i)\widehat s_i+(I-\widehat s_i\widehat s_i^\top)h_i
$$



Spoken equation reading: For a nonzero target score s, the bias adds a positive softplus-scaled component along its normalized direction to the component of a free vector h perpendicular to that direction.



The perpendicular component has zero dot product with the target gradient.



$$
b_i\cdot s_i=\operatorname{softplus}(\alpha_i)\lVert s_i\rVert\geq0
$$



Spoken equation reading: The bias–score dot product is the positive softplus coefficient times the score norm, because the perpendicular component contributes zero, so their local alignment is nonnegative.





The aligned component pushes locally toward higher target density. The perpendicular component allows the system to change direction while preserving this local alignment.



The identity uses (I−ŝŝᵀ)s=0. This defines a cone of allowed bias directions when the gradient is nonzero. At a zero gradient a normalization rule or separate branch is required; the companion uses a small-norm guard. Alignment is an instantaneous statement about the bias contribution, not a monotonicity guarantee for the full noisy dynamics.



Primary source: [Entangled Schrödinger Bridge Matching](https://arxiv.org/abs/2511.07406v1)




## Slides 563–569 · A finite inward step still needs a step-size bound

[Completed slide](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u105_b4)



For a point target, write d=x★−x and expand the squared distance after a bias-only step.



$$
\lVert d-\Delta t\,b\rVert^2-\lVert d\rVert^2=-2\Delta t\,d\cdot b+\Delta t^2\lVert b\rVert^2
$$



Spoken equation reading: The change in squared target distance after a bias-only step equals minus twice the step size times inward alignment plus the squared step size times squared bias magnitude.



A positive inward component gives a sufficient upper bound on the step size.



$$
0\leq\Delta t\leq\frac{2d\cdot b}{\lVert b\rVert^2}\quad\Longrightarrow\quad \lVert x+\Delta t b-x^\star\rVert\leq\lVert x-x^\star\rVert
$$



Spoken equation reading: A nonnegative step size no larger than twice the inward dot product divided by squared bias magnitude ensures that the bias-only update does not increase distance to the point target.





Orthogonal motion contributes a second-order increase in distance. The inward component must be strong enough for the chosen finite step.



This proof concerns a radial target direction and a position-like bias-only update. A general ∇logπB need not point toward a single target. The complete dynamics also contain the base force, friction, velocity, and noise. For independent zero-mean noise, the expected squared increment gains a positive covariance-trace term.



Primary source: [Entangled Schrödinger Bridge Matching](https://arxiv.org/abs/2511.07406v1)




## Slides 570–576 · A two-dimensional calculation shows both parts of the bias

[Completed slide](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u106_b4)



Use displacement (3,4), free vector (2,−1), and aligned magnitude 0.8.



$$
\widehat s=(0.6,0.8),\quad h_\perp=(1.76,-1.32),\quad b=0.8\widehat s+h_\perp=(2.24,-0.68)
$$



Spoken equation reading: Normalizing displacement (3,4) gives direction (0.6,0.8), projecting the free vector gives perpendicular component (1.76, minus 1.32), and adding the aligned component yields bias (2.24, minus 0.68).



The dot product is positive, and a step of 0.1 reduces squared distance.



$$
d\cdot b=4,\quad \Delta t_{\max}=1.45985,\quad \lVert d-0.1b\rVert^2=24.2548<25
$$



Spoken equation reading: The displacement–bias dot product is four, the maximum allowed step is approximately 1.45985, and using step 0.1 reduces squared target distance from 25 to 24.2548.





The bias has a substantial sideways component while still making progress toward the point target. The companion verifies the projection and the finite-step bound.



Compute h·ŝ=0.4, so h⊥=h−0.4ŝ=(1.76,−1.32). ||b||²=5.48 and 2d·b/||b||²=8/5.48. At Δt=0.1 the change is −0.8+0.0548=−0.7452. This is a geometric unit test, not a physical molecular trajectory.



Primary source: [Entangled Schrödinger Bridge Matching](https://arxiv.org/abs/2511.07406v1)




## Slides 577–583 · Cross-entropy fits the target path distribution

[Completed slide](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u107_b4)



A terminal potential g defines a path law through a reference-relative tilt.



$$
P^*(d\omega)=Z^{-1}g(X_1)R(d\omega)
$$



Spoken equation reading: The target path probability is its reference probability multiplied by the terminal potential g and divided by Z, the normalizer over reference trajectories.



Cross-entropy minimizes KL from the target path law to the learned process.



$$
\min_\theta\ -\mathbb E_{P^*}[\log p_\theta(\omega)]\quad\Longleftrightarrow\quad\min_\theta\operatorname{KL}(P^*\Vert P_\theta)
$$



Spoken equation reading: Minimizing the target expectation of negative learned path log likelihood is equivalent to minimizing KL from the fixed target path law to the learned path law.





The optimization direction differs from the earlier reward-minus-KL formulation, but both can have the same optimum when the target law is representable.



The target entropy is constant in θ. Cross-entropy is convex as a function of an unrestricted positive path density, since −log is convex. A neural parameterization and a cone-constrained drift family need not give a convex optimization problem or contain P*. Thus path-law convexity does not imply every neural training run finds a global optimum.



Primary source: [Entangled Schrödinger Bridge Matching](https://arxiv.org/abs/2511.07406v1)




## Slides 584–590 · A terminal potential need not be the terminal density

[Completed slide](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u108_b4)



The terminal tilt multiplies the reference terminal density by its potential.



$$
P_1^*(x)=\frac{R_1(x)g(x)}Z
$$



Spoken equation reading: The tilted terminal density equals the reference terminal density times the terminal potential g divided by the normalizer, so the potential itself is generally not the resulting density.



For a one-endpoint constraint, a density ratio calibrates the tilt to the desired terminal law.



$$
g(x)=\frac{\pi_B(x)}{R_1(x)}\quad\Longrightarrow\quad P_1^*(x)=\pi_B(x)
$$



Spoken equation reading: For the one-endpoint construction, choosing g as the desired density divided by the reference terminal density cancels that reference factor and gives the desired terminal density.





Using g=πB generally produces a product with the reference terminal density. Prescribing both endpoint marginals instead requires the two-potential Schrödinger system.



This distinction matters when interpreting terminal density rewards in EntangledSBM. The second formula assumes πB is absolutely continuous with respect to R₁. It fixes the terminal law but can change a random initial law. For a deterministic start it preserves the initial state; otherwise use both endpoint constraints or conditional normalization according to the intended problem.



Primary source: [Entangled Schrödinger Bridge Matching](https://arxiv.org/abs/2511.07406v1)




## Slides 591–597 · Importance weights transfer the target expectation

[Completed slide](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u109_b4)



Let Pᵛ be the process that actually generated the replay trajectories.



$$
\widetilde w(\omega)=g(X_1)\frac{dR}{dP^v}(\omega),\qquad \mathbb E_{P^*}[F]=\frac{\mathbb E_{P^v}[\widetilde wF]}{\mathbb E_{P^v}[\widetilde w]}
$$



Spoken equation reading: The unnormalized importance weight multiplies the terminal potential by the reference-to-proposal path density ratio; dividing the proposal’s weighted expectation by its mean weight recovers the target expectation.



A batch gives a self-normalized estimate of the cross-entropy objective.



$$
\widehat{\mathcal L}_{\rm CE}=\sum_i\overline w_i[-\log p_\theta(\omega_i)],\qquad \overline w_i=\operatorname{softmax}_i(\log\widetilde w)
$$



Spoken equation reading: The batch cross-entropy estimate sums each negative learned path log likelihood times its normalized importance weight, with those weights obtained by a softmax over log unnormalized weights.





Trajectory likelihoods account for the reference forces, learned proposal control, and noise. The weights remain fixed during each parameter update.



The exact ratio identity requires target support within proposal support. Finite-batch self-normalization is biased. A rapidly changing control can concentrate weights and reduce effective sample size. The same diagnostics introduced for TR2-D2 apply here, although the likelihood comes from continuous dynamics rather than token probabilities.



Primary source: [Entangled Schrödinger Bridge Matching](https://arxiv.org/abs/2511.07406v1)




## Slides 598–604 · The transition density gives a practical loss

[Completed slide](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u110_b4)



An Euler transition has a controlled mean and a known covariance.



$$
Z_{k+1}\mid Z_k\sim\mathcal N\!\left(Z_k+[a_k+\sigma_k u_\theta]\Delta t,\,\sigma_k\sigma_k^\top\Delta t\right)
$$



Spoken equation reading: The Euler transition is Gaussian with mean equal to the present state plus controlled drift times the step size, and covariance equal to σ times its transpose times that step size.



Whitening its residual leaves a squared term plus constants independent of θ.



$$
-\log p_\theta(Z_{k+1}\mid Z_k)=\frac1{2\Delta t}\left\lVert\sigma_k^+\!\left[\Delta Z_k-a_k\Delta t-\sigma_ku_\theta\Delta t\right]\right\rVert^2+C
$$



Spoken equation reading: The negative transition log likelihood is one over twice the step size times the squared noise-whitened residual after subtracting reference and control drift increments, plus constants independent of θ.





We sum the transition losses along each trajectory and then apply its importance weight. For underdamped dynamics, the likelihood is evaluated in the noisy velocity coordinates.



σ⁺ denotes an inverse on the noisy subspace, or a pseudoinverse with matching support conditions. With θ-independent covariance, its log determinant is a constant for optimization. This discretized likelihood approximates the continuous path objective as the step size decreases under regularity; it is exact for the specified Euler chain.



Primary source: [Entangled Schrödinger Bridge Matching](https://arxiv.org/abs/2511.07406v1)




## Slides 605–607 · The update uses detached trajectory weights

[Completed slide](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u111_b2)



Compute each sampled trajectory’s Gaussian transition loss under the new control.

```python

u = coupled_control(state)
mean = state + (base_drift + sigma*u)*dt
residual = (next_state-mean)/(sigma*dt**0.5)
path_nll = 0.5*residual.square().sum(dim=(-1,-2))
weight = torch.softmax(log_target-log_proposal, 0).detach()
loss = (weight*path_nll).sum()
optimizer.zero_grad()
loss.backward()

```



The companion uses interacting abstract particles to check the coupled control, geometry, likelihood weighting, and training update.



The compact code assumes elementwise positive noise scales in the modeled noisy coordinates. The companion uses a normalized overdamped synthetic system; the preceding derivation explains how to transfer the calculation to velocity coordinates in the underdamped model. This is a method demonstration, not a molecular simulation benchmark.



Primary source: [Entangled Schrödinger Bridge Matching](https://arxiv.org/abs/2511.07406v1)




## Slides 608–610 · The benchmark also tests unseen populations

[Completed slide](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u112_b2)





For the reported 50-PC Clonidine setting, trained-population W₁ falls from 5.947 for the base process to 0.342 with CE. On the unseen population, it falls from 8.217 to 0.538.



Original EntangledSBM Table 1. The variant without velocity conditioning reports 1.741 and 2.907. This comparison changes conditioning and loss, so it does not isolate the effect of CE alone. W₁ and W₂ are computed on the top two PCs; MMD uses the full PC representation. Results use the paper’s 100-step evaluation, 16 interacting particles, and five runs. These are reported data-modeling results, separate from the abstract-particle course code.



Primary source: [Entangled Schrödinger Bridge Matching](https://arxiv.org/abs/2511.07406v1)




## Slides 611–613 · Higher hit rates can come with higher energies

[Completed slide](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u113_b2)





In the protein-transition benchmark, EntangledSBM reports higher target-hit rates than TPS-DPS in the displayed systems, alongside higher transition-state energies. Both measurements are needed to assess the generated paths.



Original EntangledSBM Table 2. Alanine target-hit percentage 92.19 versus 76.00 for TPS, with energy 47.91 versus 22.79. Chignolin 64.06 versus 59.38, energy 2825.61 versus −780.18. Trpcage 82.81 versus 81.25, energy 765.74 versus −317.61; TPS has lower RMSD. BBA 96.88 versus 84.38, energy 1453.80 versus −3801.68. These values use the paper’s stated units and evaluation; the cone alone does not certify physically optimal transition paths.
The original Table 2 baseline is labelled TPS-DPS (Scalar).



Primary source: [Entangled Schrödinger Bridge Matching](https://arxiv.org/abs/2511.07406v1)




## Slides 614–616 · The reference determines the cost of changing paths

[Completed slide](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u114_b2)



A Brownian reference gives quadratic drift control. A jump reference gives a rate-relative-entropy cost. An interacting reference also encodes the original forces and noise.

The endpoint constraints and the reference together define the bridge we are trying to learn.







Primary source: [Entangled Schrödinger Bridge Matching](https://arxiv.org/abs/2511.07406v1)




## Slides 617–621 · We can now go from endpoint constraints to a sampler

[Completed slide](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u115_b4)



Optimal transport chooses a coupling that minimizes a specified movement cost.

Entropic transport adds a preference for spreading mass relative to a positive reference kernel.

A Schrödinger bridge chooses a complete path law with the required endpoints and minimum change from the reference dynamics.

Bridge matching turns conditional paths into trainable drifts, rates, or categorical transitions.







Primary source: [Foundations of Schrödinger Bridges for Generative Modeling — Tang](https://arxiv.org/abs/2603.18992v1)




## Slides 622–624 · The algorithms approximate different parts of the bridge

[Completed slide](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u116_b2)





The shared question is how to satisfy the endpoint requirements while respecting a reference. The computational route depends on whether we can represent the coupling, conditional bridge, and learned dynamics.



Course synthesis. Sinkhorn scales finite endpoint kernels. DSB alternates endpoint projections through learned reversal. DSBM and discrete variants alternate Markovian and reciprocal structure. SF²M uses an entropic coupling with conditional regression. The lab methods add reward, branching, or interacting-control structure. Consult the cited paper assumptions before transferring a theorem between rows.



Primary source: [Foundations of Schrödinger Bridges for Generative Modeling — Tang](https://arxiv.org/abs/2603.18992v1)




## Slides 625–629 · The companion lets us inspect each computation

[Completed slide](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u117_b4)



Start with the exact 2×2 OT problem, check its dual certificate, and reproduce every Sinkhorn update.

Compare an enumerated path bridge, discrete IMF, a continuous-time jump bridge, and the analytic Gaussian bridge.

Then run the small learned examples for DSB, DSBM, [SF]²M, TR2-D2, BranchSBM, and EntangledSBM.

The exercises ask which quantities are exact, which are learned, and which are changed by discretization or sampling.







Primary source: [Foundations of Schrödinger Bridges for Generative Modeling — Tang](https://arxiv.org/abs/2603.18992v1)




## Slides 630–632 · We began with a distribution over structured data

[Completed slide](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_end01_b2)



Probability, conditioning, and the state space gave us the objects to model. Autoregressive factorization showed how a sequence distribution becomes a sampler.

We then connected neural predictions to continuous motion, discrete transitions, and denoising. The network became one component of a complete generative process.



This closes the full course. Connect probability and autoregression to the later dynamics-based methods without claiming that every model is the same stochastic process.



Primary source: [Foundations of Schrödinger Bridges for Generative Modeling — Tang](https://arxiv.org/abs/2603.18992v1)




## Slides 633–642 · Motion and jumps share a probability evolution law

[Completed slide](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_end02_b6)



The generator describes how expected test functions change.



$$
\frac{d}{dt}\mathbb{E}_{p_t}[f(X)]=\mathbb{E}_{p_t}[\mathcal L_t f(X)]
$$



Spoken equation reading: The time derivative of the expectation of a test function f equals the expectation of the generator applied to f, both evaluated under the current probability law.



For continuous states, drift and noise define the generator.



$$
\mathcal L_t f=b_t\!\cdot\!\nabla f+\tfrac12\operatorname{tr}(a_t\nabla^2 f),\qquad a_t=\sigma_t\sigma_t^{\mathsf T}
$$



Spoken equation reading: For continuous states, the generator is drift dotted with the gradient of f plus one half the trace of noise covariance times the Hessian of f.



For discrete states, rates weight the possible changes.



$$
\mathcal L_t f(x)=\sum_{y\ne x}q_t(x,y)\,[f(y)-f(x)]
$$



Spoken equation reading: For discrete states, the generator sums over possible destinations the jump rate from x to y multiplied by the change in the test function from f(x) to f(y).





The continuity, Fokker–Planck, and master equations describe the corresponding probability evolution. We learned to parameterize valid dynamics and turn them into samplers.



Assume sufficient regularity and integrability for the generator identity. The deterministic flow case has a=0; the jump case has nonnegative off-diagonal rates. These identities unify the Markov dynamics studied in the course. Autoregressive factorization is summarized separately, without identifying it with a diffusion.



Primary source: [Foundations of Schrödinger Bridges for Generative Modeling — Tang](https://arxiv.org/abs/2603.18992v1)




## Slides 643–645 · The frameworks change what we learn and how we sample

[Completed slide](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_end03_b2)





The state space, training target, and sampler must agree. Changing one of them can change the distribution or path law that the method produces.



This table summarizes the course rather than asserting one-to-one equivalence among the methods. Flow maps include deterministic, categorical, latent, and stochastic constructions; their precise sampling semantics depend on the formulation.



Primary source: [Foundations of Schrödinger Bridges for Generative Modeling — Tang](https://arxiv.org/abs/2603.18992v1)




## Slides 646–648 · We can now turn an idea into a generative method

[Completed slide](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_end04_b2)



Start by specifying the state space, the desired distribution, and the process that connects them. Derive the training target and the sampling rule together.

Then test the smallest case you can solve exactly. Separate what the theory guarantees from what the implementation approximates, and what the experiments actually show.



This closes the full course. Connect probability and autoregression to the later dynamics-based methods without claiming that every model is the same stochastic process.



Primary source: [Foundations of Schrödinger Bridges for Generative Modeling — Tang](https://arxiv.org/abs/2603.18992v1)




## Slides 649–649 · If you would like to go deeper

[Completed slide](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_reference_b0)





Sophia Tang’s Foundations of Schrödinger Bridges for Generative Modeling is a detailed reference for the theory and connections developed in this course.



Optional further reading, not a research-paper walkthrough. Sophia Tang, Foundations of Schrödinger Bridges for Generative Modeling, arXiv:2603.18992v1. Computational Optimal Transport by Peyré and Cuturi (1803.00567v4) and Léonard’s survey (1308.0215v1) provide complementary foundational treatments.



Primary source: [Foundations of Schrödinger Bridges for Generative Modeling — Tang](https://arxiv.org/abs/2603.18992v1)




## Slides 650–650 · We can now derive a model, build its sampler, and test what it actually learns.

[Completed slide](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_end05_b0)





We can now derive a model, build its sampler,
and test what it actually learns.



Final course closing. Preserve the native four-corner artwork, Ubuntu centered message, and uncluttered white background used in the earlier lecture endings.



Primary source: [Foundations of Schrödinger Bridges for Generative Modeling — Tang](https://arxiv.org/abs/2603.18992v1)




## Slides 651–651 · Thank you for a great semester! I look forward to seeing what you build.

[Completed slide](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_end06_b0)





Thank you for a great semester!
I look forward to seeing what you build.



Final course closing. Preserve the native four-corner artwork, Ubuntu centered message, and uncluttered white background used in the earlier lecture endings.



Primary source: [Foundations of Schrödinger Bridges for Generative Modeling — Tang](https://arxiv.org/abs/2603.18992v1)

