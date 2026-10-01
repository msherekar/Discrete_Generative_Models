# Lecture 7 · Equation readings

The presentation reveals the equation, then its spoken reading, then intuition. This guide reproduces the literal readings; the teaching notes retain the surrounding derivations and explanations.

## A transport map chooses a destination for every starting point

[Equation reading 1](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u003_r0)

Applying the map T to a source sample must produce the target distribution.

$$
T_{\#}\mu=\nu\quad\Longleftrightarrow\quad X\sim\mu\ \Rightarrow\ T(X)\sim\nu
$$

The pushforward condition says that applying T to a random sample with law μ produces a sample with law ν; the map must transform the entire source distribution into the target distribution.

[Equation reading 2](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u003_r1)

Monge chooses the admissible map with the smallest average transport cost c.

$$
\inf_{T:\,T_{\#}\mu=\nu}\ \int c(x,T(x))\,\mu(dx)
$$

The Monge objective chooses, among maps that send μ to ν, the smallest average of the movement cost c from each source point x to its assigned destination T(x).

## A single source atom may need to split its probability

[Equation reading 1](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u004_r0)

Our source has masses 0.6 and 0.4, while the target needs masses 0.3 and 0.7.

$$
a=(0.6,0.4),\qquad b=(0.3,0.7)
$$

The source probability vector a assigns masses 0.6 and 0.4 to its two states, while the target vector b requires masses 0.3 and 0.7.

[Equation reading 2](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u004_r1)

A coupling assigns a nonnegative mass πᵢⱼ to each source and target pair.

$$
\Pi(a,b)=\{\pi\geq0:\ \pi\mathbf1=a,\ \pi^{\mathsf T}\mathbf1=b\}
$$

The set Π(a,b) contains nonnegative matrices π whose row sums equal a and whose column sums equal b; each matrix entry specifies how much probability travels between one source and one target.

## We can now minimize the cost over all valid couplings

[Equation reading 1](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u005_r0)

Kantorovich minimizes the expected cost under the joint distribution π.

$$
\operatorname{OT}_{c}(\mu,\nu)=\inf_{\pi\in\Pi(\mu,\nu)}\int c(x,y)\,\pi(dx,dy)
$$

The Kantorovich cost is the smallest average movement cost c(x,y) over joint distributions π with source marginal μ and target marginal ν.

[Equation reading 2](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u005_r1)

For finite supports, the integral becomes a sum over entries of the cost matrix C.

$$
\min_{\pi\in\Pi(a,b)}\langle C,\pi\rangle,\qquad C_{ij}=\lVert x_i-y_j\rVert^2
$$

For finite states, the objective sums each transported mass π times its corresponding matrix cost C, where C is the squared distance between the source and target locations.

## The constraints reduce our example to one unknown

[Equation reading 1](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u006_r0)

For source locations 0 and 2 and target locations 1 and 3, squared distance gives C.

$$
C=\begin{pmatrix}1&9\\1&1\end{pmatrix}
$$

The cost matrix assigns cost one to three possible moves and cost nine to the move from the first source at zero to the second target at three.

[Equation reading 2](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u006_r1)

Let r be the mass sent from the first source to the first target.

$$
\pi(r)=\begin{pmatrix}r&0.6-r\\0.3-r&0.1+r\end{pmatrix},\qquad 0\leq r\leq0.3
$$

Choosing the first entry r fixes the other three transported masses through the marginal constraints; nonnegativity restricts r to the interval from zero to 0.3.

## Increasing r removes the most expensive move

[Equation reading 1](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u007_r0)

Multiplying each transported mass by its cost gives a linear function of r.

$$
\langle C,\pi(r)\rangle=r+9(0.6-r)+(0.3-r)+(0.1+r)=5.8-8r
$$

The transport cost is the sum of four mass-times-cost terms, which simplifies to 5.8 minus eight times r; increasing r therefore decreases the objective.

[Equation reading 2](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u007_r1)

The largest feasible r gives the optimal coupling and total cost.

$$
r^*=0.3,\qquad \pi^*=\begin{pmatrix}0.3&0.3\\0&0.4\end{pmatrix},\qquad \operatorname{OT}=3.4
$$

Setting r to its largest feasible value, 0.3, gives masses 0.3, 0.3, zero, and 0.4 in the coupling matrix and yields total transport cost 3.4.

## A second optimization can certify that this cost is minimal

[Equation reading 1](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u008_r0)

Introduce source prices f and target prices g in the Lagrangian.

$$
\mathcal L(\pi,f,g)=a^{\mathsf T}f+b^{\mathsf T}g+\sum_{ij}\pi_{ij}(C_{ij}-f_i-g_j)
$$

The Lagrangian adds the source and target masses weighted by prices f and g, then sums each transported mass times its cost minus the two endpoint prices.

[Equation reading 2](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u008_r1)

Minimizing over nonnegative π leaves the dual feasibility condition.

$$
\max_{f,g}\ a^{\mathsf T}f+b^{\mathsf T}g\quad\text{subject to}\quad f_i+g_j\leq C_{ij}
$$

The dual maximizes the total mass-weighted source and target prices, subject to the condition that the two prices on every possible move sum to at most that move’s cost.

## Our example has an exact primal and dual certificate

[Equation reading 1](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u009_r0)

The source and target prices below satisfy every cost inequality.

$$
f=(0,-8),\quad g=(1,9),\quad f\mathbf1^{\mathsf T}+\mathbf1g^{\mathsf T}=\begin{pmatrix}1&9\\-7&1\end{pmatrix}\leq C
$$

The chosen source prices are zero and minus eight, and the target prices are one and nine; their pairwise sums are no greater than the corresponding entries of C.

[Equation reading 2](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u009_r1)

Their lower bound equals the cost of the coupling we already found.

$$
a^{\mathsf T}f+b^{\mathsf T}g=-3.2+0.3+6.3=3.4=\langle C,\pi^*\rangle
$$

Weighting these prices by a and b gives 3.4, exactly the cost of the proposed coupling, so the feasible dual lower bound meets the feasible primal upper bound.

## Distance costs give the Wasserstein distances

[Equation reading 1](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u010_r0)

The p-Wasserstein distance takes the pth root of the minimum pth-power distance cost.

$$
W_p(\mu,\nu)=\left[\inf_{\pi\in\Pi(\mu,\nu)}\int\lVert x-y\rVert^p\,d\pi\right]^{1/p}
$$

The p-Wasserstein distance minimizes the expected pth power of the distance between coupled samples, then takes the pth root to return to the units of distance.

[Equation reading 2](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u010_r1)

On the real line, matching equal quantiles gives the optimal cost.

$$
W_p^p(\mu,\nu)=\int_0^1\left|F_\mu^{-1}(u)-F_\nu^{-1}(u)\right|^pdu
$$

On the real line, the pth power of the Wasserstein distance averages the pth power of the gap between source and target quantiles at the same probability level u.

## The one-dimensional rule follows by uncrossing pairs

[Equation reading 1](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u011_r0)

For ordered points x₁<x₂ and y₁<y₂, compare crossing and ordered assignments.

$$
(x_1-y_2)^2+(x_2-y_1)^2-(x_1-y_1)^2-(x_2-y_2)^2=2(x_2-x_1)(y_2-y_1)>0
$$

The cost of the two crossed assignments minus the cost of the ordered assignments equals twice the source gap times the target gap, which is positive when both pairs are strictly ordered.

[Equation reading 2](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u011_r1)

Moving any shared positive mass δ from crossing edges to ordered edges lowers the cost.

$$
\Delta\operatorname{cost}=-2\delta(x_2-x_1)(y_2-y_1)<0
$$

Moving a positive mass δ from crossed to ordered assignments changes the cost by minus twice δ times the two positive gaps, so the change is strictly negative.

## We can describe the same transport through a velocity field

[Equation reading 1](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u012_r0)

Benamou and Brenier minimize the kinetic action over densities ρₜ and velocities vₜ.

$$
W_2^2(\mu,\nu)=\inf_{\rho,v}\int_0^1\!\int\lVert v_t(x)\rVert^2\rho_t(x)\,dx\,dt
$$

The squared Wasserstein distance minimizes, over density and velocity paths, the time integral of squared speed averaged under the current density ρ.

[Equation reading 2](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u012_r1)

The continuity equation moves probability while preserving the required endpoints.

$$
\partial_t\rho_t+\nabla\!\cdot(\rho_tv_t)=0,\qquad \rho_0=\mu,\quad\rho_1=\nu
$$

The continuity equation says that the time change in density plus the divergence of probability flux ρv is zero, while the initial and final densities are constrained to μ and ν.

## Straight constant-speed motion attains the dynamic optimum

[Equation reading 1](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u013_r0)

For each path, Cauchy-Schwarz bounds displacement by its integrated squared speed.

$$
\lVert X_1-X_0\rVert^2=\left\lVert\int_0^1\dot X_t\,dt\right\rVert^2\leq\int_0^1\lVert\dot X_t\rVert^2dt
$$

The squared endpoint displacement equals the squared integral of velocity and is bounded above by the integral of squared speed over the unit time interval.

[Equation reading 2](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u013_r1)

An optimal endpoint coupling achieves equality with linear interpolation.

$$
(X_0,X_1)\sim\pi^*,\qquad X_t=(1-t)X_0+tX_1,\quad\dot X_t=X_1-X_0
$$

Sampling endpoints from the optimal coupling and linearly interpolating between them gives a constant velocity equal to the endpoint difference, attaining equality in the speed bound.

## Entropy spreads mass across feasible assignments

[Equation reading 1](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u016_r0)

Let ε>0 control the entropy penalty added to the transport cost.

$$
\min_{\pi\in\Pi(a,b)}\ \langle C,\pi\rangle+\varepsilon\sum_{ij}\pi_{ij}(\log\pi_{ij}-1)
$$

The entropic objective adds the transport cost to ε times the sum of π times log π minus π, and minimizes that total over couplings with the required marginals.

[Equation reading 2](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u016_r1)

Stationarity of the Lagrangian gives the positive kernel K and its two scaling vectors.

$$
\pi_{ij}=e^{f_i/\varepsilon}e^{-C_{ij}/\varepsilon}e^{g_j/\varepsilon}=u_iK_{ij}v_j
$$

Every optimal coupling entry factors into a source scaling u, a kernel entry K equal to the exponential of negative cost divided by ε, and a target scaling v.

## Each marginal constraint gives one scaling update

[Equation reading 1](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u017_r0)

Matching the row sums gives u by dividing a by K times v.

$$
\pi\mathbf1=u\odot(Kv)=a\quad\Longrightarrow\quad u\leftarrow a\oslash(Kv)
$$

The row sums of the scaled kernel are u multiplied elementwise by Kv; matching them to a gives the update that divides each source mass by the corresponding entry of Kv.

[Equation reading 2](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u017_r1)

Matching the column sums then gives v by dividing b by K transpose times u.

$$
\pi^{\mathsf T}\mathbf1=v\odot(K^{\mathsf T}u)=b\quad\Longrightarrow\quad v\leftarrow b\oslash(K^{\mathsf T}u)
$$

The column sums are v multiplied elementwise by the transpose of K times u; matching them to b gives the update that divides each target mass by that corresponding column quantity.

## We can calculate the first Sinkhorn iteration by hand

[Equation reading 1](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u018_r0)

For ε=2, exponentiating the negative half-cost gives the kernel.

$$
K=\begin{pmatrix}0.60653&0.01111\\0.60653&0.60653\end{pmatrix},\qquad v^{(0)}=(1,1)
$$

With ε equal to two, each kernel entry is the exponential of negative half its cost, and the initial target scaling vector v is set to one in both coordinates.

[Equation reading 2](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u018_r1)

Row scaling gives u, then column scaling gives the next v.

$$
u^{(1)}\approx(0.97144,0.32974),\qquad v^{(1)}\approx(0.38013,3.32081)
$$

The first row normalization gives u approximately equal to (0.97144, 0.32974), and the following column normalization gives v approximately equal to (0.38013, 3.32081).

## A path measure assigns probabilities to entire trajectories

[Equation reading 1](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u023_r0)

A Markov path probability multiplies its initial law and transition kernels.

$$
R(x_{0:N})=r_0(x_0)\prod_{k=0}^{N-1}K_k(x_k,x_{k+1})
$$

The reference probability of a complete Markov trajectory equals the probability of its initial state multiplied by the transition probability for every consecutive pair of states.

[Equation reading 2](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u023_r1)

Relative entropy averages the log likelihood ratio of two path measures.

$$
\operatorname{KL}(P\Vert R)=\mathbb E_P\!\left[\log\frac{dP}{dR}\right]
$$

The path-space KL is the expectation under P of the logarithm of P’s density relative to R, so trajectories are weighted according to P when comparing the two path laws.

## The bridge changes the reference as little as possible

[Equation reading 1](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u024_r0)

Fix both endpoint marginals and minimize relative entropy on path space.

$$
P^*=\arg\min_{P:\,P_0=\mu,\,P_1=\nu}\operatorname{KL}(P\Vert R)
$$

The Schrödinger bridge P star minimizes KL relative to the reference path law R over all path laws whose initial marginal is μ and whose terminal marginal is ν.

[Equation reading 2](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u024_r1)

The KL chain rule separates the endpoint cost from the conditional path cost.

$$
\operatorname{KL}(P\Vert R)=\operatorname{KL}(\pi\Vert R_{01})+\mathbb E_{\pi}\operatorname{KL}(P^{xy}\Vert R^{xy})
$$

The total path KL equals the KL between endpoint couplings plus the average, over the chosen endpoint coupling π, of the KL between paths conditioned on those endpoints.

## The static bridge selects the optimal endpoint coupling

[Equation reading 1](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u025_r0)

Nonnegativity of conditional KL makes the reference conditional bridges optimal.

$$
\pi^*=\arg\min_{\pi\in\Pi(\mu,\nu)}\operatorname{KL}(\pi\Vert R_{01}),\qquad P^*(d\omega)=\int R^{xy}(d\omega)\,\pi^*(dx,dy)
$$

The optimal endpoint coupling minimizes KL relative to the reference endpoint law, and the optimal path law mixes the reference’s endpoint-conditioned bridges using that coupling.

[Equation reading 2](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u025_r1)

The optimal coupling rescales the reference endpoint law by two positive potentials.

$$
\pi^*(dx,dy)=f(x)g(y)R_{01}(dx,dy)
$$

The optimal joint endpoint law equals the reference joint law multiplied by a source potential f(x) and a terminal potential g(y), which adjust the two marginals.

## A Brownian reference recovers entropy-regularized transport

[Equation reading 1](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u026_r0)

Brownian motion with noise variance ε has a Gaussian endpoint transition density.

$$
K_{01}(x,y)=(2\pi\varepsilon)^{-d/2}\exp\!\left[-\frac{\lVert y-x\rVert^2}{2\varepsilon}\right]
$$

The Brownian transition density is an isotropic Gaussian centered at x with covariance ε times the identity, so reaching y incurs the exponential of negative squared displacement divided by twice ε.

[Equation reading 2](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u026_r1)

Substituting this density into the static KL gives squared-distance transport plus entropy.

$$
\varepsilon\operatorname{KL}(\pi\Vert R_{01})=\tfrac12\mathbb E_\pi\lVert X_1-X_0\rVert^2-\varepsilon H(\pi)+\text{constant}
$$

For fixed endpoint marginals, ε times the endpoint KL equals half the expected squared displacement minus ε times the coupling entropy, plus a constant independent of the coupling.

## A two-state chain gives a bridge we can compute exactly

[Equation reading 1](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u027_r0)

Let R use the same transition matrix A at two consecutive steps.

$$
a=(0.5,0.5),\quad b=(0.2,0.8),\quad A=\begin{pmatrix}0.8&0.2\\0.3&0.7\end{pmatrix}
$$

The reference starts with equal probabilities, the desired terminal probabilities are 0.2 and 0.8, and each row of A gives one step’s destination probabilities from its corresponding source state.

[Equation reading 2](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u027_r1)

The reference joint endpoint probabilities equal the initial masses times A squared.

$$
R_{02}=\operatorname{diag}(a)A^2=\begin{pmatrix}0.35&0.15\\0.225&0.275\end{pmatrix}
$$

Multiplying the two-step transition matrix A squared by the initial masses gives the reference joint endpoint probabilities 0.35, 0.15, 0.225, and 0.275.

## Sinkhorn rescales the endpoint probabilities of the chain

[Equation reading 1](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u028_r0)

Scaling the reference endpoint matrix to a and b gives π star.

$$
\pi^*=\begin{pmatrix}0.14&0.36\\0.06&0.44\end{pmatrix}
$$

The optimal endpoint coupling assigns joint probabilities 0.14 and 0.36 from the first source and 0.06 and 0.44 from the second, matching both prescribed marginals.

[Equation reading 2](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u028_r1)

One convenient choice of endpoint potentials gives the required scaling.

$$
f=(0.4,\,4/15),\qquad g=(1,6),\qquad \pi^*_{ij}=f_i(R_{02})_{ij}g_j
$$

Multiplying each reference endpoint probability by its source factor f and destination factor g gives the optimal coupling, with g favoring the second destination by a factor of six.

## The terminal potential can be propagated backward

[Equation reading 1](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u029_r0)

Let hₖ be the expected terminal potential given the current state.

$$
h_k(x)=\mathbb E_R[g(X_N)\mid X_k=x],\qquad h_k=A_kh_{k+1},\quad h_N=g
$$

The potential h at state x and step k is the reference expectation of the final potential g given that current state; it is computed backward by multiplying the next potential by A.

[Equation reading 2](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u029_r1)

Doob’s transform uses the ratio of future and current potentials to modify each transition.

$$
Q_k(x,y)=A_k(x,y)\frac{h_{k+1}(y)}{h_k(x)}
$$

The transformed transition Q from x to y equals the reference transition A multiplied by the next potential at y divided by the current potential at x.

## The backward recursion normalizes every transition

[Equation reading 1](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u030_r0)

Summing the transformed transitions gives one by the recursion for h.

$$
\sum_yQ_k(x,y)=\frac{\sum_yA_k(x,y)h_{k+1}(y)}{h_k(x)}=1
$$

Summing Q over all destinations divides the reference-weighted sum of next-step potentials by the current potential; the backward recursion makes this ratio exactly one.

[Equation reading 2](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u030_r1)

Multiplying transitions cancels all intermediate potentials.

$$
\prod_k\frac{Q_k(x_k,x_{k+1})}{A_k(x_k,x_{k+1})}=\frac{g(x_N)}{h_0(x_0)}
$$

Multiplying the transformed-to-reference transition ratios along a path cancels every intermediate potential, leaving only the final potential g divided by the initial future potential h.

## The transformed chain reaches the requested endpoint exactly

[Equation reading 1](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u031_r0)

Substituting the potentials gives the two transition matrices.

$$
Q_0=\begin{pmatrix}0.64&0.36\\0.16&0.84\end{pmatrix},\qquad Q_1=\begin{pmatrix}0.4&0.6\\1/15&14/15\end{pmatrix}
$$

The first transformed transition uses rows (0.64, 0.36) and (0.16, 0.84), while the second uses rows (0.4, 0.6) and (one fifteenth, fourteen fifteenths).

[Equation reading 2](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u031_r1)

Successive probability updates give the intermediate and terminal distributions.

$$
(0.5,0.5)Q_0=(0.4,0.6),\qquad (0.4,0.6)Q_1=(0.2,0.8)
$$

Multiplying the initial probability vector by the first transition gives (0.4, 0.6), and multiplying again by the second gives the required terminal probabilities (0.2, 0.8).

## For a diffusion, we can change the drift and keep the noise

[Equation reading 1](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u033_r0)

The reference drift b and noise scale √ε define the original stochastic paths.

$$
R:\ dX_t=b_t(X_t)dt+\sqrt\varepsilon\,dW_t
$$

Under the reference law R, the state increment equals drift b times the time increment plus a Brownian increment scaled by the square root of ε.

[Equation reading 2](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u033_r1)

A control u changes the drift and defines the path measure P superscript u.

$$
P^u:\ dX_t=[b_t(X_t)+u_t(X_t)]dt+\sqrt\varepsilon\,dW_t^u
$$

Under the controlled law, the increment adds control u to the reference drift b while retaining the same noise scale; the resulting trajectories define the new path law P.

## Girsanov turns drift changes into a path likelihood ratio

[Equation reading 1](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u034_r0)

Under the controlled law, the log ratio contains a martingale and control energy.

$$
\log\frac{dP^u}{dR}=\frac1{\sqrt\varepsilon}\int_0^1u_t\cdot dW_t^u+\frac1{2\varepsilon}\int_0^1\lVert u_t\rVert^2dt
$$

With matching initial laws, the controlled-to-reference log likelihood ratio is a noise-scaled stochastic integral of u plus one over twice ε times the integrated squared control.

[Equation reading 2](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u034_r1)

Taking the controlled expectation removes the martingale term.

$$
\operatorname{KL}(P^u\Vert R)=\frac1{2\varepsilon}\mathbb E_{P^u}\int_0^1\lVert u_t\rVert^2dt
$$

The KL from the controlled process to the reference equals one over twice ε times the controlled expectation of integrated squared control, because the martingale term has zero expectation.

## The backward potential determines the continuous control

[Equation reading 1](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u035_r0)

The conditional expectation h satisfies the backward Kolmogorov equation.

$$
h_t(x)=\mathbb E_R[g(X_1)\mid X_t=x],\qquad \partial_th+b\cdot\nabla h+\tfrac\varepsilon2\Delta h=0
$$

The future potential h averages the terminal potential under the reference conditioned on the current state, and its time derivative plus reference drift and diffusion terms sums to zero.

[Equation reading 2](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u035_r1)

The continuous Doob transform adds the gradient of log h to the drift.

$$
u_t^*(x)=\varepsilon\nabla\log h_t(x),\qquad dX_t=[b_t+\varepsilon\nabla\log h_t](X_t)dt+\sqrt\varepsilon\,dW_t
$$

The optimal added control is ε times the spatial gradient of log h, so the bridge drift is the reference drift plus this control and the Brownian noise scale is unchanged.

## Taking the logarithm gives an optimal-control equation

[Equation reading 1](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u036_r0)

Bellman’s equation minimizes the local control cost plus the change in value V.

$$
\partial_tV+b\cdot\nabla V+\tfrac\varepsilon2\Delta V+\min_u\{u\cdot\nabla V+\tfrac12\lVert u\rVert^2\}=0
$$

Bellman’s equation adds the time, reference-drift, and diffusion changes in value V to the minimum over controls of control–value alignment plus half the squared control norm.

[Equation reading 2](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u036_r1)

Completing the square gives u star and the Hopf-Cole transform linearizes the equation.

$$
u^*=-\nabla V,\qquad V=-\varepsilon\log h\quad\Longrightarrow\quad\partial_th+b\cdot\nabla h+\tfrac\varepsilon2\Delta h=0
$$

The minimizing control is minus the gradient of V, and substituting V equal to minus ε times log h converts the nonlinear control equation into the linear backward equation for h.

## Two potentials recover every intermediate marginal

[Equation reading 1](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u037_r0)

Propagate the initial scaling forward and the terminal scaling backward.

$$
\widehat h_t(y)=\int f(x)r_0(x)K_{0t}(x,y)dx,\qquad h_t(y)=\int K_{t1}(y,z)g(z)dz
$$

The forward potential averages the initial density weighted by f through the reference transition to time t, while the backward potential averages g through the transition from t to the endpoint.

[Equation reading 2](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u037_r1)

Their product gives the bridge density at time t.

$$
\rho_t(y)=\widehat h_t(y)h_t(y),\qquad \rho_0=\mu,\quad\rho_1=\nu
$$

The bridge density at an intermediate state is the product of its forward and backward potentials, with the two potentials chosen so that the endpoint densities equal μ and ν.

## A Brownian bridge gives an explicit training path

[Equation reading 1](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u038_r0)

Given both endpoints, the intermediate state is a Gaussian around the straight line.

$$
X_t\mid x_0,x_1\sim\mathcal N((1-t)x_0+tx_1,\ \varepsilon t(1-t)I)
$$

Conditioned on both endpoints, the Brownian bridge state at an interior time t is Gaussian with the linearly interpolated mean and covariance ε times t times one minus t.

[Equation reading 2](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u038_r1)

The conditional drift points toward the endpoint, scaled by the remaining time.

$$
\beta_t(x\mid x_1)=\frac{x_1-x}{1-t}
$$

The conditional Brownian bridge drift equals the remaining displacement from x to the terminal point divided by the remaining time one minus t.

## Bridge noise changes the Gaussian interior

[Equation reading 1](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u039_r0)

For unit-variance Gaussian endpoints and ε=1, the covariance has a closed form.

$$
\mu=\mathcal N(0,1),\quad\nu=\mathcal N(2,1),\qquad c^*=\frac{\sqrt{\varepsilon^2+4}-\varepsilon}{2}\approx0.61803
$$

For unit-variance Gaussian endpoints with means zero and two, the optimal endpoint covariance is half of the square root of ε squared plus four minus ε, approximately 0.61803 when ε is one.

[Equation reading 2](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u039_r1)

At the midpoint, conditional interpolation and Brownian variance add together.

$$
\mathbb E[X_{1/2}]=1,\qquad \operatorname{Var}(X_{1/2})=\tfrac12+\tfrac12c^*+\tfrac\varepsilon4\approx1.05902
$$

The midpoint mean is one, while its variance combines one half from the endpoint variances, one half of their covariance, and ε divided by four from the conditional Brownian bridge.

## Iterative proportional fitting corrects one endpoint at a time

[Equation reading 1](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u043_r0)

A terminal projection replaces the terminal marginal while retaining the conditional paths.

$$
P^{2n+1}(d\omega)=\frac{\nu(X_1)}{P^{2n}_1(X_1)}P^{2n}(d\omega)
$$

The terminal projection multiplies each current path probability by the desired terminal density ν divided by the current terminal density, evaluated at that path’s endpoint.

[Equation reading 2](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u043_r1)

An initial projection then restores the source marginal.

$$
P^{2n+2}(d\omega)=\frac{\mu(X_0)}{P^{2n+1}_0(X_0)}P^{2n+1}(d\omega)
$$

The next projection multiplies each path probability by the desired initial density μ divided by the current initial density, evaluated at the starting point, to restore the source marginal.

## Time reversal makes each endpoint correction useful

[Equation reading 1](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u044_r0)

Bayes’ rule gives the reverse transition of a discrete-time reference chain.

$$
K_k^{\leftarrow}(y,x)=\frac{p_k(x)K_k(x,y)}{p_{k+1}(y)}
$$

The reverse transition from y to x equals the forward joint probability of x then y divided by the marginal probability of y at the next step.

[Equation reading 2](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u044_r1)

For a diffusion, reverse time s=1−t changes the drift and adds its density score.

$$
dY_s=[-b_{1-s}(Y_s)+\varepsilon\nabla\log p_{1-s}(Y_s)]ds+\sqrt\varepsilon\,d\overline W_s
$$

In reverse time, the diffusion drift is minus the forward drift plus ε times the forward density score, while the Brownian noise scale remains the square root of ε.

## DSB learns each reverse step by regression on simulated pairs

[Equation reading 1](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u045_r0)

Let Fₖ be the current forward mean map, with a Gaussian transition around it.

$$
X_{k+1}\sim\mathcal N(F_k(X_k),\,2\gamma_{k+1}I)
$$

The next state is sampled from a Gaussian whose mean is the current forward map F applied to the present state and whose covariance is twice the step parameter γ times the identity.

[Equation reading 2](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u045_r1)

The paper regresses a backward mean map B onto a target formed from the current F.

$$
\min_B\ \mathbb E\left\lVert B(X_{k+1})-\left[X_{k+1}+F_k(X_k)-F_k(X_{k+1})\right]\right\rVert^2
$$

The backward regression minimizes the expected squared difference between B at the next state and a target formed by that next state plus F at the previous state minus F at the next state.

## An endpoint mixture can retain hidden history

[Equation reading 1](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u049_r0)

An endpoint coupling π and reference conditional paths define a reciprocal mixture Λ.

$$
\Lambda=\int R^{xy}\,\pi(dx,dy)
$$

The reciprocal mixture Λ first draws an endpoint pair from π and then draws an entire reference bridge conditioned on that pair.

[Equation reading 2](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u049_r1)

A Markov law depends on the present state; the mixture can retain its start.

$$
P(X_{t+\Delta}\mid X_{[0,t]})=P(X_{t+\Delta}\mid X_t)\quad\text{for a Markov law}
$$

For a Markov law, conditioning the future on the complete past gives the same distribution as conditioning only on the current state; the earlier history supplies no additional information.

## Markov projection averages the conditional drift

[Equation reading 1](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u050_r0)

Average the Brownian bridge drift over pairs that pass through the current state.

$$
v_t^*(x)=\mathbb E_\Lambda\!\left[\frac{X_1-X_t}{1-t}\,\middle|\,X_t=x\right]
$$

The projected drift at x is the conditional average, over mixture trajectories passing through x, of the displacement to their terminal states divided by the remaining time.

[Equation reading 2](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u050_r1)

Squared-error regression learns this conditional expectation.

$$
\min_\theta\ \mathbb E_{t,\Lambda}\left\lVert v_\theta(t,X_t)-\frac{X_1-X_t}{1-t}\right\rVert^2
$$

The drift-matching loss averages squared error between the learned drift at the current state and the endpoint-conditioned Brownian bridge drift, over times and trajectories from Λ.

## Reciprocal projection restores the reference bridges

[Equation reading 1](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u051_r0)

Keep the endpoint coupling and replace the process’s conditional bridges.

$$
\mathcal R(M)=\int R^{xy}\,M_{01}(dx,dy)
$$

The reciprocal projection retains M’s joint endpoint distribution and replaces every endpoint-conditioned path law with the corresponding reference bridge.

[Equation reading 2](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u051_r1)

Iterative Markovian fitting alternates the two projections.

$$
M^{n+1}=\mathcal M(\Lambda^n),\qquad \Lambda^{n+1}=\mathcal R(M^{n+1})
$$

One fitting cycle first takes the Markov projection of the current reciprocal mixture, then takes the reciprocal projection of that resulting Markov process.

## A KL identity explains why exact projection makes progress

[Equation reading 1](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u052_r0)

The KL identity separates the correction from the remaining distance to the SB.

$$
\operatorname{KL}(P^n\Vert P^*)=\operatorname{KL}(P^n\Vert P^{n+1})+\operatorname{KL}(P^{n+1}\Vert P^*)
$$

Under the exact projection theorem’s assumptions, the current KL to the bridge splits into the KL paid by the projection plus the new KL remaining to the bridge.

[Equation reading 2](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u052_r1)

The correction is nonnegative, so the distance to the bridge cannot increase.

$$
\operatorname{KL}(P^{n+1}\Vert P^*)\leq\operatorname{KL}(P^n\Vert P^*)
$$

Because the projection’s KL contribution is nonnegative, the next exact iterate’s KL to the bridge is no larger than the current iterate’s KL.

## The conditional Gaussian gives both a velocity and a score

[Equation reading 1](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u055_r0)

Differentiate the mean mₜ and standard deviation σₜ of the Gaussian interpolant.

$$
m_t=(1-t)x_0+tx_1,\quad\sigma_t^2=\varepsilon t(1-t),\quad v_t^c=x_1-x_0+\frac{1-2t}{2t(1-t)}(x-m_t)
$$

The conditional velocity combines endpoint displacement with a correction proportional to deviation from the interpolated mean, while the Gaussian variance is ε times t times one minus t.

[Equation reading 2](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u055_r1)

Differentiating the Gaussian log density gives the conditional score.

$$
s_t^c(x)=\nabla_x\log p_t(x\mid x_0,x_1)=-\frac{x-m_t}{\varepsilon t(1-t)}
$$

The conditional score is the gradient of the Gaussian log density, equal to the negative displacement from the interpolated mean divided by the conditional variance.

## The learned velocity and score recover either an ODE or an SDE

[Equation reading 1](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u056_r0)

Match both conditional targets with a positive score-loss weight λₜ.

$$
\mathcal L=\mathbb E\!\left[\lVert v_\theta-v_t^c\rVert^2+\lambda_t^2\lVert s_\theta-s_t^c\rVert^2\right]
$$

The loss averages the squared velocity-prediction error plus the squared score-prediction error multiplied by λ squared, with both targets supplied by the conditional Gaussian bridge.

[Equation reading 2](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u056_r1)

Adding half the diffusion variance times the score recovers the SDE drift.

$$
dX_t=\left[v_\theta(t,X_t)+\tfrac\varepsilon2s_\theta(t,X_t)\right]dt+\sqrt\varepsilon\,dW_t
$$

The sampling SDE uses learned velocity plus ε over two times the learned score as its drift, and adds Brownian noise with scale equal to the square root of ε.

## A rate describes the probability of a jump over a short interval

[Equation reading 1](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u060_r0)

For distinct states x and y, the off-diagonal generator entry is a nonnegative rate.

$$
\Pr(X_{t+dt}=y\mid X_t=x)=G_t(x,y)\,dt+o(dt),\qquad y\ne x
$$

Over a sufficiently small interval, the probability of jumping from x to a different state y is the rate G from x to y times the interval, plus a smaller-order remainder.

[Equation reading 2](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u060_r1)

The diagonal records the total rate of leaving, so each generator row sums to zero.

$$
G_t(x,x)=-\sum_{y\ne x}G_t(x,y),\qquad \dot p_t=p_tG_t
$$

The diagonal generator entry is minus the sum of all outgoing rates, and the row probability vector evolves by multiplying itself by the generator.

## Changing rates also changes the waiting times

[Equation reading 1](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u061_r0)

The log ratio sums jump terms and integrates the difference in escape rates.

$$
\log\frac{dP^Q}{dR}=\log\frac{p_0(X_0)}{r_0(X_0)}+\sum_{\tau:\,\mathrm{jump}}\log\frac{Q_\tau(X_{\tau^-},X_\tau)}{G_\tau(X_{\tau^-},X_\tau)}-\int_0^1(\lambda_t^Q-\lambda_t^G)(X_t)\,dt
$$

The path log ratio adds the initial-density log ratio and the log rate ratio at each realized jump, then subtracts the time integral of the difference in total escape rates.

[Equation reading 2](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u061_r1)

Taking its expectation gives a local control cost for each possible transition.

$$
\operatorname{KL}(P^Q\Vert R)=\operatorname{KL}(p_0\Vert r_0)+\mathbb E_{P^Q}\!\int_0^1\sum_{y\ne X_t}\!\left[Q\log\frac QG-Q+G\right](X_t,y)\,dt
$$

The controlled path KL equals the initial KL plus the controlled expectation of the time integral, summed over destinations, of Q log(Q/G) minus Q plus G.

## Conditioning a jump process also produces a Doob transform

[Equation reading 1](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u062_r0)

A positive future potential solves a backward equation on the state graph.

$$
h_t=K_{t1}g,\qquad \partial_t h_t=-G_t h_t
$$

The future potential h is the terminal potential propagated backward through the reference transition kernel, and its time derivative is minus the generator applied to h.

[Equation reading 2](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u062_r1)

The bridge multiplies each off-diagonal rate by the ratio of future potentials.

$$
Q_t(x,y)=G_t(x,y)\frac{h_t(y)}{h_t(x)},\quad y\ne x;\qquad Q_t(x,x)=-\sum_{y\ne x}Q_t(x,y)
$$

Each off-diagonal bridge rate equals the reference rate times the destination potential divided by the source potential at the same time; the diagonal is reset to minus the outgoing-rate sum.

## Our two-state example now evolves in continuous time

[Equation reading 1](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u063_r0)

Choose a generator and solve the endpoint scaling problem using its matrix exponential.

$$
G=\begin{pmatrix}-2&2\\1&-1\end{pmatrix},\quad a=(0.5,0.5),\quad b=(0.2,0.8),\quad K_{01}=e^G
$$

The generator has jump rates two from state zero and one from state one, and its matrix exponential gives the unit-time reference transition used to match a to b.

[Equation reading 2](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u063_r1)

At the midpoint, the future-potential ratio changes both jump rates.

$$
h_{1/2}=(0.91138,1.05216),\quad Q_{1/2}(0,1)=2.30894,\quad Q_{1/2}(1,0)=0.86620
$$

At the midpoint, the potential values approximately 0.91138 and 1.05216 reweight the reference rates into 2.30894 toward state one and 0.86620 toward state zero.

## The conditional bridge supplies a rate target for DDSBM

[Equation reading 1](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u066_r0)

Conditioning on terminal state z reweights the reference jump rate.

$$
G_t^z(x,y)=G_t(x,y)\frac{K_{t1}(y,z)}{K_{t1}(x,z)},\qquad y\ne x
$$

Conditioning on endpoint z multiplies the reference rate from x to y by the reference probability of reaching z from y divided by the probability of reaching z from x.

[Equation reading 2](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u066_r1)

Average this rate over destinations consistent with the current state.

$$
Q_t^*(x,y)=\mathbb E_\Lambda[G_t^{X_1}(x,y)\mid X_t=x]
$$

The projected rate averages those endpoint-conditioned reference rates over the terminal states consistent with the mixture’s current state x.

## Rate matching penalizes excessive jumping

[Equation reading 1](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u067_r0)

Match each conditional rate r with a positive predicted rate q.

$$
\mathcal L=\mathbb E\int_0^1\sum_{y\ne X_t}\left[r_t^y\log\frac{r_t^y}{q_\theta^y}-r_t^y+q_\theta^y\right]dt
$$

The rate loss averages over paths the time integral, summed over possible destinations, of target rate r times log(r/q), minus r, plus the predicted positive rate q.

[Equation reading 2](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u067_r1)

Differentiating with respect to a positive predicted rate gives its conditional mean.

$$
\frac{\partial}{\partial q}\mathbb E[r\log(r/q)-r+q\mid X_t=x]=1-\frac{\mathbb E[r\mid X_t=x]}q=0
$$

Differentiating the conditional expected rate loss with respect to q gives one minus the conditional mean target rate divided by q, which vanishes when q equals that conditional mean.

## Markov projection learns adjacent conditionals

[Equation reading 1](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u070_r0)

Project a path distribution Λ onto the Markov chain defined by its adjacent conditionals.

$$
\mathcal M(\Lambda)(x_{0:N})=\Lambda_0(x_0)\prod_{k=0}^{N-1}\Lambda(x_{k+1}\mid x_k)
$$

The Markov projection assigns a path its original initial probability multiplied by the original mixture’s adjacent conditional probabilities at every step.

[Equation reading 2](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u070_r1)

A categorical cross-entropy learns each of those conditional distributions.

$$
\min_\theta\ \mathbb E_\Lambda\left[-\sum_{k=0}^{N-1}\log q_\theta(X_{k+1}\mid X_k,k)\right]
$$

The categorical objective minimizes the expectation under Λ of the sum of negative log probabilities that the learned transitions assign to the observed next states.

## The reciprocal step restores conditional paths

[Equation reading 1](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u071_r0)

Normalize the path product by the reference endpoint transition probability.

$$
R(x_{1:N-1}\mid x_0,x_N)=\frac{\prod_{k=0}^{N-1}K_k(x_k,x_{k+1})}{K_{0N}(x_0,x_N)}
$$

The probability of the intermediate reference path conditioned on its endpoints equals the product of its one-step transitions divided by the total reference transition probability between those endpoints.

[Equation reading 2](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u071_r1)

Keep the fitted endpoint coupling and restore the reference conditional paths.

$$
\Lambda^{n+1}(x_{0:N})=M^n_{0N}(x_0,x_N)R(x_{1:N-1}\mid x_0,x_N)
$$

The next reciprocal mixture multiplies the fitted joint endpoint probability by the reference probability of the intermediate path conditioned on that same endpoint pair.

## A reward expresses a preference over trajectories

[Equation reading 1](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u077_r0)

Let R be the pretrained path law and α control the cost of changing it.

$$
\max_{P\ll R}\ \mathbb E_P[r(X_1)]-\alpha\operatorname{KL}(P\Vert R),\qquad \alpha>0
$$

The objective maximizes expected terminal reward under P minus α times the path KL from P to the pretrained reference R, restricting P to trajectories supported by R.

[Equation reading 2](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u077_r1)

Exponentiating the terminal reward gives the optimal tilted law.

$$
P^*(d\omega)=\frac{e^{r(X_1)/\alpha}}Z R(d\omega),\qquad Z=\mathbb E_R[e^{r(X_1)/\alpha}]
$$

The optimal path law multiplies each reference trajectory’s probability by the exponential of its terminal reward divided by α, then divides by the reference expectation Z of that multiplier.

## The gap to the optimum is another KL divergence

[Equation reading 1](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u078_r0)

Insert log(dP*/dR)=r(X₁)/α−log Z into the relative entropy.

$$
\mathbb E_P[r]-\alpha\operatorname{KL}(P\Vert R)=\alpha\log Z-\alpha\operatorname{KL}(P\Vert P^*)
$$

Expected reward minus the reference KL penalty equals α times log Z minus α times KL to the optimally tilted path law, so the remaining gap is itself a KL penalty.

[Equation reading 2](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u078_r1)

Its terminal law is a reward-weighted version of the pretrained terminal distribution.

$$
p_1^*(x)=\frac{p_1^R(x)e^{r(x)/\alpha}}Z
$$

The optimal terminal density is the reference terminal density multiplied by the exponential reward factor and divided by the same normalizer Z.

## Three outcomes reveal the reward–reference tradeoff

[Equation reading 1](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u079_r0)

Suppose the base probabilities are (0.5,0.3,0.2) and the reward multipliers are (1,2,4).

$$
p^R=(0.5,0.3,0.2),\quad r=\log(1,2,4),\quad \alpha=1,\quad Z=1.9
$$

The reference assigns probabilities 0.5, 0.3, and 0.2, the exponentiated rewards multiply them by one, two, and four, and their resulting total mass is Z equal to 1.9.

[Equation reading 2](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u079_r1)

Multiply first, then normalize to obtain the desired terminal probabilities.

$$
p^*=\frac{(0.5,0.6,0.8)}{1.9}=(0.26316,0.31579,0.42105)
$$

Multiplying the reference masses by their reward factors gives 0.5, 0.6, and 0.8; dividing by their sum 1.9 gives the normalized target probabilities shown.

## A random fixed source needs conditional normalization

[Equation reading 1](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u080_r0)

A global terminal tilt changes a random initial law by its future reward potential.

$$
P_0^*(x)=R_0(x)\frac{h_0(x)}Z,\qquad h_0(x)=\mathbb E_R[e^{r(X_1)/\alpha}\mid X_0=x]
$$

A global terminal reward tilt multiplies each initial-state probability by its expected future reward multiplier h, then divides by the global normalizer Z.

[Equation reading 2](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u080_r1)

Keeping that initial law fixed instead requires conditional normalization.

$$
\frac{d\widehat P}{dR}(\omega)=\frac{e^{r(X_1)/\alpha}}{h_0(X_0)}
$$

To preserve the reference initial law instead, each path’s reward multiplier is divided by the future-reward normalizer conditioned on that path’s own starting state.

## Off-policy samples estimate weighted denoising

[Equation reading 1](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u081_r0)

For trajectories sampled from a proposal Pᵛ, use the target-to-proposal likelihood ratio.

$$
w(\omega)=\frac{dP^*}{dP^v}(\omega)\propto e^{r(X_1)/\alpha}\frac{dR}{dP^v}(\omega)
$$

The importance weight is the target path density divided by the proposal path density, proportional to the reward multiplier times the reference-to-proposal path likelihood ratio.

[Equation reading 2](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u081_r1)

The target expectation becomes a weighted expectation under the proposal.

$$
\mathbb E_{P^*}[\mathcal L_{\rm denoise}(\theta;X_1)]=\mathbb E_{P^v}[w(\omega)\mathcal L_{\rm denoise}(\theta;X_1)]
$$

The expected denoising loss under the target law equals the proposal expectation of that same loss multiplied by its normalized path importance weight.

## Matched masking schedules simplify the weight

[Equation reading 1](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u082_r0)

The paper accumulates log probabilities along the unmasking trajectory.

$$
\log\widetilde w=\frac{r(X_1)}\alpha+\sum_{\ell\in\mathrm{unmasking\ events}}\log\frac{p_{\rm pre}(x_\ell\mid x_{<\ell})}{p_v(x_\ell\mid x_{<\ell})}
$$

The log weight adds terminal reward divided by α to the sum, over unmasking events, of log pretrained-token probability divided by proposal-token probability under the visible context.

[Equation reading 2](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u082_r1)

Normalizing in log space avoids overflow and gives the weighted training loss.

$$
\overline w_i=\frac{e^{\log\widetilde w_i}}{\sum_j e^{\log\widetilde w_j}},\qquad \widehat{\mathcal L}_{\rm WDCE}=\sum_i\overline w_i\,\mathcal L_{\rm denoise}(\theta;x_i)
$$

Exponentiating and normalizing the log weights gives batch weights that sum to one, and the WDCE estimate sums each example’s denoising loss multiplied by its batch weight.

## The search score balances reward and exploration

[Equation reading 1](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u084_r0)

The selection score adds an exploration bonus to the accumulated reward.

$$
U(s,a)=\frac{R(s,a)}{M\,N(s,a)}+c\,p_\theta(a\mid s)\frac{\sqrt{N(s)}}{1+N(s,a)}
$$

The search score adds accumulated reward divided by M times the action visit count to an exploration bonus scaled by c, the model’s action prior, and the parent and action visit counts.

[Equation reading 2](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u084_r1)

Effective sample size measures how many paths contribute to the weighted update.

$$
N_{\rm eff}=\frac{(\sum_i w_i)^2}{\sum_i w_i^2}=\frac1{\sum_i\overline w_i^2}
$$

Effective sample size is the squared sum of unnormalized weights divided by their squared-weight sum, or equivalently the reciprocal of the squared normalized-weight sum.

## The interpolant preserves each branch’s endpoints

[Equation reading 1](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u090_r0)

The correction vanishes at both endpoints because it is multiplied by t(1−t).

$$
I_t^k(x_0,x_1^k)=(1-t)x_0+tx_1^k+t(1-t)\phi_\eta^k(t,x_0,x_1^k)
$$

The branch interpolant adds linear motion from the source to that branch’s endpoint to a learned correction multiplied by t times one minus t, which makes the correction zero at both ends.

[Equation reading 2](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u090_r1)

Differentiating all time-dependent terms gives the conditional velocity target.

$$
\partial_t I_t^k=x_1^k-x_0+(1-2t)\phi_\eta^k+t(1-t)\partial_t\phi_\eta^k
$$

The interpolant’s time derivative equals endpoint displacement plus the derivative of the correction’s time factor and the correction network’s own time derivative, by the product rule.

## A state cost can guide the route between endpoints

[Equation reading 1](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u091_r0)

The interpolant is trained with kinetic energy and a state-dependent path cost.

$$
\mathcal L_{\rm traj}=\sum_k\mathbb E\int_0^1\left[\tfrac12\lVert\partial_t I_t^k\rVert^2+V_t(I_t^k)\right]dt
$$

The trajectory loss sums over branches the expected time integral of half the squared interpolant speed plus the state-dependent cost V evaluated along that interpolant.

[Equation reading 2](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u091_r1)

A separate velocity model then matches the fitted conditional derivative.

$$
\mathcal L_{\rm flow}=\sum_k\mathbb E\left\lVert u_\theta^k(t,I_t^k)-\partial_t I_t^k\right\rVert^2
$$

The flow loss sums over branches the expected squared difference between the learned velocity evaluated on the interpolant and the interpolant’s time derivative.

## Branch weights record how much mass follows each trajectory

[Equation reading 1](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u092_r0)

The primary branch begins with unit mass and the secondary branches with zero.

$$
w_0^0=1,\quad w_0^k=0\ (k>0),\qquad w_t^k=w_0^k+\int_0^t g_s^k\,ds
$$

The primary branch starts with weight one and every secondary branch starts with zero, and each later branch weight equals its initial weight plus the integrated growth rate.

[Equation reading 2](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u092_r1)

A weighted mixture reconstructs the total population at each time.

$$
\mu_t=\sum_k\mathbb E\!\left[w_t^k\,\delta_{X_t^k}\right],\qquad w_t^k\geq0
$$

The population measure is the sum over branches of their expected point masses at their current states, each multiplied by a nonnegative branch weight.

## The weighted continuity equation includes growth

[Equation reading 1](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u093_r0)

Differentiate a weighted test function along a branch trajectory.

$$
\frac d{dt}\mathbb E[w_t^k f(X_t^k)]=\mathbb E[g_t^k f(X_t^k)+w_t^k\nabla f(X_t^k)\cdot u_t^k]
$$

The time derivative of a branch’s weighted test-function expectation has one term from growth changing its weight and another from velocity changing the test function along its motion.

[Equation reading 2](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u093_r1)

Choosing the constant function f=1 gives the conservation condition.

$$
\frac d{dt}\mu_t(\mathcal X)=\sum_k\mathbb E[g_t^k],\qquad \sum_k g_t^k=0\ \Longrightarrow\ \mu_t(\mathcal X)=1
$$

The total population mass changes at the sum of expected branch growth rates; starting from mass one, growth rates summing to zero conserve total mass one.

## We can allocate sixty percent to one destination

[Equation reading 1](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u094_r0)

Choose linear branch weights so that the source mass is gradually reassigned.

$$
(w_t^0,w_t^1,w_t^2)=(1-t,\,0.6t,\,0.4t),\qquad (g_t^0,g_t^1,g_t^2)=(-1,0.6,0.4)
$$

The primary branch loses weight at rate one while the other two gain weight at rates 0.6 and 0.4, producing weights one minus t, 0.6t, and 0.4t.

[Equation reading 2](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u094_r1)

At the midpoint and endpoint, every weight is nonnegative and the total stays one.

$$
w_{1/2}=(0.5,0.3,0.2),\qquad w_1=(0,0.6,0.4),\qquad \sum_k w_t^k=1
$$

The branch weights are (0.5, 0.3, 0.2) halfway through and (zero, 0.6, 0.4) at the endpoint, while their sum remains one at every time.

## Growth losses encourage valid branch masses

[Equation reading 1](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u095_r0)

The terminal matching term compares each final branch weight with its target.

$$
\mathcal L_{\rm match}=\sum_k\mathbb E[(w_1^k-w_{\rm target}^k)^2]
$$

The terminal matching loss sums, across branches, the expected squared difference between the branch’s final weight and its prescribed target weight.

[Equation reading 2](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u095_r1)

A separate penalty discourages incorrect total mass and negative branch weights.

$$
\mathcal L_{\rm mass}=\mathbb E\int_0^1\left[(\sum_k w_t^k-w_{\rm total}(t))^2+\sum_k\max(0,-w_t^k)\right]dt
$$

The mass penalty integrates the squared error in total branch weight plus the sum of negative-weight violations, then averages this quantity over the sampled trajectories.

## The underdamped reference evolves position and velocity

[Equation reading 1](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u102_r0)

Position changes according to the current velocity.

$$
dr_i=v_i\,dt,\qquad \sigma_i=\sqrt{2\gamma k_BT/m_i}
$$

A particle’s position increment is its velocity times the time increment, and its thermal noise scale is the square root of twice friction times thermal energy divided by mass.

[Equation reading 2](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u102_r1)

Forces, friction, learned bias, and thermal noise determine the velocity increment.

$$
dv_i=[m_i^{-1}(-\nabla_iU(R)+b_i(R,V))-\gamma v_i]dt+\sigma_i\,dW_i
$$

The velocity increment combines potential force and learned force bias divided by mass, subtracts friction proportional to velocity, and adds a Brownian increment scaled by the particle’s thermal noise.

## The control cost must use the units of the reference noise

[Equation reading 1](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u103_r0)

Write the drift change as σu so that u is measured relative to the noise covariance.

$$
dZ_t=[a(Z_t)+\sigma(Z_t)u_\theta(Z_t)]dt+\sigma(Z_t)dW_t
$$

The controlled state increment uses reference drift a plus noise matrix σ times a dimensionless control u, and the same matrix σ scales the Brownian increment.

[Equation reading 2](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u103_r1)

For the velocity equation above, a force bias corresponds to a noise-scaled control.

$$
u_i=\frac{b_i}{\sqrt{2\gamma k_BT\,m_i}},\qquad \operatorname{KL}(P^u\Vert R)=\frac12\mathbb E_{P^u}\int_0^1\sum_i\lVert u_i\rVert^2dt
$$

The force bias divided by its mass-and-noise scale gives control u; with matching initial laws, path KL is half the controlled expectation of the time-integrated sum of squared controls.

## The bias has aligned and perpendicular components

[Equation reading 1](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u104_r0)

Normalize the target gradient and project a free vector into its perpendicular space.

$$
s_i=\nabla_i\log\pi_B,\quad \widehat s_i=s_i/\lVert s_i\rVert,\quad b_i=\operatorname{softplus}(\alpha_i)\widehat s_i+(I-\widehat s_i\widehat s_i^\top)h_i
$$

For a nonzero target score s, the bias adds a positive softplus-scaled component along its normalized direction to the component of a free vector h perpendicular to that direction.

[Equation reading 2](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u104_r1)

The perpendicular component has zero dot product with the target gradient.

$$
b_i\cdot s_i=\operatorname{softplus}(\alpha_i)\lVert s_i\rVert\geq0
$$

The bias–score dot product is the positive softplus coefficient times the score norm, because the perpendicular component contributes zero, so their local alignment is nonnegative.

## A finite inward step still needs a step-size bound

[Equation reading 1](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u105_r0)

For a point target, write d=x★−x and expand the squared distance after a bias-only step.

$$
\lVert d-\Delta t\,b\rVert^2-\lVert d\rVert^2=-2\Delta t\,d\cdot b+\Delta t^2\lVert b\rVert^2
$$

The change in squared target distance after a bias-only step equals minus twice the step size times inward alignment plus the squared step size times squared bias magnitude.

[Equation reading 2](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u105_r1)

A positive inward component gives a sufficient upper bound on the step size.

$$
0\leq\Delta t\leq\frac{2d\cdot b}{\lVert b\rVert^2}\quad\Longrightarrow\quad \lVert x+\Delta t b-x^\star\rVert\leq\lVert x-x^\star\rVert
$$

A nonnegative step size no larger than twice the inward dot product divided by squared bias magnitude ensures that the bias-only update does not increase distance to the point target.

## A two-dimensional calculation shows both parts of the bias

[Equation reading 1](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u106_r0)

Use displacement (3,4), free vector (2,−1), and aligned magnitude 0.8.

$$
\widehat s=(0.6,0.8),\quad h_\perp=(1.76,-1.32),\quad b=0.8\widehat s+h_\perp=(2.24,-0.68)
$$

Normalizing displacement (3,4) gives direction (0.6,0.8), projecting the free vector gives perpendicular component (1.76, minus 1.32), and adding the aligned component yields bias (2.24, minus 0.68).

[Equation reading 2](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u106_r1)

The dot product is positive, and a step of 0.1 reduces squared distance.

$$
d\cdot b=4,\quad \Delta t_{\max}=1.45985,\quad \lVert d-0.1b\rVert^2=24.2548<25
$$

The displacement–bias dot product is four, the maximum allowed step is approximately 1.45985, and using step 0.1 reduces squared target distance from 25 to 24.2548.

## Cross-entropy fits the target path distribution

[Equation reading 1](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u107_r0)

A terminal potential g defines a path law through a reference-relative tilt.

$$
P^*(d\omega)=Z^{-1}g(X_1)R(d\omega)
$$

The target path probability is its reference probability multiplied by the terminal potential g and divided by Z, the normalizer over reference trajectories.

[Equation reading 2](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u107_r1)

Cross-entropy minimizes KL from the target path law to the learned process.

$$
\min_\theta\ -\mathbb E_{P^*}[\log p_\theta(\omega)]\quad\Longleftrightarrow\quad\min_\theta\operatorname{KL}(P^*\Vert P_\theta)
$$

Minimizing the target expectation of negative learned path log likelihood is equivalent to minimizing KL from the fixed target path law to the learned path law.

## A terminal potential need not be the terminal density

[Equation reading 1](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u108_r0)

The terminal tilt multiplies the reference terminal density by its potential.

$$
P_1^*(x)=\frac{R_1(x)g(x)}Z
$$

The tilted terminal density equals the reference terminal density times the terminal potential g divided by the normalizer, so the potential itself is generally not the resulting density.

[Equation reading 2](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u108_r1)

For a one-endpoint constraint, a density ratio calibrates the tilt to the desired terminal law.

$$
g(x)=\frac{\pi_B(x)}{R_1(x)}\quad\Longrightarrow\quad P_1^*(x)=\pi_B(x)
$$

For the one-endpoint construction, choosing g as the desired density divided by the reference terminal density cancels that reference factor and gives the desired terminal density.

## Importance weights transfer the target expectation

[Equation reading 1](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u109_r0)

Let Pᵛ be the process that actually generated the replay trajectories.

$$
\widetilde w(\omega)=g(X_1)\frac{dR}{dP^v}(\omega),\qquad \mathbb E_{P^*}[F]=\frac{\mathbb E_{P^v}[\widetilde wF]}{\mathbb E_{P^v}[\widetilde w]}
$$

The unnormalized importance weight multiplies the terminal potential by the reference-to-proposal path density ratio; dividing the proposal’s weighted expectation by its mean weight recovers the target expectation.

[Equation reading 2](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u109_r1)

A batch gives a self-normalized estimate of the cross-entropy objective.

$$
\widehat{\mathcal L}_{\rm CE}=\sum_i\overline w_i[-\log p_\theta(\omega_i)],\qquad \overline w_i=\operatorname{softmax}_i(\log\widetilde w)
$$

The batch cross-entropy estimate sums each negative learned path log likelihood times its normalized importance weight, with those weights obtained by a softmax over log unnormalized weights.

## The transition density gives a practical loss

[Equation reading 1](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u110_r0)

An Euler transition has a controlled mean and a known covariance.

$$
Z_{k+1}\mid Z_k\sim\mathcal N\!\left(Z_k+[a_k+\sigma_k u_\theta]\Delta t,\,\sigma_k\sigma_k^\top\Delta t\right)
$$

The Euler transition is Gaussian with mean equal to the present state plus controlled drift times the step size, and covariance equal to σ times its transpose times that step size.

[Equation reading 2](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_u110_r1)

Whitening its residual leaves a squared term plus constants independent of θ.

$$
-\log p_\theta(Z_{k+1}\mid Z_k)=\frac1{2\Delta t}\left\lVert\sigma_k^+\!\left[\Delta Z_k-a_k\Delta t-\sigma_ku_\theta\Delta t\right]\right\rVert^2+C
$$

The negative transition log likelihood is one over twice the step size times the squared noise-whitened residual after subtracting reference and control drift increments, plus constants independent of θ.

## Motion and jumps share a probability evolution law

[Equation reading 1](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_end02_r0)

The generator describes how expected test functions change.

$$
\frac{d}{dt}\mathbb{E}_{p_t}[f(X)]=\mathbb{E}_{p_t}[\mathcal L_t f(X)]
$$

The time derivative of the expectation of a test function f equals the expectation of the generator applied to f, both evaluated under the current probability law.

[Equation reading 2](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_end02_r1)

For continuous states, drift and noise define the generator.

$$
\mathcal L_t f=b_t\!\cdot\!\nabla f+\tfrac12\operatorname{tr}(a_t\nabla^2 f),\qquad a_t=\sigma_t\sigma_t^{\mathsf T}
$$

For continuous states, the generator is drift dotted with the gradient of f plus one half the trace of noise covariance times the Hessian of f.

[Equation reading 3](https://docs.google.com/presentation/d/1JPu2QcklkWp77Ie5ibB7aSz15Q7wWxjGD8kh285r0Kc/edit#slide=id.os7_end02_r2)

For discrete states, rates weight the possible changes.

$$
\mathcal L_t f(x)=\sum_{y\ne x}q_t(x,y)\,[f(y)-f(x)]
$$

For discrete states, the generator sums over possible destinations the jump rate from x to y multiplied by the change in the test function from f(x) to f(y).
