# Lecture 6 · Equation readings

The presentation reveals the equation, then its spoken reading, then intuition. This guide reproduces the literal readings; the teaching notes retain the surrounding derivations and explanations.

## We can see the difficulty in a one-dimensional flow

[Equation reading 1](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u004_r0)

Start at one and let the velocity equal the current position.

$$
\dot x_t=x_t,\qquad x_0=1
$$

The time derivative of x equals x itself, and the initial value at time zero is one.

[Equation reading 2](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u004_r1)

Separate variables and integrate from zero to t.

$$
\int_1^{x_t}\frac{dz}{z}=\int_0^t d\tau\quad\Longrightarrow\quad\log x_t=t
$$

Integrating one over z from one to the current state equals integrating time from zero to t, so the logarithm of the current state equals t.

[Equation reading 3](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u004_r2)

Exponentiating gives the exact destination.

$$
x_t=e^t,\qquad x_1=e\approx2.71828
$$

The state at time t is e raised to t; at time one, the exact destination is approximately 2.71828.

## Using the initial velocity once misses that acceleration

[Equation reading 1](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u005_r0)

One Euler step uses the velocity at the starting point.

$$
x_1^{(1)}=1+1(1)=2
$$

The one-step Euler approximation adds the interval length one times the initial velocity one to the starting value one, giving two.

[Equation reading 2](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u005_r1)

Two steps update the velocity after the first move.

$$
x_{1/2}=1+\tfrac12(1)=1.5,\qquad x_1^{(2)}=1.5+\tfrac12(1.5)=2.25
$$

The first half step reaches 1.5; the second adds half of the updated velocity 1.5, giving 2.25.

[Equation reading 3](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u005_r2)

With N equal steps, the approximation approaches the exact result.

$$
x_1^{(N)}=(1+1/N)^N\longrightarrow e
$$

With N equal steps, the endpoint is one plus one over N, raised to N; this converges to e as the number of steps grows.

## First, recall how flow matching specifies the motion

[Equation reading 1](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u008_r0)

Pair a source sample with a data sample and interpolate.

$$
I_t=(1-t)X_0+tX_1,\qquad (X_0,X_1)\sim\rho
$$

The interpolant is one minus t times the source sample plus t times the data sample, with the endpoint pair drawn from the coupling ρ.

[Equation reading 2](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u008_r1)

Average the conditional velocity at the observed state.

$$
b_t(x)=\mathbb E[X_1-X_0\mid I_t=x]
$$

The velocity at state x and time t is the conditional average of the endpoint displacement, given that the interpolant currently equals x.

[Equation reading 3](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u008_r2)

The resulting ODE follows the interpolant marginals.

$$
\dot x_t=b_t(x_t),\qquad x_t\sim p_t=\operatorname{Law}(I_t)
$$

The ODE moves with velocity b at its current state, and its time-t distribution equals the law of the interpolant, under the stated regularity assumptions.

## The continuity equation explains that agreement

[Equation reading 1](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u009_r0)

Differentiate the expectation of a smooth test function.

$$
\frac{d}{dt}\mathbb E[\varphi(I_t)]=\mathbb E[\nabla\varphi(I_t)\cdot\dot I_t]
$$

The time derivative of the average test-function value equals the average dot product of its spatial gradient and the interpolant velocity.

[Equation reading 2](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u009_r1)

Condition on the current interpolant state.

$$
=\mathbb E[\nabla\varphi(I_t)\cdot\mathbb E[\dot I_t\mid I_t]]
$$

Conditioning on the current interpolant replaces its sample-specific velocity by the conditional mean velocity inside the same expected dot product.

[Equation reading 3](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u009_r2)

Integrating by parts identifies the probability evolution.

$$
\partial_t p_t+\nabla\cdot(p_t b_t)=0
$$

The time derivative of the density plus the divergence of density times velocity is zero; probability changes locally through incoming and outgoing flow.

## A flow map records where that ODE arrives

[Equation reading 1](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u010_r0)

Start the trajectory at x at time s.

$$
F_{s,t}(x)=x_t,\qquad x_s=x,\qquad \dot x_\tau=b_\tau(x_\tau)
$$

The map F from s to t returns the state reached at t by the ODE whose state at s is x and whose velocity is b.

[Equation reading 2](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u010_r1)

Integrate its velocity along the trajectory.

$$
F_{s,t}(x)=x+\int_s^t b_\tau(F_{s,\tau}(x))\,d\tau
$$

The destination equals the starting point plus the integral, from s to t, of the velocity evaluated along that same trajectory.

[Equation reading 3](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u010_r2)

A zero-length interval leaves the state unchanged.

$$
F_{s,s}(x)=x
$$

The map from time s to that same time s returns x unchanged.

## The average velocity turns this into one residual update

[Equation reading 1](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u011_r0)

Divide the total displacement by the interval length.

$$
v_{s,t}(x)=\frac{F_{s,t}(x)-x}{t-s},\qquad s<t
$$

The average velocity from s to t is the destination minus the starting point, divided by the positive interval length t minus s.

[Equation reading 2](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u011_r1)

Equivalently, average the velocity along the trajectory.

$$
v_{s,t}(x)=\frac{1}{t-s}\int_s^t b_\tau(F_{s,\tau}(x))\,d\tau
$$

The average velocity is also the integral of the instantaneous velocity along the path, divided by the interval length.

[Equation reading 3](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u011_r2)

Predict the destination using that average.

$$
F_{s,t}(x)=x+(t-s)v_{s,t}(x)
$$

The finite-time destination equals x plus the interval length times the interval-average velocity at x.

## For our example, the average is larger than the first velocity

[Equation reading 1](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u012_r0)

The solution from any starting time is known.

$$
F_{s,t}(x)=e^{t-s}x
$$

The exact exponential-flow map multiplies its starting state x by e raised to the elapsed time t minus s.

[Equation reading 2](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u012_r1)

Compute the interval average from zero to one.

$$
v_{0,1}(1)=\frac{e-1}{1}=1.71828
$$

The average velocity from zero to one, starting at one, is e minus one divided by one, approximately 1.71828.

[Equation reading 3](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u012_r2)

Use it in one update.

$$
1+(1-0)(1.71828)=2.71828
$$

Adding the full interval length times that average velocity to the initial state gives 2.71828.

## As the interval shrinks, we recover flow matching

[Equation reading 1](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u013_r0)

Expand the exact solution over a short interval.

$$
F_{t,t+h}(x)=x+h\,b_t(x)+o(h)
$$

Over a short interval h, the destination is x plus h times the current velocity, with a remainder smaller than order h.

[Equation reading 2](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u013_r1)

Divide the displacement by h.

$$
v_{t,t+h}(x)=b_t(x)+o(1)
$$

Dividing the displacement by h gives the current velocity plus an error that vanishes as h approaches zero.

[Equation reading 3](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u013_r2)

The diagonal of the two-time model is the instantaneous velocity.

$$
v_{t,t}(x)=b_t(x)
$$

On the time diagonal, the interval-average velocity equals the instantaneous velocity b at that time and state.

## First, move the end time while keeping the start fixed

[Equation reading 1](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u015_r0)

Differentiate the integral defining the map.

$$
\partial_t F_{s,t}(x)=\partial_t\!\left[x+\int_s^t b_\tau(F_{s,\tau}(x))d\tau\right]
$$

The end-time derivative of F differentiates the initial state plus the integral of velocity from the fixed start s to the moving endpoint t.

[Equation reading 2](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u015_r1)

The fundamental theorem of calculus selects the endpoint velocity.

$$
\partial_t F_{s,t}(x)=b_t(F_{s,t}(x))
$$

The end-time derivative of the map equals the instantaneous velocity evaluated at the map’s destination.

[Equation reading 3](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u015_r2)

In the residual parameterization, this gives a trainable identity.

$$
v_{s,t}+(t-s)\partial_t v_{s,t}=v_{t,t}(F_{s,t})
$$

The interval-average velocity plus the interval length times its end-time derivative equals the diagonal velocity at the predicted destination.

## Our exponential example satisfies this identity exactly

[Equation reading 1](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u016_r0)

Differentiate the map with respect to its end time.

$$
\partial_t F_{s,t}(x)=\partial_t(e^{t-s}x)=e^{t-s}x
$$

Differentiating e raised to t minus s times x with respect to t gives the same quantity, e raised to t minus s times x.

[Equation reading 2](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u016_r1)

Evaluate the velocity at the arrived state.

$$
b_t(F_{s,t}(x))=F_{s,t}(x)=e^{t-s}x
$$

Because this velocity field equals its input, evaluating b at the destination gives the destination itself, matching the end-time derivative.

## We can also stop partway and continue from there

[Equation reading 1](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u017_r0)

Reach an intermediate time u.

$$
y=F_{s,u}(x),\qquad s\leq u\leq t
$$

The intermediate state y is the result of applying the map from s to u to x, where u lies between s and t.

[Equation reading 2](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u017_r1)

Restart the same ODE from that state.

$$
F_{u,t}(y)=F_{s,t}(x)
$$

Starting at y at time u and flowing to t gives the same destination as flowing directly from x at s to t.

[Equation reading 3](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u017_r2)

This gives the composition identity.

$$
F_{s,t}=F_{u,t}\circ F_{s,u}
$$

The map from s to t is the composition of the map from s to u followed by the map from u to t.

## The same calculation works for two half intervals

[Equation reading 1](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u019_r0)

Move from zero to one half.

$$
F_{0,1/2}(1)=e^{1/2}\approx1.64872
$$

The first half-interval map sends one to e raised to one half, approximately 1.64872.

[Equation reading 2](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u019_r1)

Apply the second half map to the arrived state.

$$
F_{1/2,1}(1.64872)=e^{1/2}(1.64872)\approx2.71828
$$

The second half-interval map multiplies 1.64872 by e raised to one half, reaching approximately 2.71828.

[Equation reading 3](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u019_r2)

Compare with the direct map.

$$
e^{1/2}e^{1/2}=e=F_{0,1}(1)
$$

The product of the two half-interval multipliers is e, exactly the destination of the full-interval map from one.

## Now move the start time along the same trajectory

[Equation reading 1](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u020_r0)

Insert a very short first interval into the composition rule.

$$
F_{s,t}(x)=F_{s+h,t}(F_{s,s+h}(x))
$$

The direct map equals a short move from s to s plus h, followed by the map from that later start to t.

[Equation reading 2](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u020_r1)

Differentiate the right side at h = 0.

$$
0=\partial_sF_{s,t}(x)+J_xF_{s,t}(x)\,b_s(x)
$$

Differentiating at h equals zero gives zero as the sum of the start-time derivative and the spatial Jacobian applied to the starting velocity.

[Equation reading 3](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u020_r2)

The change in start time cancels the change in starting state.

$$
\partial_sF_{s,t}+J_xF_{s,t}\,b_s=0
$$

The start-time derivative of F plus its spatial Jacobian times the starting velocity b is zero.

## The cancellation is visible in the exponential flow

[Equation reading 1](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u021_r0)

Changing the start time gives a negative derivative.

$$
\partial_sF_{s,t}(x)=-e^{t-s}x
$$

The start-time derivative of the exponential-flow map is minus e raised to t minus s times x.

[Equation reading 2](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u021_r1)

Moving the starting state forward gives the opposite contribution.

$$
J_xF_{s,t}(x)b_s(x)=e^{t-s}x
$$

The map’s spatial Jacobian multiplies the starting velocity x by e raised to t minus s, giving the opposite contribution.

[Equation reading 3](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u021_r2)

Their sum is zero.

$$
-e^{t-s}x+e^{t-s}x=0
$$

The negative start-time contribution and the positive state-motion contribution have equal magnitude, so their sum is zero.

## Composition and the tangent condition recover the ODE

[Equation reading 1](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u022_r0)

Split off a short final interval.

$$
F_{s,t+h}(x)=F_{t,t+h}(F_{s,t}(x))
$$

The map ending at t plus h equals the map to t followed by the short final map from t to t plus h.

[Equation reading 2](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u022_r1)

Apply the tangent condition at the reached state.

$$
F_{s,t+h}(x)=F_{s,t}(x)+h\,b_t(F_{s,t}(x))+o(h)
$$

The short final map adds h times the velocity at the reached state to the original destination, plus a remainder smaller than order h.

[Equation reading 3](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u022_r2)

Take the difference quotient.

$$
\partial_tF_{s,t}(x)=b_t(F_{s,t}(x)),\qquad F_{s,s}(x)=x
$$

Taking the difference quotient recovers the endpoint ODE, with end-time derivative equal to b at the destination and identity initial condition at s.

## The maps also transport the probability distribution

[Equation reading 1](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u023_r0)

Draw a starting state from the marginal at time s.

$$
X_s\sim p_s,\qquad X_t=F_{s,t}(X_s)
$$

Draw the starting state from the time-s marginal and apply the map from s to t to obtain the later random state.

[Equation reading 2](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u023_r1)

The pushforward is the marginal at time t.

$$
(F_{s,t})_\#p_s=p_t
$$

The pushforward of the time-s distribution through F equals the time-t distribution: applying the map transports the whole law.

[Equation reading 3](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u023_r2)

For a smooth invertible map, density changes with volume.

$$
p_t(F_{s,t}(x))\,\left|\det J_xF_{s,t}(x)\right|=p_s(x)
$$

The destination density times the absolute determinant of the map’s spatial Jacobian equals the source density, accounting for local volume change.

## A Gaussian example separates translation from expansion

[Equation reading 1](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u024_r0)

Let the mean and standard deviation vary with time.

$$
p_t=\mathcal N(\mu_t,\sigma_t^2),\qquad \sigma_t>0
$$

The time-t distribution is Gaussian with mean μ at t and variance σ at t squared, where its standard deviation is strictly positive.

[Equation reading 2](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u024_r1)

Preserve the standardized coordinate of each sample.

$$
F_{s,t}(x)=\mu_t+\frac{\sigma_t}{\sigma_s}(x-\mu_s)
$$

The map subtracts the source mean, scales by the target-to-source standard-deviation ratio, and adds the target mean.

[Equation reading 3](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u024_r2)

Choose a numerical starting point.

$$
\mu_s=0,\ \sigma_s=1,\ \mu_t=2,\ \sigma_t=3,\ x=1\ \Longrightarrow\ F_{s,t}(1)=5
$$

With source mean zero and scale one, target mean two and scale three, the source point one maps to five.

## The diagonal loss retains the flow matching target

[Equation reading 1](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u027_r0)

Use the two-time model with equal time inputs.

$$
\mathcal L_{\rm diag}=\mathbb E\|v_\theta(t,t,I_t)-\dot I_t\|^2
$$

The diagonal loss averages the squared difference between the model’s equal-time velocity at the interpolant and the interpolant’s actual velocity.

[Equation reading 2](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u027_r1)

Write the conditional mean velocity plus a zero-mean residual.

$$
\dot I_t=b_t(I_t)+\eta,\qquad\mathbb E[\eta\mid I_t,t]=0
$$

The interpolant velocity is its conditional mean b plus a residual η whose average is zero when the current state and time are fixed.

[Equation reading 3](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u027_r2)

The cross term vanishes after conditioning.

$$
\mathcal L_{\rm diag}=\mathbb E\|v_\theta(t,t,I_t)-b_t(I_t)\|^2+\mathbb E\|\eta\|^2
$$

The loss splits into the squared error against the conditional mean and the residual variance; only the first term depends on the model.

## The Lagrangian loss compares two endpoint velocities

[Equation reading 1](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u028_r0)

Differentiate the learned finite-interval map.

$$
A_\theta=\partial_tF_\theta(s,t,x)
$$

A is the derivative of the learned map with respect to its destination time, with the start time and starting state held fixed.

[Equation reading 2](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u028_r1)

Evaluate the model’s local velocity at its predicted endpoint.

$$
B_\theta=v_\theta(t,t,F_\theta(s,t,x))
$$

B is the model’s diagonal velocity at time t, evaluated at its own predicted destination.

[Equation reading 3](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u028_r2)

Penalize their disagreement.

$$
\mathcal L_{\rm LSD}=\mathbb E\|A_\theta-B_\theta\|^2
$$

The Lagrangian self-distillation loss averages the squared difference between the map’s endpoint derivative A and its endpoint velocity prediction B.

## The Eulerian loss checks whether the destination changes

[Equation reading 1](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u029_r0)

Follow the starting state with the local diagonal velocity.

$$
D_sF_\theta=\partial_sF_\theta+J_xF_\theta\,v_\theta(s,s,x)
$$

The derivative along the moving start equals the partial start-time derivative plus the map’s spatial Jacobian applied to the model’s diagonal starting velocity.

[Equation reading 2](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u029_r1)

The exact destination is constant along that change.

$$
\mathcal L_{\rm ESD}=\mathbb E\|D_sF_\theta\|^2
$$

The Eulerian self-distillation loss averages the squared norm of that derivative along the moving start, penalizing changes in the predicted destination.

## The composition loss compares the two routes

[Equation reading 1](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u030_r0)

Compute the direct destination.

$$
y_{\rm direct}=F_\theta(s,t,x)
$$

The direct destination is the output of the learned map from s to t applied to the starting point x.

[Equation reading 2](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u030_r1)

Compute the destination through an intermediate time.

$$
y_{\rm split}=F_\theta(u,t,F_\theta(s,u,x))
$$

The split destination first applies the map from s to u, then applies the map from u to t to that intermediate output.

[Equation reading 3](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u030_r2)

Match the two destinations.

$$
\mathcal L_{\rm PSD}=\mathbb E\|y_{\rm direct}-y_{\rm split}\|^2
$$

The progressive self-distillation loss averages the squared distance between the direct destination and the destination obtained through the intermediate time.

## A model that always stays still exposes that missing condition

[Equation reading 1](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u031_r0)

Consider the identity map at every pair of times.

$$
\widehat F_{s,t}(x)=x
$$

The candidate map returns the input x for every choice of starting and ending times.

[Equation reading 2](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u031_r1)

It satisfies composition automatically.

$$
\widehat F_{u,t}(\widehat F_{s,u}(x))=x=\widehat F_{s,t}(x)
$$

Composing two identity maps still returns x, exactly matching the direct identity map.

[Equation reading 3](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u031_r2)

Its tangent is wrong whenever the desired velocity is nonzero.

$$
\widehat v_{t,t}(x)=0\ne b_t(x)
$$

This map’s diagonal velocity is zero, so it disagrees with the desired local velocity wherever b is nonzero.

## Detaching the target determines which branch learns

[Equation reading 1](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u032_r0)

Build a target from the two shorter intervals.

$$
y_{\rm target}=\operatorname{sg}[F_\theta(u,t,F_\theta(s,u,x))]
$$

The target is the two-interval composed prediction inside stop-gradient, so its value is used without differentiating through its construction.

[Equation reading 2](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u032_r1)

Train the direct prediction against that fixed target for this update.

$$
\ell=\|F_\theta(s,t,x)-y_{\rm target}\|^2
$$

The loss is the squared distance between the direct map prediction and that fixed target for the current update.

[Equation reading 3](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u032_r2)

Only the direct branch contributes to the gradient.

$$
\nabla_\theta\ell=2J_\theta F_\theta(s,t,x)^\top(F_\theta(s,t,x)-y_{\rm target})
$$

The parameter gradient is twice the direct map’s parameter Jacobian transpose applied to its prediction error; the detached target contributes no derivative.

## Small velocity errors accumulate along the generated path

[Equation reading 1](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u033_r0)

Let the learned path have residual r relative to a Lipschitz drift.

$$
y_t=\widehat F_{0,t}(x),\quad r_t=\dot y_t-b_t(y_t),\quad e_t=y_t-x_t
$$

The learned path y has residual r equal to its time derivative minus the desired drift at y, and state error e equal to y minus the exact path.

[Equation reading 2](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u033_r1)

A drift Lipschitz constant L controls error growth.

$$
\frac{d}{dt}\|e_t\|\leq L\|e_t\|+\|r_t\|,\qquad e_0=0
$$

The growth rate of the error norm is at most the Lipschitz constant L times the error norm plus the residual norm, starting from zero error.

[Equation reading 3](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u033_r2)

Grönwall bounds the terminal discrepancy.

$$
\|e_1\|\leq\int_0^1 e^{L(1-\tau)}\|r_\tau\|\,d\tau
$$

The endpoint error is at most the integral of each residual norm multiplied by its remaining-time amplification factor, e raised to L times one minus τ.

## Coupling both samplers turns this into a distributional bound

[Equation reading 1](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u034_r0)

Use the same initial noise for both paths.

$$
W_2^2(\widehat p_1,p_1)\leq\mathbb E\|\widehat F_{0,1}(X_0)-F_{0,1}(X_0)\|^2
$$

The squared Wasserstein distance between terminal laws is at most the expected squared endpoint difference when both maps use the same initial random sample.

[Equation reading 2](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u034_r1)

Apply Cauchy–Schwarz to the residual integral.

$$
\mathbb E\|e_1\|^2\leq\left(\int_0^1e^{2L(1-\tau)}d\tau\right)\mathbb E\int_0^1\|r_\tau\|^2d\tau
$$

The mean squared endpoint error is bounded by the integral of the squared exponential amplification times the expected time-integrated squared residual.

[Equation reading 3](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u034_r2)

Evaluate the deterministic factor for L > 0.

$$
W_2^2(\widehat p_1,p_1)\leq\frac{e^{2L}-1}{2L}\,\mathbb E\int_0^1\|r_\tau\|^2d\tau
$$

For positive L, the squared Wasserstein error is bounded by the expected integrated squared residual multiplied by e to the two L minus one, divided by two L.

## We can test the training logic on a flow with a known answer

[Equation reading 1](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u037_r0)

Train on the local ODE and check the full-interval destination.

$$
\dot x=x,\qquad F_\theta(0,1,x)=x+v_\theta(0,1,x)\approx e x
$$

For the ODE whose velocity equals x, the learned full-interval map adds its predicted average velocity to x and should approximate e times x.

## A consistency model always points to the same endpoint

[Equation reading 1](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u042_r0)

Fix the destination time at one.

$$
C_t(x)=F_{t,1}(x),\qquad C_1(x)=x
$$

The consistency function C at time t is the flow map from t to the fixed endpoint one; at time one, it returns its input.

[Equation reading 2](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u042_r1)

Points on the same trajectory have the same final prediction.

$$
C_s(x)=C_t(F_{s,t}(x))
$$

The final prediction made at s equals the final prediction made at t after transporting the state from s to t.

[Equation reading 3](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u042_r2)

Differentiating along the trajectory gives a transport equation.

$$
\partial_t C_t(x)+J_xC_t(x)b_t(x)=0
$$

The time derivative of C plus its spatial Jacobian applied to the current velocity is zero, expressing constancy along the trajectory.

## Two steps of length h define an average for length 2h

[Equation reading 1](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u044_r0)

Take a first short step from x.

$$
x_{t+h}=x+h\,a_\theta(x,t,h)
$$

The first short step adds h times the model’s average velocity for the interval starting at x and t.

[Equation reading 2](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u044_r1)

Take the second short step from the updated state.

$$
x_{t+2h}=x_{t+h}+h\,a_\theta(x_{t+h},t+h,h)
$$

The second short step adds h times a new average velocity evaluated at the updated state and the later starting time t plus h.

[Equation reading 3](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u044_r2)

Divide the combined displacement by 2h.

$$
a_{2h}^{\rm target}=\tfrac12\left[a_\theta(x,t,h)+a_\theta(x_{t+h},t+h,h)\right]
$$

The target average velocity for the interval of length two h is the arithmetic mean of the two short-interval average velocities.

## For the exponential flow, the short-step averages are different

[Equation reading 1](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u046_r0)

Use exact half-interval averages starting from one.

$$
a_1=\frac{e^{1/2}-1}{1/2}=1.29744
$$

The first half-interval average is e to the one half minus one, divided by one half, approximately 1.29744.

[Equation reading 2](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u046_r1)

The second interval starts at e to the one half.

$$
a_2=\frac{e-e^{1/2}}{1/2}=2.13912
$$

The second half-interval average is e minus e to the one half, divided by one half, approximately 2.13912.

[Equation reading 3](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u046_r2)

Their mean gives the full-interval average.

$$
\tfrac12(a_1+a_2)=1.71828=e-1
$$

Half the sum of these two averages is 1.71828, which equals the full-interval displacement e minus one.

## Write the average along the interval from r to t

[Equation reading 1](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u049_r0)

Average the instantaneous field along a trajectory.

$$
u(z_t,r,t)=\frac{1}{t-r}\int_r^t v(z_\tau,\tau)\,d\tau
$$

The MeanFlow average u is the integral of instantaneous velocity along the trajectory from r to t, divided by t minus r.

[Equation reading 2](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u049_r1)

The integral is the forward displacement.

$$
(t-r)u(z_t,r,t)=z_t-z_r
$$

The interval length times u equals the later state z at t minus the earlier state z at r.

[Equation reading 3](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u049_r2)

Generate toward the earlier data time.

$$
z_r=z_t-(t-r)u(z_t,r,t)
$$

To recover the earlier state, subtract the interval length times the predicted average velocity from the later state.

## Differentiating the interval identity gives the MeanFlow target

[Equation reading 1](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u050_r0)

Differentiate while the later state follows its trajectory.

$$
\frac{d}{dt}[(t-r)u(z_t,r,t)]=v(z_t,t)
$$

The total derivative of interval length times u, while the later state follows the trajectory, equals the instantaneous velocity at that later state.

[Equation reading 2](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u050_r1)

Use the product and chain rules.

$$
u+(t-r)(\partial_tu+J_zu\,v)=v
$$

The product and chain rules give u plus the interval length times the sum of its partial time derivative and spatial Jacobian applied to v.

[Equation reading 3](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u050_r2)

Isolate the average velocity.

$$
u=v-(t-r)(\partial_tu+J_zu\,v)
$$

The average velocity equals the instantaneous velocity minus the interval length times that total derivative correction.

## Encoding defines the target distribution for the latent map

[Equation reading 1](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u055_r0)

Encode data into a latent representation.

$$
Y\sim p_{\rm data},\qquad Z_1=E(Y),\qquad p_1^Z=E_\#p_{\rm data}
$$

A data sample Y is encoded into the latent Z at time one, and the latent target law is the pushforward of the data law through the encoder E.

[Equation reading 2](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u055_r1)

Learn a flow map from latent noise to encoded data.

$$
Z_0\sim p_0^Z,\qquad Z_1^{\rm gen}=F^Z_{0,1}(Z_0)
$$

A latent source sample is drawn from its noise law and transformed by the latent flow map into a generated endpoint latent.

[Equation reading 3](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u055_r2)

Decode the generated latent.

$$
Y^{\rm gen}=D(Z_1^{\rm gen}),\qquad p_{\rm gen}=D_\#(F^Z_{0,1})_\#p_0^Z
$$

Decoding the generated latent produces Y; its law is obtained by pushing the noise law through the latent map and then through the decoder D.

## Decoder sensitivity affects the visible generation error

[Equation reading 1](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u057_r0)

Assume the decoder is L-sub-D Lipschitz.

$$
\|D(z)-D(z')\|\leq L_D\|z-z'\|
$$

The distance between two decoded outputs is at most the decoder’s Lipschitz constant times the distance between their latent inputs.

[Equation reading 2](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u057_r1)

Push a coupling of the latent distributions through the decoder.

$$
W_2(D_\#\widehat p^Z,D_\#p_1^Z)\leq L_DW_2(\widehat p^Z,p_1^Z)
$$

The Wasserstein distance between decoded latent laws is at most that same constant times the Wasserstein distance between the latent laws.

[Equation reading 3](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u057_r2)

Include the encoder–decoder reconstruction discrepancy.

$$
W_2(p_{\rm gen},p_{\rm data})\leq L_DW_2(\widehat p^Z,p_1^Z)+W_2((D\circ E)_\#p_{\rm data},p_{\rm data})
$$

The data-space error is bounded by amplified latent distribution error plus the Wasserstein discrepancy introduced by encoding and then decoding real data.

## A simple decoder shows why latent error can be amplified

[Equation reading 1](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u058_r0)

Choose a one-dimensional decoder.

$$
D(z)=3z+2,\qquad L_D=3
$$

The decoder multiplies z by three and adds two, so its Lipschitz constant is three.

[Equation reading 2](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u058_r1)

Compare a target latent with a perturbed prediction.

$$
z^*=1,\quad\widehat z=1.1,\quad|\widehat z-z^*|=0.1
$$

The target latent is one and the predicted latent is 1.1, giving an absolute latent error of 0.1.

[Equation reading 3](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u058_r2)

The visible error is three times as large.

$$
D(z^*)=5,\quad D(\widehat z)=5.3,\quad|5.3-5|=0.3
$$

The two decoded values are five and 5.3, so the absolute output error is 0.3.

## Two equal source atoms cannot produce every target law

[Equation reading 1](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u061_r0)

Start from two atoms of equal probability.

$$
p_0(a)=p_0(b)=\tfrac12
$$

The source distribution assigns probability one half to each of the two atoms a and b.

[Equation reading 2](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u061_r1)

A deterministic map assigns each atom to one destination.

$$
p_1(A)=\tfrac12\mathbf1\{F(a)=A\}+\tfrac12\mathbf1\{F(b)=A\}
$$

The probability of destination A is half the indicator that a maps to A plus half the indicator that b maps to A.

[Equation reading 3](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u061_r2)

Only three destination probabilities are possible.

$$
p_1(A)\in\{0,\tfrac12,1\},\qquad p_1(A)=0.3\ \text{is impossible}
$$

A deterministic map can therefore assign A only probability zero, one half, or one; probability 0.3 is unavailable from these two source atoms.

## Represent each token as a vertex, then add continuous noise

[Equation reading 1](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u063_r0)

Embed a length-L sequence using V-dimensional one-hot vectors.

$$
X_1\in\{e_1,\ldots,e_V\}^{L}\subset\mathbb R^{L\times V}
$$

The clean sequence consists of L one-hot vectors chosen from V vocabulary vertices, embedded in a continuous space with L times V coordinates.

[Equation reading 2](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u063_r1)

Use Gaussian noise in the same ambient coordinates.

$$
X_0\sim\mathcal N(0,I),\qquad I_t=(1-t)X_0+tX_1
$$

The source is standard Gaussian noise, and the noisy sequence interpolates between that noise and the clean one-hot sequence with weights one minus t and t.

[Equation reading 3](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u063_r2)

The denoiser predicts token probabilities from the entire noisy sequence.

$$
D_t(x)=\mathbb E[X_1\mid I_t=x]\in(\Delta^{V-1})^L
$$

The denoiser is the conditional mean clean sequence given the noisy state, returning one categorical probability vector per position.

## The posterior mean gives the marginal velocity

[Equation reading 1](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u065_r0)

Solve the linear interpolant for its noise endpoint.

$$
X_0=\frac{x-tX_1}{1-t}\quad\text{when }I_t=x
$$

When the interpolant equals x, its noise endpoint is x minus t times the clean endpoint, divided by one minus t.

[Equation reading 2](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u065_r1)

Substitute into the endpoint displacement.

$$
X_1-X_0=\frac{X_1-x}{1-t}
$$

Substituting that expression makes the clean-minus-noise displacement equal to clean endpoint minus x, divided by one minus t.

[Equation reading 3](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u065_r2)

Average over the conditional clean endpoint.

$$
b_t(x)=\frac{D_t(x)-x}{1-t},\qquad t<1
$$

The marginal velocity is the denoiser’s conditional mean minus the current state, divided by the remaining time one minus t.

## Cross-entropy learns that posterior mean one token at a time

[Equation reading 1](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u066_r0)

Train each position against its observed clean token.

$$
\mathcal L_{\rm CE}=\mathbb E\left[-\sum_{\ell=1}^{L}\log D_\theta^{\ell,Y_\ell}(t,I_t)\right]
$$

The cross-entropy loss averages the negative sum, over sequence positions, of the log probability assigned to each observed clean token.

[Equation reading 2](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u066_r1)

At a fixed noisy state, write its true token marginal as q.

$$
\mathbb E[-\log D^Y\mid I_t=x]=H(q)+D_{\rm KL}(q\|D)
$$

At a fixed noisy state, expected negative log probability equals the true token distribution’s entropy plus its KL divergence to the predicted distribution.

[Equation reading 3](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u066_r2)

The minimum occurs at the conditional categorical distribution.

$$
D^{\ell,k}_*(t,x)=\Pr(Y_\ell=k\mid I_t=x)
$$

The minimizing probability for category k at position ℓ is the conditional probability that the clean token there is k, given the noisy sequence.

## Independent draws can destroy a simple two-word relationship

[Equation reading 1](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u068_r0)

Suppose only two phrases occur, each with equal probability.

$$
p(\text{New York})=p(\text{San Diego})=\tfrac12
$$

The joint distribution assigns probability one half to New York and one half to San Diego.

[Equation reading 2](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u068_r1)

At complete uncertainty, each position has two equally likely words.

$$
p_1(\text{New})=p_1(\text{San})=p_2(\text{York})=p_2(\text{Diego})=\tfrac12
$$

The first-position words New and San and the second-position words York and Diego each have marginal probability one half.

[Equation reading 3](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u068_r2)

Independent token draws also produce two invalid combinations.

$$
p(\text{New Diego})+p(\text{San York})=\tfrac14+\tfrac14=\tfrac12
$$

Independent draws assign one quarter to New Diego and one quarter to San York, so the total probability of an invalid phrase is one half.

## The finite map can also be written using an endpoint prediction

[Equation reading 1](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u069_r0)

Define a two-time mean denoiser from the interval average.

$$
\psi_{s,t}(x)=x+(1-s)v_{s,t}(x)
$$

The two-time mean denoiser ψ equals the current state plus the remaining source-time interval, one minus s, times the average velocity.

[Equation reading 2](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u069_r1)

Substitute into the residual map.

$$
F_{s,t}(x)=x+\frac{t-s}{1-s}(\psi_{s,t}(x)-x)
$$

The map adds ψ minus x to x, scaled by the elapsed interval divided by the time remaining at the start.

[Equation reading 3](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u069_r2)

Separate the two coefficients.

$$
F_{s,t}(x)=\frac{1-t}{1-s}x+\frac{t-s}{1-s}\psi_{s,t}(x)
$$

The destination is a weighted sum of x and ψ, with weights one minus t over one minus s and t minus s over one minus s.

## Why does the two-time prediction stay on the simplex?

[Equation reading 1](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u070_r0)

Use the ODE and an integrating factor.

$$
\frac{d}{du}\left[\frac{F_{s,u}(x)}{1-u}\right]=\frac{D_u(F_{s,u}(x))}{(1-u)^2}
$$

The derivative of the current flow state divided by its remaining time equals the denoiser at that state divided by remaining time squared.

[Equation reading 2](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u070_r1)

Integrate from s to t, with t < 1.

$$
F_{s,t}(x)=\frac{1-t}{1-s}x+(1-t)\int_s^t\frac{D_u(F_{s,u}(x))}{(1-u)^2}\,du
$$

Integrating gives a scaled starting state plus one minus t times the integral of denoiser values weighted by inverse remaining time squared.

[Equation reading 3](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u070_r2)

Compare with the mean-denoiser parameterization.

$$
\psi_{s,t}(x)=\int_s^t\frac{(1-s)(1-t)}{(t-s)(1-u)^2}\,D_u(F_{s,u}(x))\,du
$$

The two-time denoiser is the integral of along-path denoisers weighted by the displayed factor involving s, t, and the intermediate time u.

## The weights integrate to one

[Equation reading 1](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u071_r0)

Write the nonnegative weighting function.

$$
w(u)=\frac{(1-s)(1-t)}{(t-s)(1-u)^2}\geq0
$$

The weight at intermediate time u is the product of one minus s and one minus t, divided by t minus s and one minus u squared; it is nonnegative.

[Equation reading 2](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u071_r1)

Integrate it explicitly.

$$
\int_s^tw(u)du=\frac{(1-s)(1-t)}{t-s}\left(\frac1{1-t}-\frac1{1-s}\right)
$$

The integral of the weight equals its constant prefactor times the difference between the inverse remaining times at t and s.

[Equation reading 3](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u071_r2)

Cancel the interval factors.

$$
\int_s^tw(u)du=1\quad\Longrightarrow\quad\psi_{s,t}(x)\in\Delta^{V-1}
$$

The weights integrate to one, so averaging categorical probability vectors with them keeps ψ inside the probability simplex.

## We can calculate one step in these coordinates

[Equation reading 1](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u072_r0)

Choose an ambient state and a simplex-valued prediction.

$$
s=0.25,\ t=0.75,\quad x=(-0.2,0.6,1.1),\quad\psi=(0.1,0.7,0.2)
$$

The interval runs from 0.25 to 0.75; the ambient state is minus 0.2, 0.6, 1.1, and the denoiser predicts probabilities 0.1, 0.7, 0.2.

[Equation reading 2](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u072_r1)

Compute the interpolation weight.

$$
\frac{t-s}{1-s}=\frac{0.5}{0.75}=\frac23
$$

The elapsed interval 0.5 divided by the remaining starting interval 0.75 gives interpolation weight two thirds.

[Equation reading 3](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u072_r2)

Apply the finite map.

$$
F_{s,t}(x)=\tfrac13x+\tfrac23\psi=(0,\,0.66667,\,0.5)
$$

The destination is one third of the ambient state plus two thirds of the denoiser, giving coordinates zero, 0.66667, and 0.5.

## Composition induces a weighted consistency rule for denoisers

[Equation reading 1](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u073_r0)

Compose the two affine map expressions.

$$
F_{s,t}(x)=F_{u,t}(F_{s,u}(x))
$$

The map from s to t equals the map from s to u followed by the map from u to t.

[Equation reading 2](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u073_r1)

Match their coefficients after subtracting the shared x term.

$$
\psi_{s,t}(x)=\gamma\psi_{s,u}(x)+(1-\gamma)\psi_{u,t}(F_{s,u}(x))
$$

The direct denoiser is γ times the first-interval denoiser plus one minus γ times the second-interval denoiser evaluated at the intermediate destination.

[Equation reading 3](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u073_r2)

The coefficient depends on all three times.

$$
\gamma=\frac{(1-t)(u-s)}{(1-u)(t-s)},\qquad0\leq\gamma\leq1
$$

The weight γ is one minus t times u minus s, divided by one minus u times t minus s, and lies between zero and one.

## The same two shorter predictions receive unequal weights

[Equation reading 1](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u074_r0)

Choose s = 0, u = 0.5, and t = 0.75.

$$
\gamma=\frac{(0.25)(0.5)}{(0.5)(0.75)}=\frac13
$$

For these three times, γ is 0.25 times 0.5 divided by 0.5 times 0.75, giving one third.

[Equation reading 2](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u074_r1)

Suppose the two predictions are different probability vectors.

$$
\psi_{s,u}=(0.8,0.2),\qquad\psi_{u,t}=(0.2,0.8)
$$

The first short-interval denoiser predicts 0.8 and 0.2, while the second predicts 0.2 and 0.8.

[Equation reading 3](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u074_r2)

Their consistency target is still normalized.

$$
\psi_{s,t}^{\rm target}=\tfrac13(0.8,0.2)+\tfrac23(0.2,0.8)=(0.4,0.6)
$$

The direct target takes one third of the first prediction and two thirds of the second, yielding the normalized vector 0.4, 0.6.

## A KL loss matches this probability-valued target

[Equation reading 1](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u075_r0)

Detach the composed denoiser target.

$$
q=\operatorname{sg}[\gamma\psi_\theta(s,u,x)+(1-\gamma)\psi_\theta(u,t,F_\theta(s,u,x))]
$$

The target q is the γ-weighted mixture of the two composed denoiser predictions, enclosed in stop-gradient to freeze the target during the update.

[Equation reading 2](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u075_r1)

Match it with the direct prediction at each position.

$$
\mathcal L_{\rm SG}=\mathbb E\sum_{\ell=1}^L D_{\rm KL}(q^\ell\|\psi_\theta^\ell(s,t,x))
$$

The consistency loss averages the sum, over positions, of KL divergence from the detached target q to the direct two-time denoiser prediction.

[Equation reading 3](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u075_r2)

Retain the diagonal cross-entropy anchor.

$$
\mathcal L=\mathcal L_{\rm CE,diag}+\lambda\mathcal L_{\rm SG}
$$

The total loss adds diagonal cross-entropy to λ times the semigroup consistency loss.

## The decoding error defines a more evenly spaced clock

[Equation reading 1](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u077_r0)

Measure the probability of decoding the wrong token.

$$
P_e(t)=\Pr[\arg\max I_t\ne\arg\max X_1]
$$

The decoding error probability is the chance that the largest coordinate of the noisy interpolant identifies a different token from the clean endpoint.

[Equation reading 2](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u077_r1)

Normalize between random guessing and perfect recovery.

$$
\tau(t)=1-\frac{V}{V-1}P_e(t)
$$

The new clock τ is one minus the error probability multiplied by V over V minus one, normalizing random guessing and perfect recovery.

[Equation reading 3](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u077_r2)

Sample a value in the new clock and invert the schedule.

$$
q\sim\mathcal U(0,1),\qquad t=\tau^{-1}(q)
$$

Draw q uniformly between zero and one, then use the inverse clock to convert q into the original interpolation time t.

## Let’s implement the probability-preserving map update

[Equation reading 1](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u079_r0)

The code should move the state using a valid categorical prediction.

$$
F_{s,t}(x)=(1-h)x+h\psi_{s,t}(x),\qquad h=\frac{t-s}{1-s},\quad\psi_{s,t}\in\Delta^{V-1}
$$

The map mixes the current state with a simplex-valued denoiser using weight h equal to the elapsed interval divided by the remaining starting time.

## Their map has the same affine structure

[Equation reading 1](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u085_r0)

Use the paper’s endpoint prediction π.

$$
F_{s,t}(x)=x+\eta_{s,t}(\pi_{s,t}(x)-x),\qquad\eta_{s,t}=\frac{t-s}{1-s}
$$

The categorical flow map adds the endpoint prediction π minus the current state, scaled by η, the elapsed interval divided by the remaining starting time.

[Equation reading 2](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u085_r1)

The diagonal prediction supplies the velocity.

$$
b_t(x)=\frac{\pi_{t,t}(x)-x}{1-t}
$$

The diagonal endpoint prediction minus x, divided by one minus t, gives the instantaneous velocity.

[Equation reading 3](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u085_r2)

Multiply the Lagrangian identity by the remaining time.

$$
(1-t)\partial_tF_{s,t}(x)=\pi_{t,t}(F_{s,t}(x))-F_{s,t}(x)
$$

The remaining time times the map’s end-time derivative equals the diagonal endpoint prediction at the destination minus that destination.

## Differentiate the map to isolate the two sources of residual

[Equation reading 1](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u086_r0)

The time derivative acts on both the coefficient and the prediction.

$$
\partial_tF_{s,t}=\frac{\pi_{s,t}-x}{1-s}+\eta_{s,t}\partial_t\pi_{s,t}
$$

The end-time derivative is π minus x divided by one minus s, plus η times the end-time derivative of π.

[Equation reading 2](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u086_r1)

Define the scaled Lagrangian residual.

$$
r=(1-t)\partial_tF_{s,t}-\pi_{t,t}(F_{s,t})+F_{s,t}
$$

The residual r is one minus t times the map’s derivative, minus the diagonal endpoint prediction at the destination, plus the destination itself.

[Equation reading 3](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u086_r2)

Collect terms.

$$
r=\pi_{s,t}-\pi_{t,t}(F_{s,t})+(1-t)\eta_{s,t}\partial_t\pi_{s,t}
$$

Collecting terms leaves the difference between the two endpoint predictions plus one minus t times η times the derivative of the finite-interval prediction.

## KL divergence can bound the endpoint mismatch

[Equation reading 1](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u087_r0)

Divide by remaining time and use the squared-sum inequality.

$$
\left\|\frac r{1-t}\right\|^2\leq\frac{2\|\pi_{s,t}-\pi_{t,t}(F)\|^2}{(1-t)^2}+2\eta_{s,t}^2\|\partial_t\pi_{s,t}\|^2
$$

The squared residual divided by remaining time squared is at most twice the scaled squared endpoint mismatch plus twice η squared times the squared prediction derivative.

[Equation reading 2](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u087_r1)

Pinsker bounds squared Euclidean discrepancy by KL.

$$
\|p-q\|_2^2\leq\|p-q\|_1^2\leq2D_{\rm KL}(p\|q)
$$

The squared Euclidean distance between probability vectors is at most their squared L1 distance, which Pinsker bounds by twice their KL divergence.

[Equation reading 3](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u087_r2)

Apply it to the endpoint term.

$$
\mathbb E\left\|\frac r{1-t}\right\|^2\leq4\mathcal L_{\rm EC}+2\mathcal L_{\rm TD}
$$

The expected squared scaled residual is therefore at most four times the endpoint-consistency loss plus twice the time-derivative loss.

## The two penalties measure different failures

[Equation reading 1](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u088_r0)

Suppose the current endpoint predictions agree.

$$
\pi_{s,t}=\pi_{t,t}(F_{s,t})=(0.4,0.6)
$$

The finite-interval endpoint prediction and the diagonal prediction at its destination both equal the probability vector 0.4, 0.6.

[Equation reading 2](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u088_r1)

Allow the finite-interval prediction to vary with t.

$$
t=0.5,\ \eta=0.5,\quad\partial_t\pi_{s,t}=(0.2,-0.2)
$$

At time 0.5 with interpolation weight 0.5, the endpoint prediction changes with derivative 0.2, minus 0.2.

[Equation reading 3](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u088_r2)

A nonzero residual remains.

$$
r=(0.5)(0.5)(0.2,-0.2)=(0.05,-0.05),\qquad\|r\|^2=0.005
$$

The derivative correction produces residual 0.05, minus 0.05, whose squared norm is 0.005 despite agreement between the endpoint predictions.

## The Lagrangian identity becomes a relation between denoisers

[Equation reading 1](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u094_r0)

Use the same affine map and mean denoiser ψ.

$$
F_{s,t}=\frac{1-t}{1-s}x+\frac{t-s}{1-s}\psi_{s,t}
$$

The destination combines x and the mean denoiser ψ with weights one minus t over one minus s and t minus s over one minus s.

[Equation reading 2](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u094_r1)

Substitute into the Lagrangian equation and collect terms.

$$
\psi_{s,t}+c_{s,t}\partial_t\psi_{s,t}=\psi_{t,t}(F_{s,t})
$$

The finite-interval denoiser plus c times its end-time derivative equals the diagonal denoiser evaluated at the predicted destination.

[Equation reading 3](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u094_r2)

The derivative coefficient is determined by the times.

$$
c_{s,t}=\frac{(t-s)(1-t)}{1-s}
$$

The derivative coefficient c is the elapsed interval times the remaining end-time interval, divided by the remaining starting interval.

## A raw derivative correction can leave the simplex

[Equation reading 1](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u095_r0)

A tentative teacher subtracts a derivative from a probability vector.

$$
q_{\rm raw}=\psi_{t,t}(F)-c_{s,t}\partial_t\psi_{s,t}
$$

The raw teacher subtracts c times the finite-interval denoiser’s end-time derivative from the diagonal denoiser at the destination.

[Equation reading 2](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u095_r1)

Consider a two-category numerical example.

$$
\psi_{t,t}(F)=(0.1,0.9),\quad c=0.2,\quad\partial_t\psi=(1,-1)
$$

The diagonal prediction is 0.1, 0.9, the coefficient is 0.2, and the derivative is one, minus one.

[Equation reading 3](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u095_r2)

The first coordinate becomes negative.

$$
q_{\rm raw}=(-0.1,1.1)
$$

The corrected vector is minus 0.1, 1.1, so its first coordinate is negative even though the coordinates still sum to one.

## The softmax derivative gives a multiplicative correction

[Equation reading 1](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u096_r0)

Write the prediction using logits z.

$$
\psi_k=\frac{e^{z_k}}{\sum_j e^{z_j}}
$$

The probability of category k is the exponential of its logit divided by the sum of exponentiated logits over all categories.

[Equation reading 2](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u096_r1)

Differentiate and subtract the probability-weighted logit derivative.

$$
\partial_t\psi_k=\psi_k\left(\partial_tz_k-\sum_j\psi_j\partial_tz_j\right)=\psi_k\delta_k
$$

The derivative of probability k equals that probability times its logit derivative minus the probability-weighted average logit derivative; this centered difference is δ.

[Equation reading 3](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u096_r2)

The Lagrangian identity becomes coordinate-wise multiplication.

$$
\psi_k(1+c\delta_k)=\psi_{t,t,k}(F)
$$

For each category, the finite-interval probability multiplied by one plus c times δ equals the diagonal probability at the destination.

## Taking logs produces a probability-valued Lagrangian teacher

[Equation reading 1](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u097_r0)

Assume each correction denominator is positive.

$$
a_k=1+c\delta_k>0
$$

For each category k, the correction denominator a is one plus c times δ and must be strictly positive.

[Equation reading 2](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u097_r1)

Solve the coordinate relation and normalize.

$$
\psi_k\propto\frac{\psi_{t,t,k}(F)}{a_k}
$$

The finite-interval probability is proportional to the diagonal destination probability divided by its correction denominator, followed by normalization.

[Equation reading 3](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u097_r2)

Express the normalized result using softmax.

$$
T^{\rm LSD}=\operatorname{softmax}\!\left(z_{t,t}(F)-\log(1+c\delta)\right)
$$

The Lagrangian teacher applies softmax to the diagonal destination logits minus the coordinate-wise logarithm of one plus c times δ.

## The Eulerian teacher uses a derivative along the starting flow

[Equation reading 1](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u098_r0)

Differentiate logits with respect to the moving start.

$$
D_sz=\partial_sz+J_xz\,b_s,\qquad\delta=D_sz-\langle\psi,D_sz\rangle\mathbf1
$$

The start-flow derivative of the logits adds their partial start-time derivative to their spatial Jacobian times b; subtracting its probability-weighted mean gives δ.

[Equation reading 2](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u098_r1)

The probability identity determines its coefficient.

$$
D_s\psi=\kappa(\psi-\psi_{s,s}),\qquad\kappa=\frac{1-t}{(1-s)(t-s)}
$$

The start-flow derivative of ψ equals κ times the difference between the finite-interval prediction and its starting diagonal prediction, with κ given by the time ratio.

[Equation reading 3](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u098_r2)

Solve in logits on the positive-denominator domain.

$$
T^{\rm ESD}=\operatorname{softmax}\!\left(z_{s,s}-\log(\mathbf1-\kappa^{-1}\delta)\right)
$$

The Eulerian teacher is softmax of the starting diagonal logits minus the logarithm of one minus δ divided by κ, where every denominator is positive.

## A common scalar can be removed before taking softmax

[Equation reading 1](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u099_r0)

Write the Eulerian denominator without an explicit inverse ratio.

$$
1-\kappa^{-1}\delta=\frac{(1-t)\mathbf1-(1-s)(t-s)\delta}{1-t}
$$

One minus δ divided by κ equals one minus t times the all-ones vector, minus one minus s times t minus s times δ, all divided by one minus t.

[Equation reading 2](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u099_r1)

The denominator contributes the same logit shift to every category.

$$
\operatorname{softmax}(z+c\mathbf1)=\operatorname{softmax}(z)
$$

Adding the same scalar c to every logit leaves its softmax probabilities unchanged.

[Equation reading 3](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u099_r2)

Use the equivalent target for t < 1.

$$
T^{\rm ESD}=\operatorname{softmax}\!\left(z_{s,s}-\log[(1-t)\mathbf1-(1-s)(t-s)\delta]\right)
$$

The equivalent Eulerian teacher subtracts the logarithm of the unscaled positive correction vector from the starting diagonal logits before applying softmax.

## The KL gradient has a simple interpretation in logit space

[Equation reading 1](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u100_r0)

Let p be the detached teacher and q the student softmax.

$$
\ell=D_{\rm KL}(p\|q),\qquad q=\operatorname{softmax}(z)
$$

The loss is KL divergence from the detached teacher p to the student q, where q is softmax of the trainable logits z.

[Equation reading 2](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u100_r1)

Differentiate the student cross-entropy term.

$$
\frac{\partial\ell}{\partial z_k}=q_k-p_k
$$

The derivative with respect to logit k is the student probability q at k minus the teacher probability p at k.

[Equation reading 3](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u100_r2)

A detached adaptive weight rescales this mismatch.

$$
w=\operatorname{sg}[(\|q-p\|^2+c_0)^{-r}],\qquad\nabla_z(w\ell)=w(q-p)
$$

The detached weight is a negative power of squared probability mismatch plus c zero; the weighted logit gradient is that fixed weight times q minus p.

## The desired distribution reweights data by its reward

[Equation reading 1](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u104_r0)

Let r assign a scalar reward to a clean sample z.

$$
p_1^r(z)=\frac{p_1(z)e^{r(z)}}{Z},\qquad Z=\mathbb E_{p_1}[e^{r(Z_1)}]
$$

The tilted endpoint law multiplies the original probability by exponential reward and divides by Z, the original-law average of exponential reward.

[Equation reading 2](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u104_r1)

At a noisy state, average the exponentiated future reward.

$$
h_t(x)=\mathbb E[e^{r(Z_1)}\mid X_t=x],\qquad V_t(x)=\log h_t(x)
$$

The function h at x and t averages exponentiated endpoint reward conditional on the current state x; the value V is the logarithm of h.

[Equation reading 3](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u104_r2)

The intermediate tilted marginal follows from Bayes’ rule.

$$
p_t^r(x)=\frac{p_t(x)h_t(x)}{Z}
$$

The tilted intermediate density is the original intermediate density multiplied by its conditional reward average h and divided by the same normalizer Z.

## A small posterior makes the exponential weighting explicit

[Equation reading 1](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u105_r0)

Suppose two clean endpoints are equally likely.

$$
p(z_A\mid x)=p(z_B\mid x)=\tfrac12,\qquad r_A=0,\quad r_B=\log4
$$

The two clean endpoints each have conditional probability one half, with rewards zero and log four.

[Equation reading 2](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u105_r1)

Average exponentiated reward before taking the logarithm.

$$
h=\tfrac12(1)+\tfrac12(4)=2.5,\qquad V=\log2.5\approx0.91629
$$

The conditional exponential-reward average is half of one plus half of four, giving 2.5; its logarithm is approximately 0.91629.

[Equation reading 3](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u105_r2)

The reward-tilted posterior probabilities are different.

$$
p^r(z_A\mid x)=\frac{0.5}{2.5}=0.2,\qquad p^r(z_B\mid x)=\frac2{2.5}=0.8
$$

Dividing the reward-weighted masses 0.5 and two by 2.5 gives tilted posterior probabilities 0.2 and 0.8.

## Even the denoiser mean can hide the available outcomes

[Equation reading 1](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u106_r0)

Let the clean posterior place mass at minus one and plus one.

$$
\Pr(Z_1=-1\mid x)=\Pr(Z_1=1\mid x)=\tfrac12
$$

The clean posterior places probability one half at minus one and probability one half at plus one.

[Equation reading 2](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u106_r1)

Choose a reward that favors magnitude.

$$
D_t(x)=\mathbb E[Z_1\mid x]=0,\qquad r(z)=z^2
$$

The denoiser’s conditional mean is zero, while the reward of a candidate z is its square.

[Equation reading 3](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u106_r2)

Compare mean-based lookahead with the true value.

$$
r(D_t(x))=0,\qquad\log\mathbb E[e^{Z_1^2}\mid x]=1
$$

Reward at the denoiser mean is zero, but the logarithm of the posterior average of exponentiated squared endpoint values is one.

## Two clocks distinguish generation from posterior sampling

[Equation reading 1](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u109_r0)

Fix the outer noisy observation and its time.

$$
(x,t)\quad\text{is held fixed}
$$

The outer noisy state x and its outer time t are fixed conditioning information throughout this inner sampling calculation.

[Equation reading 2](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u109_r1)

An inner flow acts on a separate state and inner times a, b.

$$
G_{a,b}(\bar x\mid x,t)
$$

The conditional map G moves a separate inner state from inner time a to inner time b while conditioning on the fixed outer observation x and t.

[Equation reading 3](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u109_r2)

Fresh inner noise gives a clean conditional sample.

$$
\epsilon\sim\mathcal N(0,I),\qquad G_{0,1}(\epsilon\mid x,t)\sim p_{1\mid t}(\cdot\mid x)
$$

Drawing fresh Gaussian inner noise and applying G from zero to one produces a sample from the clean endpoint distribution conditional on the outer observation.

## Several posterior samples estimate the value function

[Equation reading 1](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u110_r0)

Draw K independent inner noise samples.

$$
z_k=G_{0,1}(\epsilon_k\mid x,t),\qquad\epsilon_k\overset{\rm iid}{\sim}\mathcal N(0,I)
$$

Each candidate z is the conditional map applied to an independent standard Gaussian inner-noise sample, using the same outer state and time.

[Equation reading 2](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u110_r1)

Estimate the logarithm of the conditional reward average.

$$
\widehat V_t(x)=\log\left(\frac1K\sum_{k=1}^K e^{r(z_k)}\right)
$$

The estimated value is the logarithm of the arithmetic average of the K exponentiated candidate rewards.

[Equation reading 3](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u110_r2)

Differentiate through the posterior sampler.

$$
\nabla_x\widehat V_t=\sum_k\operatorname{softmax}(r(z_{1:K}))_k\,\nabla_x r(z_k)
$$

The value gradient is a weighted sum of candidate reward gradients, with weights given by softmax of their rewards and derivatives passing through the conditional sampler.

## The value gradient changes the transport velocity

[Equation reading 1](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u111_r0)

Write the noisy state using signal and noise schedules.

$$
X_t=\alpha_tZ_1+\sigma_t\epsilon
$$

The noisy state equals the clean endpoint multiplied by signal scale α plus standard Gaussian noise multiplied by noise scale σ.

[Equation reading 2](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u111_r1)

Define the schedule-dependent guidance coefficient.

$$
c_t=\sigma_t^2\frac{\dot\alpha_t}{\alpha_t}-\dot\sigma_t\sigma_t
$$

The guidance coefficient is noise variance times the signal’s relative time derivative, minus noise scale times its own time derivative.

[Equation reading 3](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u111_r2)

The reward-tilted probability-flow velocity adds the value gradient.

$$
b_t^r(x)=b_t(x)+c_t\nabla_xV_t(x)
$$

The reward-tilted probability-flow velocity equals the original velocity plus that schedule coefficient times the spatial gradient of the value function.

## Two noisy observations add their Gaussian precisions

[Equation reading 1](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u113_r0)

Condition on outer x and inner x-bar with independent noises.

$$
x=\alpha_tz+\sigma_t\epsilon,\qquad\bar x=\alpha_az+\sigma_a\bar\epsilon
$$

The outer and inner observations are separately scaled versions of the same clean endpoint z, each corrupted by its own independent Gaussian noise.

[Equation reading 2](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u113_r1)

Collect the quadratic coefficient and the linear coefficient in z.

$$
\Lambda=\frac{\alpha_t^2}{\sigma_t^2}+\frac{\alpha_a^2}{\sigma_a^2},\qquad m=\Lambda^{-1}\left(\frac{\alpha_tx}{\sigma_t^2}+\frac{\alpha_a\bar x}{\sigma_a^2}\right)
$$

The precision Λ sums the two squared signal-to-noise ratios; m is the sum of their signal-weighted, inverse-variance observations divided by Λ.

[Equation reading 3](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u113_r2)

Complete the square in the product likelihood.

$$
p(x,\bar x\mid z)\propto\exp\!\left[-\tfrac12\Lambda\|z-m\|^2\right]
$$

The joint observation likelihood, as a function of z, is proportional to the exponential of minus one half Λ times the squared distance from z to m.

## An effective noise level turns this into a denoiser query

[Equation reading 1](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u114_r0)

Choose an effective time with matching noise-to-signal ratio.

$$
\frac{\sigma_{t_*}^2}{\alpha_{t_*}^2}=\Lambda^{-1}
$$

The effective time is chosen so that its noise variance divided by squared signal scale equals the inverse combined precision.

[Equation reading 2](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u114_r1)

Rescale the effective observation into the original path coordinates.

$$
\widetilde x=\alpha_{t_*}m
$$

The effective observation is the combined likelihood center m multiplied by the signal scale at that effective time.

[Equation reading 3](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u114_r2)

The original denoiser supplies the joint posterior mean.

$$
\mathbb E[Z_1\mid x,\bar x]=D_{t_*}(\widetilde x)
$$

The conditional mean clean endpoint given both observations equals the original denoiser evaluated at the effective time and effective observation.

## That posterior mean defines the inner conditional velocity

[Equation reading 1](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u115_r0)

Differentiate the inner interpolant while holding outer x and t fixed.

$$
\dot{\bar X}_a=\dot\alpha_a Z_1+\dot\sigma_a\bar\epsilon
$$

The inner interpolant’s time derivative is the signal-scale derivative times the clean endpoint plus the noise-scale derivative times the fixed inner noise.

[Equation reading 2](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u115_r1)

Eliminate the inner noise using its observed state.

$$
\bar\epsilon=(\bar x-\alpha_a Z_1)/\sigma_a
$$

The inner noise is the observed inner state minus its signal-scaled clean endpoint, divided by the inner noise scale.

[Equation reading 3](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u115_r2)

Average over the posterior conditioned on both observations.

$$
\bar b_a(\bar x\mid x,t)=\frac{\dot\sigma_a}{\sigma_a}\bar x+\left(\dot\alpha_a-\frac{\dot\sigma_a\alpha_a}{\sigma_a}\right)D_{t_*}(\widetilde x)
$$

The inner conditional velocity combines the inner state times the relative noise-scale derivative with the joint posterior mean times the remaining signal derivative coefficient.

## The posterior map follows its own composition rule

[Equation reading 1](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u116_r0)

Keep the same outer observation throughout the inner trajectory.

$$
G_{a,c}(\bar x\mid x,t)=G_{b,c}(G_{a,b}(\bar x\mid x,t)\mid x,t)
$$

The conditional map from a to c equals the map from a to b followed by b to c, using the same outer observation in both maps.

[Equation reading 2](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u116_r1)

Anchor the diagonal to the conditional velocity.

$$
\partial_bG_{a,b}(\bar x\mid x,t)\big|_{b=a}=\bar b_a(\bar x\mid x,t)
$$

The end-time derivative on the inner time diagonal equals the conditional inner velocity at the starting inner state.

[Equation reading 3](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u116_r2)

Distill the inner Lagrangian identity.

$$
\mathcal L=\mathbb E\|\partial_bG_{a,b}-\bar b_b(G_{a,b}\mid x,t)\|^2
$$

The loss averages the squared difference between the inner map’s end-time derivative and the conditional velocity at its predicted endpoint.

## Re-noising defines a proposal from an earlier outer time

[Equation reading 1](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u118_r0)

Choose an earlier, noisier time q < t.

$$
x_q=kx+\sqrt{\sigma_q^2-k^2\sigma_t^2}\,\epsilon,\qquad k=\alpha_q/\alpha_t
$$

The earlier noisy proposal scales x by the signal ratio k and adds Gaussian noise with variance equal to σ at q squared minus k squared times σ at t squared.

[Equation reading 2](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u118_r1)

Map each perturbed state to a clean endpoint.

$$
z=F_{q,1}(x_q)
$$

The proposed earlier state is mapped from time q to a clean endpoint using the outer flow map.

[Equation reading 3](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u118_r2)

The endpoint law depends on the re-noising proposal.

$$
z\sim Q(\cdot\mid x),\qquad Q(\cdot\mid x)\ \text{need not equal}\ p_{1\mid t}(\cdot\mid x)
$$

That endpoint has the proposal law Q conditional on x, which need not be the true clean posterior conditional on x at time t.

## A recovery term enforces compatibility with the observation

[Equation reading 1](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u119_r0)

Include the likelihood of the current noisy observation.

$$
r_{\rm rec}(x,z)=r(z)-\frac{\|x-\alpha_tz\|^2}{2\sigma_t^2}
$$

The recovery reward is endpoint reward minus the squared observation mismatch divided by twice the observation-noise variance.

[Equation reading 2](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u119_r1)

Use the score to recover the change in the earlier marginal density.

$$
\Gamma=(x_q-x)^\top\int_0^1\nabla\log p_q(x+u(x_q-x))\,du
$$

The density correction Γ is the proposal displacement dotted with the integral of the earlier-time score along the straight segment connecting the two states.

[Equation reading 3](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u119_r2)

The exact-design log weight includes the proposal correction.

$$
\ell=r_{\rm rec}+\Gamma+\tfrac12\|\epsilon\|^2
$$

The corrected log weight adds recovery reward, the density correction Γ, and one half the squared norm of the Gaussian proposal noise.

## The gradient correction also depends on the noisy-state scores

[Equation reading 1](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u120_r0)

Define the additional score difference.

$$
\Delta s=\frac{\alpha_q}{\alpha_t}\nabla\log p_q(x_q)-\nabla\log p_t(x)
$$

The score correction is the earlier-time score at the proposal multiplied by the signal-scale ratio, minus the current-time score at x.

[Equation reading 2](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u120_r1)

Normalize the corrected log weights.

$$
w_k=\operatorname{softmax}(\ell_{1:K})_k
$$

The candidate weights are the softmax of their corrected log weights, so they are positive and sum to one.

[Equation reading 3](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u120_r2)

Combine recovery gradients with the score correction.

$$
\widehat{\nabla V_t}(x)=\sum_kw_k\left[\nabla_xr_{\rm rec}(x,z_k)+\Delta s_k\right]
$$

The estimated value gradient averages each candidate’s recovery-reward gradient plus its score correction using those normalized weights.

## Both observations share the same clean endpoint

[Equation reading 1](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u126_r0)

Draw a clean example and two independent noise samples.

$$
Z_1\sim p_1,\qquad Z_0,\bar Z_0\overset{\rm iid}{\sim}p_0
$$

Draw one clean endpoint from the data law and two independent source samples from the noise law.

[Equation reading 2](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u126_r1)

Construct the outer observation.

$$
I_t=\alpha_tZ_0+\beta_tZ_1
$$

The outer interpolant combines its noise sample with the clean endpoint using the outer noise coefficient α and signal coefficient β.

[Equation reading 3](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u126_r2)

Construct the inner interpolant using the same clean example.

$$
\bar I_s=\alpha_s\bar Z_0+\beta_sZ_1
$$

The inner interpolant combines its separate noise sample with the same clean endpoint using the inner-time coefficients α and β.

## Conditional flow matching now needs no solved trajectories

[Equation reading 1](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u127_r0)

Differentiate the inner interpolant.

$$
\dot{\bar I}_s=\dot\alpha_s\bar Z_0+\dot\beta_sZ_1
$$

The inner interpolant’s derivative is the derivative of α times the inner noise sample plus the derivative of β times the shared clean endpoint.

[Equation reading 2](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u127_r1)

Train its diagonal velocity conditioned on the outer observation.

$$
\mathcal L_{\rm diag}=\mathbb E\|v_\theta(s,s,\bar I_s;t,I_t)-\dot{\bar I}_s\|^2
$$

The diagonal loss averages squared error between the conditional equal-time velocity prediction and that directly computed inner interpolant derivative.

[Equation reading 3](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u127_r2)

The conditional mean is the minimizing velocity.

$$
\bar b_s(\bar x;t,x)=\mathbb E[\dot{\bar I}_s\mid\bar I_s=\bar x,I_t=x]
$$

The minimizing conditional velocity is the mean inner interpolant derivative given both the inner state and the outer observation.

## Finite inner maps reuse the same consistency identities

[Equation reading 1](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u128_r0)

Parameterize the inner destination with an average velocity.

$$
G_{s,u}(\bar x;t,x)=\bar x+(u-s)v_\theta(s,u,\bar x;t,x)
$$

The inner map adds the inner interval length times its learned average velocity to the inner state while conditioning on outer time t and state x.

[Equation reading 2](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u128_r1)

Compose inner intervals while keeping the outer context fixed.

$$
G_{s,v}(\bar x;t,x)=G_{u,v}(G_{s,u}(\bar x;t,x);t,x)
$$

The inner map from s to v equals the map from s to u followed by u to v, with the outer conditioning unchanged.

[Equation reading 3](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u128_r2)

At the final inner time, generate a conditional clean endpoint.

$$
G_{0,1}(\bar Z_0;t,x)\sim p_{1\mid t}(\cdot\mid x)
$$

Applying the inner map from zero to one to a fresh source sample produces the clean endpoint law conditional on the outer observation.

## The unconditional generator is included at the noise endpoint

[Equation reading 1](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u130_r0)

At t = 0, the outer observation is independent source noise.

$$
I_0=Z_0,\qquad Z_0\perp Z_1
$$

At outer time zero, the observation is the source noise Z zero, which is independent of the clean endpoint Z one.

[Equation reading 2](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u130_r1)

Conditioning on it does not change the clean distribution.

$$
p_{1\mid0}(z\mid x_0)=p_1(z)
$$

The clean endpoint law conditioned on a time-zero observation is therefore just the unconditional clean data law.

[Equation reading 3](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u130_r2)

The conditional family therefore contains an unconditional flow.

$$
G_{0,1}(\bar Z_0;0,x_0)\sim p_1
$$

Running the inner map with outer time zero consequently produces an unconditional data sample, even though the model accepts an outer observation.

## Reward weighting gives us another guidance estimator

[Equation reading 1](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u131_r0)

Define the tilted posterior mean.

$$
D_t^r(x)=\frac{\mathbb E[Z_1e^{r(Z_1)}\mid I_t=x]}{\mathbb E[e^{r(Z_1)}\mid I_t=x]}
$$

The reward-tilted denoiser is the conditional average of clean endpoint times exponential reward divided by the conditional average of exponential reward.

[Equation reading 2](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u131_r1)

Differentiate the Gaussian likelihood of the outer observation.

$$
\nabla_xV_t(x)=\frac{\beta_t}{\alpha_t^2}\left(D_t^r(x)-D_t(x)\right)
$$

The value gradient equals the signal coefficient β divided by noise coefficient α squared, multiplied by tilted posterior mean minus ordinary posterior mean.

[Equation reading 3](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u131_r2)

Insert the tilted mean directly into the velocity.

$$
b_t^r(x)=\frac{\dot\alpha_t}{\alpha_t}x+\left(\dot\beta_t-\frac{\dot\alpha_t\beta_t}{\alpha_t}\right)D_t^r(x)
$$

The tilted velocity combines x times the relative noise derivative with the tilted denoiser times the signal derivative corrected for that relative noise change.

## Our two-outcome example gives that tilted mean immediately

[Equation reading 1](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u132_r0)

Use clean endpoints minus one and plus one with equal posterior mass.

$$
z_A=-1,\quad z_B=1,\quad e^{r_A}=1,\quad e^{r_B}=4
$$

The two endpoint values are minus one and plus one, with exponentiated reward weights one and four.

[Equation reading 2](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u132_r1)

The untilted mean is zero and the tilted mean favors plus one.

$$
D=0,\qquad D^r=0.2(-1)+0.8(1)=0.6
$$

The ordinary mean is zero, while the tilted mean weights minus one by 0.2 and plus one by 0.8, giving 0.6.

[Equation reading 3](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u132_r2)

Choose α = β = 0.5 to calculate the value gradient.

$$
\nabla_xV=\frac{0.5}{0.5^2}(0.6-0)=1.2
$$

With signal and noise coefficients both 0.5, the value gradient is 0.5 divided by 0.5 squared times the mean difference 0.6, giving 1.2.

## Weighted samples estimate the tilted mean

[Equation reading 1](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u133_r0)

Draw clean candidates from the conditional map.

$$
z_k=G_{0,1}(\epsilon_k;t,x)
$$

Each clean candidate z is generated by applying the inner conditional map to an independent noise input at the fixed outer context.

[Equation reading 2](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u133_r1)

Normalize the exponentiated rewards.

$$
w_k=\frac{e^{r(z_k)}}{\sum_je^{r(z_j)}}
$$

The weight of candidate k is its exponentiated reward divided by the sum of all candidates’ exponentiated rewards.

[Equation reading 3](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u133_r2)

Average the candidates using these weights.

$$
\widehat D_t^r(x)=\sum_kw_kz_k
$$

The estimated tilted denoiser is the sum of candidate endpoints multiplied by these normalized reward weights.

## Adding diffusion requires a drift correction

[Equation reading 1](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u135_r0)

Start from a probability-flow velocity b and density p.

$$
\partial_tp=-\nabla\cdot(bp)
$$

The density’s time derivative is minus the divergence of density times probability-flow velocity b.

[Equation reading 2](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u135_r1)

Add scalar diffusion g and a score correction to the drift.

$$
dX_t=\left[b_t(X_t)+\tfrac12g_t^2\nabla\log p_t(X_t)\right]dt+g_t\,dW_t
$$

The SDE adds Brownian noise with scale g and uses drift b plus one half g squared times the score of the current density.

[Equation reading 3](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u135_r2)

The score term cancels the new diffusion term in Fokker–Planck.

$$
-\nabla\cdot\left(\tfrac12g_t^2p_t\nabla\log p_t\right)+\tfrac12g_t^2\Delta p_t=0
$$

The negative divergence of the added score-weighted probability flux exactly cancels one half g squared times the density Laplacian.

## The Doob transform determines stochastic guidance

[Equation reading 1](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u136_r0)

A reward tilt adds the value gradient to the SDE drift.

$$
f_t^r=f_t+g_t^2\nabla V_t
$$

The Doob-transformed SDE drift equals its original drift plus diffusion variance g squared times the gradient of the value function.

[Equation reading 2](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u136_r1)

The tilted score also changes.

$$
\nabla\log p_t^r=\nabla\log p_t+\nabla V_t
$$

The tilted density’s score equals the original score plus the value gradient.

[Equation reading 3](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u136_r2)

Subtract half the diffusion-weighted score to obtain probability flow.

$$
b_t^r=b_t+\tfrac12g_t^2\nabla V_t
$$

Subtracting half the diffusion-weighted tilted score from the tilted SDE drift leaves probability-flow velocity b plus one half g squared times the value gradient.

## Fine-tuning can avoid a normalized Monte Carlo ratio

[Equation reading 1](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u137_r0)

Write the desired change in velocity as d and the reward weight as w.

$$
d=b_{\rm student}-b_{\rm teacher},\qquad w=e^{r(G_{0,1}(\epsilon;t,x))},\qquad a=\tfrac12g_t^2
$$

The drift difference d is student minus teacher; w is exponentiated reward of a conditional sample, and a is one half the diffusion variance.

[Equation reading 2](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u137_r1)

The desired drift relation can be written without dividing by E[w].

$$
\mathbb E[w]d=a\,\mathbb E[\nabla_xw]
$$

The desired relation multiplies d by the expected reward weight and sets it equal to a times the expected spatial gradient of that weight.

[Equation reading 3](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u137_r2)

Equivalently, its estimating equation has zero mean.

$$
\mathbb E[w\,d-a\nabla_xw]=0
$$

Equivalently, the expected residual w times d minus a times the weight gradient is zero.

## The detached surrogate has exactly that expected gradient

[Equation reading 1](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u138_r0)

Treat the posterior sampler and teacher as frozen.

$$
R=d+(w-1)\operatorname{sg}(d)-a\nabla_xw
$$

The surrogate residual is d plus w minus one times a detached copy of d, minus a times the spatial gradient of w.

[Equation reading 2](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u138_r1)

Use the squared surrogate residual.

$$
\mathcal L_{\rm FT}=\mathbb E\|R\|^2
$$

The fine-tuning loss is the expected squared norm of that surrogate residual R.

[Equation reading 3](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u138_r2)

Differentiate with respect to the trainable drift difference.

$$
\nabla_d\mathcal L_{\rm FT}=2\mathbb E[w\,d-a\nabla_xw]
$$

Differentiating with respect to the trainable drift difference gives twice the expected weighted drift residual, because the extra copy of d is detached.

## Factor the finite move into expansion and transport

[Equation reading 1](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u146_r0)

At time s, the active state has dimension d(s).

$$
x_s\in\mathbb R^{d(s)},\qquad d(t)\geq d(s)
$$

The active state at time s lies in a space of dimension d at s, and the later dimension is at least as large.

[Equation reading 2](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u146_r1)

Inject conditional noise into the additional coordinates.

$$
E_{s,t}(x_s,\epsilon)\in\mathbb R^{d(t)}
$$

The expansion map takes the current state and a fresh noise input and returns a state in the later, possibly larger-dimensional space.

[Equation reading 3](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u146_r2)

Transport the augmented state to the later time.

$$
\Phi_{s,t}(x_s,\epsilon)=X_{s,t}(E_{s,t}(x_s,\epsilon))
$$

The full finite move first applies expansion E and then applies transport X to the augmented state.

## A two-coordinate example makes the change of space explicit

[Equation reading 1](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u147_r0)

Begin with a single observed coordinate.

$$
x_s=(2)\in\mathbb R
$$

The starting state contains the single coordinate two and lies on the real line.

[Equation reading 2](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u147_r1)

Append one noise coordinate.

$$
\epsilon=-0.4,\qquad E_{s,t}(x_s,\epsilon)=(2,-0.4)\in\mathbb R^2
$$

Appending the noise coordinate minus 0.4 gives the two-dimensional augmented state two, minus 0.4.

[Equation reading 3](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u147_r2)

Apply a transport in the enlarged space.

$$
X_{s,t}(z_1,z_2)=(z_1+0.5,z_2+0.8)\quad\Longrightarrow\quad\Phi=(2.5,0.4)
$$

The transport adds 0.5 to the first coordinate and 0.8 to the second, producing the destination 2.5, 0.4.

## Each coordinate needs a clock that begins at its birth

[Equation reading 1](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u148_r0)

Let coordinate i enter the state at global time b-i.

$$
b_i\in[0,1),\qquad i\text{ is inactive when }t<b_i
$$

The birth time of coordinate i lies between zero and one, and that coordinate is inactive before its birth.

[Equation reading 2](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u148_r1)

Rescale its remaining lifetime to a local unit interval.

$$
\tau_i(t)=\frac{t-b_i}{1-b_i},\qquad t\geq b_i
$$

After birth, the coordinate’s local time is global time minus birth time divided by one minus birth time.

[Equation reading 3](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u148_r2)

Interpolate from new noise to its clean value.

$$
I_t^i=(1-\tau_i(t))Z_0^i+\tau_i(t)Z_1^i
$$

The coordinate interpolant mixes its fresh noise value and clean value with weights one minus its local time and its local time.

## Late-born coordinates must move faster in global time

[Equation reading 1](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u149_r0)

Differentiate the local clock.

$$
\frac{d\tau_i}{dt}=\frac1{1-b_i}
$$

The local clock advances at rate one divided by one minus the coordinate’s birth time.

[Equation reading 2](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u149_r1)

Apply the chain rule to the coordinate interpolant.

$$
\frac{dI_t^i}{dt}=\frac{Z_1^i-Z_0^i}{1-b_i}
$$

The coordinate velocity is its clean-minus-noise displacement divided by the global time remaining after birth.

[Equation reading 3](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u149_r2)

For a birth at b = 0.75, the scale factor is four.

$$
b_i=0.75\quad\Longrightarrow\quad\frac{dI_t^i}{dt}=4(Z_1^i-Z_0^i)
$$

A coordinate born at time 0.75 has one quarter of the global interval left, so its velocity is four times its clean-minus-noise displacement.

## Coordinates can have different progress at the same time

[Equation reading 1](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u150_r0)

Use one coordinate born at zero and another born at one half.

$$
b_1=0,\qquad b_2=0.5,\qquad t=0.75
$$

The first coordinate is born at zero, the second at 0.5, and both are observed at global time 0.75.

[Equation reading 2](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u150_r1)

Compute their local times.

$$
\tau_1=0.75,\qquad\tau_2=\frac{0.75-0.5}{1-0.5}=0.5
$$

The first coordinate’s local time is 0.75, while the second has used 0.25 of its 0.5 lifetime, giving local time 0.5.

[Equation reading 3](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u150_r2)

With shared noise value zero and clean value one, their states differ.

$$
I_t^1=0.75,\qquad I_t^2=0.5
$$

With noise zero and clean value one, the two coordinate states equal their local times: 0.75 and 0.5.

## The generator combines transport and expansion events

[Equation reading 1](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u153_r0)

Let b be the within-stratum velocity and λ the birth rate.

$$
b(t,x)\in\mathbb R^{d},\qquad\lambda(t,x)\geq0
$$

The within-space velocity b has the current state’s dimension d, while the expansion-event rate λ is nonnegative.

[Equation reading 2](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u153_r1)

An event augments the state using a noise mark ε.

$$
x\longmapsto E(x,\epsilon),\qquad\epsilon\sim q(d\epsilon\mid t,x)
$$

At an expansion event, x is replaced by E of x and a noise mark ε drawn from the conditional mark law q.

[Equation reading 3](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u153_r2)

The generator describes the expected change of a test function.

$$
\mathcal Af=b\cdot\nabla f+\lambda\int[f(E(x,\epsilon))-f(x)]q(d\epsilon\mid t,x)
$$

The generator adds the directional derivative along b to λ times the average change in the test function caused by a marked expansion event.

## The jump term follows from a short-time calculation

[Equation reading 1](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u154_r0)

Over a short interval h, one event occurs with probability λh + o(h).

$$
\Pr(\text{one event})=\lambda h+o(h),\qquad\Pr(\text{no event})=1-\lambda h+o(h)
$$

Over a short interval h, the probability of one event is λh plus a smaller-order term, while the probability of no event is one minus λh plus a smaller-order term.

[Equation reading 2](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u154_r1)

Average the test-function change over both possibilities.

$$
\mathbb E[f(X_{t+h})-f(x)]\!=h\,b\cdot\nabla f+\lambda h\!\int[f(E(x,\epsilon))-f(x)]q(d\epsilon)+o(h)
$$

The expected test-function change is h times its transport derivative plus λh times its average jump change, up to a remainder smaller than h.

[Equation reading 3](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u154_r2)

Divide by h and let h approach zero.

$$
\mathcal Af=\lim_{h\downarrow0}\frac{\mathbb E[f(X_{t+h})\mid X_t=x]-f(x)}h
$$

The generator is the limit of the conditional expected test-function increment divided by h as the time interval shrinks to zero.

## The residual map now begins from the augmented state

[Equation reading 1](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u155_r0)

Create the state on the destination canvas.

$$
z=E_{s,t}(x,\epsilon)
$$

The augmented state z is the result of applying the expansion map to the starting state and its expansion noise.

[Equation reading 2](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u155_r1)

Predict its interval-average transport.

$$
\Phi_{s,t}(x,\epsilon)=z+(t-s)v_\theta(s,t,z)
$$

The destination adds the interval length times the learned average transport velocity to the augmented state z.

[Equation reading 3](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u155_r2)

If no coordinates are added, the fixed-dimensional case is recovered.

$$
E_{s,t}(x)=x\quad\Longrightarrow\quad\Phi_{s,t}(x)=x+(t-s)v_\theta(s,t,x)
$$

When expansion leaves x unchanged, the expression reduces to x plus the interval length times its ordinary fixed-dimensional average velocity.

## Composition must reuse the same expansion randomness

[Equation reading 1](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u156_r0)

Associate each birth with a noise mark that is reused under refinement.

$$
\omega=\{(b_i,\epsilon_i)\}_i
$$

The marked history ω records each coordinate’s birth time together with its associated noise sample.

[Equation reading 2](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u156_r1)

Split the same marked expansion history at u.

$$
\omega_{s,t}=\omega_{s,u}\cup\omega_{u,t}
$$

The marks for the whole interval are the union of the marks occurring before and after the intermediate time u.

[Equation reading 3](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u156_r2)

A pathwise composition uses those same marks on both routes.

$$
\Phi_{s,t}(x;\omega)=\Phi_{u,t}(\Phi_{s,u}(x;\omega_{s,u});\omega_{u,t})
$$

Pathwise composition applies the first map with the first set of marks and the second map with the remaining marks, matching the direct map with their union.

## Include the noise density before applying a square Jacobian

[Equation reading 1](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u158_r0)

For a simple concatenation expansion, form the augmented density.

$$
z=(x,\epsilon),\qquad q(z)=p_s(x)q(\epsilon\mid x)
$$

The augmented state concatenates x and ε, and its joint density is the original state density multiplied by the conditional density of the added noise.

[Equation reading 2](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u158_r1)

Apply a same-dimensional invertible transport y = X(z).

$$
p_t(y)=q(X^{-1}(y))\left|\det JX^{-1}(y)\right|
$$

The destination density is the augmented density evaluated at the inverse transport, multiplied by the absolute determinant of the inverse Jacobian.

[Equation reading 3](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u158_r2)

The noise and transport both contribute to the log density.

$$
\log p_t(X(z))=\log p_s(x)+\log q(\epsilon\mid x)-\log|\det JX(z)|
$$

The destination log density is source log density plus conditional noise log density minus the log absolute Jacobian determinant of the forward transport.

## The schedule determines the chance of insertion

[Equation reading 1](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u161_r0)

Let a(t) be the cumulative probability that a position has been inserted.

$$
\Pr(B\leq t)=a(t),\qquad a(0)=0,\quad a(1)=1
$$

The insertion time B has cumulative distribution a, starting at zero probability at time zero and reaching probability one at time one.

[Equation reading 2](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u161_r1)

Condition on the position still being absent at s.

$$
\Pr(s<B\leq t\mid B>s)=\frac{a(t)-a(s)}{1-a(s)}
$$

Given that insertion has not occurred by s, its probability of occurring by t is the cumulative probability increase divided by the remaining probability at s.

[Equation reading 3](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u161_r2)

Denote the fraction of remaining positions inserted over the interval.

$$
\rho_{s,t}=\frac{a(t)-a(s)}{1-a(s)}
$$

The interval insertion fraction ρ is a at t minus a at s, divided by one minus a at s.

## The expected gap count scales with that insertion probability

[Equation reading 1](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u162_r0)

Let g-i be the number of clean positions still missing from gap i.

$$
m_i(s,x_s)=\mathbb E[g_i(s)\mid X_s=x_s]
$$

The predicted remaining gap count m is the conditional mean of the clean positions still missing from that gap, given the current state.

[Equation reading 2](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u162_r1)

Each missing position is inserted with probability rho.

$$
I_{s,t}(x_s)[i]=\rho_{s,t}\,m_i(s,x_s)
$$

The expected inserted count in gap i is the interval insertion probability ρ multiplied by that gap’s predicted remaining count.

[Equation reading 3](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u162_r2)

In the infinitesimal limit, recover the birth hazard.

$$
\lambda_t=\frac{\dot a(t)}{1-a(t)}
$$

The instantaneous birth hazard is the schedule derivative divided by the probability that insertion has not yet occurred.

## A concrete gap example gives two expected insertions

[Equation reading 1](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u163_r0)

Use a linear insertion schedule and three expected missing positions.

$$
a(t)=t,\qquad s=0.25,\quad t=0.75,\quad m_i=3
$$

The cumulative insertion schedule equals time, the interval runs from 0.25 to 0.75, and the predicted remaining gap count is three.

[Equation reading 2](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u163_r1)

Calculate the conditional insertion probability.

$$
\rho_{s,t}=\frac{0.75-0.25}{1-0.25}=\frac23
$$

The conditional insertion probability is 0.75 minus 0.25 divided by one minus 0.25, giving two thirds.

[Equation reading 3](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u163_r2)

Multiply by the expected remaining count.

$$
I_{s,t}[i]=\frac23(3)=2
$$

Multiplying insertion probability two thirds by remaining count three gives an expected inserted count of two.

## The count loss learns a real-valued insertion mean

[Equation reading 1](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u164_r0)

Use the nonnegative count divergence for target a and predicted mean b.

$$
\phi(a,b)=b-a+a\log(a/b),\qquad b>0
$$

The count divergence is predicted mean b minus target count a plus a times the logarithm of a divided by b, with b strictly positive.

[Equation reading 2](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u164_r1)

Differentiate the conditional expected loss.

$$
\partial_b\mathbb E[\phi(A,b)\mid x]=1-\frac{\mathbb E[A\mid x]}b
$$

The derivative of its conditional expected loss is one minus the conditional expected target count divided by the predicted mean.

[Equation reading 3](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u164_r2)

Set the derivative to zero.

$$
b^*(x)=\mathbb E[A\mid x]
$$

The minimizing prediction b equals the conditional mean of the target count given x.

## The sampler turns that mean into a bounded count

[Equation reading 1](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u165_r0)

Let B be the remaining token budget and μ the predicted gap mean.

$$
0\leq\mu\leq B,\qquad p=\mu/B
$$

The predicted mean μ lies between zero and the remaining budget B, and the binomial success probability is μ divided by B.

[Equation reading 2](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u165_r1)

Use the paper’s capped binomial count proposal.

$$
L_i\sim\operatorname{Binomial}(B,p),\qquad\mathbb E[L_i]=\mu
$$

The proposed inserted count is binomial with B trials and success probability p, so its expectation equals μ.

[Equation reading 3](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u165_r2)

For B = 4 and μ = 2, the variance is one.

$$
p=\tfrac12,\qquad\operatorname{Var}(L_i)=4(\tfrac12)(\tfrac12)=1
$$

With budget four and mean two, the success probability is one half and the count variance is four times one half times one half, giving one.

## Each token uses its own local interpolation weight

[Equation reading 1](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u166_r0)

For coordinate i, start from its current local time tau-s.

$$
\eta_i=\frac{\tau_i(t)-\tau_i(s)}{1-\tau_i(s)}
$$

Coordinate i uses interpolation weight equal to its local-time advance divided by its remaining local time at the start.

[Equation reading 2](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u166_r1)

Apply the categorical mean-denoiser update coordinate-wise.

$$
F^i=(1-\eta_i)z^i+\eta_i\psi^i
$$

The coordinate destination is one minus η times its current value plus η times its categorical denoiser prediction.

[Equation reading 3](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u166_r2)

A newly inserted token starts from local time zero.

$$
\tau_i(s)=0\quad\Longrightarrow\quad F^i=(1-\tau_i(t))z^i+\tau_i(t)\psi^i
$$

For a newly inserted coordinate with starting local time zero, η equals its destination local time, giving the displayed noise-to-denoiser interpolation.

## Let’s first implement one insertion interval

[Equation reading 1](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u167_r0)

Expand the state and preserve the matching coordinate clocks.

$$
(x_s,\{b_i\})\xrightarrow{E_{s,t}}(z,\{b_i^{\rm expanded}\}),\qquad\tau_i(t)=\frac{t-b_i}{1-b_i}
$$

The expansion carries both the state and its coordinate birth times into the larger space; each coordinate’s local time is then computed from its own birth time.

## The SDE solution depends on an entire driving path

[Equation reading 1](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u176_r0)

Consider the additive-noise Itô SDE.

$$
dX_t=f(t,X_t)\,dt+g(t)\,dW_t
$$

The state increment consists of drift f at the current time and state times dt, plus time-dependent noise scale g times a Brownian increment.

[Equation reading 2](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u176_r1)

Write its integral form from a starting state x.

$$
X_t=x+\int_s^t f(u,X_u)\,du+\int_s^tg(u)\,dW_u
$$

The endpoint equals the starting state plus the ordinary integral of drift along the path and the Itô integral of noise scale against Brownian motion.

[Equation reading 3](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u176_r2)

The Itô map takes the initial state and that noise realization.

$$
\Psi_{s,t}(x,W_{[s,t]})=X_t
$$

The Itô map Ψ takes the starting state and the entire Brownian path segment from s to t and returns the resulting SDE endpoint.

## Strong and weak error ask different questions

[Equation reading 1](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u177_r0)

Strong error compares solutions driven by the same Brownian path.

$$
\mathbb E\|\widehat\Psi(x,W)-\Psi(x,W)\|^2
$$

The strong squared error averages the squared distance between learned and exact endpoints driven by the same starting state and Brownian path.

[Equation reading 2](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u177_r1)

Weak error compares expectations of test functions.

$$
\left|\mathbb E[\varphi(\widehat X_t)]-\mathbb E[\varphi(X_t)]\right|
$$

The weak error is the absolute difference between the expected test-function values under the learned and exact terminal distributions.

[Equation reading 3](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u177_r2)

Equal marginal laws can coexist with a poor pathwise coupling.

$$
\widehat X_t\overset d=X_t\quad\not\Longrightarrow\quad\widehat X_t=X_t\ \text{for the same }W
$$

Equality in distribution of the two endpoints does not imply that they agree when both are evaluated using the same Brownian realization.

## A constant-coefficient SDE gives an exact first example

[Equation reading 1](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u178_r0)

Choose constant drift and diffusion.

$$
dX_t=a\,dt+\sigma\,dW_t
$$

The SDE has constant drift a and constant noise scale σ multiplying Brownian motion.

[Equation reading 2](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u178_r1)

Integrate both terms over the interval.

$$
\Psi_{s,t}(x,W)=x+a(t-s)+\sigma(W_t-W_s)
$$

The exact endpoint is x plus a times the elapsed interval plus σ times the Brownian increment from s to t.

[Equation reading 3](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u178_r2)

Only the total Brownian increment is needed in this special case.

$$
\Delta W_{s,t}=W_t-W_s\sim\mathcal N(0,t-s)
$$

The total Brownian increment is the difference W at t minus W at s, normally distributed with mean zero and variance t minus s.

## Two half steps agree when their noise increments add

[Equation reading 1](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u179_r0)

Choose x = 1, a = 0.5, sigma = 0.8, and two half intervals.

$$
\Delta W_L=0.2,\qquad\Delta W_R=-0.1,\qquad\Delta W=0.1
$$

The left and right Brownian increments are 0.2 and minus 0.1, so the whole-interval increment is their sum, 0.1.

[Equation reading 2](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u179_r1)

The direct move uses the summed increment.

$$
X_1=1+0.5(1)+0.8(0.1)=1.58
$$

The direct endpoint adds drift 0.5 and noise contribution 0.8 times 0.1 to the starting value one, giving 1.58.

[Equation reading 3](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u179_r2)

The split move reaches the same endpoint.

$$
X_{1/2}=1+0.25+0.16=1.41,\qquad X_1=1.41+0.25-0.08=1.58
$$

The first half reaches 1.41; the second adds drift 0.25 and noise minus 0.08, reaching the same endpoint 1.58.

## Independent coarse noise breaks that agreement

[Equation reading 1](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u180_r0)

Keep the fine increments from the preceding example.

$$
\Delta W_L+\Delta W_R=0.1
$$

The two fine-interval Brownian increments sum to 0.1, fixing the whole-interval increment for that same path.

[Equation reading 2](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u180_r1)

Independently draw a coarse increment equal to minus 0.3.

$$
\widetilde{\Delta W}=-0.3
$$

The independently drawn coarse increment is minus 0.3, which differs from the fine increments’ sum.

[Equation reading 3](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u180_r2)

The coarse endpoint is now different.

$$
\widetilde X_1=1+0.5+0.8(-0.3)=1.26\ne1.58
$$

Using that independent increment gives endpoint 1.26, differing from the coupled fine-path endpoint 1.58.

## A changing drift can require more path information

[Equation reading 1](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u181_r0)

Consider the Ornstein–Uhlenbeck SDE.

$$
dX_t=-\lambda X_t\,dt+\sigma\,dW_t
$$

The Ornstein–Uhlenbeck drift is minus λ times the current state, with constant noise scale σ multiplying Brownian motion.

[Equation reading 2](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u181_r1)

An integrating factor gives the exact solution.

$$
X_t=e^{-\lambda(t-s)}x+\sigma\int_s^t e^{-\lambda(t-u)}\,dW_u
$$

The exact solution exponentially damps the starting state and adds a Brownian integral whose later increments receive larger exponential weights.

[Equation reading 3](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u181_r2)

The marginal variance has a closed form.

$$
\operatorname{Var}(X_t\mid X_s=x)=\frac{\sigma^2}{2\lambda}(1-e^{-2\lambda(t-s)})
$$

The conditional variance is σ squared divided by two λ, multiplied by one minus e raised to minus two λ times the interval length.

## The weighted-noise variance follows from Itô isometry

[Equation reading 1](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u182_r0)

Let J be the stochastic integral in the OU solution.

$$
\mathbb E[J^2]=\sigma^2\int_s^t e^{-2\lambda(t-u)}\,du
$$

The second moment of the noise contribution J equals σ squared times the integral of the squared exponential weighting function.

[Equation reading 2](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u182_r1)

Evaluate the ordinary integral.

$$
\sigma^2\int_0^{t-s}e^{-2\lambda r}dr=\frac{\sigma^2}{2\lambda}(1-e^{-2\lambda(t-s)})
$$

Evaluating this integral gives σ squared over two λ times one minus the exponential of minus two λ times the elapsed interval.

## Shifted Legendre polynomials define the path coefficients

[Equation reading 1](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u184_r0)

Normalize time within an interval of length h.

$$
h=t-s,\qquad q=(u-s)/h\in[0,1]
$$

The interval length h is t minus s, and the normalized inner time q is u minus s divided by h, lying between zero and one.

[Equation reading 2](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u184_r1)

The first two shifted Legendre polynomials are simple.

$$
\widetilde P_0(q)=1,\qquad\widetilde P_1(q)=2q-1
$$

The zeroth shifted Legendre polynomial is one, and the first is two q minus one.

[Equation reading 3](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u184_r2)

Project Brownian increments onto each polynomial.

$$
I^{(n)}_{s,t}=\int_s^t\widetilde P_n\!\left(\frac{u-s}{h}\right)dW_u
$$

Coefficient n is the Itô integral of the nth shifted Legendre polynomial, evaluated at normalized interval time, against Brownian increments.

## Orthogonality gives independent Gaussian coefficients

[Equation reading 1](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u185_r0)

Apply Itô isometry to a pair of deterministic integrands.

$$
\mathbb E[I^{(n)}I^{(m)}]=h\int_0^1\widetilde P_n(q)\widetilde P_m(q)dq
$$

The expected product of coefficients n and m equals h times the integral, from zero to one, of the product of their shifted Legendre polynomials.

[Equation reading 2](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u185_r1)

Use the orthogonality of shifted Legendre polynomials.

$$
\mathbb E[I^{(n)}I^{(m)}]=\frac{h}{2n+1}\,\mathbf1\{n=m\}
$$

Orthogonality makes that covariance zero for different indices and h divided by two n plus one when the indices agree.

[Equation reading 3](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u185_r2)

Joint Gaussianity turns zero covariance into independence.

$$
I^{(n)}_{s,t}\overset{\rm independent}{\sim}\mathcal N\!\left(0,\frac{h}{2n+1}\right)
$$

The coefficients are independent zero-mean Gaussians, with the nth coefficient having variance h divided by two n plus one.

## Integrating the basis reconstructs a polynomial noise path

[Equation reading 1](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u186_r0)

Keep N coefficients on the interval.

$$
W^{(N)}_{u,v}=\sum_{n=0}^{N-1}\frac{2n+1}{h}I^{(n)}_{s,t}\int_u^v\widetilde P_n\!\left(\frac{r-s}{h}\right)dr
$$

The reconstructed noise increment sums each retained coefficient times its integrated polynomial basis, scaled by two n plus one divided by h.

[Equation reading 2](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u186_r1)

One coefficient produces a straight interpolation of the increment.

$$
W^{(1)}_{s,s+qh}=qI^{(0)}
$$

With only the zeroth coefficient, the reconstructed path at fractional time q is q times the total Brownian increment.

[Equation reading 3](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u186_r2)

Two coefficients add an interior curvature term.

$$
W^{(2)}_{s,s+qh}=qI^{(0)}+3I^{(1)}(q^2-q)
$$

With two coefficients, the path adds three times the first-order coefficient times q squared minus q to that straight interpolation.

## The next coefficient changes the interior of the path

[Equation reading 1](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u187_r0)

Choose a unit interval and two coefficients.

$$
I^{(0)}=0.1,\qquad I^{(1)}=0.2
$$

On the unit interval, choose total increment 0.1 and first-order path coefficient 0.2.

[Equation reading 2](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u187_r1)

Evaluate the two-coefficient path at its midpoint.

$$
W^{(2)}_{0,1/2}=0.5(0.1)+3(0.2)(0.25-0.5)=-0.1
$$

At the midpoint, the linear term is 0.05 and the curvature term is minus 0.15, so the reconstructed path value is minus 0.1.

[Equation reading 3](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u187_r2)

At the endpoint, the correction vanishes.

$$
W^{(2)}_{0,1}=0.1+3(0.2)(1-1)=0.1
$$

At the endpoint q equals one, the curvature factor is zero, so the path still ends at the total increment 0.1.

## The zeroth coefficient combines by adding increments

[Equation reading 1](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u189_r0)

Split the interval at u.

$$
[s,t]=[s,u]\cup[u,t]
$$

The full interval from s to t is split into the left interval from s to u and the right interval from u to t.

[Equation reading 2](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u189_r1)

Split the stochastic integral of the constant basis function.

$$
I^{(0)}_{s,t}=\int_s^u dW+\int_u^t dW
$$

The zeroth coefficient is the sum of the Brownian integrals over the left and right intervals.

[Equation reading 3](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u189_r2)

The coarse coefficient is the sum of the fine coefficients.

$$
I^{(0)}_{s,t}=I^{(0)}_{s,u}+I^{(0)}_{u,t}
$$

The whole-interval zeroth coefficient equals the left zeroth coefficient plus the right zeroth coefficient.

## The first-order coefficient also has an exact composition rule

[Equation reading 1](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u190_r0)

Let the left and right lengths be h-L and h-R.

$$
h_L=u-s,\qquad h_R=t-u,\qquad h=h_L+h_R
$$

The left length is u minus s, the right length is t minus u, and the total length is their sum.

[Equation reading 2](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u190_r1)

Restrict the global linear basis to each local interval.

$$
\widetilde P_1^{\rm global}=\frac{h_L}{h}\widetilde P_1^L-\frac{h_R}{h}\ \text{on L},\qquad\frac{h_R}{h}\widetilde P_1^R+\frac{h_L}{h}\ \text{on R}
$$

On each half, the global linear polynomial is its local polynomial scaled by that half’s length fraction, with the displayed constant offset.

[Equation reading 3](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u190_r2)

Integrate those expressions against the same Brownian path.

$$
I^{(1)}_{s,t}=\frac{h_L}{h}I_L^{(1)}+\frac{h_R}{h}I_R^{(1)}-\frac{h_R}{h}I_L^{(0)}+\frac{h_L}{h}I_R^{(0)}
$$

The coarse first-order coefficient combines length-weighted fine first-order coefficients, subtracts the right fraction of the left increment, and adds the left fraction of the right increment.

## We can calculate the coefficients for two equal halves

[Equation reading 1](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u191_r0)

Use the previously chosen fine increments and two first-order terms.

$$
I_L^{(0)}=0.2,\quad I_R^{(0)}=-0.1,\quad I_L^{(1)}=0.04,\quad I_R^{(1)}=-0.02
$$

The fine zeroth coefficients are 0.2 and minus 0.1, while their first-order coefficients are 0.04 and minus 0.02.

[Equation reading 2](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u191_r1)

The coarse increment remains their sum.

$$
I^{(0)}=0.2-0.1=0.1
$$

Adding the zeroth coefficients gives the coarse total increment 0.1.

[Equation reading 3](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u191_r2)

Combine the first-order coefficients with the equal-half rule.

$$
I^{(1)}=\tfrac12(0.04-0.02-0.2-0.1)=-0.14
$$

The equal-half formula takes half of 0.04 minus 0.02 minus 0.2 minus 0.1, giving the coarse first-order coefficient minus 0.14.

## The composition rule also preserves the correct variance

[Equation reading 1](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u192_r0)

On half intervals, increment variance is one half and first-order variance is one sixth.

$$
\operatorname{Var}(I_L^{(0)})=\operatorname{Var}(I_R^{(0)})=\tfrac12,\quad\operatorname{Var}(I_L^{(1)})=\operatorname{Var}(I_R^{(1)})=\tfrac16
$$

Each half-interval zeroth coefficient has variance one half, and each first-order coefficient has variance one sixth.

[Equation reading 2](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u192_r1)

The four coefficients are independent.

$$
\operatorname{Var}(I^{(1)})=\tfrac14\left(\tfrac16+\tfrac16+\tfrac12+\tfrac12\right)
$$

Because the four terms are independent and have coefficients of magnitude one half, the coarse variance is one quarter of the sum of their variances.

[Equation reading 3](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u192_r2)

This equals the unit-interval first-order variance.

$$
\operatorname{Var}(I^{(1)})=\tfrac13
$$

The resulting coarse first-order variance is one third, matching the first-order coefficient on a unit interval.

## The map learns interval drift and diffusion terms

[Equation reading 1](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u193_r0)

Use the paper’s Euler–Maruyama-like parameterization.

$$
\Psi_{s,t}(x,W)=x+(t-s)f_{s,t}(x,W)+g_{s,t}(W)(W_t-W_s)
$$

The learned stochastic map adds interval length times its interval drift and the Brownian increment multiplied by its interval diffusion coefficient to x.

[Equation reading 2](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u193_r1)

On the diagonal, the coefficients recover the SDE.

$$
f_{t,t}(x)=f(t,x),\qquad g_{t,t}=g(t)
$$

On the time diagonal, the map’s drift and diffusion coefficients equal the instantaneous coefficients of the target SDE.

[Equation reading 3](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u193_r2)

In practice, pass retained path coefficients to the networks.

$$
W\longrightarrow(I^{(0)},\ldots,I^{(N-1)})
$$

In the implementation, the noise path is represented by its retained coefficients from index zero through N minus one.

## Pathwise composition uses the same noise realization

[Equation reading 1](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u194_r0)

Predict the whole interval directly.

$$
y=\Psi_{s,t}(x,W_{[s,t]})
$$

The direct destination y is the stochastic map over the full interval using that interval’s Brownian path.

[Equation reading 2](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u194_r1)

Predict the first part, then continue with the remaining noise.

$$
\widetilde y=\Psi_{u,t}(\Psi_{s,u}(x,W_{[s,u]}),W_{[u,t]})
$$

The split destination first uses the left Brownian segment, then continues from that result using the right segment.

[Equation reading 3](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u194_r2)

The exact solution satisfies the stochastic composition identity.

$$
\Psi_{s,t}(x,W)=\Psi_{u,t}(\Psi_{s,u}(x,W_L),W_R)
$$

The exact stochastic map equals this composition when the two segments belong to the same whole-interval Brownian realization.

## These conditions identify the strong solution

[Equation reading 1](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u195_r0)

Compose over a partition and separate the diagonal terms.

$$
\widehat X_t-x=\sum_i f(t_i,\widehat X_{t_i})h_i+\sum_i g(t_i)\Delta W_i+R_f+R_g
$$

The composed endpoint displacement is the sum of diagonal drift times interval lengths and diagonal diffusion times Brownian increments, plus drift and diffusion remainders.

[Equation reading 2](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u195_r1)

The time-Lipschitz bounds force both remainders to vanish.

$$
|R_f|\leq L_f(t-s)|\pi|,\qquad |R_g|\leq L_g(t-s)\max_i|\Delta W_i|
$$

The drift remainder is bounded by a constant times total duration and maximum step size; the diffusion remainder is bounded by a constant times duration and the largest Brownian increment.

[Equation reading 3](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u195_r2)

The remaining sums converge to the defining SDE integrals.

$$
\widehat X_t=x+\int_s^t f(u,\widehat X_u)\,du+\int_s^t g(u)\,dW_u
$$

As the partition is refined, the endpoint equals x plus the drift integral and the Itô noise integral, recovering the defining SDE.

## Training anchors the SDE and enforces composition

[Equation reading 1](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u196_r0)

Match the instantaneous SDE coefficients on the diagonal.

$$
\mathcal L_{f,g}=\mathbb E[\|\widehat f_{t,t}(X_t)-f(t,X_t)\|^2+\|\widehat g_{t,t}-g(t)\|^2]
$$

The diagonal loss averages the sum of squared errors in the learned instantaneous drift and diffusion coefficients against their target SDE values.

[Equation reading 2](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u196_r1)

Use coupled noise representations in the off-diagonal loss.

$$
\mathcal L_D=\mathbb E\|\widehat\Psi_{s,t}(x,W)-\widehat\Psi_{u,t}(\widehat\Psi_{s,u}(x,W_L),W_R)\|^2
$$

The composition loss averages the squared difference between the direct stochastic map and its two-interval composition, using coupled representations of the same noise.

[Equation reading 3](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u196_r2)

Combine both requirements.

$$
\mathcal L_{\rm SSFM}=\mathcal L_{f,g}+\mathcal L_D
$$

The strong stochastic flow-map loss is the sum of the diagonal coefficient loss and the coupled composition loss.

## Let’s implement the coupled noise inputs first

[Equation reading 1](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u197_r0)

Construct the coarse noise from its two fine intervals.

$$
I^{(0)}=I_L^{(0)}+I_R^{(0)},\qquad I^{(1)}=\tfrac12(I_L^{(1)}+I_R^{(1)}-I_L^{(0)}+I_R^{(0)})
$$

The coarse zeroth coefficient adds the fine increments; the first-order coefficient adds the two fine first-order terms, subtracts the left increment, adds the right increment, and halves that total.

## Local dynamics and interval consistency remain central

[Equation reading 1](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u208_r0)

The diagonal fixes the intended local motion.

$$
v_{t,t}(x)=b_t(x),\qquad F_{t,t}(x)=x
$$

The diagonal average velocity equals the intended instantaneous velocity, and the zero-length flow map returns its input unchanged.

[Equation reading 2](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u208_r1)

Composition connects the finite intervals.

$$
F_{s,t}=F_{u,t}\circ F_{s,u}
$$

The full finite map equals the first interval’s map followed by the second interval’s map.

[Equation reading 3](https://docs.google.com/presentation/d/1wfLAazqYveMoyvdy0y5ILeionp7eUjcPgxu-5QFlWwI/edit#slide=id.fm6_u208_r2)

For a strong stochastic map, retain the same driving noise.

$$
\Psi_{s,t}(x,W)=\Psi_{u,t}(\Psi_{s,u}(x,W_L),W_R)
$$

The stochastic composition uses the left and right pieces of the same driving path to reproduce the full-interval map.
