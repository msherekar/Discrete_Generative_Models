# From the lecture equations to the running code

The [slide code map](SLIDE_CODE_MAP.md) supplies the corresponding slide links.
Unless stated otherwise, time zero denotes Gaussian noise and time one denotes
data. All expectations below are population quantities; the training code uses
finite minibatches and learned approximations.

## We first learn the motion at one time

For a noise sample $X_0$ and a data sample $X_1$, construct

$$I_t=(1-t)X_0+tX_1,\qquad \dot I_t=X_1-X_0.$$

Here $I_t$ is the sampled interpolant and $t$ is its progress from noise to data.
Regressing the displacement against the noisy state learns

$$b_t(x)=\mathbb E[X_1-X_0\mid I_t=x].$$

`interpolate` constructs the pair; `diagonal_loss` performs the regression.
The conditional-mean field transports the same one-time marginals as the
interpolant. Individual interpolant lines generally differ from its ODE paths.

## Now predict the destination over an interval

$$F_{s,t}(x)=x+(t-s)v_{s,t}(x).$$

$s$ is the start, $t$ is the arrival time, and $v_{s,t}$ is the average velocity
along that ODE trajectory. `finite_map` uses this residual parameterization,
which gives $F_{s,s}(x)=x$ exactly. The diagonal average $v_{s,s}$ must still be
trained to equal $b_s$.

For $\dot x=x$, the exact map is $e^{t-s}x$. From $x=1$ over a unit interval,
the average velocity is $e-1\approx1.71828$. The residual update reaches
$2.71828$, while one Euler step using the initial velocity reaches only $2$.
`numerical_examples.py` trains this scalar map and reports its actual error.

## The ODE supplies three finite-interval identities

$$\partial_t F_{s,t}(x)=b_t(F_{s,t}(x)),$$

$$\partial_sF_{s,t}(x)+J_xF_{s,t}(x)b_s(x)=0,$$

$$F_{s,t}(x)=F_{u,t}(F_{s,u}(x)).$$

The first changes the destination time. The second advances the start along the
same path, leaving the destination fixed. The third splits the interval at an
intermediate time $u$. `lagrangian_residual` and `eulerian_residual` use JVPs;
`semigroup_loss` evaluates the direct and split paths.

The identity map satisfies composition for every interval but has zero velocity.
This is why composition needs the diagonal anchor. A detached target also
changes the optimization: only the student branch receives its gradient.

## Shortcut and MeanFlow choose particular targets

Shortcut learns a velocity for a longer interval from two shorter intervals:

$$v_{s,s+2d}(x)\approx\operatorname{sg}\left[
\tfrac12 v_{s,s+d}(x)+\tfrac12 v_{s+d,s+2d}(x+d\,v_{s,s+d}(x))\right].$$

$d$ is the half-step length. The second prediction receives the moved state.
`shortcut_loss` samples dyadic lengths and builds that detached target.

MeanFlow uses the opposite clock. Its $z_t=(1-t)X_{\rm data}+tX_{\rm noise}$
has data at zero and noise at one. For $r\leq t$,

$$z_r=z_t-(t-r)u(z_t,r,t),$$

$$u=v-(t-r)(\partial_tu+J_zu\,v).$$

$v=X_{\rm noise}-X_{\rm data}$ is the conditional training velocity.
`meanflow_loss` evaluates the JVP in direction $(v,0,1)$ for inputs $(z,r,t)$,
then detaches the complete right-hand target. The sampler retains the minus sign.

## A latent representation changes the coordinates

For encoder $E$, decoder $D$, and data distribution $p$, fit a map to $E_\#p$.
Generation is $D(F_{0,1}(Z_0))$. The code first trains an autoencoder and freezes
it; a stored mean and standard deviation normalize its latent coordinates.
`train_latent` saves the encoder, decoder, normalization, and map together.

For an $L_D$-Lipschitz decoder, the distributional error is bounded by latent
transport error multiplied by $L_D$, plus reconstruction error. The code
reports reconstruction MSE separately; it does not assert a measured global
Lipschitz constant or equate that MSE with a Wasserstein distance.

## Categorical maps constrain their predictions

For one-hot endpoints, the posterior mean denoiser $D_s(x)$ lies on the
probability simplex. The noisy state $x$ may lie outside it. The finite map is

$$F_{s,t}(x)=\frac{1-t}{1-s}x+\frac{t-s}{1-s}\psi_{s,t}(x),$$

where $\psi_{s,t}$ is the two-time softmax prediction. `categorical_map` implements
this expression. At $s=.25,t=.75$, state $(-.2,.6,1.1)$ and prediction
$(.1,.7,.2)$ produce $(0,2/3,.5)$; the intermediate state is not normalized.

Composition induces

$$q=\gamma\psi_{s,u}(x)+(1-\gamma)\psi_{u,t}(F_{s,u}(x)),\qquad
\gamma=\frac{(1-t)(u-s)}{(1-u)(t-s)}.$$

At $(s,u,t)=(0,.5,.75)$, the weight is $1/3$. Combining predictions $(.8,.2)$
and $(.2,.8)$ gives $(.4,.6)$. `composition_target` builds this probability-valued
target; a detached KL objective has logit gradient $p_{\rm student}-q$.

The decoding clock is $\tau(t)=1-\frac{V}{V-1}P_e(t)$, with vocabulary size $V$
and single-token corruption error $P_e$. `DecodingClock` numerically inverts it.
Endpoint enforcement prevents floating-point accuracy saturation from stopping
the sampler before physical time one.

## Categorical and discrete consistency constrain different residuals

Categorical ECLD combines endpoint agreement with the time derivative of the
finite denoiser. With $\eta=(t-s)/(1-s)$, the scaled Lagrangian residual is

$$r=\psi_{s,t}-\psi_{t,t}(F_{s,t})+(1-t)\eta\partial_t\psi_{s,t}.$$

The endpoint term can vanish while the derivative term remains nonzero. The
implementation includes both and follows the released code's finite weights.

For Discrete Flow Maps, write $\psi=\operatorname{softmax}(z)$ and center the
arrival-time logit derivative:

$$\delta_k=\partial_tz_k-\sum_j\psi_j\partial_tz_j,\qquad
c=\frac{(t-s)(1-t)}{1-s}.$$

The Lagrangian teacher is

$$T_{\rm LSD}=\operatorname{softmax}\left[z_{t,t}(F_{s,t})-\log(1+c\delta)\right].$$

For the Eulerian teacher, replace the derivative with
$D_sz=\partial_sz+J_xz\,b_s$, center it in the same way, and use

$$T_{\rm ESD}=\operatorname{softmax}\left[z_{s,s}-
\log\left(1-\frac{(1-s)(t-s)}{1-t}\delta\right)\right].$$

`corrected_logit_teacher` computes both versions. Log arguments must be positive;
the code records its stabilization frequency so that violations remain visible.

## Posterior maps need a second noise draw

Meta training constructs

$$I_t=(1-t)X_0+tX_1,\qquad \bar I_s=(1-s)\bar X_0+sX_1,$$

with independent $X_0,\bar X_0$. The outer observation $(t,I_t)$ stays fixed
while the inner map advances from $s$ to its target time. At that fixed context,
the shared $X_1$ has exactly the desired posterior distribution.

Diamond distillation obtains the corresponding conditional velocity through
GLASS. For two independent observations, the likelihood precisions add:

$$\Lambda=\frac{s^2}{(1-s)^2}+\frac{t^2}{(1-t)^2},\qquad
S=\Lambda^{-1}\left[\frac{s\bar x}{(1-s)^2}+\frac{tx}{(1-t)^2}\right].$$

The equivalent linear-interpolant time is
$t^*=\sqrt\Lambda/(1+\sqrt\Lambda)$. Evaluating the original denoiser at
$(t^*,t^*S)$ supplies the posterior mean conditioned on both observations.
`glass_denoiser` is checked against an independent Gaussian conditioning formula.

## Posterior samples make reward averages computable

For reward $r$ and conditional samples $Z_k$,

$$\widehat V_t(x)=\log\left(\frac1K\sum_{k=1}^K e^{r(Z_k)}\right),\qquad
\widehat D_t^r(x)=\sum_k\operatorname{softmax}(r(Z))_k Z_k.$$

`posterior_value` differentiates through the learned conditional samples.
`guided_samples` uses $(\widehat D_t^r(x)-x)/(1-t)$ as the guided velocity.
The ratio is biased at finite $K$, even with an exact conditional sampler.

For Meta fine-tuning, let $d=b_{\rm student}-b_{\rm base}$, $w=e^{r(Z)}$, and
$a=g_t^2/2$. The detached surrogate is

$$\ell=\left\|d+(w-1)\operatorname{sg}(d)-a\operatorname{sg}(\nabla_xw)\right\|^2.$$

Its expected gradient in $d$ is $2\mathbb E[wd-a\nabla_xw]$. Squaring $wd-a\nabla w$
without that stop-gradient placement introduces an extra weight. The test suite
checks this distinction directly. For the linear path, $a=(1-t)/t$; the optional
training demonstration therefore excludes the singular initial time.

## Expansion gives each new token its own clock

A token born at $b_i$ has local time

$$\tau_i(t)=\max\left(0,\frac{t-b_i}{1-b_i}\right),\qquad
\frac{d\tau_i}{dt}=\frac1{1-b_i}\quad(t>b_i).$$

`local_clock` and `local_map` preserve this information during transport.
`insert_tokens` keeps token order, inserted noise, and birth metadata aligned.
The linear insertion CDF gives conditional birth fraction
$\rho_{s,t}=(t-s)/(1-s)$. Predicted remaining gap means are converted into
interval means using this factor.

The count divergence is

$$\phi(a,b)=b-a+a\log(a/b),\qquad b>0,$$

where $a$ is a realized count and $b$ is the predicted mean. Its population
minimum is $b=\mathbb E[a]$. The zero-count branch is implemented with a finite
autodiff expression. Sampling uses bounded binomial proposals and reports
counts removed by global budget capping.

## Strong maps must retain the same driving path

For an interval of length $h$, the first two Brownian integrals satisfy

$$I_0\sim\mathcal N(0,h),\qquad I_1\sim\mathcal N(0,h/3),\qquad I_0\perp I_1.$$

For left/right subintervals of lengths $h_L,h_R$, Chen composition is

$$I_0=I_0^L+I_0^R,$$

$$I_1=\frac{h_LI_1^L+h_RI_1^R-h_RI_0^L+h_LI_0^R}{h_L+h_R}.$$

With equal halves, $(I_0^L,I_1^L)=(.2,.04)$ and
$(I_0^R,I_1^R)=(-.1,-.02)$ combine into $(.1,-.14)$.
`chen_two` and `aggregate_tree` ensure the direct and split maps refer to that
same history. Independent coarse noise would compare different sample paths.

The SSFM objective combines a short stochastic matching step with same-noise
composition. The reported strong error compares the learned result to a fine
Euler-Maruyama trajectory using the same increments. Terminal variance checks
the marginal separately; matching variance cannot establish pathwise accuracy.
