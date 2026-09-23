# Lecture 3 · Continuous Generative Models

These examples accompany Lecture 3 of CIS 6270. We implement flow matching and DDPM diffusion using frozen ESM-2
residue embeddings, with shared property definitions and normalization to
compare guidance across the two sampling processes.

Each example includes data loading, latent and property normalization, a small
conditional model, training, classifier-free guidance, single-objective reward
steering, weighted multi-objective steering, and final sequence decoding with a
hard residue-count constraint.

## Examples

| Example | What the model learns | Sampling |
| --- | --- | --- |
| [Flow matching](esm2_flow_guidance.py) | A velocity field between noise and clean ESM-2 latents | Euler integration from noise to data |
| [Diffusion](esm2_diffusion_guidance.py) | The noise added to clean ESM-2 latents | A 1,000-step DDPM reverse chain |
| [Mathematical and implementation notes](GUIDANCE_NOTES.md) | Normalization, guidance conventions, decoding, and limitations | Companion reading for both scripts |

Each script contains the complete training and sampling implementation,
including the final sequence decoder, and can run independently.

## Quick start

Use **Python 3.11**, or another compatible Python version at least 3.10. A fresh
virtual environment is recommended. The dependencies are pinned to the versions
used to test these examples: PyTorch 2.9.1 and Transformers 4.57.6.

### 1. Download and install

```bash
git clone https://huggingface.co/ChatterjeeLab/CIS6270
cd CIS6270

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell, use `python -m venv .venv` and activate with
`.venv\Scripts\Activate.ps1`.

### 2. Train and sample

Run either example, or both:

```bash
python lecture_3/esm2_flow_guidance.py --epochs 200 --samples 8
python lecture_3/esm2_diffusion_guidance.py --epochs 200 --samples 8
```

Each command trains a generator and a property predictor from scratch, then
generates sequences using three guidance settings. The first run downloads the
public [ESM-2 8M checkpoint](https://huggingface.co/facebook/esm2_t6_8M_UR50D)
into `lecture_3/.esm2_cache/`, which subsequent runs reuse. Both the repository
and checkpoint are publicly accessible.

The scripts use CUDA when available and CPU otherwise. Apple Silicon currently
uses the CPU path. Runtime depends on hardware and the first-run download.

For a short end-to-end installation check:

```bash
python lecture_3/esm2_flow_guidance.py --epochs 2 --samples 2 --output outputs/flow_smoke
python lecture_3/esm2_diffusion_guidance.py --epochs 2 --samples 2 --output outputs/diffusion_smoke
```

The two-epoch runs exercise encoding, training, sampling, and decoding.

### 3. Find the generated sequences

By default, outputs are written beside the scripts:

```text
lecture_3/
├── esm2_flow_outputs/
│   ├── cfg.fasta
│   ├── single.fasta
│   ├── multi.fasta
│   └── results.pt
└── esm2_diffusion_outputs/
    ├── cfg.fasta
    ├── single.fasta
    ├── multi.fasta
    └── results.pt
```

The FASTA files contain decoded sequences. `results.pt` contains generated
standardized latents, generator and reward-predictor state dictionaries,
normalization statistics, the ESM checkpoint name, latent dimensions, and
constraint settings. The terminal also prints re-evaluated composition
properties of the decoded sequences.

Each run initializes new generator and reward-predictor parameters. Repeating
a command with the same output directory replaces its saved files; choose a
new `--output` directory to retain results from separate experiments.

## Implementation

1. **Encode sequences.** Use frozen ESM-2 to obtain one 320-dimensional vector
   per residue, retaining position information and removing BOS/EOS tokens.
2. **Normalize.** Standardize latent coordinates and each property using
   training-set statistics.
3. **Train.** Learn a conditional generative field and a time-conditioned
   predictor of the clean sequence's standardized properties.
4. **Guide generation.** Compare CFG, one reward, and a weighted reward sum.
5. **Decode once at the end.** Undo latent normalization, apply the frozen ESM-2
   language-model head, and enforce the selected residue-count constraint.
6. **Re-evaluate.** Calculate the composition properties on the actual decoded
   sequences so comparisons between guidance settings reflect the final
   amino-acid outputs.

## Sequence data and property labels

For the teaching examples, we use [64 synthetic sequences of length
24](esm2_example.csv). We calculate both property labels directly
from residue counts and assign the conditioning class according to the first
property:

| Column | Definition | Interpretation |
| --- | --- | --- |
| `sequence` | A canonical amino-acid sequence | Input to frozen ESM-2 |
| `r1` | `(count(K) + count(R) - count(D) - count(E)) / length` | A simple charge-count proxy |
| `r2` | `count(residues in DEHKNQRST) / length` | A defined polar/charged fraction |
| `c` | `1` if `r1 > 0`, otherwise `0` | The binary conditioning class |

### Use your own data

Supply a CSV with `sequence,c,r1,r2` columns:

```csv
sequence,c,r1,r2
QYWSDSWWESQMMSPWYPMSPLSV,0,-0.08333333,0.41666667
CKSEFQPPHLMGHDFFACEMRNFK,0,0.00000000,0.45833333
WPYGEHMLADNNVVKKRLQQWCFI,1,0.04166667,0.41666667
KVVGALPIESFYTAKMESIAVEVI,0,-0.04166667,0.33333333
```

```bash
python lecture_3/esm2_flow_guidance.py --data my_sequences.csv --output outputs/my_flow
python lecture_3/esm2_diffusion_guidance.py --data my_sequences.csv --output outputs/my_diffusion
```

- Include at least four sequences and both class labels, 0 and 1.
- Use one fixed sequence length, at most 128 residues, and the 20 canonical
  amino acids to match the fixed-length latent tensors in both models.
- Supply finite property values and orient each objective so **larger is better**.
  For a quantity to minimize, negate it before writing the CSV.
- If both `r1` and `r2` are omitted, `composition_proxies()` calculates the
  two example properties directly. Supplied reward columns override them.
- Experimental measurements or scores from a separate predictor can fill the
  reward columns. We fit a differentiable surrogate to these labels and
  evaluate its gradients with respect to the intermediate latent during sampling.
- The printed composition proxies always retain their defined count-based
  meaning, even when custom reward labels are supplied. Re-evaluate custom
  objectives with the corresponding assay or scorer after decoding.

The bundled demonstration uses all 64 sequences for training. For studies with
held-out evaluation, partition sequences by cluster before fitting normalization
statistics or model parameters.

## Property normalization and scalarization

Each objective is standardized independently:

```text
r̃ₘ(s) = [rₘ(s) − μₘ] / max(σₘ, 10⁻⁶)
```

Here μₘ and σₘ are the training-set mean and standard deviation of property m.
Standardization expresses each property in units of its training-set variation,
so the scalarization weights specify relative preferences on a common scale.
Reuse the saved training statistics for new examples to maintain that scale
throughout evaluation and generation.

The reward predictor estimates standardized clean-sequence properties from an
intermediate latent and its time. During sampling, we combine its predictions:

```text
Rλ(z,t) = λ₁ r̂₁(z,t) + λ₂ r̂₂(z,t)
λ₁ ≥ 0, λ₂ ≥ 0, λ₁ + λ₂ = 1
```

The nonnegative tradeoff weights λ are normalized to sum to one. Overall
steering strength η is a separate parameter.

## Guidance configurations

Both scripts run these settings in `main()`:

| Output | CFG strength `w` | Reward strength `eta` | Property weights `lambdas` |
| --- | ---: | ---: | --- |
| `cfg.fasta` | 2.0, class 1 | 0.0 | Unused |
| `single.fasta` | 0.0 | 1.0 | `(1.0, 0.0)` |
| `multi.fasta` | 0.0 | 1.0 | `(0.7, 0.3)` |

**Classifier-free guidance.** During training, we replace 20% of class labels
with null class index 2. At sampling time, we combine the conditional and
unconditional predictions as `F_uncond + w * (F_cond - F_uncond)`.
Thus `w=0` is unconditional, `w=1` is ordinary conditional sampling, and `w>1` amplifies the conditional
difference. F is a velocity for flow matching and predicted noise for DDPM.

**Reward steering.** Freeze both networks, enable gradients only for the
current latent, calculate the scalarized reward, and differentiate it with
respect to that latent. Set `(1, 0)` for the first objective alone or change
the weights to express a tradeoff. This correction uses the property predictor,
while CFG uses the conditional and unconditional generative predictions.

To change these settings, edit the three `sample(...)` calls in `main()`.
To combine CFG with reward steering, set both strengths:

```python
latent = sample(model, reward_model, n=8, c=1,
                w=2.0, eta=1.0, lambdas=(0.7, 0.3))
```

Each call resets the sampling seed to compare methods using the same random
draws for the same batch size.

### Guidance in the sampling dynamics

| | Flow matching | DDPM diffusion |
| --- | --- | --- |
| Clean-data endpoint | Z₁ | Z₀ |
| Training target | Velocity Z₁ − Z₀ | Injected Gaussian noise ε |
| Generation direction | t = 0 → 1 | k = 1000 → 1 |
| Reward correction | Add κ(t)∇Rλ to velocity | Subtract η√(1−ᾱₖ)∇Rλ from predicted noise |
| Numerical step | Euler ODE update | Reverse mean plus posterior Gaussian noise |

For the flow, we choose `κ(t)=4ηt(1−t)` to taper reward steering near the noise
and data endpoints. For DDPM, we convert the reward-gradient score correction
into a noise-prediction correction using `s=−ε/√(1−ᾱₖ)`. Both implementations
use heuristic gradient steering based on learned estimates of endpoint properties.

The DDPM predictor includes a Gaussian-reference skip plus a learned correction
so the small MLP can carry high-dimensional noise. The training target remains
noise, and the sampler uses the standard DDPM reverse equations.

## Constrained sequence decoding

The default decoder requires **at least 12 of the 24 output residues** to
belong to:

```text
𝒫 = {D, E, H, K, N, Q, R, S, T}
```

We begin with the highest-logit residue at each position and count the members
of 𝒫. When the count falls below the specified minimum, we replace the required
number of residues using the lowest-cost substitutions into 𝒫. This discrete
decoding procedure maximizes the sum of the fixed per-position logits subject
to the minimum-count constraint.

```bash
# Require at least 16 selected residues.
python lecture_3/esm2_flow_guidance.py --min-polar 16

# Remove the minimum-count constraint.
python lecture_3/esm2_diffusion_guidance.py --min-polar 0
```

Set `--min-polar` between zero and the sequence length to specify the minimum
number of selected polar or charged residues in each decoded sequence.

## Verification

Run the offline unit checks after installing dependencies:

```bash
python -m unittest discover -s tests -v
```

The five unit tests cover dataset annotations, scalarization weights, reward
input gradients, DDPM schedule indexing, and constrained decoding. For the
decoder test, we compare the selected sequences with exhaustive solutions on
small examples. All unit tests run locally with the installed dependencies.

We also checked both scripts end to end using the ESM-2 checkpoint, 200 training
epochs, and all three guidance modes. All 48 decoded sequences in that run
satisfied the specified minimum residue count.

## Troubleshooting

- **Download failure:** the first run needs internet access to the ESM-2
  checkpoint. Cached subsequent runs can use `HF_HUB_OFFLINE=1` once all files
  have been downloaded.
- **Dependency conflicts:** use a clean virtual environment and the pinned
  requirements. For a specific CUDA build, follow the official
  [PyTorch installation instructions](https://pytorch.org/get-started/locally/).
- **Input validation error:** check equal lengths, canonical residues, both
  class labels, finite rewards, and a minimum count no greater than the length.
- **Weak or repetitive samples:** inspect training convergence, evaluate
  reconstruction through the ESM-2 head, and assess property-predictor accuracy
  on held-out sequences before adjusting guidance strength. Compare guidance
  settings using properties recalculated after sequence decoding.

## References

- [ESM-2 checkpoint](https://huggingface.co/facebook/esm2_t6_8M_UR50D)
  and [Transformers ESM documentation](https://huggingface.co/docs/transformers/model_doc/esm).
- [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747).
- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239).
- [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).

## License

Repository code is distributed under the [MIT License](https://huggingface.co/ChatterjeeLab/CIS6270/blob/main/LICENSE), matching the
repository's license setting. ESM-2 weights are downloaded separately and remain
subject to their original distribution terms.
