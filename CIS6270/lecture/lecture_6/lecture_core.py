"""Stable imports connecting the lecture's code walkthroughs to full methods.

Implementation lives in the named modules so training and sampling use the
same functions demonstrated on the slides. See SLIDE_CODE_MAP.md.
"""
from common import finite_map
from continuous import (diagonal_loss,lagrangian_residual,eulerian_residual,
                        semigroup_loss,meanflow_loss,shortcut_loss)
from categorical import (categorical_map,composition_target,corrected_logit_teacher,
                         probability_kl,DecodingClock)
from posterior import (glass_denoiser,posterior_samples,posterior_value,
                       fine_tune_surrogate,weighted_diamond_samples)
from expanding import (local_clock,local_map,count_divergence,bounded_counts,insert_tokens)
from stochastic import chen_two,sample_coefficients
