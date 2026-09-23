# Code walkthrough map

Function names are stable even if the combined deck is later split or reordered. Each method has a full training and sampling command in the README.

| Slide unit | Code | Location |
| --- | --- | --- |
| [Code for the four-letter vocabulary](https://docs.google.com/presentation/d/1qcPdF4SYTq4VdTN_5w9GjviLRPr2HFtbuYMpbfLrc_M/edit#slide=id.cis4r2_u141_s5) | `complete experiment` | `lecture_core.py` |
| [Code for a small shared DNA network](https://docs.google.com/presentation/d/1qcPdF4SYTq4VdTN_5w9GjviLRPr2HFtbuYMpbfLrc_M/edit#slide=id.cis4r2_u142_s6) | `DNA` | `lecture_core.py` |
| [Code for the network forward pass](https://docs.google.com/presentation/d/1qcPdF4SYTq4VdTN_5w9GjviLRPr2HFtbuYMpbfLrc_M/edit#slide=id.cis4r2_u143_s5) | `forward` | `lecture_core.py` |
| [Code for one cross-entropy per DNA position](https://docs.google.com/presentation/d/1qcPdF4SYTq4VdTN_5w9GjviLRPr2HFtbuYMpbfLrc_M/edit#slide=id.cis4r2_u144_s3) | `token_ce` | `lecture_core.py` |
| [Code for MDLM training loss](https://docs.google.com/presentation/d/1qcPdF4SYTq4VdTN_5w9GjviLRPr2HFtbuYMpbfLrc_M/edit#slide=id.cis4r2_u145_s10) | `mdlm_loss` | `lecture_core.py` |
| [Code for one reusable optimization step](https://docs.google.com/presentation/d/1qcPdF4SYTq4VdTN_5w9GjviLRPr2HFtbuYMpbfLrc_M/edit#slide=id.cis4r2_u146_s8) | `train_step` | `lecture_core.py` |
| [Code for sample each categorical DNA vector](https://docs.google.com/presentation/d/1qcPdF4SYTq4VdTN_5w9GjviLRPr2HFtbuYMpbfLrc_M/edit#slide=id.cis4r2_u147_s5) | `draw` | `lecture_core.py` |
| [Code for MDLM generation](https://docs.google.com/presentation/d/1qcPdF4SYTq4VdTN_5w9GjviLRPr2HFtbuYMpbfLrc_M/edit#slide=id.cis4r2_u148_s13) | `mdlm_sample` | `lecture_core.py` |
| [Code for UDLM reverse rates](https://docs.google.com/presentation/d/1qcPdF4SYTq4VdTN_5w9GjviLRPr2HFtbuYMpbfLrc_M/edit#slide=id.cis4r2_u149_s7) | `udlm_rates` | `lecture_core.py` |
| [Code for the continuous UDLM loss](https://docs.google.com/presentation/d/1qcPdF4SYTq4VdTN_5w9GjviLRPr2HFtbuYMpbfLrc_M/edit#slide=id.cis4r2_u150_s10) | `udlm_loss` | `lecture_core.py` |
| [Code for train one conditional DNA block](https://docs.google.com/presentation/d/1qcPdF4SYTq4VdTN_5w9GjviLRPr2HFtbuYMpbfLrc_M/edit#slide=id.cis4r2_u151_s9) | `block_loss` | `lecture_core.py` |
| [Code for classifier-free categorical guidance](https://docs.google.com/presentation/d/1qcPdF4SYTq4VdTN_5w9GjviLRPr2HFtbuYMpbfLrc_M/edit#slide=id.cis4r2_u152_s5) | `geometric_cfg` | `lecture_core.py` |
| [Code for predictor guidance as a rate multiplier](https://docs.google.com/presentation/d/1qcPdF4SYTq4VdTN_5w9GjviLRPr2HFtbuYMpbfLrc_M/edit#slide=id.cis4r2_u153_s4) | `guide_rates` | `lecture_core.py` |
| [Code for preserve non-dominated DNA candidates](https://docs.google.com/presentation/d/1qcPdF4SYTq4VdTN_5w9GjviLRPr2HFtbuYMpbfLrc_M/edit#slide=id.cis4r2_u154_s6) | `pareto_filter` | `lecture_core.py` |
| [Code for run a complete MDLM experiment](https://docs.google.com/presentation/d/1qcPdF4SYTq4VdTN_5w9GjviLRPr2HFtbuYMpbfLrc_M/edit#slide=id.cis4r2_u180_s6) | `complete experiment` | `lecture_core.py` and `run.py` |
