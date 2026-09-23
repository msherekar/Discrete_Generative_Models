# Code walkthrough map

Function names are stable even if the combined deck is later split or reordered. Each method has a full training and sampling command in the README.

| Slide unit | Code | Location |
| --- | --- | --- |
| [Code for classic DFM training pairs](https://docs.google.com/presentation/d/1qcPdF4SYTq4VdTN_5w9GjviLRPr2HFtbuYMpbfLrc_M/edit#slide=id.cis4r2_u155_s9) | `gat_loss` | `lecture_core.py` |
| [Code for convert a posterior into Gat jump rates](https://docs.google.com/presentation/d/1qcPdF4SYTq4VdTN_5w9GjviLRPr2HFtbuYMpbfLrc_M/edit#slide=id.cis4r2_u156_s4) | `gat_rates` | `lecture_core.py` |
| [Code for one valid categorical Euler step](https://docs.google.com/presentation/d/1qcPdF4SYTq4VdTN_5w9GjviLRPr2HFtbuYMpbfLrc_M/edit#slide=id.cis4r2_u157_s7) | `rate_step` | `lecture_core.py` |
| [Code for classic DFM generation](https://docs.google.com/presentation/d/1qcPdF4SYTq4VdTN_5w9GjviLRPr2HFtbuYMpbfLrc_M/edit#slide=id.cis4r2_u158_s12) | `gat_sample` | `lecture_core.py` |
| [Code for Dirichlet posterior training](https://docs.google.com/presentation/d/1qcPdF4SYTq4VdTN_5w9GjviLRPr2HFtbuYMpbfLrc_M/edit#slide=id.cis4r2_u159_s7) | `dirichlet_loss` | `lecture_core.py` |
| [Code for the Dirichlet velocity module](https://docs.google.com/presentation/d/1qcPdF4SYTq4VdTN_5w9GjviLRPr2HFtbuYMpbfLrc_M/edit#slide=id.cis4r2_u160_s9) | `dirichlet_velocity` | `lecture_core.py` |
| [Code for a Fisher conditional path](https://docs.google.com/presentation/d/1qcPdF4SYTq4VdTN_5w9GjviLRPr2HFtbuYMpbfLrc_M/edit#slide=id.cis4r2_u161_s9) | `fisher_path` | `lecture_core.py` |
| [Code for Fisher velocity regression](https://docs.google.com/presentation/d/1qcPdF4SYTq4VdTN_5w9GjviLRPr2HFtbuYMpbfLrc_M/edit#slide=id.cis4r2_u162_s9) | `fisher_loss` | `lecture_core.py` |
| [Code for the Gumbel conditional path](https://docs.google.com/presentation/d/1qcPdF4SYTq4VdTN_5w9GjviLRPr2HFtbuYMpbfLrc_M/edit#slide=id.cis4r2_u163_s9) | `gumbel_path` | `lecture_core.py` |
| [Code for Gumbel velocity regression](https://docs.google.com/presentation/d/1qcPdF4SYTq4VdTN_5w9GjviLRPr2HFtbuYMpbfLrc_M/edit#slide=id.cis4r2_u164_s7) | `gumbel_loss` | `lecture_core.py` |
| [Code for objective scores modify the base rates](https://docs.google.com/presentation/d/1qcPdF4SYTq4VdTN_5w9GjviLRPr2HFtbuYMpbfLrc_M/edit#slide=id.cis4r2_u165_s12) | `mog_rates` | `lecture_core.py` |
| [Code for continuous rectified-flow training](https://docs.google.com/presentation/d/1qcPdF4SYTq4VdTN_5w9GjviLRPr2HFtbuYMpbfLrc_M/edit#slide=id.cis4r2_u166_s8) | `rectified_loss` | `lecture_core.py` |
| [Code for ReDi replaces the endpoint pairing](https://docs.google.com/presentation/d/1qcPdF4SYTq4VdTN_5w9GjviLRPr2HFtbuYMpbfLrc_M/edit#slide=id.cis4r2_u167_s6) | `redi_pairs` | `lecture_core.py` |
| [Code for the preference scalarization](https://docs.google.com/presentation/d/1qcPdF4SYTq4VdTN_5w9GjviLRPr2HFtbuYMpbfLrc_M/edit#slide=id.cis4r2_u168_s3) | `maxmin` | `lecture_core.py` |
| [Code for enumerate every single-base DNA edit](https://docs.google.com/presentation/d/1qcPdF4SYTq4VdTN_5w9GjviLRPr2HFtbuYMpbfLrc_M/edit#slide=id.cis4r2_u169_s3) | `dna_neighbors` | `lecture_core.py` |
| [Code for a normalized locally balanced proposal](https://docs.google.com/presentation/d/1qcPdF4SYTq4VdTN_5w9GjviLRPr2HFtbuYMpbfLrc_M/edit#slide=id.cis4r2_u170_s6) | `proposal` | `lecture_core.py` |
| [Code for correct the refinement proposal](https://docs.google.com/presentation/d/1qcPdF4SYTq4VdTN_5w9GjviLRPr2HFtbuYMpbfLrc_M/edit#slide=id.cis4r2_u171_s12) | `mh_refine` | `lecture_core.py` |
| [Code for run a complete Gat DFM experiment](https://docs.google.com/presentation/d/1qcPdF4SYTq4VdTN_5w9GjviLRPr2HFtbuYMpbfLrc_M/edit#slide=id.cis4r2_u181_s5) | `complete experiment` | `lecture_core.py` and `run.py` |
