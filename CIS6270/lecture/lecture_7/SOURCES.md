# Primary sources and original code

The versions below match the downloaded PDFs used for the lecture. Original figure locations are recorded in `source_manifest.json`. Code below was consulted or linked as the authoritative full implementation; course code is an independent synthetic teaching implementation.

| Reference | Paper | Original code |

|---|---|---|

| Computational Optimal Transport — Peyré and Cuturi | [Paper](https://arxiv.org/abs/1803.00567v4) | — |

| A computational fluid mechanics solution to the Monge–Kantorovich mass transfer problem — Benamou and Brenier | [Paper](https://doi.org/10.1007/s002110050002) | — |

| Sinkhorn Distances — Cuturi | [Paper](https://arxiv.org/abs/1306.0895v1) | — |

| Foundations of Schrödinger Bridges for Generative Modeling — Tang | [Guide](https://arxiv.org/abs/2603.18992v1) | — |

| Diffusion Schrödinger Bridge with Applications to Score-Based Generative Modeling | [Paper](https://arxiv.org/abs/2106.01357v5) | [Repository](https://github.com/JTT94/diffusion_schrodinger_bridge) |

| Diffusion Schrödinger Bridge Matching | [Paper](https://arxiv.org/abs/2303.16852v3) | [Repository](https://github.com/yuyang-shi/dsbm-pytorch) |

| Simulation-Free Schrödinger Bridges via Score and Flow Matching | [Paper](https://arxiv.org/abs/2307.03672v3) | [Repository](https://github.com/atong01/conditional-flow-matching) |

| Discrete Diffusion Schrödinger Bridge Matching for Graph Transformation | [Paper](https://arxiv.org/abs/2410.01500v2) | [Repository](https://github.com/junhkim1226/DDSBM) |

| Categorical Schrödinger Bridge Matching | [Paper](https://arxiv.org/abs/2502.01416v2) | [Repository](https://github.com/gregkseno/csbm) |

| TR2-D2: Tree Search Guided Trajectory-Aware Fine-Tuning for Discrete Diffusion | [Paper](https://arxiv.org/abs/2509.25171v1) | [Repository](https://huggingface.co/ChatterjeeLab/TR2-D2) |

| Branched Schrödinger Bridge Matching | [Paper](https://arxiv.org/abs/2506.09007v2) | [Repository](https://huggingface.co/ChatterjeeLab/BranchSBM) |

| Entangled Schrödinger Bridge Matching | [Paper](https://arxiv.org/abs/2511.07406v1) | [Repository](https://huggingface.co/ChatterjeeLab/EntangledSBM) |

| A survey of the Schrödinger problem and some of its connections with optimal transport — Léonard | [Paper](https://arxiv.org/abs/1308.0215v1) | — |



Historical context: Schrödinger (1931), Fortet (1940), Beurling (1960), and Sinkhorn (1964) are discussed and referenced in the Léonard, Peyré–Cuturi, and Tang readings. The slides derive the finite marginal constraints, dual certificate, kernel scaling, path-KL chain rule, Doob transforms, control cost, and matching projections before using them in the recent methods.



Reported benchmark comparisons retain their metric definitions and computational differences. In particular, CSBM’s time-step and representation choices differ from its baselines; BranchSBM’s intermediate-time results are mixed; EntangledSBM’s control variants differ in conditioning and loss.