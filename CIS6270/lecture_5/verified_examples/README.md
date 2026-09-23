# Verified CPU examples

Actual seed-7 runs: 160 optimizer steps, length 4, batch size 16, 40 sampling steps, and 8 requested samples. ReDi/AReUReDi additionally use 160 teacher updates and 128 recoupled pairs. PepTune returns its archive, whose size can differ from the requested sample count.

| Method | Mean first 20 losses | Mean last 20 losses | Valid DNA |
| --- | ---: | ---: | --- |
| gat | 4.2671 | 2.6596 | True |
| dirichlet | 5.1298 | 1.7297 | True |
| fisher | 3.8173 | 2.2441 | True |
| gumbel | 0.8744 | 0.3556 | True |
| rectified | 0.8557 | 0.4507 | True |
| redi | 3.9590 | 2.1978 | True |
| mog-dfm | 4.2671 | 2.6596 | True |
| areuredi | 3.9590 | 2.1978 | True |

These are execution and learning checks on toy data, not paper benchmark results. Different method losses have different meanings and scales. The sample counts are too small for statistical quality claims.
