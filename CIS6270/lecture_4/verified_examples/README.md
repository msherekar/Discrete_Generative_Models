# Verified CPU examples

Actual seed-7 runs: 160 optimizer steps, length 4, batch size 16, 40 sampling steps, and 8 requested samples. ReDi/AReUReDi additionally use 160 teacher updates and 128 recoupled pairs. PepTune returns its archive, whose size can differ from the requested sample count.

| Method | Mean first 20 losses | Mean last 20 losses | Valid DNA |
| --- | ---: | ---: | --- |
| mdlm | 4.8388 | 2.6139 | True |
| udlm | 5.1313 | 2.2786 | True |
| block | 5.3755 | 3.2929 | True |
| cfg | 4.3721 | 2.3383 | True |
| classifier-exact | 4.8388 | 2.6139 | True |
| classifier-gradient | 4.8388 | 2.6139 | True |
| peptune | 4.8388 | 2.6139 | True |

These are execution and learning checks on toy data, not paper benchmark results. Different method losses have different meanings and scales. The sample counts are too small for statistical quality claims.
