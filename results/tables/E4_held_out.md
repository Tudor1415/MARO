| Algorithm | success rate [95% CI over instances] | instances solved in every run | instances never solved | mean gap (%) | PAR-10 (s) | median ERT (ms) | median TTT of successes (ms) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RRLS (control) | 0.999 [0.997, 1.000] | 34/35 | 0 | 2.3e-06 | 0.320 | 74.4 | 49.3 |
| GRASP | 0.999 [0.997, 1.000] | 34/35 | 0 | 1.0e-06 | 0.422 | 82.0 | 54.5 |
| ILS | 1.000 [1.000, 1.000] | 35/35 | 0 | 0.0000 | 0.081 | 53.7 | 38.7 |
| ILS, random start | 1.000 [1.000, 1.000] | 35/35 | 0 | 0.0000 | 0.084 | 64.7 | 42.9 |
| Tabu Search | 0.979 [0.942, 1.000] | 32/35 | 0 | 0.0030 | 2.450 | 82.2 | 51.0 |
| TS, random start | 0.971 [0.931, 1.000] | 32/35 | 0 | 0.0032 | 3.263 | 124.9 | 69.4 |
| TS, restarts without frequency memory | 0.977 [0.935, 1.000] | 32/35 | 0 | 0.0034 | 2.564 | 74.8 | 41.8 |
| TS, short-term memory only | 0.487 [0.332, 0.654] | 14/35 | 17 | 0.1171 | 51.449 | 33071.9 | 9.7 |
