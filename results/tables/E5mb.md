| Algorithm | success rate [95% CI over instances] | instances solved in every run | instances never solved | mean gap (%) | PAR-10 (s) | median ERT (ms) | median TTT of successes (ms) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RRLS (control) | 0.513 [0.357, 0.667] | 10/30 | 9 | 0.0010 | 99.475 | 23338.2 | 2159.4 |
| GRASP | 0.563 [0.413, 0.710] | 11/30 | 5 | 0.0009 | 89.754 | 15368.7 | 2159.2 |
| ILS | 0.980 [0.940, 1.000] | 29/30 | 0 | 2.0e-05 | 5.415 | 859.6 | 761.6 |
| ILS, random start | 0.980 [0.940, 1.000] | 29/30 | 0 | 2.0e-05 | 5.414 | 1004.5 | 756.7 |
| ILS, kick scaled with n | 0.923 [0.853, 0.977] | 22/30 | 0 | 2.5e-05 | 18.576 | 2366.4 | 1309.5 |
| Tabu Search | 0.760 [0.633, 0.870] | 15/30 | 3 | 0.0003 | 51.310 | 6507.9 | 2472.9 |
| TS, random start | 0.763 [0.673, 0.850] | 12/30 | 0 | 0.0004 | 51.139 | 7660.7 | 2809.4 |
| TS, restarts without frequency memory | 0.743 [0.613, 0.860] | 16/30 | 2 | 0.0004 | 54.750 | 6044.2 | 2718.1 |
| TS, short-term memory only | 0.063 [0.000, 0.163] | 1/30 | 28 | 0.0127 | 187.493 | ∞ | 335.6 |
| TS, tenure and kick scaled with n | 0.167 [0.060, 0.293] | 3/30 | 21 | 0.0065 | 167.087 | ∞ | 881.9 |
