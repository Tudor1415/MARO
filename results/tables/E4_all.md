| Algorithm | success rate [95% CI over instances] | instances solved in every run | instances never solved | mean gap (%) | PAR-10 (s) | median ERT (ms) | median TTT of successes (ms) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RRLS (control) | 0.993 [0.984, 1.000] | 46/49 | 0 | 2.6e-05 | 1.114 | 85.8 | 67.7 |
| GRASP | 0.999 [0.998, 1.000] | 48/49 | 0 | 7.5e-07 | 0.544 | 100.4 | 76.0 |
| ILS | 0.993 [0.978, 1.000] | 48/49 | 0 | 2.1e-06 | 0.879 | 70.3 | 46.3 |
| ILS, random start | 0.995 [0.986, 1.000] | 48/49 | 0 | 1.3e-06 | 0.614 | 78.1 | 55.7 |
| Tabu Search | 0.984 [0.956, 0.999] | 45/49 | 0 | 0.0021 | 2.002 | 82.1 | 50.0 |
| TS, random start | 0.963 [0.929, 0.991] | 43/49 | 0 | 0.0030 | 4.163 | 157.2 | 93.5 |
| TS, restarts without frequency memory | 0.967 [0.924, 0.997] | 44/49 | 0 | 0.0025 | 3.566 | 74.8 | 45.8 |
| TS, short-term memory only | 0.490 [0.352, 0.633] | 21/49 | 24 | 0.1011 | 51.042 | 33071.9 | 9.3 |
