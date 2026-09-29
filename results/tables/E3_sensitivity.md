**GRASP** (sorted by success rate, then PAR-10; top 8 of 8)

| configuration | success | mean gap (%) | PAR-10 (s) | median TTT (ms) |
| :--- | ---: | ---: | ---: | ---: |
| alpha=0.3 | 0.880 | 0.0015 | 1.380 | 80.8 |
| alpha=1.0 | 0.850 | 0.0022 | 1.674 | 73.8 |
| alpha=0.2 | 0.830 | 0.0026 | 1.888 | 80.9 |
| alpha=0.5 | 0.825 | 0.0069 | 1.897 | 58.6 |
| alpha=0.1 | 0.780 | 0.0027 | 2.347 | 65.5 |
| alpha=0.75 | 0.775 | 0.0061 | 2.412 | 110.0 |
| alpha=0.05 | 0.705 | 0.0050 | 3.099 | 86.0 |
| alpha=0.0 | 0.695 | 0.0130 | 3.162 | 55.2 |

**ILS** (sorted by success rate, then PAR-10; top 12 of 27)

| configuration | success | mean gap (%) | PAR-10 (s) | median TTT (ms) |
| :--- | ---: | ---: | ---: | ---: |
| acceptance=better_equal,strength=12 | 0.955 | 1.3e-05 | 0.550 | 45.9 |
| acceptance=better,strength=8 | 0.940 | 0.0001 | 0.714 | 47.4 |
| acceptance=better_equal,strength=30 | 0.930 | 0.0005 | 0.851 | 54.7 |
| acceptance=better_equal,strength=8 | 0.925 | 0.0003 | 0.866 | 38.8 |
| acceptance=better_equal,strength=20 | 0.925 | 2.2e-05 | 0.893 | 59.2 |
| acceptance=better_equal,pivot=first,strength=5 | 0.925 | 0.0003 | 0.893 | 60.6 |
| acceptance=better,strength=20 | 0.920 | 0.0005 | 0.920 | 51.7 |
| acceptance=better,strength=12 | 0.920 | 0.0007 | 0.908 | 41.6 |
| acceptance=better,strength=30 | 0.920 | 0.0001 | 0.935 | 69.0 |
| acceptance=better_equal,strength=5 | 0.910 | 0.0008 | 1.016 | 41.4 |
| acceptance=better,strength=5 | 0.910 | 0.0004 | 1.033 | 47.5 |
| acceptance=always,strength=30 | 0.910 | 0.0013 | 1.079 | 54.4 |

**TABU** (sorted by success rate, then PAR-10; top 12 of 77)

| configuration | success | mean gap (%) | PAR-10 (s) | median TTT (ms) |
| :--- | ---: | ---: | ---: | ---: |
| attribute=pair,diversification=300.0,restart_after=25,restart_strength=10,tenure=7 | 0.970 | 0.0007 | 0.396 | 33.6 |
| attribute=pair,diversification=10.0,restart_after=25,restart_strength=10,tenure=7 | 0.940 | 0.0044 | 0.666 | 25.4 |
| attribute=element,diversification=30.0,restart_after=100,restart_strength=10,tenure=20 | 0.940 | 0.0089 | 0.667 | 15.8 |
| attribute=pair,diversification=10.0,restart_after=50,restart_strength=10,tenure=7 | 0.940 | 0.0074 | 0.683 | 16.1 |
| attribute=pair,diversification=30.0,restart_after=25,restart_strength=10,tenure=7 | 0.935 | 0.0015 | 0.733 | 24.5 |
| attribute=pair,diversification=30.0,restart_after=100,restart_strength=40,tenure=7 | 0.935 | 0.0007 | 0.747 | 19.6 |
| attribute=pair,diversification=100.0,restart_after=25,restart_strength=10,tenure=7 | 0.925 | 0.0015 | 0.834 | 27.2 |
| attribute=pair,diversification=0.0,restart_after=25,restart_strength=10,tenure=7 | 0.920 | 2.4e-05 | 0.886 | 43.4 |
| attribute=element,diversification=30.0,restart_after=100,restart_strength=10,tenure=15 | 0.915 | 0.0125 | 0.912 | 21.8 |
| attribute=pair,diversification=3.0,restart_after=25,restart_strength=10,tenure=7 | 0.915 | 0.0015 | 0.942 | 26.4 |
| attribute=pair,diversification=10.0,restart_after=100,restart_strength=10,tenure=7 | 0.910 | 0.0107 | 0.975 | 14.9 |
| attribute=pair,diversification=10.0,restart_after=200,restart_strength=10,tenure=7 | 0.910 | 0.0121 | 1.012 | 14.9 |
