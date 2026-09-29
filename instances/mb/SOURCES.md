# MB (Mitchell-Borchers) LOP instances: files and optima

Fetched 2026-09-29. There are 30 instances: r100[a-e]2, r150[a-e][01], r200[a-e][01] and r250[a-e]0.

## Files
- `raw_mitchell/<name>`: the original generator output, unchanged, from Mitchell's `problems.tar.gz`.
  - Source: https://web.archive.org/web/20100603100823id_/http://www.rpi.edu/~mitchj/generators/linord/problems.tar.gz
  - Format:
    - a text header with MATSIZE, Seed, Ratio, the proportion of zero coefficients, and the hidden "generating permutation" together with its value (NOT the optimum);
    - then the line "The entries of the matrix:";
    - then n*n integers, one per line, row-major. The diagonal is 0.
- `raw_mitchell/values_mitchell.txt`: Mitchell's list of optimal values.
  - Source: https://web.archive.org/web/20010729011601id_/http://www.rpi.edu:80/~mitchj/generators/linord/values
  - It also lists r50/r75 and other r100 instances that are not in the tarball.
- `optsicom_normalised/N-<name>`: the Optsicom normal-form versions, unchanged, from the Optsicom MB.zip.
  - Source: https://web.archive.org/web/20110902144500id_/http://www.optsicom.es/lolib/lop/MB.zip
  - Format: n, then n*n integers row-major.
- `optima.json`: {name: RAW optimum}. RAW means the maximisation objective, the sum of strictly-above-diagonal entries of the permuted `raw_mitchell` matrix.
- `optima_detail.json`: for each instance, n, the raw optimum, the Optsicom normalised value, the offset sum_{i<j} min(a_ij, a_ji) computed from the raw matrix, and Mitchell's listed value.
- `r200d1_witness_order_1based.json`: an ordering of r200d1 whose raw value is 888563.
- `optima_normalised.json`: {name: optimum of the `optsicom_normalised` matrix} = raw optimum − offset. These normalised files and values are the ones used by the study (the same convention as xLOLIB); the witness ordering reproduces 616617 on the normalised r200d1 matrix.

## Verification
- For all 30 instances, the Optsicom N- matrix equals the raw matrix normalised by a_ij - min(a_ij, a_ji), exactly.
- raw = Optsicom normalised value + offset matches Mitchell's listed optimum for 29/30 instances.
- r200d1 is the exception: Mitchell lists 888562, while Optsicom's 616617 + offset 271946 = 888563.
  - My ILS found an ordering with normalised value 616617 and raw value 888563, recomputed directly on the raw matrix; it is saved in the witness file.
  - So Mitchell's listed 888562 is not the optimum; it is most likely a typo or rounding. `optima.json` uses 888563.
  - That this is optimal rests on Optsicom (which lists it as a single value, not an interval). I also tried to prove it myself with a cutting-plane LP (HiGHS), but stopped it for time: the LP bound had only come down to about 904889, so it is not a proof.
- Optimality of the other 29 rests on Mitchell-Borchers' exact solves and agrees exactly with Optsicom.
