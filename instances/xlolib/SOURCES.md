# xLOLIB (Schiavinotto & Stützle 2004): instances and best-known values

Fetched 2026-09-29.

## Files
- `N-<name>_150` and `N-<name>_250`: 78 instances, 39 of size 150 and 39 of size 250. They are unchanged from the Optsicom LOLIB `xLOLIB.zip`.
  - Source: https://web.archive.org/web/20110902144610id_/http://www.optsicom.es/lolib/lop/xLOLIB.zip
  - The Wayback SHA-1 digest LUYJHT3N6TELBPDI73RW2CBEC4MPOBCG matches the downloaded zip.
  - The live links on grafo.etsii.urjc.es/optsicom/lolib are broken: they redirect to grafo-services.etsii.urjc.es, which does not resolve.
- File format: the first token is n, followed by n*n whitespace-separated integers, row-major, one matrix row per line. There is no title line, so this is NOT the Heidelberg .mat layout, which has a title line first.
- Why 78 and not 98: Optsicom states that 20 of the original 98 xLOLIB instances were removed because their entry sums overflowed 4-byte integers. That drops t59i11xx, t65i11xx, t65w11xx, t70i11xx, t70k11xx, t70u11xx, t70w11xx, t70x11xx, t75i11xx and t75u11xx at both sizes.
- `best_known.json`: {instance_name (without the "N-" prefix): best-known value}.
- `upper_bounds.json`: the matching Optsicom upper bounds. Every instance is open: there is a gap of roughly 0.4% to 3% between best known and upper bound, so NONE of these values is a proven optimum.

## Convention (important)
- The Optsicom matrices are in "normal form": a_ij' = a_ij - min(a_ij, a_ji). I checked all 78 files, and in every one min(a_ij, a_ji) = 0 for all i != j. The diagonal is nonzero but irrelevant.
- The values are therefore the raw objective OF THESE FILES: the sum of strictly-above-diagonal entries of the permuted matrix, for the matrices exactly as stored here. The offset sum_{i<j} min(a_ij, a_ji) computed from these files is 0 for every instance.
- The original, un-normalised Schiavinotto-Stützle matrices do not appear to be online. I checked the IRIDIA ~stuetzle pages, Wayback copies of the TU Darmstadt ~tom pages, and Heidelberg LOLIB, which has no xLOLIB. Their offsets cannot be recovered from normal-form data. Any values quoted in the 2004 paper for the original matrices will therefore differ from these by an unknown per-instance constant.

## Value sources
1. `lolib_bestvalues.zip` -> `XLOLIB.pdf` ("Bounds" [best known, upper bound]).
   - Source: https://web.archive.org/web/20110902144424id_/http://www.optsicom.es/lolib/lop/lolib_bestvalues.zip
2. `lolib_method_exp.xls`, sheet xLOLIB ("Best Known Norm", "Upper Bound Norm"; the file was last saved 2009-08-05).
   - Source: https://web.archive.org/web/20110902144519id_/http://www.optsicom.es/lolib/lop/lolib_method_exp.xls
- The two sources agree on 76/78 best-known values and on 78/78 upper bounds. The two disagreements are below; in both, the xls value is higher and was reached by the MA column of the xls itself.
  - t59n11xx_150: PDF 318951, xls 318960 -> used 318960
  - tiw56r54_150: PDF 958139, xls 958192 -> used 958192
- No solution orderings are published, so I could not recompute any value. A short (4 min) home-made ILS did not reach either disputed value (318504 and 955236), which says nothing about the sources, only that the ILS is weak.
- The values date from 2009-2011. Later papers may have improved best-knowns; I did not search the literature for them.
