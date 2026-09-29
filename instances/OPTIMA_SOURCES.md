# LOLIB IO optima: sources and verification (2026-09-29)

Found: 49/49 of the requested instances. Missing: none.

## Primary source (all 49 values)
Heidelberg LOLIB, still online at https://comopt.ifi.uni-heidelberg.de/software/LOLIB/
- matrices: iomat/<name>.mat
- optimum solutions: iosol/<name>.opt ("Value" field = sum above the diagonal, diagonal excluded; for t59b11xx, t65b11xx and t70b11xx the file uses an older format: "optimum solution (value N)" followed by the permutation)
- 46 files report "Number of b&c nodes" (1 or 3), i.e. these are branch-and-cut optimality certificates. The 3 old-format files (t59b11xx, t65b11xx, t70b11xx) are labelled "optimum solution".

## Checks
1. Recomputed: I applied each .opt permutation to its .mat matrix and summed the strictly upper triangle. It equals the stated Value for all 49. Where the file gives "Value + Diagonals", that also equals the upper sum plus the trace.
2. Second source, Optsicom LOLIB best-values PDF (lolib_bestvalues.zip, IO.pdf, "Bounds for IO problems"), from the Wayback Machine copy of www.optsicom.es/lolib/lop/ dated 2011-09-02. Optsicom distributes NORMALISED matrices ("N-" files, IO.zip), with a_ij' = a_ij - min(a_ij, a_ji), so its values are lower by the constant sum_{i<j} min(a_ij, a_ji). For all 49 instances:
   Heidelberg value - sum_{i<j} min(a_ij, a_ji) = the Optsicom bound = the Heidelberg permutation scored on the Optsicom N- matrix, exactly.
   Name mapping: stabu1/2/3 = Optsicom stabu70/74/75; t70d11xn = Optsicom t70d11xxb. Optsicom also lists usa79, which was not requested.
3. Disagreements: none. Do NOT use the Optsicom numbers directly: they are the normalised objective (e.g. be75eec 236464 vs the original 264940).

## Notes
- The live grafo.etsii.urjc.es/optsicom/lolib download links are broken (the host they redirect to does not resolve). I used the Wayback copies instead.
- be75np/be75oi/be75tot .mat files end with a stray repeated title line after the matrix. It does not matter because only the first n*n numbers are read.
- The optimal orderings from the .opt files are stored (0-based) in `optimal_orderings.json`; the study re-verified that each one scores exactly the value in `optima.json` on the matrices in this directory (49/49).
