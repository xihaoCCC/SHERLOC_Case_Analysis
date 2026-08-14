# Demonstration Candidate Search v2

Prepared 2026-08-13 with `07_prepare_demo_review_v2.py` v1.0.0. This is review assistance only: no final six were selected, no model/API was run, and the frozen benchmark and provisional A1/A2 splits were not changed.

## Result

The combined sheet retains **14** prior Agree/Hold cases and adds **15** new candidates: **5 U.S.**, **5 other high-support**, and **5 other/reserve**. The two prior Skip cases remain excluded.

### United States

- Rank 1178, **United States v. Fermin Pedro Ramos-Ramos et al** (United States of America; 83 words): AMP 7/7 CLEAR; Form POSSIBLE.
- Rank 1477, **United States v. Esperanza Vargas** (United States of America; 168 words): AMP 6/6 CLEAR; Form CLEAR.
- Rank 1242, **United States v. Lucilene Felipe Dos Santos** (United States of America; 115 words): AMP 4/4 CLEAR; Form CLEAR.
- Rank 692, **United States v. Fu Sheng Kuo** (United States of America; 313 words): AMP 5/5 CLEAR; Form CLEAR.
- Rank 498, **United States v. Edk Kenit** (United States of America; 174 words): AMP 7/7 CLEAR; Form CLEAR.

### Other high-support jurisdictions

- Rank 936, **Dosar nr. 1ra – 511/2009** (Republic of Moldova; 88 words): AMP 6/6 CLEAR; Form CLEAR.
- Rank 391, **Case n352012** (Belgium; 282 words): AMP 7/7 CLEAR; Form CLEAR.
- Rank 1343, **Case No B 4385-05** (Sweden; 175 words): AMP 6/6 CLEAR; Form CLEAR.
- Rank 828, **Proceso No. 2006-01458** (Colombia; 266 words): AMP 8/8 CLEAR; Form CLEAR.
- Rank 641, **Processo n 2007.05.00.088769-6** (Brazil; 74 words): AMP 5/5 CLEAR; Form POSSIBLE.

### Other/reserve jurisdictions

- Rank 334, **Case n180912** (North Macedonia; 103 words): AMP 7/7 CLEAR; Form CLEAR.
- Rank 761, **6B_81/2010 and 6B_126/2010** (Switzerland; 191 words): AMP 5/5 CLEAR; Form CLEAR.
- Rank 972, **Danish Supreme Court judgment 23 March 2009** (Denmark; 363 words): AMP 5/5 CLEAR; Form CLEAR.
- Rank 338, **RUC 1100440193-1 RIT199-2012** (Chile; 704 words): AMP 9/9 CLEAR; Form CLEAR.
- Rank 157, **The State of Israel v. Teddy Ness** (Israel; 248 words): AMP 4/4 CLEAR; Form POSSIBLE.

## Fidelity and next review

All **15/15 new cases** have every displayed Legacy AMP label screened `CLEAR`. **12/15** have `CLEAR` Geographic Form. Ranks 641 and 1178 are `POSSIBLE` because only one endpoint's country is explicit; rank 157 is also `POSSIBLE` because Georgia and Tbilisi are not explicitly linked without outside geographic knowledge. CLEAR-Form ranks: 1477, 1242, 692, 498, 936, 391, 1343, 828, 334, 761, 972, 338.

Recommended strongest cases for the next human inspection are ranks **1178, 1477, 936, 391, 334, 761, 828**, plus retained Agree seed **1487**. This eight-case inspection set covers **14/17 AMP labels**. Rank 1178 is the best compact U.S. option, with ranks 1477 and 1242 as strong U.S. alternatives. Ranks 334 and 761 are the cleanest reserve options; rank 338 is the high-coverage reserve. This is not a final bank.

Across every screened case whose displayed AMP references are all CLEAR, the attainable union is **14/17**. Missing labels are `MEANS_ABDUCTION`, `MEANS_FRAUD`, `PURPOSE_REMOVAL_OF_ORGANS`. Thus 17/17 does **not** appear achievable within this reviewed pool without accepting an ambiguous demonstration. The strict search intentionally did not force rare-label coverage.

The near-miss Argentina rank 913 was not added: its summary calls victims vulnerable and notes irregular stay, but does not show clearly how that vulnerability was used or constrained alternatives.

## Ekweremadu audit and blockers

`Rex and Obinna Obeta, Ike and Beatrice Ekweremadu` is **not eligible**. It is downloaded/parser rank 5 with a usable English Fact Summary, but it is absent from the frozen 1,263-case cohort because Legacy Acts and Means are `SECTION_ABSENT` / `SECTION_ABSENT`. Its only Legacy AMP value is `Removal of organs`, which the narrative clearly supports; Legacy Geographic Form is `SECTION_ABSENT`. Recruitment, vulnerability, and transnational values exist only in the trafficking sidebar and cannot be backfilled into the Legacy reference.

The remaining blockers are human/HT-professional adjudication of positive-label fidelity, possible source under-labeling, sensitive examples, and later fold-specific jurisdiction substitutions. Those decisions belong to the separate finalization step before split regeneration.

## Reproducibility guardrails

- Corpus search source: the full frozen N=1,263 benchmark, not provisional split roles.
- Candidate generation: full-cohort filtering followed by a manually curated sentence-level fidelity audit; the frozen audit table rebuild is deterministic, but the qualitative ranking is not model-derived.
- Evidence IDs: `sherloc_sentence_splitter_v1`; all IDs are range-validated.
- Reference source: Legacy Keywords only; sidebar values never define demo targets.
- `Organized Criminal Group` remains in raw Form JSON where supplied but is not a geographic target.
- `rare_label_coverage` uses a pre-screen frequency threshold below 10% of the frozen cohort.
- Candidate screening does not alter silver-reference benchmark membership.
