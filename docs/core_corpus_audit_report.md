# SHERLOC core corpus audit

Snapshot: `2026-08-09`  
Corpus: 1,590 cases returned by SHERLOC under `Crime Type = Trafficking in persons`

## Audit scope and definitions

This is the focused audit needed to close parser engineering and choose the next AMP benchmark-design steps. It uses the unnormalized parser-v2 output in `data/interim/sherloc_cases_raw.jsonl`. No labels were normalized, reconciled, or semantically interpreted, and no benchmark inclusion decisions or splits were made.

- A usable English Fact Summary is a nonempty `narrative.fact_summary.english_text_raw` value.
- A field is available when its source contains at least one nonempty raw value: Acts (A), Means (M), or Purpose (P).
- `Union` is only a field-level availability test: a field is present in either source. It does not merge or reconcile raw labels.
- Within-case repeated identical strings are counted once for label-frequency and set-comparison calculations. The parser output itself remains unchanged.
- Legacy-versus-Sidebar agreement uses exact raw strings. Jaccard similarity is the equally weighted mean or median of `|Legacy ∩ Sidebar| / |Legacy ∪ Sidebar|` among cases where both sources contain the field.

## 1. Parser-quality conclusion

**Parser v2 can be considered closed for the core paper workflow. No parser fix is required before label normalization and modeling.**

All 24 `PARTIAL` cases and all 77 warning-severity diagnostics were reviewed against the parsed records and source HTML structures. There were no unreliable English Fact Summary extractions, no unreliable Legacy AMP extractions, and no unreliable Sidebar AMP values.

- Fact Summary: no warning-severity diagnostic targets this section. Of the 24 `PARTIAL` cases, 23 have usable English text. Rank 307 correctly reports `SECTION_ABSENT`; the source page has no Fact Summary structure. The two INFO-only Fact Summary cases, ranks 62 and 104, preserve separate English/French and English/Spanish panes and passed raw-text checks.
- Legacy Keywords: no warning-severity diagnostic targets Legacy AMP. Automated raw-marker, source, and value checks found no mismatches across the 24 cases.
- Trafficking sidebar: rank 205 has the only core-related warning, `MULTIPLE_TRAFFICKING_BADGES`. Its page contains one empty trafficking badge and one populated badge. Both are preserved, and the populated badge reliably yields four Acts, three Means, and one Purpose.
- The remaining warnings concern secondary or irregular structures: duplicate/malformed tabs in participants, procedure, charges, sources, jurisdiction, or unheaded sections; nested charge subjects; and recovered Court/Attachments containers. These issues remain visible in diagnostics and do not compromise the core fields audited here.

For context, parser v2 produced 1,566 `FOUND`, 24 `PARTIAL`, and 0 `PARSE_ERROR` case records. It recorded 98 diagnostics in total: 77 `WARNING` and 21 `INFO`.

## 2. Fact Summary availability

| Fact Summary state | Cases | Percent of corpus |
|---|---:|---:|
| Usable English Fact Summary | 1,565 | 98.43% |
| No usable English Fact Summary | 25 | 1.57% |
| Total | 1,590 | 100.00% |

All 25 absences were previously validated as genuine structural absences rather than parser failures.

## 3. AMP availability and realistic benchmark size

### Full corpus

| Source | A | M | P | A+M | A+P | M+P | A+M+P |
|---|---:|---:|---:|---:|---:|---:|---:|
| Legacy Keywords | 1,404 | 1,319 | 1,412 | 1,299 | 1,365 | 1,289 | 1,273 |
| Trafficking sidebar | 365 | 326 | 348 | 318 | 329 | 300 | 293 |
| Source-availability union | 1,507 | 1,405 | 1,501 | 1,387 | 1,457 | 1,369 | 1,353 |

### Cases with a usable English Fact Summary

| Source | A | M | P | A+M | A+P | M+P | A+M+P |
|---|---:|---:|---:|---:|---:|---:|---:|
| Legacy Keywords | 1,387 | 1,307 | 1,394 | 1,288 | 1,350 | 1,279 | **1,263** |
| Trafficking sidebar | 362 | 326 | 344 | 318 | 326 | 300 | **293** |
| Source-availability union | 1,489 | 1,393 | 1,483 | 1,376 | 1,441 | 1,359 | **1,343** |

Thus, **1,263 of 1,565 English-summary cases (80.70%)** have complete Legacy A+M+P and form the strongest current estimate of the Topic 1 benchmark ceiling before any later eligibility or quality decisions. Sidebar A+M+P covers 293 English-summary cases (18.72%). The availability union covers 1,343 (85.81%), an increase of exactly **80 cases** or 5.11 percentage points over Legacy alone.

The source-complete relationship is unusually clean: 213 English-summary cases are A+M+P-complete in both sources, and all 80 union additions are Sidebar-complete but not Legacy-complete. There are **zero** cases whose union completeness is achieved only by mixing incomplete fields across the two sources.

### Mutually exclusive Legacy availability

| Population | A only | M only | P only | A+M only | A+P only | M+P only | A+M+P | None |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| All cases (N=1,590) | 13 | 4 | 31 | 26 | 92 | 16 | 1,273 | 135 |
| English summary (N=1,565) | 12 | 3 | 28 | 25 | 87 | 16 | 1,263 | 131 |

## 4. Legacy versus Sidebar comparison

### Field availability in the full corpus

| Field | Legacy only | Sidebar only | Both | Neither |
|---|---:|---:|---:|---:|
| Act | 1,142 | 103 | 262 | 83 |
| Means | 1,079 | 86 | 240 | 185 |
| Purpose | 1,153 | 89 | 259 | 89 |

Legacy has much broader coverage. The Sidebar therefore has substantially lower coverage at the corpus level, but its Sidebar-only cases show that it is not wholly redundant.

### Exact raw-set agreement when both sources are present

| Field | Both-source cases | Exact | Partial overlap | Disjoint | Mean Jaccard | Median Jaccard |
|---|---:|---:|---:|---:|---:|---:|
| Act | 262 | 50 (19.08%) | 170 (64.89%) | 42 (16.03%) | 0.503 | 0.500 |
| Means | 240 | 95 (39.58%) | 102 (42.50%) | 43 (17.92%) | 0.579 | 0.500 |
| Purpose | 259 | 248 (95.75%) | 10 (3.86%) | 1 (0.39%) | 0.976 | 1.000 |

Among partial-overlap cases, Act has 16 Legacy-proper-subset-of-Sidebar cases and 154 cases where neither source is a subset; no Sidebar Act set is a proper subset of Legacy. Means has 3 Legacy-subset, 1 Sidebar-subset, and 98 neither-subset cases. Purpose has 2 Legacy-subset, 6 Sidebar-subset, and 2 neither-subset cases.

The descriptive conclusion is field-specific. Purpose is almost exactly redundant when both sources exist. Act and Means frequently share labels but differ in their complete raw sets, partly because the Sidebar uses some differently named or additional categories. Disagreement is not treated as annotation error. Overall, Legacy is the strongest primary-target candidate because of its much greater coverage; Sidebar is best retained independently as a secondary, augmentation, or sensitivity source pending a formal reconciliation policy.

## 5. Raw AMP vocabularies and candidate normalization issues

| Field | Legacy unique labels | Legacy available cases | Sidebar unique labels | Sidebar available cases |
|---|---:|---:|---:|---:|
| Act | 5 | 1,404 | 8 | 365 |
| Means | 6 | 1,319 | 7 | 326 |
| Purpose | 6 | 1,412 | 7 | 348 |

The vocabularies are small. Major labels are:

- Acts: Legacy `Recruitment` (1,112 cases), `Transportation` (905), and `Harbouring` (648); Sidebar `Recruitment/Hiring` (274), `Harbouring` (193), and `Transportation` (189).
- Means: Legacy `Abuse of power or a position of vulnerability` (797), `Deception` (695), and the combined threat/use-of-force/coercion label (684); Sidebar abuse of power (183), its threat/coercion wording (181), and `Deception` (166).
- Purpose: exploitation of prostitution/other sexual exploitation dominates both Legacy (1,123) and Sidebar (263), followed by forced labour/services (267 and 88 respectively).

The only labels occurring in five or fewer cases are Sidebar Purpose `Forced begging` and `Forced criminality (e.g. theft, pickpocketing, etc.)`, each in one case.

Automated surface review flags two candidate groups for human review, not automatic merging:

1. Act: `Recruitment` versus `Recruitment/Hiring`.
2. Means: Legacy `Threat or use of force or other forms of coercion` versus Sidebar `Threat of the use of force or of other forms of coercion` and `Use of force or of other forms of coercion`.

No capitalization-, punctuation-, or whitespace-only alias groups were found. Legacy Purpose `Other` should not be mechanically equated with Sidebar `Forced begging` or `Forced criminality`; that would be a semantic ontology decision, not a surface normalization.

Normalization is therefore **small in inventory but not purely mechanical**. The important complexity is the combined-versus-split coercion category and source-specific Act/Purpose categories, not a large or noisy string vocabulary. Two case-level Sidebar field lists also contain exact within-case repetitions (Acts at rank 48 and Means at rank 334); these remain preserved in raw parser output and were case-deduplicated only for this audit's frequency and set calculations.

## 6. Jurisdiction feasibility

The parser output contains 115 nonmissing exact `country_raw` jurisdiction/category strings.

| Rank | Jurisdiction | Total | English summary | English + complete Legacy AMP |
|---:|---|---:|---:|---:|
| 1 | United States of America | 192 | 191 | 160 |
| 2 | Brazil | 132 | 131 | 103 |
| 3 | Philippines | 88 | 87 | 73 |
| 4 | Argentina | 85 | 85 | 75 |
| 5 | Republic of Moldova | 65 | 65 | 57 |
| 6 | Romania | 55 | 55 | 52 |
| 7 | Slovakia | 49 | 49 | 48 |
| 8 | Belgium | 41 | 31 | 24 |
| 9 | Colombia | 40 | 40 | 38 |
| 10 | Serbia | 40 | 38 | 36 |

The top five account for 562 cases (35.35%); the top ten account for 787 (49.50%).

| Minimum English + complete Legacy AMP cases | Jurisdictions meeting threshold |
|---:|---:|
| 5 | 53 |
| 10 | 31 |
| 20 | 18 |
| 30 | 11 |
| 50 | 6 |

On sample-size grounds, jurisdiction-held-out evaluation **appears feasible with restrictions**. There are enough well-represented jurisdictions to consider held-out evaluation, but the median jurisdiction has only four complete cases; 62 jurisdictions have fewer than five and 15 have none. A later split design will need an explicit minimum-support rule and a treatment for sparse jurisdictions. `International and Regional Bodies` is one of the 115 raw categories (12 total cases; 7 complete) but is not a national jurisdiction, so its eligibility also requires an explicit decision.

## 7. Recommended decisions for the next phase

The core audit answers the requested research-design questions as follows:

- **A. Parser closure:** yes. Parser v2 is closed for English Fact Summary and raw Legacy/Sidebar AMP extraction; remaining warnings affect secondary/template structures.
- **B. Primary source:** yes. On descriptive coverage alone, Legacy Keywords is the strongest primary AMP-target candidate. Sidebar should remain source-separated for sensitivity or augmentation analysis.
- **C. Primary benchmark ceiling:** exactly **1,263** cases have a usable English Fact Summary plus complete Legacy A+M+P.
- **D. Union gain:** Sidebar has 293 complete English-summary cases; the source union has 1,343 and adds exactly **80** beyond Legacy. No additional completion arises only from mixing partial sources.
- **E. Cross-source similarity:** Purpose is nearly identical when both are present (mean Jaccard 0.976; 95.75% exact), while Acts (0.503; 19.08% exact) and Means (0.579; 39.58% exact) show materially different raw sets.
- **F. Vocabulary complexity:** Legacy has 5/6/6 unique Act/Means/Purpose labels; Sidebar has 8/7/7. The inventory is small, but combined/split and source-specific categories require explicit expert decisions.
- **G. Jurisdiction evaluation:** on sample-size grounds, it appears feasible with restrictions. Thirty-one jurisdictions have at least 10 complete primary-source cases, but the long tail is sparse.

The next 3–5 researcher decisions should be:

1. Confirm Legacy Keywords as the primary raw AMP target and define the Sidebar's separate role (secondary evaluation, augmentation cohort, or sensitivity analysis).
2. Approve a versioned, expert-reviewed normalization/ontology policy, beginning with recruitment wording, coercion combined-versus-split labels, and source-specific categories; implement it outside the parser.
3. Treat 1,263 as the pre-exclusion primary benchmark ceiling and decide whether the 80 Sidebar-complete additions should form a separately reported augmented cohort. This audit does not make that inclusion decision.
4. Define jurisdiction eligibility and minimum-support rules, including how to handle `International and Regional Bodies`, before constructing any held-out split.
5. Fix the multi-label target representation and evaluation rules before normalization or modeling begins.

## Machine-readable outputs

- `outputs/tables/core_amp_audit.csv`: tidy availability, mutually exclusive Legacy patterns, source overlap, exact raw-set agreement, overlap subtypes, and Jaccard statistics for all cases and the English-summary subset.
- `outputs/tables/raw_amp_label_audit.csv`: exact raw label frequencies and candidate-alias flags.
- `outputs/tables/jurisdiction_core_audit.csv`: one row per raw jurisdiction/category.
