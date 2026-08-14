# SHERLOC benchmark v1 construction report

Build freeze date: `2026-08-11`  
SHERLOC snapshot: `2026-08-09`  
Builder: `src/sherloc/05_build_benchmark.py` version `1.0.0`

## Decision summary

The frozen primary AMP cohort remains exactly **1,263 cases** under cohort ID
`sherloc-tip-2026-08-09-en-legacy-amp-complete-n1263-097ce2027171ebc9`. Legacy Keywords alone define primary Act, Means, and
Purpose targets. Sidebar AMP is retained only as source-separated secondary
metadata. No modeling or evaluation split was created.

| Target | Task type | Reference source | Eligible N | Status |
|---|---|---|---:|---|
| Act | 5-label multi-label | Legacy Keywords Acts | 1,263 | PRIMARY |
| Means | 6-label multi-label | Legacy Keywords Means | 1,263 | PRIMARY |
| Purpose | 6-label multi-label | Legacy Keywords Purpose of Exploitation | 1,263 | PRIMARY |
| Geographic Form | 2-label multi-label | Legacy Form: `Internal`/`Transnational` only | 1,156 | AUXILIARY |
| Victim multiplicity | SINGLE/MULTIPLE; UNKNOWN abstains | Conservative feasibility policy v1 | 965 provisional | AUXILIARY |
| Child/minor involvement | TRUE/FALSE; UNKNOWN abstains | Strict structured-reference plus narrative-support screen | 355 | EXPLORATORY |

Sector, exact victim count, and Organized Criminal Group as a geographic Form
label are excluded from benchmark v1.

## 1. Frozen universes and provenance

- Complete parser-v2 corpus: **1,590** cases.
- Usable English Fact Summary universe: **1,565** cases.
- Primary complete-Legacy-AMP cohort: **1,263** cases.
- Primary membership SHA-256: `097ce2027171ebc9cac5ad6dfdbf6e854729f81a8ede78e8401086fe5d5ed48c`.
- Parser-v2 JSONL SHA-256: `ea0592fcb633a0eee55e5feacb02fc1ef119cfcbb0f594566b4da6420eb184df`.
- Prior feasibility-review CSV SHA-256: `d644ee79983f6720c78a92939748f8fcdc7701c5b739f8bb4d13d9278cd4b360`.
- Every benchmark record retains canonical URL, raw HTML filename/checksum,
  parser version/status, download timestamp, and API identity.

Parser v2 does not expose a validated decision/verdict date. The benchmark
therefore leaves `decision_or_verdict_year` null and separately retains the
case-page URL year where one exists; it does not reinterpret that URL segment.

## 2. Primary Legacy AMP ontology and frequencies

The machine ontology is `sherloc-legacy-amp-v1`, with stable zero-based indices
and exactly **5 Act, 6 Means, and 6 Purpose labels**. Every raw Legacy AMP value
in all 1,263 cases maps exactly once. The ontology preserves raw SHERLOC strings
without merging or relabeling them.

### Act

| Raw Legacy label | Cases | Percent |
|---|---:|---:|
| `Recruitment` | 1,025 | 81.16% |
| `Transportation` | 825 | 65.32% |
| `Transfer` | 489 | 38.72% |
| `Harbouring` | 608 | 48.14% |
| `Receipt` | 352 | 27.87% |

### Means

| Raw Legacy label | Cases | Percent |
|---|---:|---:|
| `Threat or use of force or other forms of coercion` | 664 | 52.57% |
| `Fraud` | 304 | 24.07% |
| `Deception` | 673 | 53.29% |
| `Abuse of power or a position of vulnerability` | 769 | 60.89% |
| `Abduction` | 115 | 9.11% |
| `Giving or receiving payments or benefits to achieve the consent of a person having control over another person` | 141 | 11.16% |

### Purpose

| Raw Legacy label | Cases | Percent |
|---|---:|---:|
| `Forced labour or services` | 249 | 19.71% |
| `Slavery or practices similar to slavery` | 64 | 5.07% |
| `Exploitation of the prostitution of others or other forms of sexual exploitation` | 1,007 | 79.73% |
| `Servitude` | 72 | 5.70% |
| `Other` | 66 | 5.23% |
| `Removal of organs` | 10 | 0.79% |

## 3. Geographic Form

| Raw Legacy Form combination | Cases |
|---|---:|
| Internal only | 362 |
| Transnational only | 614 |
| Internal + Transnational | 34 |
| Organized Criminal Group only | 30 |
| Internal + Organized Criminal Group | 23 |
| Transnational + Organized Criminal Group | 109 |
| Internal + Transnational + Organized Criminal Group | 14 |
| Other observed combination | 0 |
| Missing Form | 77 |

- Geographic eligible N: **1,156** (91.53% of primary cohort).
- INTERNAL: **433**.
- TRANSNATIONAL: **771**.
- Both: **48**.
- Organized Criminal Group appears as raw metadata in **176** cases.
- OCG-only cases excluded from the geographic target: **30**.

Internal and Transnational are independent binary labels. Co-occurrence is
preserved. OCG never becomes a geographic label, and OCG-only or missing-Form
cases are not geographic-Form eligible.

## 4. Provisional victim multiplicity

| Provisional label | Cases |
|---|---:|
| SINGLE | 183 |
| MULTIPLE | 782 |
| UNKNOWN | 298 |
| Eligible SINGLE or MULTIPLE | **965** |

The compact minimum review queue contains **250** cases:
183 SINGLE,
37 flagged MULTIPLE, and
30 UNKNOWN. Of these,
**220** are currently performance-eligible
but still require human confirmation. All 183 provisional SINGLE cases are in
the queue. Other flags cover duplicate person records, mixed Victim/Plaintiff
roles, `Migrants` headings, absent person sections, and multi-record semantic
exceptions. UNKNOWN remains outside the main performance cohort.

## 5. Exploratory child/minor target

| Strict label | Cases |
|---|---:|
| TRUE | 337 |
| FALSE | 18 |
| UNKNOWN | 908 |
| Exploratory eligible | **355** |

The eligible class ratio is **18.72:1** TRUE:FALSE
(94.93% TRUE). This is an automated
intersection of the conservative structured policy and a strict Fact Summary
support screen, not a fully human-adjudicated focal-victim gold set. It remains
exploratory and does not alter the 1,263-case AMP cohort.

Known role-linkage limitation: search rank **448**
(`ECLI:NL:HR:2011:BP9394`) currently screens `TRUE` because both structured
person metadata and the narrative mention an infant, but the infant is the
manslaughter victim while the focal trafficking victim appears to be the adult
appellant. The frozen automated count is retained here as a candidate ceiling;
this case, and any similar role-ambiguous case, requires human adjudication
before label-level use.

## 6. Jurisdiction composition

The primary cohort contains **100** nonempty exact
jurisdiction/category values. Counts meeting later support thresholds are:
**31** with at least 10 cases,
**18** with at least 20, and
**11** with at least 30.

| Rank | Jurisdiction/category raw value | Cases |
|---:|---|---:|
| 1 | United States of America | 160 |
| 2 | Brazil | 103 |
| 3 | Argentina | 75 |
| 4 | Philippines | 73 |
| 5 | Republic of Moldova | 57 |
| 6 | Romania | 52 |
| 7 | Slovakia | 48 |
| 8 | Colombia | 38 |
| 9 | Serbia | 36 |
| 10 | Sweden | 31 |

No jurisdiction-held-out or other evaluation split was constructed.

## 7. Blinded 100-case human reliability sample

The deterministic sample is drawn from the 1,565-case usable-English universe,
not only the primary AMP cohort. Membership SHA-256 is
`39bb96284b94ac5dd95e89d12c57dfdf09593d1add6b7d6737172ea01c32cd4b`.

| Sampling bucket | Cases |
|---|---:|
| A. Representative clean cases | 35 |
| B. Structured metadata present but narrative support incomplete | 20 |
| C. Structured metadata missing but narrative informative | 15 |
| D. Narrative genuinely insufficient or abstention candidate | 15 |
| E. Rare or challenge cases | 15 |

- Unique jurisdictions/categories: **51**.
- Complete primary AMP members: **89**;
  outside-primary usable-English cases: **11**.
- All 5/6/6 Legacy AMP labels occur somewhere in the selected sample.
- Multiplicity-flagged primary cases: **26**.
- Child uncertainty/attempt/conflict anchors: **6**.
- Parser structural-warning cases with usable English: **3**.

Reviewer order is the stable SHA-256 order of
`sherloc-reliability-v1-2026-08-11|review-order|canonical_url`, exposed only as
neutral IDs `HRV1-001` through `HRV1-100`. The reviewer template contains no
rank, title, URL, jurisdiction, sampling bucket, SHERLOC structured target,
provisional multiplicity/child label, or hidden audit judgment. Reviewers must
not receive the project-management sample or researcher-only reference key
until both independent annotations are complete.

Sentence evidence uses `sherloc_sentence_splitter_v1`. Original Fact Summaries
remain unchanged in benchmark/project files; reviewer text adds deterministic
`[S1]`, `[S2]`, ... display identifiers only.

## 8. Remaining blockers before modeling

1. Human-confirm the 220 currently eligible multiplicity queue cases before
   treating the provisional cohort as final gold; preserve UNKNOWN when unclear.
2. Complete both blinded annotations, calculate agreement, and adjudicate before
   creating human-grounded reliability labels.
3. Freeze the evaluation protocol and jurisdiction-support rules only after the
   reviewed target distributions are available.
4. Human-adjudicate focal-victim linkage for the child set, beginning with the
   known rank-448 exception, and keep the 355 cases exploratory unless that
   review supports a stronger claim.

The generated files deliberately contain no model outputs, folds, random split,
held-out-jurisdiction split, or cross-validation assignment.
