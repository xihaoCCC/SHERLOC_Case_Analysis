# SHERLOC auxiliary-feature feasibility audit

Audit date: `2026-08-11`  
Frozen SHERLOC snapshot: `2026-08-09`

## Decision summary

| Feature | Candidate/reference ceiling | Fact Summary recoverability evidence | Main ambiguity | Recommendation |
|---|---:|---|---|---|
| Child/minor involvement | 551 provisional cases (461 TRUE, 90 FALSE; 43.63%) | Obvious support screen: 355/551 (64.43%); manual policy-eligible sample: 17/23 answerable | Adult-only evidence is weak; age timing, groups, attempts, and source conflicts | **Not worth adding as a large-scale target under the current time limit**; retain for the human reliability set |
| Victim multiplicity | 965 provisional cases (183 SINGLE, 782 MULTIPLE; 76.41%) | Stratified manual sample: 38/42 YES, 3 PARTIAL, 1 NO | One record can be a group; multiple records are not invariably distinct focal victims; SINGLE is partly narrative-adjudicated | **Feasible with restrictions** |
| Form of Trafficking | 1,186 Legacy-reference cases (93.90%) | Among 32 structured-present manual cases: 20 fully and 26 at least partially supported | `Internal` is often implicit; `Organized Criminal Group` is conceptually unlike geographic Form | **Feasible with restrictions** |
| Sector of Exploitation | 1,183 Legacy-reference cases (93.67%) | Among 31 structured-present manual cases: 18 fully and 21 at least partially supported | Severe class imbalance, `Other sectors`, and coarse or false-promise mappings | **Feasible with restrictions**; optional third feature |

The strongest two-feature package under the time constraint is **Form of Trafficking plus victim multiplicity**. Form supplies the cleanest high-coverage structured reference and a three-label Legacy taxonomy; multiplicity supplies the strongest observed narrative recoverability and direct operational meaning. If a third feature is possible, add **Sector of Exploitation**. Child/minor involvement should remain in the later human-validated reliability/abstention set rather than become a large-scale target now.

The manual samples are purposeful challenge samples, not random prevalence samples. Their percentages describe the reviewed cases and must not be extrapolated directly to all 1,263 cases.

## 1. Frozen primary cohort

The frozen rule reproduced exactly **1,263 cases**:

```text
nonempty English Fact Summary
AND nonempty Legacy Keywords Acts
AND nonempty Legacy Keywords Means
AND nonempty Legacy Keywords Exploitative Purposes
```

No Sidebar AMP value was used to construct or modify membership.

- Cohort ID: `sherloc-tip-2026-08-09-en-legacy-amp-complete-n1263-097ce2027171ebc9`
- Full membership SHA-256: `097ce2027171ebc9cac5ad6dfdbf6e854729f81a8ede78e8401086fe5d5ed48c`
- Membership hash serialization: ascending `search_rank`, one UTF-8 line per case as `<search_rank>\t<canonical_url>\n`.
- Parser-v2 JSONL SHA-256: `ea0592fcb633a0eee55e5feacb02fc1ef119cfcbb0f594566b4da6420eb184df`

All results below are restricted to this cohort. Feature availability never redefines the primary AMP corpus.

## 2. Audit method and reference/narrative distinction

Structured values were retained as exact parser-v2 strings. Legacy and Sidebar Form/Sector values were never normalized, merged, or reconciled. A source `.person` record was treated as a source container, not as an exact victim count.

The manual feasibility file contains 156 feature-case reviews:

| Feature | Manually reviewed cases |
|---|---:|
| Child/minor involvement | 42 |
| Victim multiplicity | 42 |
| Form of Trafficking | 36 |
| Sector of Exploitation | 36 |

For every selected row, the relevant structured record and complete English Fact Summary were inspected. The CSV retains a short evidence excerpt. Sampling deliberately covered clear, missing, multi-label, rare, grouped, conflicting, and unusual cases.

The four required structured/narrative states are:

- `TYPE_1`: structured reference present and Fact Summary supports the value.
- `TYPE_2`: structured reference present but Fact Summary does not fully support a fair Fact-Summary-only target.
- `TYPE_3`: structured reference missing but Fact Summary appears informative.
- `TYPE_4`: structured reference missing and Fact Summary is also insufficient.

| Feature | TYPE_1 | TYPE_2 | TYPE_3 | TYPE_4 |
|---|---:|---:|---:|---:|
| Child/minor involvement | 17 | 16 | 7 | 2 |
| Victim multiplicity | 32 | 4 | 5 | 1 |
| Form of Trafficking | 20 | 12 | 4 | 0 |
| Sector of Exploitation | 18 | 13 | 4 | 1 |

This confirms that structured missingness and narrative answerability are different phenomena. Metadata absence cannot become a negative label, and metadata presence does not guarantee a fair narrative-only prediction target.

## 3. Child/minor involvement

### Structured coverage and representations

- Person-role section: 1,241 cases (98.26%); absent in 22.
- Source person records: 2,892.
- Case-level record counts: 22 with zero, 658 with one, and 583 with two or more.
- Explicit grouped source representations: 323 records across 295 cases (23.36%).
- Visible headings: 1,238 `Victims / Plaintiffs in the first instance` containers and 5 `Migrants` containers.

SHERLOC's raw age/gender representation is inconsistent:

| Raw representation | Count |
|---|---:|
| `Gender: Child` | 755 fields across 442 cases |
| `Gender: Female` | 1,666 fields |
| `Gender: Male` | 196 fields |
| `Age:` fields | 745 |
| Numeric `Age:` values | 743 |
| Numeric ages under 18 | 426 fields across 238 cases |
| Numeric ages 18–120 | 317 fields |
| `Born:` values | 222 |
| Unlabelled fields carried in the source `age` class | 333 |

`Child` is therefore sometimes stored in a field visibly labelled `Gender`. Conversely, `Female`, `Male`, `woman`, or `man` does not establish adulthood. One `Age:` value is blank and one is `1988`; birth year alone cannot establish age at trafficking without a reliable event date. Some numeric ages are explicitly at trial rather than at the offence, and grouped records may attach one age to several people.

### Sidebar Offences as a separate signal

Sidebar Offences is available in only 176 cases:

| Raw Sidebar Offence value/pattern | Cases |
|---|---:|
| `Trafficking in children (under 18 years)` | 71 |
| `Trafficking in persons (adults)` | 132 |
| Child tag only | 44 |
| Adult tag only | 105 |
| Both child and adult tags | 27 |

The child tag is a useful positive signal but is not infallible. Rank 850 is an attempted-child-trafficking case involving an undercover officer; the source explicitly states that no actual child was used or trafficked. The adult-only tag cannot safely define FALSE: of its 105 cases, the conservative person-record policy finds 3 TRUE, 11 defensible FALSE, and 91 UNKNOWN. Ranks 177, 836, and 1055 carry adult-only Sidebar tags while other source evidence indicates child involvement.

### Conservative feasibility states

The provisional audit policy used explicit participant child/age language plus the Sidebar child tag for TRUE, with reviewed conflict/uncertainty exceptions. FALSE required explicit adulthood for every enumerated victim/person record, no positive signal, and no conflicting narrative evidence. Missing age, `Female`/`Male`, and vague `young` wording remained UNKNOWN.

| Provisional state | Cases | Percent |
|---|---:|---:|
| TRUE | 461 | 36.50% |
| FALSE | 90 | 7.13% |
| UNKNOWN | 712 | 56.37% |
| TRUE or FALSE candidate | **551** | **43.63%** |

These are feasibility candidates, not production labels. The proposed cohort barely exceeds 500 and is highly imbalanced: 83.67% of eligible cases are TRUE.

An intentionally strict Fact Summary support screen found obvious support for 337/461 TRUE candidates and only 18/90 FALSE candidates: **355/551 (64.43%)**, below the 500-case practical threshold for an obviously narrative-grounded benchmark.

### Manual findings and recommendation

The 42-case challenge review produced 22 TRUE, 9 FALSE, and 11 UNKNOWN judgments; Fact Summary answerability was 26 YES, 13 NO, and 3 AMBIGUOUS. Among 23 sampled policy-eligible cases, all agreed with the manual judgment, but only 17 were answerable from the Fact Summary. Eight policy-UNKNOWN cases became resolvable through narrative review, demonstrating that metadata is incomplete.

Common failure modes are:

- age at trial versus age during trafficking;
- grouped records with mixed or incompletely stated ages;
- `Gender: Child` and other schema misplacements;
- Sidebar adult/child tags that disagree with person fields or narrative facts;
- attempted offences without an actual child victim;
- victim records that mention the victim's own children or childcare work;
- disputed or unproven minority; and
- `Plaintiff`, `Complainant`, or `Migrants` roles whose trafficking-victim semantics need review.

**Recommendation:** do not add child/minor involvement as a large-scale auxiliary benchmark under the current time limit. Preserve it as a high-value target in the later human-validated reliability/abstention set, with explicit evidence spans and adjudicated TRUE/FALSE/UNKNOWN labels.

## 4. Victim multiplicity

### Structural audit

The same 1,241 cases have person-role sections, with source record-count distribution 22/658/583 for 0/1/2+ records. Record count alone is not the target:

- 208 one-record cases contain an explicit aggregate role value such as a number, `several`, or a plural group.
- A separate broad Fact Summary lexical screen flags literal plural-victim wording in 187/658 one-record cases (28.42%). This may overlap the structured aggregate screen and is not, by itself, a reliable multiplicity label.
- 24 multi-record cases contain exact duplicate normalized record payloads. Twenty-three have independent narrative/group evidence of multiple victims; rank 468 is a genuine singular counterexample.
- Of 583 multi-record cases, the provisional structural screen classifies 574 as MULTIPLE, flags 3 singular exceptions, and leaves 6 UNKNOWN because of role semantics.
- Five cases visibly use the heading `Migrants`; all were included in the challenge review.

The conservative candidate policy required explicit grouping or independently distinguishable victim records for MULTIPLE. SINGLE required an individual source record plus consistently singular narrative evidence; absence of plural wording alone was insufficient.

| Provisional state | Cases | Percent |
|---|---:|---:|
| SINGLE | 183 | 14.49% |
| MULTIPLE | 782 | 61.92% |
| UNKNOWN | 298 | 23.59% |
| SINGLE or MULTIPLE candidate | **965** | **76.41%** |

The candidate ceiling comfortably exceeds 700 but has a 4.27:1 MULTIPLE:SINGLE imbalance. SINGLE is partly a narrative-adjudicated conclusion rather than an independent SHERLOC count, so all 183 SINGLE candidates require confirmation before benchmark use.

### Manual findings and recommendation

The 42-case challenge review judged 24 MULTIPLE, 14 SINGLE, and 4 UNKNOWN. Answerability was 38 YES, 3 PARTIAL, and 1 NO. This 90.48% YES rate is encouraging but is a stratified validation observation, not a population estimate.

Ambiguities include aggregate victims in one record, duplicate records, victim plus non-victim plaintiffs, policy-case plaintiffs, narrative-only victims, plural references to clients or family members, and `Migrants` sections that do not clearly identify focal trafficking victims.

**Recommendation:** victim multiplicity is feasible with restrictions and is one of the top two auxiliary targets. Before construction, double-review all SINGLE candidates and challenge cases involving duplicates, mixed roles, absent person sections, or `Migrants`. Preserve UNKNOWN. Exact victim count is not recommended because aggregate descriptions, incomplete lists, and narrative-only quantities prevent a consistent reference.

## 5. Form of Trafficking

### Source-separated coverage and vocabulary

| Source availability | Cases | Percent |
|---|---:|---:|
| Legacy available | 1,186 | 93.90% |
| Sidebar available | 4 | 0.32% |
| Either source available | 1,186 | 93.90% |
| Legacy only / Sidebar only / both / neither | 1,182 / 0 / 4 / 77 | — |

| Source | Raw value | Cases | Percent among source-available cases |
|---|---|---:|---:|
| Legacy | `Transnational` | 771 | 65.01% |
| Legacy | `Internal` | 433 | 36.51% |
| Legacy | `Organized Criminal Group` | 176 | 14.84% |
| Sidebar | `Transnational trafficking` | 3 | 75.00% |
| Sidebar | `Internal trafficking` | 1 | 25.00% |

Legacy cardinality is 1,006 single-label, 180 multi-label, and 77 missing cases. The multi-label cases comprise 166 with two values and 14 with all three. All four Sidebar-present cases are single-label.

Among the four both-source cases, exact raw-set agreement is 0, partial overlap 0, and disjoint 4; mean and median raw Jaccard are 0. This raw disagreement is driven by strings such as `Transnational` versus `Transnational trafficking`. It is reported without alias merging or semantic reconciliation.

### Manual findings and recommendation

The 36-case review included 32 structured-present and 4 structured-missing cases. Of the 32 structured-present cases, 20 were fully supported, 6 partially supported, and 6 unsupported by the Fact Summary. Across all 36 rows, answerability was 23 YES, 7 PARTIAL, and 6 NO. All four missing-reference cases contained potentially informative narrative evidence, although one was only partial.

Transnational movement is usually explicit. `Internal` is often recorded even when the summary only describes a local investigation and does not state victim origin or route. `Organized Criminal Group` sometimes has explicit organization/network evidence but is conceptually different from geographic `Internal`/`Transnational` Form and is sometimes inferred from multiple actors.

**Recommendation:** Form is feasible with restrictions and is the strongest first auxiliary target. Use Legacy values as the source-separated reference ceiling of 1,186; do not use the four Sidebar rows to expand coverage. Before benchmarking, define answerability rules for `Internal`, adjudicate `Organized Criminal Group`, and preserve multi-label values.

## 6. Sector of Exploitation

### Source-separated coverage, vocabulary, and imbalance

| Source availability | Cases | Percent |
|---|---:|---:|
| Legacy available | 1,183 | 93.67% |
| Sidebar available | 5 | 0.40% |
| Either source available | 1,183 | 93.67% |
| Legacy only / Sidebar only / both / neither | 1,178 / 0 / 5 / 80 | — |

| Legacy raw value | Cases | Percent among 1,183 available |
|---|---:|---:|
| `Commercial sexual exploitation` | 920 | 77.77% |
| `Hotel/Restaurant/Bar` | 119 | 10.06% |
| `Domestic servitude` | 80 | 6.76% |
| `Other sectors` | 73 | 6.17% |
| `Begging` | 39 | 3.30% |
| `Agriculture` | 34 | 2.87% |
| `Construction` | 19 | 1.61% |
| `Factory/Manufacturing` | 19 | 1.61% |
| `Organ/tissue removal` | 10 | 0.85% |
| `Hair/Beauty Salon` | 6 | 0.51% |
| `Mining` | 2 | 0.17% |

Sidebar vocabulary consists of `Prostitution / sex work / pornography industry` (3), `Construction` (1), and `Agriculture` (1). It adds no cases beyond Legacy.

Legacy cardinality is 1,058 single-label, 125 multi-label, and 80 missing cases. Five labels occur in fewer than 20 cases, and the leading class appears in 77.77% of available cases. The taxonomy is compact enough for extraction but not balanced enough for naive aggregate accuracy.

Among five both-source cases, exact raw-set agreement is 2, partial overlap 0, and disjoint 3; mean raw Jaccard is 0.4 and median 0. The three disjoint cases include source-specific sexual-sector wording. No values were normalized.

### Manual findings and recommendation

The 36-case review included 31 structured-present and 5 missing cases. Of the 31 structured-present cases, 18 were fully supported, 3 partially supported, and 10 unsupported. Across all rows, answerability was 22 YES, 3 PARTIAL, and 11 NO. Four of five missing-reference narratives contained sector information; one was insufficient.

Sexual exploitation, begging, organ removal, and several concrete work settings are usually explicit. Problems include:

- `Other sectors`, which is not an extractable substantive category;
- a promised recruitment job being recorded as the exploitation sector even when the actual exploitation differs;
- coarse mappings such as fishing work to `Agriculture`;
- structured multi-label sets that omit additional narrative sectors; and
- very rare labels that cannot support stable per-class quantitative evaluation.

**Recommendation:** Sector is feasible with restrictions and is the optional third auxiliary feature. Use the 1,183 Legacy-reference cases only as a candidate ceiling. A production cohort needs an answerability screen, explicit handling of `Other sectors`, multi-label metrics, and rare-label reporting; Sidebar does not expand coverage.

## 7. Field-specific cohorts and practical thresholds

| Cohort | Candidate cases | Exceeds 500? | Exceeds 700? | Status |
|---|---:|---:|---:|---|
| Primary AMP | 1,263 | Yes | Yes | Frozen; unchanged |
| Child/minor TRUE or FALSE | 551 | Barely | No | Provisional; not recommended at scale |
| Multiplicity SINGLE or MULTIPLE | 965 | Yes | Yes | Provisional; feasible with confirmation |
| Legacy Form available | 1,186 | Yes | Yes | Reference ceiling; answerability restrictions needed |
| Legacy Sector available | 1,183 | Yes | Yes | Reference ceiling; imbalance/answerability restrictions needed |

The provisional intersection of all four auxiliary eligibility conditions is **371 cases (29.37%)**. Other useful intersections are 1,126 with both Legacy Form and Sector, 412 with provisional Minor and Multiplicity, and 864 with provisional Multiplicity plus both Legacy Form and Sector.

These intersections are descriptive only. The correct design is field-specific cohorts, not `AMP AND Minor AND Multiplicity AND Form AND Sector`.

Direct answers to the selection questions:

1. All four candidate/reference ceilings exceed 500 structurally or provisionally. For Minor, however, the conservative screen establishes obvious summary support for only 355 cases; this audit therefore does not establish a clean narrative-grounded cohort of at least 500.
2. The Multiplicity candidate ceiling and the Form/Sector structured-availability ceilings exceed 700; Minor's provisional ceiling does not.
3. Form has the cleanest combination of source-defined ground truth, high N, and small vocabulary. Sector also has source-defined labels but more semantic mismatch and imbalance.
4. Multiplicity was most often answerable in its challenge review, followed by Form. Its reference construction is less independent because SINGLE requires narrative adjudication.
5. Minor has the most consequential positive/negative reference ambiguity. Sector shows the largest structured-taxonomy versus actual-exploitation mismatch; Form's main mismatch is implicit `Internal` and the mixed meaning of `Organized Criminal Group`.
6. Include Minor only in the human reliability/abstention subset for now.
7. Include Multiplicity with restrictions and double review.
8. Include Form with restrictions.
9. Include Sector as the optional third feature, with imbalance and answerability controls.

## 8. Proposed 140-case human reliability/abstention set

Select approximately 140 unique cases and annotate all four auxiliary fields where applicable. Do not require every feature to be answerable. Use five mutually exclusive sampling buckets:

| Bucket | Proposed cases | Purpose |
|---|---:|---|
| A. Representative clean cases | 50 | Common classes with structured references and clear narrative evidence |
| B. Structured present, narrative incomplete | 25 | Directly measure unfair-target and abstention behavior |
| C. Structured missing, narrative informative | 20 | Measure metadata incompleteness and narrative-only recovery |
| D. Both insufficient | 20 | Genuine UNKNOWN/abstention cases |
| E. Rare or challenge cases | 25 | Deliberate stress coverage |
| **Total** | **140** | Within the requested 120–150 range |

Bucket E should deliberately include:

- Minor: age-at-trial/offence conflicts, grouped mixed ages, adult-only/child Sidebar conflicts, disputed age, and rank 850-like attempts.
- Multiplicity: one-record aggregates, duplicate records, mixed plaintiff/victim roles, absent person sections, and all five `Migrants`-heading cases where possible.
- Form: `Organized Criminal Group` only, all-three-label cases, implicit `Internal`, and missing metadata with clear routes.
- Sector: `Other sectors`, rare labels, multi-label cases, false-promise sectors, and coarse mappings.

Use two independent reviewers followed by adjudication. Record, per feature: structured source/value, narrative label, evidence sentence/span, answerability (`YES`/`PARTIAL`/`NO`), abstention decision, confidence, disagreement type, and adjudicated result. This is a future plan only; no 140-case set was selected or annotated here.

## 9. Issues to resolve before benchmark construction

1. Freeze a versioned auxiliary-reference policy without modifying parser-v2 raw fields.
2. Confirm Legacy as the Form/Sector source and define narrative-answerability gates independently of structured availability.
3. Double-review all provisional multiplicity SINGLE cases and preserve UNKNOWN; do not create an exact-count target.
4. Confirm that Minor remains reliability-set-only and define how attempts, disputed minority, temporal age, and Sidebar conflicts are adjudicated.
5. Define multi-label evaluation for Form/Sector and reporting rules for rare Sector labels and `Other sectors` without silently normalizing the raw taxonomy.

No LLM, transformer, conventional ML, benchmark split, AMP normalization, auxiliary production labeling, or final benchmark construction was performed.

## Machine-readable outputs

- `outputs/tables/auxiliary_feature_feasibility.csv`: cohort provenance, structural coverage, candidate eligibility, raw Form/Sector vocabularies and cardinality, source agreement, manual-review aggregates, intersections, recommendations, and the proposed reliability-set allocation.
- `outputs/tables/auxiliary_feature_manual_review.csv`: 156 feature-case manual feasibility reviews with source references, evidence excerpts, judgments, answerability, ambiguity, and TYPE_1–TYPE_4 status.
