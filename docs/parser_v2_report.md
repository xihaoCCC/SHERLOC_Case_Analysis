# SHERLOC parser v2 report

Generated: `2026-08-10T20:14:20Z`

## Outcome

Parser v2 passed the 19-page regression-fixture gate and the 46-page deterministic challenge gate before parsing all 1,590 frozen-manifest records. The run completed in 64.60 seconds with 0 case-level parse errors.

Corpus membership remains defined only by `data/manifests/case_urls.csv` (the frozen 2026-08-09 SHERLOC `Crime Type = Trafficking in persons` result set). URL-path crime type and page badges are audit fields only.

## Validation gates

- Manual fixtures: **PASS** (19/19 parsed; 0 failed checks).
- Deterministic challenge: **PASS** (46 cases; 0 failed checks).
- Full corpus: **1,590** records written; status counts `{"FOUND": 1566, "PARTIAL": 24}`.

The fixture checks cover B637 English/French separation, Causa 2422 multilingual sections, Twitter as a corporate respondent, strict trafficking-badge scoping, independent legacy Keywords, Sentencia 298/2015's visible `Migrants` role, repeated entities, conservative charges, and direct-main-section Court scoping.

The challenge contains all 25 genuine Fact Summary absences plus every URL-path category, temporal and byte-size extremes, malformed/duplicate tab identities, rare sidebar fields, multilingual party sections, legal entities, `Jurisdiction`, and maximum-defendant stress cases.

## Corpus extraction coverage

- Fact Summary: English 1,565; any usable variant 1,565; no usable Fact Summary 25; statuses `{"FOUND": 1565, "SECTION_ABSENT": 25}`.
- Trafficking sidebar: Acts 365 cases / 954 values; Means 326 / 637; Exploitative Purposes 348 / 402.
- Legacy Keywords: Acts 1,404 cases / 3,571 values; Means 1,319 / 2,767; Purpose 1,412 / 1,628.
- Multilingual/tabbed markup: 218 cases, 878 tab groups, 1,757 panes; 179 cases have at least one group with multiple detected languages.
- Person-role sections: 1,528 cases, 1,534 sections, 3,464 source records.
- Defendant/respondent sections: 1,585 cases, 1,585 sections, 3,372 source records.
- Charges / Claims / Decisions: 1,541 cases, 3,293 subject records, 6,157 charge blocks.
- Main-record Court: 1,478 cases.

## Missing Fact Summary pages

All 25 known pages without `.factSummary` were confirmed as `SECTION_ABSENT`, not parser failures. Twenty-three retain case-specific prose in commentary, procedural, legal-reasoning, proceeding, or appellate-decision structures. Ranks 431 and 923 are genuinely structured-only records. Parser v2 does not substitute those sections into the dedicated Fact Summary field.

## Warnings and unfamiliar structures

There are 98 recorded diagnostics across 30 cases; 77 are warning/error severity (the remainder are informational). Warning-code counts: `{"DUPLICATE_TAB_HREF": 28, "DUPLICATE_TAB_PANE_ID": 28, "MULTIPLE_TRAFFICKING_BADGES": 1, "NESTED_CASE_LAW_DETAIL": 4, "NESTED_CHARGE_SUBJECT_RECORDS": 2, "TAB_GROUP_MULTIPLE_ACTIVE_PANES": 14, "TAB_GROUP_NO_ACTIVE_PANE": 21}`.

Unfamiliar direct main-section headings, all preserved under `main_record_sections.other`: `{}`.

Sections marked `PARTIAL`: `{"charges_claims_decisions": 4, "jurisdiction": 1, "participants:defendantsRespondents": 9, "procedural_information": 6, "sources_citations": 3, "trafficking_sidebar": 1, "unheaded": 2}`. A partial status means usable source content was retained but the source markup was ambiguous (for example duplicate tab IDs/hrefs, multiple active panes, an orphan charge, or duplicate trafficking badges).

Notable source structures retained for later audit include rank 205's two trafficking badges; rank 307's distinct panes sharing blank labels and the same ID/href; rank 574's three-pane duplicate-ID group; recovered nested main-record sections on ranks 1 and 63; malformed nested charge subjects on ranks 63 and 1489, whose record-local text and literal DOM-subtree fallback are stored separately; repeated defendant-level legal reasoning/statutes; companies and grouped legal persons encoded in ordinary `.person` nodes; and `victimsPlaintiffs` containers visibly headed `Migrants`.

## Fields intentionally only conservatively parsed

- Charges, claims, verdicts, statutes, sentences, and person-level dispositions are ordered within their source DOM containers, with complete per-record/per-pane raw text retained. No defendant-charge-verdict relationships are inferred beyond explicit containment.
- Procedural, commentary, source, attachment, cross-cutting, jurisdiction, appellate, and unheaded metadata sections retain ordered fields, pane provenance, and raw fallback text. Their heterogeneous subtypes are not semantically normalized.
- Entity type is not inferred. DOM role, visible heading, original labels, values, and source record text are preserved for people, groups, companies, authorities, and other organizations.
- The decorative SHERLOC sidebar bullet is removed only in `value_raw`; its exact displayed form remains in `source_text_raw` with `decorative_prefix_removed` recorded.

## Before label normalization

The next audit should define explicit, versioned policies for label aliases, duplicate/repeated translations, grouped parties, entity typing, and any reconciliation of trafficking-sidebar versus legacy Keyword annotations. Those decisions must occur in a separate normalization layer; parser v2 deliberately leaves the two annotation sources independent.

No Act/Means/Purpose normalization, exploitation-type derivation, source reconciliation, benchmark inclusion decision, LLM extraction, evidence-grounding experiment, abstention model, or scientific interpretation was performed here.

## Outputs

- `data/interim/sherloc_cases_raw.jsonl`: one nested record per frozen-manifest case.
- `outputs/metrics/parser_coverage.csv`: one coverage row per case.
- `logs/parser_diagnostics.json`: gates, challenge membership, aggregate diagnostics, and per-case warning/error records.
- `docs/sherloc_extraction_contract_v2.md`: schema and extraction rules.
- `tests/test_sherloc_parser_v2.py`: offline regression tests.
