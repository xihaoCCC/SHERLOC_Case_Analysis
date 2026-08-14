# SHERLOC Extraction Contract v2

Status: normative contract for `src/sherloc/03_parse_pages.py` version 2.x.

This document supersedes `sherloc_extraction_contract_v1.md` for parser-v2
outputs. The keywords **MUST**, **MUST NOT**, **SHOULD**, and **MAY** are
normative. The contract describes the parser's emitted data, not a future
normalized or semantically enriched schema.

## 1. Purpose and corpus boundary

Parser v2 converts the frozen SHERLOC case HTML into a loss-aware structural
representation for later legal and trafficking-feature work. It is not a
semantic annotation system.

Corpus membership MUST come only from `data/manifests/case_urls.csv`, the
frozen 2026-08-09 result set for SHERLOC's `Crime Type = Trafficking in
persons` filter. The parser MUST NOT add or remove records based on a URL-path
crime type, page badge, title, or extracted text. Cross-classified paths such
as migrant smuggling, criminal group, drug, fraud, corruption, or cybercrime
remain corpus members when present in the frozen manifest.

The current frozen manifest has 1,590 ordered rows and SHA-256:

```text
1df2256dbfc063f88b23c2d85062f6650e91e49bd7a21cdb7a5fc11c94988fd5
```

The implementation MUST derive output cardinality from the manifest rather
than hard-code 1,590 as a general parser rule.

## 2. Inputs and integrity gates

The provenance chain is:

```text
case_urls.csv row
  -> page_download_manifest.csv row
    -> immutable data/raw_html response bytes
      -> parser-v2 case envelope
```

Before corpus parsing, the parser MUST validate that:

- source `search_rank` values are ordered, unique, and contiguous from 1;
- source `canonical_url` values are present and unique;
- the download manifest has one row per source rank and no extra ranks;
- joined `canonical_url` and `api_result_id` values agree;
- every joined download has `download_status = HTTP_OK_VALID`; and
- each production raw file's observed byte count and SHA-256 are compared with
  its download-manifest values during parsing.

The raw HTML files are authoritative source bytes. Parser v2 MUST NOT rewrite,
repair, normalize, or overwrite them. A duplicated doctype is a known SHERLOC
template artifact and is tolerated by the in-memory HTML tree builder.

Every successfully completed corpus run MUST emit one envelope per source row,
in `search_rank` order. A case-level parsing exception MUST produce an explicit
failure envelope at that rank instead of silently omitting the case.

## 3. Status vocabulary

All status-bearing sections, tab groups, panes, categories, and structured
field groups use this vocabulary:

- `FOUND`: the relevant source structure contains usable content;
- `SECTION_ABSENT`: the relevant source structure is not present;
- `EMPTY`: the source structure exists but contains no usable value;
- `PARTIAL`: usable content was retained, but a structural warning makes the
  extraction incomplete or ambiguous; and
- `PARSE_ERROR`: extraction could not be completed for the case or section.

These five strings are the complete parser-v2 section/field-group status
vocabulary. `SECTION_ABSENT` is expected missingness, not an error. The frozen
corpus contains 25 valid pages with no `.factSummary`; their dedicated Fact
Summary status is `SECTION_ABSENT`.

`parser_provenance.parse_status` uses only:

- `FOUND` when the case has no warning- or error-severity parser diagnostic;
- `PARTIAL` when at least one `WARNING` diagnostic exists and no `ERROR`
  diagnostic exists; or
- `PARSE_ERROR` when an `ERROR` diagnostic exists or the case could not be
  parsed.

Informational diagnostics do not by themselves change overall status.

Diagnostics have exactly these fields:

```text
code
message
location
severity        INFO | WARNING | ERROR
```

They are retained in source encounter order. Section-local warnings are also
collected in `parser_provenance.warnings`.

## 4. Top-level case envelope

The physical corpus serialization is UTF-8 JSONL, one object per manifest row.
Every normal parsed object has this top-level envelope:

```text
schema_version                 "sherloc-extraction-contract-v2"
corpus_membership              object
provenance                     object
source_input                   object
case_identity                  object
narrative
  fact_summary                 FactSummary
trafficking_sidebar            TraffickingSidebar
legacy_keywords                LegacyKeywords
participants
  sections[]                   ParticipantSection
charges_claims_decisions       ChargesSection
crime_type_badges[]            CrimeTypeBadge
main_record_sections           object of section arrays
parser_provenance              object
```

Parser v2 MUST NOT move the trafficking sidebar under a generic annotations
object, rename participants to parties, or flatten direct main-record
sections. Consumers SHOULD use the envelope above rather than v1 field names.

### 4.1 `corpus_membership`

```text
membership_rule                string
snapshot_date                  "2026-08-09"
source_manifest_path           "data/manifests/case_urls.csv"
source_manifest_record         complete joined CSV row, with parser-v2 scalar typing
```

The source row is retained in full. Integer-like manifest columns used by the
parser become integers where present, and `is_canonical_duplicate` becomes a
boolean.

### 4.2 `provenance`

```text
search_rank                    integer
api_result_id                  string | null
api_result_uri                 string | null
unodc_case_number              string | null
canonical_url                  string | null
requested_url                  string | null
resolved_url                   string | null
download_manifest_raw_filename string | null
download_manifest_sha256       string | null
download_timestamp             string | null
download_validation
  download_status              string | null
  http_status                  integer | null
  content_type                 string | null
  final_url_relation           string | null
  og_url_relation              string | null
  structural_markers           parsed JSON value | string | null
  warnings[]                   strings
download_manifest_record       complete joined CSV row, with parser-v2 scalar/JSON typing
```

The successful-record envelope carries both selected convenience provenance
and the full joined download row. A fatal failure envelope MAY omit convenience
members that could not be evaluated, but MUST retain the source row, joined
download row when available, rank, canonical URL, and failure diagnostic.

### 4.3 `source_input`

```text
input_kind                     production_raw_html | manual_regression_fixture
actual_path                    repository-relative path when inside the repository
computed_byte_count            integer
computed_sha256                64-character lowercase hexadecimal string
utf8_replacement_character_count integer
```

The checksum is computed over the unchanged response bytes, before decoding.
A failure envelope retains `input_kind` and `actual_path`; values unavailable
because reading or parsing failed MAY be absent.

### 4.4 `case_identity`

```text
title_raw                      string | null
document_title_raw             string | null
manifest_title_raw             string | null
country_raw                    string | null
page_locale                    string | null
page_locale_detection_method   pageLocale_script | html_lang | url_locale | unknown
html_lang_raw                  string | null
og_url                         string | null
og_url_relation_to_canonical   EXACT_MATCH | CANONICAL_EQUIVALENT | MISMATCH | MISSING
url_path_crime_type            string | null
```

`title_raw` is the visible direct case-title heading. Manifest identity fields
remain provenance and MUST NOT be reconstructed from page prose.

## 5. Text and whitespace policy

The parser reads original bytes and decodes them as UTF-8 with replacement on
invalid sequences. It MUST count resulting U+FFFD replacement characters in
`source_input.utf8_replacement_character_count` and issue
`UTF8_REPLACEMENT_CHARACTERS` when the count is nonzero.

The standard-library HTML parser decodes character references. Extracted text:

- maps non-breaking space U+00A0 to an ordinary space;
- maps CRLF and CR line endings to LF;
- collapses spaces, tabs, form feeds, and vertical tabs within each line;
- trims line edges and redundant leading/trailing blank lines;
- retains HTML block and `<br>` boundaries as newlines where block-preserving
  extraction is requested; and
- excludes script, style, noscript, and template text.

Parser v2 performs no Unicode normalization, translation, paraphrasing, label
aliasing, taxonomy reconciliation, entity resolution, or legal inference.
Fields named `*_raw` are source-facing text after the stated mechanical HTML
and whitespace policy; they are not claims of byte-for-byte substring
identity.

SHERLOC trafficking-sidebar values may begin with a decorative `•`. For
each such value, `value_raw` removes exactly one leading decorative bullet,
while `source_text_raw` retains the displayed source representation and
`decorative_prefix_removed` records `•`. No other semantic prefix stripping
is performed.

## 6. Reusable structural records

### 6.1 Ordered field record

Fields are arrays, never lossy label-to-value maps:

```text
ordinal                        one-based integer
dom_classes[]                  strings
class_key                      string | null
parent_field_ordinal           integer | null
label_raw                      string | null
value_raw                      string | null
raw_text                       string | null
status                         FOUND | EMPTY
```

Repeated labels, blank labels, blank values, nested fields, and source order
MUST be retained. No label/value relationship is inferred beyond source DOM
containment.

### 6.2 Tab group and pane

```text
TabGroup
  group_index                  one-based integer
  status                       status vocabulary
  pane_count                   integer
  panes[]                      TabPane
  warnings[]                   diagnostics

TabPane
  group_index                  one-based integer
  pane_index                   one-based integer
  language                     ISO-like source language code | null
  language_detection_method    string
  pane_id_raw                  string | null
  tab_href_raw                 string | null
  tab_label_raw                string | null
  is_active_in_html            boolean
  status                       FOUND | EMPTY
  text_raw                     string | null
  fields[]                     ordered field records
```

Language is inferred only from structural signals, in order: explicit tab
label, pane-ID suffix, tab-href suffix, then page locale. It is not inferred
from prose. Every pane is paired with its tab link by DOM ordinal. Duplicate
IDs, duplicate hrefs, link/pane-count mismatches, no-active-pane groups, and
multiple-active-pane groups MUST be diagnosed, and every pane MUST remain in
DOM order. Duplicate identifiers are never deduplication keys.

## 7. Narrative Fact Summary

`narrative.fact_summary` is:

```text
status                         status vocabulary
heading_raw                    string | null
variants[]                     TabPane-shaped records
english_text_raw               string | null
english_variant_indices[]      one-based integers
warnings[]                     diagnostics
```

For tabbed summaries, every pane becomes a variant. For an un-tabbed
`.factSummary`, one variant is emitted with `group_index = null`,
`pane_index = 1`, and the page-locale structural signal when available.
`english_text_raw` is the first nonempty variant structurally classified as
English; all variants remain available even when more than one is English.

Parser v2 MUST NOT substitute commentary, procedure, legal reasoning, or other
case prose into a missing Fact Summary. A Fact Summary heading without a
`.factSummary` container is `PARTIAL`; a missing Fact Summary section is
`SECTION_ABSENT`.

## 8. Crime badges and trafficking sidebar

`crime_type_badges` retains every top-level `.crimeType-details-badge` in DOM
order:

```text
ordinal
badge_type
dom_classes[]
label_raw
fields[]
raw_text
```

Each badge field retains:

```text
ordinal, badge_ordinal, structural_class, heading_raw, heading_classes[],
status, values_raw[], value_records[], container_present, raw_text
```

Each `value_records` item contains `ordinal`, `value_raw`, `source_text_raw`,
and `decorative_prefix_removed`.

`trafficking_sidebar` MUST be derived exclusively from badges whose structural
badge type is `traffickingPersonsCrimeType`:

```text
status
badge_count
badge_ordinals[]
fields
  offences
  acts
  means
  exploitative_purposes
  form_of_trafficking
  sector
  keywords
additional_fields[]
warnings[]
```

Each named field contains `status`, `values_raw[]`, and `sources[]`; each
source retains badge ordinal, field ordinal, heading, structural class, status,
values, and value records. Fields from migrant-smuggling, fraud, drug,
criminal-group, or other badges MUST NOT leak into this object. All badges
remain independently available in `crime_type_badges`.

Multiple trafficking badges are retained as multiple sources and make the
sidebar `PARTIAL`. Missing trafficking badges are `SECTION_ABSENT` with a
diagnostic; they do not alter corpus membership.

## 9. Legacy Keywords

The main-record section headed `Keywords` is a separate annotation source:

```text
status
heading_raw
non_pane_text_raw
categories[]
core_fields
  acts
  means
  exploitative_purposes
  form_of_trafficking
  sector
tab_groups[]
warnings[]
```

Each category retains `ordinal`, `label_raw`, `values_raw[]`, `status`, and
`raw_text`. Each core field retains `status`, `values_raw[]`, and
`category_ordinals[]`. Matching is limited to the parser's explicit source
labels; unmatched categories still remain in `categories`.

Legacy Keywords and the structured trafficking sidebar MUST remain
independent. Parser v2 MUST NOT merge, reconcile, prefer, overwrite, or infer
agreement between them.

## 10. Participants

`participants.sections` contains one record per source role container in DOM
order:

```text
status
dom_role_container_type        victimsPlaintiffs | defendantsRespondents
role_family                    person_role | defendant_respondent
visible_section_heading_raw
container_dom_classes[]
non_pane_text_raw
tab_groups[]
container_metadata_fields[]
records[]
record_count
warnings[]
```

Each repeated record contains `ordinal`, `status`, `dom_classes[]`,
`pane_provenance`, `fields[]`, and `raw_text`. `pane_provenance` is null outside a pane, otherwise
it copies the pane's structural identity and language fields.

The visible heading MUST be preserved independently of the DOM role class.
Thus `Sentencia 298/2015` remains a `victimsPlaintiffs` / `person_role`
container visibly headed `Migrants`. A `.person` is a source record, not a
guarantee of one natural person: it may represent a company, authority, group,
or aggregate. Parser v2 MUST NOT infer entity type or split aggregate records.

## 11. Charges, claims, and decisions

`charges_claims_decisions` is:

```text
status
heading_raw
non_pane_text_raw
tab_groups[]
section_metadata_fields[]
subject_records[]
orphan_charge_records[]
warnings[]
```

Each subject record retains `ordinal`, `status`, `dom_classes[]`, `pane_provenance`,
`subject_and_disposition_fields[]`, `charges[]`, `charge_count`, and
record-local `raw_text`. It also records
`raw_text_excluded_nested_subject_count`; when malformed markup nests later
subjects, the literal inclusive subtree fallback is retained separately as
`dom_subtree_text_raw`. Each nested or orphan charge retains `ordinal`, `status`, `dom_classes[]`,
`pane_provenance`, `fields[]`, and `raw_text`.

Containment in the source DOM is the only relationship asserted. The parser
MUST NOT infer cross-record defendant-charge-verdict links. A charge outside a
subject is preserved in `orphan_charge_records` with a warning instead of being
assigned to a person.

Malformed pages may omit closing tags and therefore nest later explicit
`.charges .person` blocks inside an earlier one. Every explicit `.person`
source block MUST still become an ordered subject record; the nesting is
reported as `NESTED_CHARGE_SUBJECT_RECORDS`. Charges belong only to their
nearest source-person ancestor, so preserving the malformed subjects does not
invent cross-subject links.

## 12. Main-record sections

Direct top-level `.case-law-detail` sections not routed to identity, Fact
Summary, Keywords, participants, or charges are retained under:

```text
main_record_sections
  attachments[]
  commentary_significant_features[]
  court[]
  cross_cutting_issues[]
  jurisdiction[]
  procedural_information[]
  sources_citations[]
  victims_witnesses_summary[]
  other[]
  unheaded[]
```

Every section record contains `status`, `heading_raw`, `non_pane_text_raw`,
`fields[]`, `tab_groups[]`, and `warnings[]`. Unknown headed sections go to
`other` with their source heading and an informational diagnostic. Nonempty
unheaded sections go to `unheaded`.

Heading routing is scoped to a direct main-record section. A nested field or
procedural subheading named `Court` MUST NOT create a page-level Court section.
Repeated direct sections are arrays and remain in source order.

An explicit `.case-law-detail` may itself be nested when missing closing tags
cause later source sections to fall inside an earlier DOM block. Such a block
is recovered as an independent main-record section and diagnosed with
`NESTED_CASE_LAW_DETAIL`; its contents are excluded from the enclosing
section's fallback text and fields. This recovery applies to explicit section
containers only, not nested procedural `h4` labels such as `Court`.

## 13. Parser provenance and failures

A normal record's `parser_provenance` is:

```text
parser_version                 string
parsed_at                      UTC timestamp
whitespace_policy              exact policy-description string
parse_status                   FOUND | PARTIAL | PARSE_ERROR
warning_count                  integer
warnings[]                     diagnostics
```

Parser v2 does not emit a parser-configuration hash, source-manifest hash,
download-manifest hash, HTML-library metadata, or per-field text hash in the
case envelope. Provenance claims MUST be limited to fields actually emitted.

When a case raises an exception, the failure envelope preserves the same
top-level family of objects, sets `parser_provenance.parse_status` to
`PARSE_ERROR`, records a `CASE_PARSE_ERROR` diagnostic, gives Fact Summary,
trafficking sidebar, legacy Keywords, and charges `PARSE_ERROR` placeholders,
and retains empty participant, badge, and main-section collections as
applicable.

## 14. Validation gates and regression expectations

The production pipeline MUST pass the checked-in 19-fixture gate before the
deterministic corpus challenge gate, and both before a full parse.

The fixture gate covers:

- 19/19 manifest identity joins and usable Fact Summaries;
- separate B637 English/French and Causa 2422 English/Spanish variants;
- Twitter, Inc. retained as a corporate respondent source record;
- strict cross-classified badge scoping and independent legacy Keywords;
- `Sentencia 298/2015`'s visible `Migrants` role;
- 54 person-role records, 45 defendant/respondent records, 45 charge subjects,
  111 charge blocks, and 16 direct main-record Court sections;
- United States v Robinson's 17 defendants, 17 charge subjects, and 63 charge
  blocks; and
- no fixture-level `PARSE_ERROR`.

The deterministic challenge MUST include all 25 known missing-Fact ranks plus
fixed structural anchors and any additional first-rank anchors required to
cover every URL-path crime type. It validates that all 25 missing Fact
Summaries are `SECTION_ABSENT`, duplicate tab IDs/hrefs retain every pane in
DOM order, high-cardinality records are not truncated, raw checksums match the
download manifest, and parsed `og:url` identities remain canonically aligned.

Coverage helpers and `outputs/metrics/parser_coverage.csv` count source records,
not inferred real-world people, charges, or legal events. A full run is valid
only when parsed-record and coverage-row cardinality both equal source-manifest
cardinality.

## 15. Explicitly out of scope

Parser v2 MUST NOT perform:

- corpus rediscovery, membership changes, downloading, or raw HTML mutation;
- sidebar/legacy reconciliation or controlled-vocabulary normalization;
- translation, summarization, paraphrasing, or prose-based language detection;
- entity resolution, entity-type inference, or aggregate-person splitting;
- inferred defendant-charge-verdict relationships;
- Fact Summary imputation from other prose;
- legal or trafficking-element inference, LLM prompting, benchmark scoring, or
  gold-label creation;
- attachment, judgment, PDF, or external-citation downloading/parsing; or
- case deduplication based on title, path, badges, narrative, or checksum.

Those are downstream transformations and require separately versioned schemas
and provenance.

## 16. Acceptance criteria

Parser v2 is contract-compliant only when:

1. manifest/download integrity gates pass;
2. the fixture and deterministic challenge gates pass offline;
3. one case envelope and one coverage row are emitted per source row in rank
   order;
4. failed cases remain explicit `PARSE_ERROR` envelopes;
5. multilingual, repeated, duplicate-ID, unfamiliar, and absent structures are
   retained and diagnosed under the rules above;
6. unchanged inputs produce the same extracted content apart from run
   timestamps; and
7. no raw HTML, source manifest, or download manifest is modified.
