# SHERLOC Extraction Contract v1

## 1. Design principles

1. Preserve raw SHERLOC content before normalization.
2. Treat corpus membership as determined by the SHERLOC "Trafficking in persons" search result set, not by URL path.
3. Keep SHERLOC trafficking-sidebar annotations separate from legacy/general `Keywords` annotations.
4. Distinguish:
   - section absent,
   - section present but empty,
   - parser error,
   - download error.
5. Preserve multilingual Fact Summary variants separately. Do not concatenate languages.
6. Use only the English Fact Summary as the default LLM benchmark input when an English variant is available.
7. Save raw HTML independently from parsed JSON so parsing can be rerun offline.

## 2. Case-level raw schema

```text
case_identity
  source_file
  source_url
  title
  country_raw
  url_path_crime_type
  page_locale

narrative
  fact_summaries[]
      language
      text_raw
      source_pane_id
      tab_label_raw
      language_detection_method
      is_active_in_html
  fact_summary_en_raw
  commentary_raw
  cross_cutting_issues_raw

trafficking_sidebar
  offences_raw[]
  acts_raw[]
  means_raw[]
  purpose_raw[]
  form_raw[]
  sector_raw[]
  keywords_raw[]

legacy_keywords
  categories_raw {
      <SHERLOC category label>: [values...]
  }
  acts_raw[]
  means_raw[]
  purpose_raw[]
  form_raw[]
  sector_raw[]

victims[]
  fields_raw {label: value}
  raw_text

defendants[]
  fields_raw {label: value}
  raw_text

procedural_information
  fields_raw
  raw_text

charges_claims_decisions
  raw_text

court
  raw_text

availability
  fact_summary
  commentary
  acts_sidebar
  means_sidebar
  purpose_sidebar
  legacy_keywords
  victims
  defendants
  charges
  court

parser_provenance
  parser_version
  parse_warnings[]
```

## 3. Canonical source precedence

No source is silently overwritten.

For later research-ready fields:

```text
acts_shERLOC_sources = {
    sidebar: [...],
    legacy_keywords: [...]
}
```

The same applies to Means, Purpose, Form, and Sector.

Only after a data audit will a separate normalization script decide whether/how to reconcile them.

## 4. Fact Summary rules

Preferred source: `.factSummary`.

If multilingual tabs exist:
1. extract each `.tab-pane` separately;
2. map explicit labels (English, Français, Español, etc.);
3. if a tab is unlabeled, infer the default language from `pageLocale`, HTML `lang`, or URL locale;
4. never concatenate language panes.

If no multilingual panes exist:
- extract the direct Fact Summary body as the page-locale version.

## 5. Trafficking-sidebar rules

Scope extraction strictly to:

`div.traffickingPersonsCrimeType-details-badge`

Then parse the value block following these headings/classes:

- `crimeTypes_traffickingPersonsCrimeType_offences`
- `crimeTypes_traffickingPersonsCrimeType_actsInvolved`
- `crimeTypes_traffickingPersonsCrimeType_meansUsed`
- `crimeTypes_traffickingPersonsCrimeType_exploitativePurposes`
- `crimeTypes_traffickingPersonsCrimeType_formOfTrafficking`
- `crimeTypes_traffickingPersonsCrimeType_sectorsInWhichExploitationTakesPlace`
- `crimeTypes_traffickingPersonsCrimeType_keywords`

This prevents data from other crime-type badges on cross-classified pages from leaking into trafficking fields.

## 6. Legacy Keywords rules

Within the main case record, locate the `Keywords` section and parse each:

`div.keywordCategory.field`

Store the original label and all values.

Canonical aliases initially include:

- `Acts:` -> Acts
- `Means:` -> Means
- `Purpose of Exploitation:` -> Purpose
- `Form of Trafficking:` -> Form
- `Sector in which exploitation takes place:` -> Sector

Unknown keyword categories must be retained rather than dropped.

## 7. Victim/defendant rules

Victims:
- locate `div.victimsPlaintiffs`
- each `div.person` is one source record
- preserve all label/value pairs exactly

Defendants:
- locate the Defendant/Respondent section
- parse repeated person/entity blocks where possible
- always preserve the full raw section text because organization/legal-person records are heterogeneous

No assumption that a defendant is a natural person.

## 8. Missingness/status vocabulary

Use:

- `FOUND`
- `SECTION_ABSENT`
- `EMPTY`
- `PARSE_ERROR`

A failed page download is handled before parsing and must use a download-specific status such as:

- `HTTP_OK`
- `TEMPORARY_FAILURE`
- `HTTP_ERROR`
- `NETWORK_ERROR`

## 9. Validation before full scrape

The prototype must be run on the manually saved HTML test set.

Required checks:
- title and canonical URL extracted;
- Fact Summary detected in all pages where visibly/source-present;
- multilingual panes separated;
- trafficking sidebar never mixes values from other crime badges;
- legacy keyword categories recovered;
- victim repeated records preserved;
- parser warnings raised instead of silently failing.

After the full scrape, conduct:
1. automated coverage/outlier audit;
2. random manual parser audit;
3. separate 100–200 case semantic annotation audit for Topic 2.