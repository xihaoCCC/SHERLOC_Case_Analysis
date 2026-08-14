# SHERLOC trafficking case URL discovery report

## Status and scope

Reconnaissance and URL-manifest discovery are complete. No production case-detail
pages were downloaded. The only case-level network checks were 20 `HEAD`
requests, which do not retrieve page bodies.

This report describes a live snapshot collected on **2026-08-09 CDT**
(`2026-08-10T00:19:26Z` through `2026-08-10T00:25:07Z`). SHERLOC is a live
database, so the total and membership may change on a later run. The collector
detects the returned total; it does not hard-code it.

## Finding: direct JSON endpoint, no browser required

The least brittle available mechanism is a same-origin JSON search endpoint.
The static search HTML is an Angular application shell: it contains result-row
templates and `ng-repeat` bindings, but no case result records. The client script
[`waxs-ng.js`](https://www.unodc.org/cld/misc/v3/waxs-ng.js) sets
`jsonSourceBaseUrl = "data.json"` and performs a `GET` request with `lng` and a
JSON-encoded `criteria` query parameter.

The collector uses the endpoint resolved relative to the English search page:

```text
https://www.unodc.org/cld/en/v3/sherloc/cldb/data.json
```

The live search page is:

```text
https://www.unodc.org/cld/en/v3/sherloc/cldb/search.html
```

The locale-neutral `/cld/v3/sherloc/cldb/...?...lng=en` route also works, but
the locale-explicit route above exactly matches the relative endpoint used by
the inspected English page. `www.unodc.org` is used because a headers-only probe
showed that `sherloc.unodc.org` redirects to it.

Playwright is **not required**. Direct standard-library HTTP requests work
without authentication, a prior cookie, browser headers, or a referrer.

## Exact trafficking filter and request structure

The decoded `criteria` value is:

```json
{
  "filters": [
    {
      "fieldName": "en#__el.caseLaw.crimeTypes_s",
      "value": "Trafficking in persons"
    }
  ],
  "startAt": 0,
  "sortings": ""
}
```

The request is a `GET` with the following query parameters:

```text
lng=en
criteria=<URL-encoded compact JSON shown above>
```

Equivalent request shape:

```text
GET /cld/en/v3/sherloc/cldb/data.json
    ?lng=en
    &criteria=%7B%22filters%22%3A%5B...
```

The UI stores the same criteria in the search-page URL fragment under `c`; the
`#` in the field name is encoded as `%23`. That fragment is client-side state,
not the data response itself. Angular decodes it and issues the endpoint request.

The initial response echoes the applied filter as `Crime Type = Trafficking in
persons`. The collector requires that exact filter echo before it will publish a
manifest.

## Current count and pagination

The exact returned/displayed count in this snapshot is **1,590**.

This was independently present in both:

- top-level JSON `.found = 1590`; and
- the selected trafficking crime-type facet count, also `1590`.

The search UI renders `Found {{responseData.found}} cases`, so the count is not
embedded in the static HTML; it appears after the JSON response is loaded.

Pagination behavior:

| Requested `criteria.startAt` | Results returned | `.found` |
|---:|---:|---:|
| 0 | 10 | 1,590 |
| 10 | 10 | 1,590 |
| 15 | 10 | 1,590 |
| 1,580 | 10 | 1,590 |
| 1,590 | 0 | 1,590 |

The server batch size is 10. The client exposes no page-size, rows, or limit
parameter. `criteria.startAt` is a zero-based result offset, and the UI advances
it to the number of results accumulated so far.

The interface uses automatic infinite scrolling, not numbered pages. A scroll
handler clicks the `#moreResults` element when it becomes visible; that calls
`doSearch()`, appends the next batch, and advances the offset.

Offset 0 returns criteria, facets, result-field metadata, total, and results.
Later offsets omit criteria/facets and return total, result fields, and results.
Although the request sends a blank sorting value, the server normalizes it to:

```json
[
  {
    "item": "caseLaw@decisionVerdictDate_d1",
    "order": "desc"
  }
]
```

This sort lacks an explicit unique tie-breaker. The collector therefore checks
total consistency on every batch, repeated API IDs and URIs, a terminal boundary
request, and a repeated first-page snapshot before publishing.

## Case title and URL representation

Each result has this relevant shape:

```json
{
  "id": "2200,en,/case-law-doc/...html",
  "uri": "/case-law-doc/...html",
  "values": {
    "page_title": "Case title",
    "uri": "/case-law-doc/...html"
  }
}
```

- Title: `results[].values.page_title`
- Relative path: both `results[].uri` and `results[].values.uri`
- Search identity: `results[].id`

The collector flags disagreement between the two URI fields. None occurred in
this snapshot.

The UI's presentation link adds `?lng=en&tmpl=sherloc`. The manifest therefore
preserves:

- `result_url`: normalized absolute UI presentation URL, including those query
  parameters;
- `canonical_url`: `https://www.unodc.org/cld` plus the normalized API URI,
  without presentation query parameters or a fragment; and
- `api_result_uri`: the original relative URI returned by SHERLOC.

Canonicalization standardizes the scheme/host, removes duplicate slashes and
presentation parameters, normalizes percent escapes for unreserved characters,
and preserves path case and punctuation. It never creates a URL from a title.
Malformed or off-domain paths are flagged and retained as result rows rather
than silently discarded.

Most importantly, canonicalization does **not** require
`traffickingpersonscrimetype` in the path. Corpus membership comes exclusively
from the filtered SHERLOC result set.

## Full discovery validation

The production walk made 159 result-batch requests, followed by one terminal
boundary request, one first-page snapshot request, and 20 random `HEAD`
validations: 181 requests in total.

| Check | Result |
|---|---:|
| SHERLOC returned/facet total | 1,590 |
| Result links collected | 1,590 |
| Unique canonical URLs | 1,590 |
| Unique API result IDs | 1,590 |
| Unique API result URIs | 1,590 |
| Canonical duplicate rows/groups | 0 / 0 |
| Duplicate API IDs or URIs | 0 |
| Missing or malformed canonical URLs | 0 |
| Other flagged result rows | 0 |
| Boundary check at offset 1,590 | 0 results; passed |
| Repeated first-page snapshot | Same 10 URIs and total; passed |
| Random `HEAD` sample | 20/20 HTTP 200 |

The random sample was selected uniformly from unique canonical URLs with seed
`20260809`. It included three cross-classified URL paths (two
`criminalgroupcrimetype` and one `moneylaunderingcrimetype`); all returned HTTP
200. Full sample ranks and URLs are preserved in
`logs/url_discovery_diagnostics.json`.

The manifest's SHA-256 over ordered `search_rank + NUL + canonical_url + LF`
records is:

```text
7eac69b18a6b975d1f225bcbf7efba92f8b670ddc32a7f14761119d4dd404b33
```

The CSV is deliberately a **result-hit manifest**: it preserves every returned
rank. If a future run contains canonical duplicates, later occurrences remain in
the CSV with `is_canonical_duplicate`, `duplicate_of_search_rank`, and flags,
rather than disappearing silently. This run has a one-to-one mapping between
result hits and canonical URLs.

## URL-path crime-type distribution

| URL-path crime type | Count |
|---|---:|
| `traffickingpersonscrimetype` | 1,406 |
| `criminalgroupcrimetype` | 103 |
| `migrantsmugglingcrimetype` | 30 |
| `drugcrimetype` | 24 |
| `moneylaunderingcrimetype` | 18 |
| `illicitfirearmscrimetype` | 6 |
| `corruptioncrimetype` | 3 |
| **Total** | **1,590** |

There are **184** valid trafficking-filter results whose URL path names another
crime type. These are not errors and were not removed. They are strong evidence
that URL-path text cannot define the trafficking corpus.

## Anti-bot and rate-limit observations

All 181 production requests returned HTTP 200. Observed diagnostics:

- no HTTP 403;
- no HTTP 429;
- no CAPTCHA or challenge page;
- no `Retry-After` header;
- no retries; and
- no authentication requirement.

Response times ranged from 0.966 to 2.997 seconds, with a 1.629-second mean.
The walk was sequential and added a 0.25-second pause after each response.
Ordinary `JSESSIONID` and load-balancer cookies were observed during
reconnaissance, but no cookie was required before the first request.

The JSON endpoint transferred 3,909,596 response bytes during the production
run. SHERLOC includes Fact Summary HTML inside result JSON even though discovery
needs only title and URI. The collector deliberately discards those summaries
and saves none of them. It did not request or save any case-detail body.

## Outputs and reproduction

Outputs:

- `src/sherloc/01_collect_case_urls.py`
- `data/manifests/case_urls.csv`
- `logs/url_discovery_diagnostics.json`
- this report

The collector uses only the Python standard library. A production rerun is:

```bash
python3 src/sherloc/01_collect_case_urls.py --verbose
```

The default delay is one second. The reviewed snapshot used:

```bash
python3 src/sherloc/01_collect_case_urls.py \
  --delay-seconds 0.25 \
  --validation-sample-size 20 \
  --random-seed 20260809 \
  --verbose
```

For limited reconnaissance, use `--max-pages N`. A partial run writes to
`logs/url_discovery_diagnostics_partial.json` by default and will not create or
overwrite the production manifest or its paired production diagnostics. The
collector also preserves request-event diagnostics if a network or response
failure exhausts its retries. Skipped or zero-size URL validation cannot publish
a production manifest.

## Parsing/schema observations for a later task

No parser was rewritten during this task. The file named in the task,
`src/sherloc/03_parse_pages.py`, is not present in the current working tree; the
available parser is `src/sherloc/sherloc_parser_prototype_v1.py`.

Recommended later changes, kept separate from URL discovery:

1. Scope main-section heading searches to the main record under
   `#case-law-content` and direct `.case-law-detail` sections. The current global
   `h2`/`h3`/`h4` search selects the trafficking-sidebar `Keywords` heading first
   in four samples, causing real legacy keyword categories to be missed. It can
   also select a nested procedural `Court` heading rather than the main Court
   section.
2. Discover repeated-person containers structurally and preserve their original
   section heading/role. `Sentencia 298_2015.html` uses a `Migrants` heading with
   `div.victimsPlaintiffs`; exact victim-heading matching currently loses that
   record and should not silently relabel it as a victim.
3. Generalize language-pane preservation beyond Fact Summary. `Causa 2422.html`
   has English/Spanish panes for commentary, procedural information, and
   sources/citations; the prototype currently concatenates some translations.
4. Link each parsed record to manifest identity and download provenance:
   search rank, canonical URL, requested URL, resolved URL, `og:url`, HTTP
   status, raw-file checksum, and URL-mismatch warnings. The samples have
   `og:url` but no formal `link[rel=canonical]`, so treat it as a validation
   signal.
5. Extend structural availability statuses to offences/form/sector/keywords,
   cross-cutting issues, procedure, sources, and attachments. Distinguish empty
   structure from absent structure rather than relying only on text truthiness.
6. Incrementally align the extraction contract and parser output for
   `defendants[]`, ordered/repeated labeled fields, charges, and court fields;
   retain raw sections for backward compatibility.
7. Preserve all crime-type badges as audit metadata, but never use a badge or
   URL-path crime type to redefine filtered-result corpus membership.
