# SHERLOC case-detail page download report

## Status and scope

This task downloads the English SHERLOC presentation/detail page for every row
in the frozen trafficking-filter result manifest. It does not discover, add, or
remove corpus members, and it does not parse or normalize case content.

The deterministic 80-case pilot, guarded full-corpus pass, and resume-safe
cleanup pass completed successfully. All 1,590 source rows now have a distinct
raw file with status `HTTP_OK_VALID`.

The sole corpus-membership source is:

```text
data/manifests/case_urls.csv
```

It is the frozen 2026-08-09 CDT SHERLOC snapshot containing 1,590 ordered result
rows returned by the filter `Crime Type = Trafficking in persons`. Its file
SHA-256, recorded by the downloader, is:

```text
1df2256dbfc063f88b23c2d85062f6650e91e49bd7a21cdb7a5fc11c94988fd5
```

No case is included or excluded based on the crime type in its URL path. This is
important because 184 records in the manifest use a cross-classified path such
as `criminalgroupcrimetype`, `migrantsmugglingcrimetype`, or `drugcrimetype`.

## Exact download mechanism

`src/sherloc/02_download_pages.py` is a sequential, Python-standard-library
downloader. For each source row it issues an HTTPS `GET` to the row's exact
`result_url`, normally the canonical case path plus:

```text
?lng=en&tmpl=sherloc
```

The exact requested URL is retained independently of the canonical URL and the
final URL after redirects. The client requests `Accept-Encoding: identity` and
saves the response body returned by SHERLOC as original bytes; it does not
decode, re-encode, tidy, normalize, or otherwise alter successful HTML. Neither
browser automation nor Playwright is required.

The default request interval is one second, measured between request start
times. The default timeout is 60 seconds. The user agent identifies the process
as sequential, authorized academic research.

### Raw filenames and writes

Successful pages are stored under `data/raw_html/` with a deterministic name:

```text
<zero-padded-search-rank>_<first-12-hex-of-SHA256(canonical-url)>.html
```

For example, rank 1 is saved as:

```text
data/raw_html/000001_3b88fff50ec4.html
```

Titles never enter filenames. The rank preserves the frozen result order while
the URL hash protects against accidental identity confusion. Response bytes are
first written to a same-directory temporary file, flushed and `fsync`ed, and
then atomically moved into place. An invalid nonempty response is preserved
under `data/raw_html/_failed/` with a `.failed.html` suffix instead of being
misclassified as case HTML. Bulk HTML, temporary partial files, and quarantined
failure bodies are ignored by Git.

## Plausibility and identity validation

HTTP 200 alone is not considered a successful case download. A successful row
has download status `HTTP_OK_VALID` and must satisfy all of these checks:

- HTTP status 200;
- HTML/XHTML content type;
- absent or identity content encoding, consistent with preserving original
  response bytes;
- at least 10,000 response bytes;
- an HTML root and an English signal (`html[lang]` or SHERLOC's
  `pageLocale = "en"`);
- SHERLOC database header `#db-headder`;
- case container `#case-law-content`;
- case-detail container `.case-law-detail`;
- a nonempty `h2 span.title` case title;
- a final URL on an allowed SHERLOC/UNODC HTTPS host and canonically equivalent
  to the manifest URL; and
- if `og:url` exists, an `og:url` canonically equivalent to the manifest URL.

The canonical comparison deliberately treats SHERLOC's locale-prefixed
`/cld/en/case-law-doc/...` form, locale-neutral `/cld/case-law-doc/...` form,
and presentation query parameters as equivalent. It still flags a genuinely
different case URL. Error/challenge titles such as service unavailable, access
denied, too many requests, internal server error, or not found are also checked
so a service page cannot silently pass as a case.

The following conditions are warnings, not hard failures:

- missing `.factSummary`;
- missing trafficking badge;
- missing `og:url`;
- a case-title text difference from the discovery manifest;
- unusual doctype count;
- page size below 100,000 bytes or above 2,000,000 bytes; and
- a retry or redirect before eventual success.

Fact Summary and trafficking fields are optional at the page-template level;
requiring either would incorrectly reject otherwise valid SHERLOC cases.

## Retry, backoff, redirects, and circuit breaking

Network errors and HTTP 408, 425, 429, 500, 502, 503, and 504 are transient.
The default configuration permits four retries after the initial attempt. The
backoff starts at two seconds and doubles per attempt. If SHERLOC returns
`Retry-After` as either seconds or an HTTP date, the downloader waits for at
least that interval. A structurally invalid HTTP 200 response is retried as a
possible transient service/challenge response.

Redirect chains are recorded. Redirects to a non-HTTPS URL, a nonstandard port,
or a host outside `www.unodc.org`/`sherloc.unodc.org` are blocked. Three
consecutive network, transient-HTTP, or challenge-like failures trigger a
circuit breaker so a site-wide failure is not followed by hundreds of futile
requests. Every targeted source row remains in the download manifest even when
the run is interrupted or the circuit breaker stops it.

## Resume safety and provenance

`logs/page_download_manifest.csv` always contains all 1,590 source rows, not
only downloaded rows. Source-controlled identity fields are reloaded from the
frozen URL manifest on every run. For each row the download manifest records:

- search rank, case title, URL-path crime type, API result identity, and UNODC
  case number;
- canonical, exact requested, final/resolved, and `og:url` values plus their
  identity relations;
- status, content type/encoding, timestamp, raw filename, byte count, SHA-256,
  attempts, redirect chain, and elapsed time;
- pilot-selection status/reason, structural-marker results, warnings, error,
  last action, and last validation time.

Without `--force`, an existing raw file is skipped only after its recorded
byte count and SHA-256 (when present) and its current structural/URL validation
all pass. A valid unlogged or incompletely logged file can be recovered into
the manifest. Missing, changed, or invalid files are requested again. CSV and
JSON state is written atomically, with the CSV checkpointed every ten processed
targets by default. This makes normal reruns idempotent and interruption-safe.

Full mode is guarded: unless explicitly overridden, it requires a completed
pilot for the same source-manifest SHA-256, the same deterministic pilot ranks,
and currently valid files for every pilot row.

## Deterministic 80-case pilot design

The pilot does not rely on a random draw. It combines rank-stratified primary
paths, explicit cross-classified-path coverage, and temporal anchors:

1. Divide the 1,590 ordered results into ten equal rank bands and choose five
   quantile-spaced `traffickingpersonscrimetype` records per band.
2. Select quantile-spaced records from every observed cross-classified URL-path
   type. Allocate five to each type, then assign the remaining two slots to the
   largest cross-classified group (`criminalgroupcrimetype`).
3. Require the earliest and latest years visible in canonical URL paths (1981
   and 2024), replacing the nearest eligible primary-path record when needed.
4. Retain yearless URLs; five were selected. This tests the tail where SHERLOC
   paths do not contain a year segment.

Path coverage was:

| URL-path crime type | Manifest rows | Pilot rows |
|---|---:|---:|
| `traffickingpersonscrimetype` | 1,406 | 50 |
| `criminalgroupcrimetype` | 103 | 7 |
| `migrantsmugglingcrimetype` | 30 | 5 |
| `drugcrimetype` | 24 | 5 |
| `moneylaunderingcrimetype` | 18 | 5 |
| `illicitfirearmscrimetype` | 6 | 5 |
| `corruptioncrimetype` | 3 | 3 |
| **Total** | **1,590** | **80** |

The ten rank bands contained 13, 8, 6, 6, 8, 6, 7, 9, 9, and 8 pilot records,
respectively. The selected URL-year segments spanned 1981 through 2024 and
included five yearless paths.

Exact selected ranks:

```text
1, 3, 4, 35, 40, 52, 69, 80, 86, 122, 138, 158, 159, 160, 185, 197,
218, 234, 275, 290, 318, 319, 358, 400, 437, 461, 477, 478, 517, 529,
559, 598, 636, 637, 662, 677, 717, 751, 754, 757, 795, 796, 833, 875,
913, 937, 954, 955, 992, 1001, 1035, 1067, 1074, 1113, 1114, 1150,
1153, 1155, 1163, 1193, 1207, 1234, 1272, 1274, 1295, 1312, 1341,
1354, 1370, 1394, 1423, 1431, 1432, 1471, 1483, 1523, 1529, 1553,
1574, 1590
```

## Pilot results

The pilot ran from `2026-08-10T01:54:00Z` to `2026-08-10T01:57:49Z`
(229.475 seconds, approximately 3 minutes 49 seconds) with the production
defaults: one-second request spacing, 60-second timeout, four allowed retries,
and two-second exponential-backoff base.

| Check | Pilot result |
|---|---:|
| Source/download manifest rows | 1,590 / 1,590 |
| Pilot successes | 80 / 80 |
| Pilot failures | 0 |
| HTTP request attempts | 80 |
| HTTP 200 responses | 80 |
| Retries | 0 |
| HTTP 403 / 429 | 0 / 0 |
| `Retry-After` responses | 0 |
| Redirected responses | 0 |
| Final-URL true mismatches | 0 |
| `og:url` true mismatches | 0 |
| Unique raw SHA-256 values | 80 |
| Duplicate-checksum groups | 0 |
| Missing required structural markers | 0 |

All 80 final URLs and all 80 `og:url` values were
`CANONICAL_EQUIVALENT` to their manifest canonical URL. SHERLOC added the
English locale segment to `og:url`, while the requested/final presentation URL
retained `?lng=en&tmpl=sherloc`; these are expected representation differences,
not case-identity mismatches.

Pilot page-size distribution:

| Statistic | Bytes |
|---|---:|
| Total | 21,013,255 |
| Minimum | 246,942 |
| p01 | 247,154.51 |
| p05 | 248,630.90 |
| Q1 | 253,237.50 |
| Median | 256,601.50 |
| Mean | 262,665.69 |
| Q3 | 266,141.75 |
| p95 | 281,766.85 |
| p99 | 352,347.63 |
| Maximum | 399,118 |

No pilot page crossed the fixed suspicious-size thresholds of 100,000 and
2,000,000 bytes. Four legitimate pages were high Tukey-IQR outliers: ranks 138
(339,915 bytes), 751 (288,433), 937 (399,118), and 1295 (298,505). All passed
the required SHERLOC marker and URL-identity checks, so size alone was not used
to reject them.

The pilot exposed two nonfatal template conditions:

- all 80 pages contain two doctype declarations (`DOCTYPE_COUNT_2`); and
- ranks 318, 1295, and 1423 lack `.factSummary` but otherwise passed as valid
  SHERLOC case pages.

There were no missing trafficking badges, `og:url` metadata, case containers,
case-detail containers, title markers, database headers, or English-locale
signals in the pilot. With no systematic failure, the pilot satisfied the
precondition for the complete run.

## Full-corpus results

The first full pass ran from `2026-08-10T02:01:20Z` to
`2026-08-10T03:33:45Z` (5,539.387 seconds, approximately 1 hour 32 minutes 19
seconds). A temporary local Wi-Fi interruption caused ranks 943 and 944 to
exhaust their retries with `Connection refused`; they remained explicit
`NETWORK_ERROR` rows while the pass continued. That pass ended with 1,588
validated downloads and two failures.

The same full-mode command was then rerun. It verified and skipped all existing
valid byte files, retried the two failed rows, and downloaded both successfully
on their first cleanup attempt. The cleanup pass took 43.419 seconds. Production
full-pass plus cleanup elapsed time was 5,582.806 seconds (approximately 1 hour
33 minutes 3 seconds); including the pilot, end-to-end active runtime was
5,812.281 seconds (approximately 1 hour 36 minutes 52 seconds).

| Check | Full-run result |
|---|---:|
| Expected source rows | 1,590 |
| Download-manifest rows | 1,590 |
| Successful validated downloads | 1,590 |
| Failed/not-successful rows after cleanup | 0 |
| Full-mode target successes / failures | 1,590 / 0 |
| Production active runtime | 5,582.806 seconds |
| Production request attempts | 1,525 |
| HTTP 200 / local network-error attempts | 1,510 / 15 |
| Retry-indexed request events / `Retry-After` events | 13 / 0 |
| HTTP 403 / 429 events | 0 / 0 |
| Redirected successful pages | 0 |
| Final-URL true mismatches | 0 |
| `og:url` true mismatches | 0 |
| Unique successful SHA-256 values | 1,590 |
| Duplicate-checksum groups/rows | 0 / 0 |
| Missing required structural markers | 0 |

An independent final audit reread every saved file. All 1,590 requested URLs
exactly match the frozen manifest's `result_url`; all 1,590 recorded byte counts
and SHA-256 values match the files; all filenames match the stable rank/hash
pattern; and all raw paths are distinct. The raw directory contains exactly
1,590 successful `.html` files (411,442,034 bytes), no quarantined failure
bodies, and no partial files.

The 15 network-error attempts were local connectivity failures, not SHERLOC
HTTP responses. Four cases recovered within the first pass after retry: ranks
267 (two attempts), 945 (three), 1430 (two), and 1541 (two). Ranks 943 and 944
each exhausted five attempts during the Wi-Fi outage and succeeded with one
attempt each in the cleanup pass. The diagnostics preserve the pilot, first
full pass, and final cleanup summaries so the temporary failures are not erased
by eventual success.

Final page-size distribution:

| Statistic | Bytes |
|---|---:|
| Total | 411,442,034 |
| Minimum | 241,704 |
| p01 | 245,221.78 |
| p05 | 247,588.45 |
| Q1 | 251,247.25 |
| Median | 255,477.50 |
| Mean | 258,768.57 |
| Q3 | 262,526.25 |
| p95 | 280,166.30 |
| p99 | 301,358.64 |
| Maximum | 399,118 |

No page crossed the fixed suspicious-size thresholds of 100,000 and 2,000,000
bytes. Eighty-three pages were high Tukey-IQR outliers above 279,444.75 bytes;
none was a low outlier, and all passed structural, URL-identity, and checksum
validation. The complete outlier rank/URL list is retained in
`logs/page_download_diagnostics.json`.

All 1,590 pages contain the SHERLOC template's duplicate doctype declaration.
Twenty-five pages lack `.factSummary`: ranks 84, 307, 318, 380, 431, 543, 575,
592, 593, 845, 871, 911, 923, 952, 965, 1007, 1022, 1231, 1255, 1295, 1317,
1419, 1423, 1435, and 1482. This is an optional-template condition, not a bad
download. No page lacks the trafficking badge, `og:url`, case container,
case-detail container, title marker, database header, or English-locale signal.

No HTTP rate limit, anti-bot response, `Retry-After`, redirect, or HTTP error was
observed. The only interruption was the documented local Wi-Fi disconnection.

## HTML/template observations for parser v2

The parser was not modified in this task. Downloader validation and the existing
19-page manual sample audit support these recommendations for the subsequent
parser-v2 work:

1. Treat duplicate doctypes as a SHERLOC template artifact, not a broken
   download. All 1,590 production pages and all 19 manually saved samples have
   two doctype declarations. The raw duplication must remain untouched, and the
   parser should be tolerant of it.
2. Treat `.factSummary` as optional. Twenty-five production pages lack that
   marker. Absence should become an explicit availability state, not a download
   or parser failure.
3. Scope main-section heading searches to `#case-law-content` and direct
   `.case-law-detail` sections. A global heading search can select a trafficking
   sidebar `Keywords` heading before the legacy main-record Keywords section;
   similarly, a nested procedural `Court` heading can mask the main Court
   section.
4. Discover repeated-person containers structurally and preserve their source
   heading/role. The manual sample `Sentencia 298_2015.html` labels a
   `div.victimsPlaintiffs` block as `Migrants`; exact victim-heading matching
   loses the record, while silently relabeling it as a victim loses source
   meaning.
5. Preserve multilingual panes beyond Fact Summary. `Causa 2422.html` contains
   English/Spanish panes for commentary, procedural information, and
   sources/citations; concatenating translations would contaminate language-
   specific benchmark inputs.
6. Do not require parties, charges, decisions, court, Fact Summary, or a
   particular sidebar field for page-level validity. These case sections are
   heterogeneous and optional. Preserve section absent, section empty, parser
   error, and download error as different states.
7. Link every parsed record to download provenance: search rank, API identity,
   canonical/requested/final/`og:url`, raw filename, byte count, SHA-256, and
   download warnings. The `og:url` locale variation should be normalized for
   identity comparison but preserved verbatim as provenance.
8. Preserve every crime-type badge for audit, but continue to derive corpus
   membership solely from the frozen trafficking-filter manifest. A URL path or
   badge must never cause a cross-classified result to be dropped.

The manual samples ranged from 248,907 to 310,593 bytes (median 260,243), had 19
unique SHA-256 checksums, and all contained the downloader's required core
SHERLOC structural markers. Fact Summary and trafficking-specific content
remain warning-level/parse-level features rather than universal download gates.

## Outputs and reproduction

Primary outputs are:

- `src/sherloc/02_download_pages.py`;
- `logs/page_download_manifest.csv`;
- `logs/page_download_diagnostics.json`;
- ignored original HTML under `data/raw_html/`; and
- this report.

Run or resume the deterministic pilot with:

```bash
python3 src/sherloc/02_download_pages.py --mode pilot
```

After the pilot passes, run or resume the complete corpus with:

```bash
python3 src/sherloc/02_download_pages.py --mode full
```

Both commands default to the frozen manifest, one-second request spacing, and
resume-safe validation. `--force` intentionally redownloads selected existing
pages and should not be used for an ordinary resume.
