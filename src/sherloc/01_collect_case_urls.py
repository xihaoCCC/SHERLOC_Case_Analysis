#!/usr/bin/env python3
"""Discover SHERLOC trafficking-in-persons case URLs from the search API.

This script queries search-result metadata only.  It does not download or save
case-detail page bodies.  By default it walks the complete filtered result set,
validates a deterministic random sample with HEAD requests, and writes:

* data/manifests/case_urls.csv
* logs/url_discovery_diagnostics.json

Use ``--max-pages`` for a reconnaissance-only run.  Partial runs write
diagnostics but deliberately do not overwrite the production manifest.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import ssl
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlsplit
from urllib.request import HTTPSHandler, Request, build_opener


COLLECTOR_VERSION = "1.0.0"

SEARCH_PAGE_URL = (
    "https://www.unodc.org/cld/en/v3/sherloc/cldb/search.html"
)
DATA_ENDPOINT_URL = (
    "https://www.unodc.org/cld/en/v3/sherloc/cldb/data.json"
)
CANONICAL_ORIGIN = "https://www.unodc.org"
FILTER_FIELD = "en#__el.caseLaw.crimeTypes_s"
FILTER_VALUE = "Trafficking in persons"
DISCOVERY_METHOD = "sherloc_cldb_json_endpoint"
DEFAULT_USER_AGENT = (
    "SHERLOC-Case-Analysis-url-discovery/1.0 "
    "(academic research; search metadata and HEAD validation only)"
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "data" / "manifests" / "case_urls.csv"
DEFAULT_DIAGNOSTICS = REPO_ROOT / "logs" / "url_discovery_diagnostics.json"
DEFAULT_PARTIAL_DIAGNOSTICS = (
    REPO_ROOT / "logs" / "url_discovery_diagnostics_partial.json"
)

MANIFEST_FIELDS = [
    "search_rank",
    "case_title",
    "result_url",
    "canonical_url",
    "url_path_crime_type",
    "result_page_or_offset",
    "result_page_number",
    "discovery_method",
    "discovered_at",
    "api_result_uri",
    "api_result_id",
    "unodc_case_number",
    "is_canonical_duplicate",
    "duplicate_of_search_rank",
    "discovery_flags",
    "validation_status",
    "validation_http_status",
    "validation_final_url",
    "validation_error",
]

UNRESERVED = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~"
)
VALID_PERCENT_ESCAPE = re.compile(r"%[0-9A-Fa-f]{2}")
CONTROL_CHARACTER = re.compile(r"[\x00-\x1f\x7f]")
CRIME_TYPE_FROM_PATH = re.compile(r"^/case-law-doc/([^/]+)/")


class DiscoveryError(RuntimeError):
    """Raised when the search response cannot support a reliable manifest."""


@dataclass
class FetchResult:
    payload: Dict[str, Any]
    status: int
    final_url: str
    headers: Dict[str, str]
    elapsed_seconds: float
    response_bytes: int
    attempts: int


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def clean_title(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def normalize_percent_escapes(path: str) -> Tuple[str, List[str]]:
    """Normalize percent escapes while decoding only RFC 3986 unreserved bytes."""

    flags: List[str] = []
    output: List[str] = []
    index = 0
    while index < len(path):
        char = path[index]
        if char != "%":
            output.append(char)
            index += 1
            continue

        token = path[index : index + 3]
        if len(token) != 3 or not VALID_PERCENT_ESCAPE.fullmatch(token):
            flags.append("MALFORMED_PERCENT_ESCAPE")
            output.append("%25")
            index += 1
            continue

        byte_value = int(token[1:], 16)
        decoded = chr(byte_value)
        if decoded in UNRESERVED:
            output.append(decoded)
        else:
            output.append(f"%{byte_value:02X}")
        index += 3

    return "".join(output), flags


def canonicalize_result_uri(raw_uri: Any) -> Tuple[str, str, Optional[str], List[str]]:
    """Return result URL, canonical URL, URL-path crime type, and flags.

    SHERLOC's API returns locale-neutral paths such as
    ``/case-law-doc/traffickingpersonscrimetype/...html``.  The stable identity
    used here is ``https://www.unodc.org/cld`` plus that path.  ``result_url``
    retains the English UI presentation parameters while ``canonical_url``
    drops them.  Fragments, duplicate slashes, and the redirecting
    ``sherloc.unodc.org`` host are intentionally excluded.
    """

    flags: List[str] = []
    if not isinstance(raw_uri, str) or not raw_uri.strip():
        return "", "", None, ["MISSING_RESULT_URI"]

    raw = raw_uri.strip()
    if CONTROL_CHARACTER.search(raw):
        flags.append("CONTROL_CHARACTER_IN_URI")

    parsed = urlsplit(raw)
    if parsed.scheme or parsed.netloc:
        if parsed.scheme.lower() not in {"http", "https"}:
            flags.append("UNEXPECTED_URI_SCHEME")
        if parsed.hostname and parsed.hostname.lower() not in {
            "www.unodc.org",
            "sherloc.unodc.org",
        }:
            flags.append("UNEXPECTED_URI_HOST")
        path = parsed.path
    else:
        path = parsed.path

    path = path.replace("\\", "/")
    if "\\" in raw:
        flags.append("BACKSLASH_IN_URI")
    path = re.sub(r"/{2,}", "/", path)

    # Reduce all observed result forms to the locale-neutral API identity.
    for prefix in ("/cld/en", "/cld"):
        if path.startswith(prefix + "/case-law-doc/"):
            path = path[len(prefix) :]
            break

    if not path.startswith("/"):
        path = "/" + path

    segments = path.split("/")
    if any(segment in {".", ".."} for segment in segments):
        flags.append("DOT_SEGMENT_IN_URI")

    path, percent_flags = normalize_percent_escapes(path)
    flags.extend(percent_flags)
    if any(segment in {".", ".."} for segment in path.split("/")):
        flags.append("DOT_SEGMENT_IN_URI")
    path = quote(path, safe="/%:@!$&'()*+,;=-._~")

    match = CRIME_TYPE_FROM_PATH.match(path)
    crime_type = match.group(1) if match else None
    if not match:
        flags.append("UNEXPECTED_CASE_PATH")
    if not path.lower().endswith(".html"):
        flags.append("NON_HTML_RESULT_PATH")

    fatal_flags = {
        "MISSING_RESULT_URI",
        "CONTROL_CHARACTER_IN_URI",
        "UNEXPECTED_URI_SCHEME",
        "UNEXPECTED_URI_HOST",
        "DOT_SEGMENT_IN_URI",
        "MALFORMED_PERCENT_ESCAPE",
        "UNEXPECTED_CASE_PATH",
    }
    if fatal_flags.intersection(flags):
        return "", "", crime_type, sorted(set(flags))

    canonical_url = f"{CANONICAL_ORIGIN}/cld{path}"
    result_url = f"{canonical_url}?lng=en&tmpl=sherloc"
    return result_url, canonical_url, crime_type, sorted(set(flags))


def parse_retry_after(value: Optional[str]) -> float:
    if not value:
        return 0.0
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            parsed = parsedate_to_datetime(value)
            now = datetime.now(parsed.tzinfo or timezone.utc)
            return max(0.0, (parsed - now).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return 0.0


def resolve_ca_bundle(explicit: Optional[Path]) -> Optional[Path]:
    """Find a trusted CA bundle without ever disabling TLS verification."""

    if explicit is not None:
        candidate = explicit.expanduser().resolve()
        if not candidate.is_file():
            raise DiscoveryError(f"CA bundle does not exist: {candidate}")
        return candidate

    candidates: List[Optional[str]] = [
        os.environ.get("SSL_CERT_FILE"),
        ssl.get_default_verify_paths().cafile,
        "/etc/ssl/cert.pem",
        "/etc/ssl/certs/ca-certificates.crt",
        "/etc/pki/tls/certs/ca-bundle.crt",
    ]
    for raw_candidate in candidates:
        if raw_candidate:
            candidate = Path(raw_candidate)
            if candidate.is_file():
                return candidate
    return None


class PoliteHttpClient:
    """Sequential stdlib HTTP client with delay, retries, and diagnostics."""

    def __init__(
        self,
        *,
        delay_seconds: float,
        timeout_seconds: float,
        max_retries: int,
        backoff_base_seconds: float,
        user_agent: str,
        ca_bundle: Optional[Path],
        verbose: bool,
    ) -> None:
        self.delay_seconds = max(0.0, delay_seconds)
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.backoff_base_seconds = max(0.0, backoff_base_seconds)
        self.user_agent = user_agent
        self.verbose = verbose
        self.ca_bundle = resolve_ca_bundle(ca_bundle)
        ssl_context = ssl.create_default_context(
            cafile=str(self.ca_bundle) if self.ca_bundle else None
        )
        self.opener = build_opener(HTTPSHandler(context=ssl_context))
        self.last_request_finished_at: Optional[float] = None
        self.events: List[Dict[str, Any]] = []

    def _polite_wait(self) -> None:
        if self.last_request_finished_at is None:
            return
        elapsed = time.monotonic() - self.last_request_finished_at
        remaining = self.delay_seconds - elapsed
        if remaining > 0:
            time.sleep(remaining)

    def _request(
        self,
        request: Request,
        *,
        purpose: str,
        offset: Optional[int] = None,
        expect_json: bool = False,
    ) -> FetchResult:
        retriable_statuses = {408, 425, 429, 500, 502, 503, 504}
        last_error: Optional[BaseException] = None

        for attempt in range(1, self.max_retries + 2):
            self._polite_wait()
            started = time.monotonic()
            event: Dict[str, Any] = {
                "purpose": purpose,
                "offset": offset,
                "method": request.get_method(),
                "attempt": attempt,
                "started_at": utc_now(),
            }
            try:
                with self.opener.open(request, timeout=self.timeout_seconds) as response:
                    body = response.read()
                    elapsed = time.monotonic() - started
                    status = int(getattr(response, "status", response.getcode()))
                    headers = {key.lower(): value for key, value in response.headers.items()}
                    final_url = response.geturl()
                    event.update(
                        {
                            "status": status,
                            "elapsed_seconds": round(elapsed, 3),
                            "response_bytes": len(body),
                            "content_type": headers.get("content-type", ""),
                            "final_url": final_url,
                        }
                    )
                    self.events.append(event)
                    self.last_request_finished_at = time.monotonic()

                    if expect_json:
                        try:
                            payload = json.loads(body.decode("utf-8"))
                        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                            raise DiscoveryError(
                                f"Expected JSON for offset {offset}, got "
                                f"{headers.get('content-type', 'unknown content type')}"
                            ) from exc
                        if not isinstance(payload, dict):
                            raise DiscoveryError(
                                f"Expected a JSON object for offset {offset}"
                            )
                    else:
                        payload = {}

                    return FetchResult(
                        payload=payload,
                        status=status,
                        final_url=final_url,
                        headers=headers,
                        elapsed_seconds=elapsed,
                        response_bytes=len(body),
                        attempts=attempt,
                    )

            except HTTPError as exc:
                elapsed = time.monotonic() - started
                retry_after = exc.headers.get("Retry-After") if exc.headers else None
                event.update(
                    {
                        "status": int(exc.code),
                        "elapsed_seconds": round(elapsed, 3),
                        "error": str(exc),
                        "retry_after": retry_after,
                    }
                )
                self.events.append(event)
                self.last_request_finished_at = time.monotonic()
                last_error = exc
                if exc.code not in retriable_statuses or attempt > self.max_retries:
                    raise
                retry_delay = max(
                    parse_retry_after(retry_after),
                    self.backoff_base_seconds * (2 ** (attempt - 1)),
                )
            except (URLError, TimeoutError, OSError) as exc:
                elapsed = time.monotonic() - started
                event.update(
                    {
                        "status": None,
                        "elapsed_seconds": round(elapsed, 3),
                        "error": repr(exc),
                    }
                )
                self.events.append(event)
                self.last_request_finished_at = time.monotonic()
                last_error = exc
                if attempt > self.max_retries:
                    raise
                retry_delay = self.backoff_base_seconds * (2 ** (attempt - 1))

            if self.verbose:
                print(
                    f"Retrying {purpose} after {retry_delay:.1f}s "
                    f"(attempt {attempt + 1})",
                    file=sys.stderr,
                )
            time.sleep(retry_delay)

        assert last_error is not None
        raise last_error

    def get_search_page(self, offset: int) -> FetchResult:
        criteria = {
            "filters": [{"fieldName": FILTER_FIELD, "value": FILTER_VALUE}],
            "startAt": offset,
            "sortings": "",
        }
        query = urlencode({"lng": "en", "criteria": compact_json(criteria)})
        request = Request(
            f"{DATA_ENDPOINT_URL}?{query}",
            headers={
                "Accept": "application/json",
                "User-Agent": self.user_agent,
            },
            method="GET",
        )
        return self._request(
            request,
            purpose="search_metadata",
            offset=offset,
            expect_json=True,
        )

    def head(self, url: str, search_rank: int) -> FetchResult:
        request = Request(
            url,
            headers={
                "Accept": "text/html,application/xhtml+xml",
                "User-Agent": self.user_agent,
            },
            method="HEAD",
        )
        return self._request(
            request,
            purpose="sample_url_head_validation",
            offset=search_rank,
            expect_json=False,
        )


def require_nonnegative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise DiscoveryError(f"{field_name} must be an integer, not bool")
    try:
        converted = int(value)
    except (TypeError, ValueError) as exc:
        raise DiscoveryError(f"{field_name} is not an integer: {value!r}") from exc
    if converted < 0:
        raise DiscoveryError(f"{field_name} is negative: {converted}")
    return converted


def find_filter_facet_count(payload: Dict[str, Any]) -> Optional[int]:
    for facet in payload.get("facetFields") or []:
        if not isinstance(facet, dict) or facet.get("name") != FILTER_FIELD:
            continue
        for entry in facet.get("values") or []:
            if isinstance(entry, dict) and entry.get("value") == FILTER_VALUE:
                try:
                    return int(entry.get("count"))
                except (TypeError, ValueError):
                    return None
    return None


def validate_applied_filter(payload: Dict[str, Any]) -> List[str]:
    flags: List[str] = []
    criteria = payload.get("criteria")
    if not isinstance(criteria, dict):
        return ["INITIAL_RESPONSE_MISSING_CRITERIA"]
    filters = criteria.get("filters")
    if not isinstance(filters, list):
        return ["INITIAL_RESPONSE_MISSING_FILTERS"]
    exact = [
        item
        for item in filters
        if isinstance(item, dict)
        and item.get("fieldName") == FILTER_FIELD
        and item.get("value") == FILTER_VALUE
    ]
    if len(exact) != 1:
        flags.append("TRAFFICKING_FILTER_NOT_ECHOED_EXACTLY_ONCE")
    if len(filters) != 1:
        flags.append("UNEXPECTED_ADDITIONAL_FILTERS")
    return flags


def result_to_record(
    result: Any,
    *,
    search_rank: int,
    offset: int,
    page_number: int,
    discovered_at: str,
) -> Dict[str, Any]:
    flags: List[str] = []
    if not isinstance(result, dict):
        result = {}
        flags.append("RESULT_NOT_OBJECT")
    values = result.get("values")
    if not isinstance(values, dict):
        values = {}
        flags.append("MISSING_VALUES_OBJECT")

    top_uri = result.get("uri")
    values_uri = values.get("uri")
    if top_uri and values_uri and top_uri != values_uri:
        flags.append("URI_FIELDS_DISAGREE")
    raw_uri = values_uri or top_uri
    result_url, canonical_url, crime_type, url_flags = canonicalize_result_uri(raw_uri)
    flags.extend(url_flags)

    title = clean_title(values.get("page_title"))
    if not title:
        flags.append("MISSING_CASE_TITLE")

    return {
        "search_rank": search_rank,
        "case_title": title,
        "result_url": result_url,
        "canonical_url": canonical_url,
        "url_path_crime_type": crime_type or "",
        "result_page_or_offset": offset,
        "result_page_number": page_number,
        "discovery_method": DISCOVERY_METHOD,
        "discovered_at": discovered_at,
        "api_result_uri": raw_uri if isinstance(raw_uri, str) else "",
        "api_result_id": clean_title(result.get("id")),
        "unodc_case_number": clean_title(values.get("caseLaw@unodcNo_s1")),
        "is_canonical_duplicate": "false",
        "duplicate_of_search_rank": "",
        "discovery_flags": "|".join(sorted(set(flags))),
        "validation_status": "NOT_SAMPLED",
        "validation_http_status": "",
        "validation_final_url": "",
        "validation_error": "",
    }


def mark_duplicates(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    first_rank_by_url: Dict[str, int] = {}
    duplicate_groups: Dict[str, List[int]] = defaultdict(list)
    for record in records:
        canonical = str(record.get("canonical_url") or "")
        if not canonical:
            continue
        rank = int(record["search_rank"])
        duplicate_groups[canonical].append(rank)
        if canonical in first_rank_by_url:
            record["is_canonical_duplicate"] = "true"
            record["duplicate_of_search_rank"] = first_rank_by_url[canonical]
            flags = set(filter(None, str(record["discovery_flags"]).split("|")))
            flags.add("CANONICAL_DUPLICATE")
            record["discovery_flags"] = "|".join(sorted(flags))
        else:
            first_rank_by_url[canonical] = rank

    return [
        {"canonical_url": url, "search_ranks": ranks, "occurrences": len(ranks)}
        for url, ranks in duplicate_groups.items()
        if len(ranks) > 1
    ]


def find_identity_duplicates(
    records: List[Dict[str, Any]], field_name: str, flag_name: str
) -> List[Dict[str, Any]]:
    """Find repeated non-empty API identities and flag every involved row."""

    ranks_by_value: Dict[str, List[int]] = defaultdict(list)
    record_by_rank: Dict[int, Dict[str, Any]] = {}
    for record in records:
        rank = int(record["search_rank"])
        record_by_rank[rank] = record
        value = str(record.get(field_name) or "")
        if value:
            ranks_by_value[value].append(rank)

    groups = [
        {"value": value, "search_ranks": ranks, "occurrences": len(ranks)}
        for value, ranks in ranks_by_value.items()
        if len(ranks) > 1
    ]
    for group in groups:
        for rank in group["search_ranks"]:
            record = record_by_rank[int(rank)]
            flags = set(filter(None, str(record["discovery_flags"]).split("|")))
            flags.add(flag_name)
            record["discovery_flags"] = "|".join(sorted(flags))
    return groups


def validate_sample_urls(
    records: List[Dict[str, Any]],
    *,
    client: PoliteHttpClient,
    sample_size: int,
    seed: int,
    verbose: bool,
) -> Dict[str, Any]:
    first_by_url: Dict[str, Dict[str, Any]] = {}
    for record in records:
        canonical = str(record.get("canonical_url") or "")
        if canonical and canonical not in first_by_url:
            first_by_url[canonical] = record

    population = list(first_by_url.values())
    count = min(max(0, sample_size), len(population))
    selected = random.Random(seed).sample(population, count) if count else []
    selected.sort(key=lambda row: int(row["search_rank"]))
    status_counts: Counter[str] = Counter()
    details: List[Dict[str, Any]] = []

    for index, record in enumerate(selected, start=1):
        rank = int(record["search_rank"])
        url = str(record["canonical_url"])
        if verbose:
            print(
                f"HEAD validation {index}/{count}: rank {rank} {url}",
                file=sys.stderr,
                flush=True,
            )
        try:
            response = client.head(url, rank)
            status = response.status
            final_url = response.final_url
            if 200 <= status < 300:
                validation_status = "HTTP_OK"
            elif 300 <= status < 400:
                validation_status = "HTTP_REDIRECT"
            else:
                validation_status = "HTTP_ERROR"
            error = ""
        except HTTPError as exc:
            status = int(exc.code)
            final_url = exc.geturl()
            validation_status = "HTTP_ERROR"
            error = str(exc)
        except (URLError, TimeoutError, OSError) as exc:
            status = None
            final_url = ""
            validation_status = "NETWORK_ERROR"
            error = repr(exc)

        status_counts[validation_status] += 1
        for matching in records:
            if matching.get("canonical_url") == url:
                matching["validation_status"] = validation_status
                matching["validation_http_status"] = status or ""
                matching["validation_final_url"] = final_url
                matching["validation_error"] = error
                if validation_status != "HTTP_OK":
                    flags = set(
                        filter(None, str(matching["discovery_flags"]).split("|"))
                    )
                    flags.add("SAMPLED_URL_VALIDATION_FAILED")
                    matching["discovery_flags"] = "|".join(sorted(flags))

        details.append(
            {
                "search_rank": rank,
                "case_title": record["case_title"],
                "canonical_url": url,
                "url_path_crime_type": record["url_path_crime_type"],
                "status": validation_status,
                "http_status": status,
                "final_url": final_url,
                "error": error or None,
            }
        )

    return {
        "method": "HEAD",
        "seed": seed,
        "population_unique_canonical_urls": len(population),
        "requested_sample_size": sample_size,
        "actual_sample_size": count,
        "status_counts": dict(sorted(status_counts.items())),
        "records": details,
    }


def summarize_http_events(events: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    status_counts: Counter[str] = Counter()
    retry_count = 0
    retry_after_count = 0
    elapsed: List[float] = []
    bytes_total = 0
    for event in events:
        status = event.get("status")
        status_counts[str(status) if status is not None else "NETWORK_ERROR"] += 1
        if int(event.get("attempt", 1)) > 1:
            retry_count += 1
        if event.get("retry_after"):
            retry_after_count += 1
        if isinstance(event.get("elapsed_seconds"), (int, float)):
            elapsed.append(float(event["elapsed_seconds"]))
        if isinstance(event.get("response_bytes"), int):
            bytes_total += int(event["response_bytes"])

    return {
        "request_attempts": len(events),
        "status_counts": dict(sorted(status_counts.items())),
        "retry_attempts": retry_count,
        "retry_after_headers_seen": retry_after_count,
        "http_403_seen": status_counts.get("403", 0),
        "http_429_seen": status_counts.get("429", 0),
        "response_bytes_total": bytes_total,
        "elapsed_seconds_min": round(min(elapsed), 3) if elapsed else None,
        "elapsed_seconds_max": round(max(elapsed), 3) if elapsed else None,
        "elapsed_seconds_mean": (
            round(sum(elapsed) / len(elapsed), 3) if elapsed else None
        ),
    }


def stable_manifest_hash(records: Sequence[Dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(str(record.get("search_rank", "")).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(record.get("canonical_url", "")).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def atomic_write_csv(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=False)
        handle.write("\n")
    os.replace(temporary, path)


def _collect_with_client(
    args: argparse.Namespace,
    client: PoliteHttpClient,
    started_at: str,
    discovered_at: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], bool]:
    records: List[Dict[str, Any]] = []
    page_observations: List[Dict[str, Any]] = []
    found_values: List[int] = []
    initial_payload: Optional[Dict[str, Any]] = None
    initial_result_uris: List[str] = []
    stop_reason = ""
    offset = 0
    page_number = 0

    while True:
        if args.max_pages is not None and page_number >= args.max_pages:
            stop_reason = "max_pages_reached"
            break

        page_number += 1
        if args.verbose:
            print(
                f"Fetching search metadata page {page_number} at offset {offset}",
                file=sys.stderr,
                flush=True,
            )
        fetch = client.get_search_page(offset)
        payload = fetch.payload
        found = require_nonnegative_int(payload.get("found"), "response.found")
        found_values.append(found)
        results = payload.get("results")
        if not isinstance(results, list):
            raise DiscoveryError(f"response.results is not a list at offset {offset}")

        if initial_payload is None:
            initial_payload = payload
            initial_result_uris = [
                str((item.get("values") or {}).get("uri") or item.get("uri") or "")
                for item in results
                if isinstance(item, dict)
            ]

        page_observations.append(
            {
                "page_number": page_number,
                "offset": offset,
                "found": found,
                "returned": len(results),
                "http_status": fetch.status,
                "elapsed_seconds": round(fetch.elapsed_seconds, 3),
                "response_bytes": fetch.response_bytes,
                "attempts": fetch.attempts,
            }
        )

        for result in results:
            records.append(
                result_to_record(
                    result,
                    search_rank=len(records) + 1,
                    offset=offset,
                    page_number=page_number,
                    discovered_at=discovered_at,
                )
            )

        if not results:
            stop_reason = "empty_page"
            break
        if len(records) >= found:
            stop_reason = "returned_total_reached"
            break

        next_offset = len(records)
        if next_offset <= offset:
            raise DiscoveryError(
                f"Pagination made no progress: offset {offset}, records {len(records)}"
            )
        offset = next_offset

    if initial_payload is None:
        raise DiscoveryError("No initial response was obtained")

    initial_total = found_values[0]
    total_consistent = len(set(found_values)) == 1
    full_retrieval = (
        args.max_pages is None
        and stop_reason == "returned_total_reached"
        and len(records) == initial_total
        and total_consistent
    )

    boundary_check: Optional[Dict[str, Any]] = None
    snapshot_check: Optional[Dict[str, Any]] = None
    if full_retrieval:
        boundary_fetch = client.get_search_page(initial_total)
        boundary_payload = boundary_fetch.payload
        boundary_results = boundary_payload.get("results")
        boundary_found = require_nonnegative_int(
            boundary_payload.get("found"), "boundary_response.found"
        )
        boundary_check = {
            "offset": initial_total,
            "found": boundary_found,
            "returned": len(boundary_results) if isinstance(boundary_results, list) else None,
            "http_status": boundary_fetch.status,
            "passed": (
                isinstance(boundary_results, list)
                and len(boundary_results) == 0
                and boundary_found == initial_total
            ),
        }

        snapshot_fetch = client.get_search_page(0)
        snapshot_payload = snapshot_fetch.payload
        snapshot_results = snapshot_payload.get("results")
        snapshot_found = require_nonnegative_int(
            snapshot_payload.get("found"), "snapshot_response.found"
        )
        snapshot_uris = [
            str((item.get("values") or {}).get("uri") or item.get("uri") or "")
            for item in (snapshot_results if isinstance(snapshot_results, list) else [])
            if isinstance(item, dict)
        ]
        snapshot_check = {
            "offset": 0,
            "found": snapshot_found,
            "returned": len(snapshot_results) if isinstance(snapshot_results, list) else None,
            "first_page_uris_match": snapshot_uris == initial_result_uris,
            "http_status": snapshot_fetch.status,
            "passed": (
                isinstance(snapshot_results, list)
                and snapshot_found == initial_total
                and snapshot_uris == initial_result_uris
            ),
        }

        full_retrieval = bool(
            full_retrieval
            and boundary_check["passed"]
            and snapshot_check["passed"]
        )

    duplicate_groups = mark_duplicates(records)
    api_id_duplicate_groups = find_identity_duplicates(
        records, "api_result_id", "DUPLICATE_API_RESULT_ID"
    )
    api_uri_duplicate_groups = find_identity_duplicates(
        records, "api_result_uri", "DUPLICATE_API_RESULT_URI"
    )
    unique_canonical_urls = len(
        {record["canonical_url"] for record in records if record["canonical_url"]}
    )
    unique_api_result_ids = len(
        {record["api_result_id"] for record in records if record["api_result_id"]}
    )
    unique_api_result_uris = len(
        {record["api_result_uri"] for record in records if record["api_result_uri"]}
    )
    missing_canonical_count = sum(
        1 for record in records if not record["canonical_url"]
    )
    path_counts = Counter(
        record["url_path_crime_type"] or "[missing_or_malformed]"
        for record in records
    )

    sample_validation: Dict[str, Any]
    if full_retrieval and not args.skip_url_validation:
        sample_validation = validate_sample_urls(
            records,
            client=client,
            sample_size=args.validation_sample_size,
            seed=args.random_seed,
            verbose=args.verbose,
        )
    else:
        sample_validation = {
            "method": "HEAD",
            "seed": args.random_seed,
            "requested_sample_size": args.validation_sample_size,
            "actual_sample_size": 0,
            "status_counts": {},
            "records": [],
            "skipped_reason": (
                "--skip-url-validation"
                if args.skip_url_validation
                else "retrieval_not_complete"
            ),
        }

    sample_failures = sum(
        count
        for status, count in sample_validation.get("status_counts", {}).items()
        if status != "HTTP_OK"
    )
    facet_total = find_filter_facet_count(initial_payload)
    applied_filter_flags = validate_applied_filter(initial_payload)
    validation_completed = bool(
        not args.skip_url_validation
        and sample_validation.get("actual_sample_size", 0) > 0
    )
    publishable = bool(
        full_retrieval
        and validation_completed
        and sample_failures == 0
        and not applied_filter_flags
        and (facet_total is None or facet_total == initial_total)
        and not api_id_duplicate_groups
        and not api_uri_duplicate_groups
    )
    normalized_sort = (initial_payload.get("criteria") or {}).get("sortings")
    flagged_records = [record for record in records if record["discovery_flags"]]
    finished_at = utc_now()
    http_summary = summarize_http_events(client.events)
    diagnostics: Dict[str, Any] = {
        "schema_version": "1.0",
        "collector_version": COLLECTOR_VERSION,
        "run": {
            "started_at": started_at,
            "finished_at": finished_at,
            "discovered_at_used_in_manifest": discovered_at,
            "complete_result_walk": full_retrieval,
            "sample_url_validation_completed": validation_completed,
            "publishable_manifest": publishable,
            "stop_reason": stop_reason,
            "max_pages": args.max_pages,
            "delay_seconds": args.delay_seconds,
            "timeout_seconds": args.timeout_seconds,
            "max_retries": args.max_retries,
            "backoff_base_seconds": args.backoff_base_seconds,
            "user_agent": args.user_agent,
            "tls_ca_bundle": str(client.ca_bundle) if client.ca_bundle else None,
        },
        "discovery": {
            "method": DISCOVERY_METHOD,
            "search_page_url": SEARCH_PAGE_URL,
            "data_endpoint_url": DATA_ENDPOINT_URL,
            "request_method": "GET",
            "query_parameters": {
                "lng": "en",
                "criteria": {
                    "filters": [
                        {"fieldName": FILTER_FIELD, "value": FILTER_VALUE}
                    ],
                    "startAt": "<zero-based result offset>",
                    "sortings": "",
                },
            },
            "applied_filter_flags": applied_filter_flags,
            "server_normalized_sort": normalized_sort,
            "results_embedded_in_static_search_html": False,
            "browser_or_playwright_required": False,
            "ui_pagination": "automatic infinite scroll / load-more",
            "endpoint_pagination": "criteria.startAt zero-based offset",
            "page_size_parameter_exposed": False,
            "detected_page_size": len(initial_payload.get("results") or []),
            "title_field": "results[].values.page_title",
            "uri_fields": ["results[].uri", "results[].values.uri"],
            "note": (
                "The endpoint also returns Fact Summary HTML in result values; "
                "the collector deliberately does not persist it."
            ),
        },
        "counts": {
            "initial_returned_total": initial_total,
            "trafficking_facet_count": facet_total,
            "total_values_consistent_across_pages": total_consistent,
            "result_rows_retrieved": len(records),
            "unique_canonical_urls": unique_canonical_urls,
            "unique_api_result_ids": unique_api_result_ids,
            "unique_api_result_uris": unique_api_result_uris,
            "canonical_duplicate_rows": sum(
                len(group["search_ranks"]) - 1 for group in duplicate_groups
            ),
            "canonical_duplicate_groups": len(duplicate_groups),
            "api_result_id_duplicate_rows": sum(
                len(group["search_ranks"]) - 1
                for group in api_id_duplicate_groups
            ),
            "api_result_id_duplicate_groups": len(api_id_duplicate_groups),
            "api_result_uri_duplicate_rows": sum(
                len(group["search_ranks"]) - 1
                for group in api_uri_duplicate_groups
            ),
            "api_result_uri_duplicate_groups": len(api_uri_duplicate_groups),
            "missing_or_malformed_canonical_urls": missing_canonical_count,
            "flagged_result_rows": len(flagged_records),
            "counts_match_returned_total": len(records) == initial_total,
            "facet_matches_returned_total": (
                facet_total == initial_total if facet_total is not None else None
            ),
        },
        "url_path_crime_type_counts": dict(sorted(path_counts.items())),
        "non_trafficking_url_path_count": sum(
            count
            for path_type, count in path_counts.items()
            if path_type != "traffickingpersonscrimetype"
        ),
        "duplicates": duplicate_groups,
        "api_result_id_duplicates": api_id_duplicate_groups,
        "api_result_uri_duplicates": api_uri_duplicate_groups,
        "flagged_results": [
            {
                "search_rank": record["search_rank"],
                "case_title": record["case_title"],
                "api_result_uri": record["api_result_uri"],
                "canonical_url": record["canonical_url"],
                "flags": str(record["discovery_flags"]).split("|"),
            }
            for record in flagged_records
        ],
        "page_observations": page_observations,
        "boundary_check": boundary_check,
        "snapshot_check": snapshot_check,
        "random_url_validation": sample_validation,
        "http": {
            **http_summary,
            "anti_bot_or_rate_limit_observed": bool(
                http_summary["http_403_seen"]
                or http_summary["http_429_seen"]
                or http_summary["retry_after_headers_seen"]
            ),
        },
        "request_events": client.events,
        "manifest": {
            "path": str(args.manifest_path),
            "rows_preserve_all_result_hits": True,
            "duplicate_rows_are_flagged_not_dropped": True,
            "rank_and_canonical_url_sha256": stable_manifest_hash(records),
            "written": False,
        },
    }

    return records, diagnostics, publishable


def collect(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], Dict[str, Any], bool]:
    """Run discovery and preserve request evidence even if the run fails."""

    started_at = utc_now()
    discovered_at = started_at
    client = PoliteHttpClient(
        delay_seconds=args.delay_seconds,
        timeout_seconds=args.timeout_seconds,
        max_retries=args.max_retries,
        backoff_base_seconds=args.backoff_base_seconds,
        user_agent=args.user_agent,
        ca_bundle=args.ca_bundle,
        verbose=args.verbose,
    )
    try:
        return _collect_with_client(args, client, started_at, discovered_at)
    except Exception as exc:
        http_summary = summarize_http_events(client.events)
        failure_diagnostics: Dict[str, Any] = {
            "schema_version": "1.0",
            "collector_version": COLLECTOR_VERSION,
            "run": {
                "started_at": started_at,
                "failed_at": utc_now(),
                "complete_result_walk": False,
                "publishable_manifest": False,
                "status": "FAILED",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "max_pages": args.max_pages,
                "delay_seconds": args.delay_seconds,
                "timeout_seconds": args.timeout_seconds,
                "max_retries": args.max_retries,
                "backoff_base_seconds": args.backoff_base_seconds,
                "user_agent": args.user_agent,
                "tls_ca_bundle": str(client.ca_bundle) if client.ca_bundle else None,
            },
            "discovery": {
                "method": DISCOVERY_METHOD,
                "search_page_url": SEARCH_PAGE_URL,
                "data_endpoint_url": DATA_ENDPOINT_URL,
                "request_method": "GET",
                "query_parameters": {
                    "lng": "en",
                    "criteria": {
                        "filters": [
                            {"fieldName": FILTER_FIELD, "value": FILTER_VALUE}
                        ],
                        "startAt": "<zero-based result offset>",
                        "sortings": "",
                    },
                },
            },
            "http": {
                **http_summary,
                "anti_bot_or_rate_limit_observed": bool(
                    http_summary["http_403_seen"]
                    or http_summary["http_429_seen"]
                    or http_summary["retry_after_headers_seen"]
                ),
            },
            "request_events": client.events,
            "manifest": {
                "path": str(args.manifest_path),
                "written": False,
                "not_written_reason": "discovery run failed",
            },
        }
        try:
            atomic_write_json(args.diagnostics_path, failure_diagnostics)
        except OSError as diagnostics_error:
            print(
                f"Could not write failure diagnostics: {diagnostics_error}",
                file=sys.stderr,
            )
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Collect URLs and titles from the SHERLOC Trafficking in persons "
            "filtered JSON result set; never downloads case-page bodies."
        )
    )
    parser.add_argument(
        "--manifest",
        dest="manifest_path",
        type=Path,
        default=DEFAULT_MANIFEST,
        help=f"production CSV path (default: {DEFAULT_MANIFEST})",
    )
    parser.add_argument(
        "--diagnostics",
        dest="diagnostics_path",
        type=Path,
        default=DEFAULT_DIAGNOSTICS,
        help=f"diagnostics JSON path (default: {DEFAULT_DIAGNOSTICS})",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help=(
            "reconnaissance limit; partial runs do not overwrite the manifest or "
            "production diagnostics"
        ),
    )
    parser.add_argument("--delay-seconds", type=float, default=1.0)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--backoff-base-seconds", type=float, default=2.0)
    parser.add_argument("--validation-sample-size", type=int, default=20)
    parser.add_argument("--random-seed", type=int, default=20260809)
    parser.add_argument(
        "--ca-bundle",
        type=Path,
        default=None,
        help=(
            "trusted TLS CA bundle; auto-detected from Python/system paths by default"
        ),
    )
    parser.add_argument(
        "--skip-url-validation",
        action="store_true",
        help="skip HEAD validation; a production manifest will not be published",
    )
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT)
    parser.add_argument("--verbose", action="store_true")
    return parser


def validate_arguments(args: argparse.Namespace) -> None:
    if args.max_pages is not None and args.max_pages < 1:
        raise SystemExit("--max-pages must be at least 1")
    if args.delay_seconds < 0:
        raise SystemExit("--delay-seconds cannot be negative")
    if args.timeout_seconds <= 0:
        raise SystemExit("--timeout-seconds must be positive")
    if args.max_retries < 0:
        raise SystemExit("--max-retries cannot be negative")
    if args.backoff_base_seconds < 0:
        raise SystemExit("--backoff-base-seconds cannot be negative")
    if args.validation_sample_size < 0:
        raise SystemExit("--validation-sample-size cannot be negative")
    if args.max_pages is not None and args.diagnostics_path == DEFAULT_DIAGNOSTICS:
        args.diagnostics_path = DEFAULT_PARTIAL_DIAGNOSTICS


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_arguments(args)

    try:
        records, diagnostics, publishable = collect(args)
    except (DiscoveryError, HTTPError, URLError, TimeoutError, OSError) as exc:
        print(f"URL discovery failed: {exc}", file=sys.stderr)
        return 1

    if publishable:
        atomic_write_csv(args.manifest_path, records)
        diagnostics["manifest"]["written"] = True
    else:
        diagnostics["manifest"]["not_written_reason"] = (
            "partial retrieval or sampled URL validation failure"
        )

    atomic_write_json(args.diagnostics_path, diagnostics)

    counts = diagnostics["counts"]
    print(
        "SHERLOC URL discovery: "
        f"total={counts['initial_returned_total']}, "
        f"retrieved={counts['result_rows_retrieved']}, "
        f"unique={counts['unique_canonical_urls']}, "
        f"duplicate_rows={counts['canonical_duplicate_rows']}, "
        f"manifest_written={diagnostics['manifest']['written']}"
    )
    print(f"Diagnostics: {args.diagnostics_path}")
    if diagnostics["manifest"]["written"]:
        print(f"Manifest: {args.manifest_path}")
        return 0
    return 2 if args.max_pages is None else 0


if __name__ == "__main__":
    raise SystemExit(main())
