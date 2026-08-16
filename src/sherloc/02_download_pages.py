#!/usr/bin/env python3
"""Download and validate SHERLOC case-detail HTML without modifying its bytes.

Corpus membership comes exclusively from ``data/manifests/case_urls.csv``.
The default mode is an 80-case deterministic pilot.  Use ``--mode full`` only
after that pilot succeeds.  Downloads, state, and diagnostics are resume-safe;
validated files are skipped unless ``--force`` is supplied.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
import re
import ssl
import statistics
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from http.client import HTTPException
from http.cookiejar import CookieJar
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import quote, unquote, urljoin, urlsplit
from urllib.request import (
    HTTPCookieProcessor,
    HTTPRedirectHandler,
    HTTPSHandler,
    Request,
    build_opener,
)


DOWNLOADER_VERSION = "1.0.0"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_MANIFEST = REPO_ROOT / "data" / "manifests" / "case_urls.csv"
DEFAULT_RAW_DIR = REPO_ROOT / "data" / "raw_html"
DEFAULT_DOWNLOAD_MANIFEST = REPO_ROOT / "logs" / "page_download_manifest.csv"
DEFAULT_DIAGNOSTICS = REPO_ROOT / "logs" / "page_download_diagnostics.json"

DEFAULT_PILOT_SIZE = 80
DEFAULT_RANK_BANDS = 10
PRIMARY_PATH_TYPE = "traffickingpersonscrimetype"
ALLOWED_HOSTS = {"www.unodc.org", "sherloc.unodc.org"}
SUCCESS_STATUS = "HTTP_OK_VALID"
TRANSIENT_HTTP_STATUSES = {408, 425, 429, 500, 502, 503, 504}
HARD_MINIMUM_BYTES = 10_000
SMALL_PAGE_WARNING_BYTES = 100_000
LARGE_PAGE_WARNING_BYTES = 2_000_000
USER_AGENT = (
    "SHERLOC-Case-Analysis-page-downloader/1.0 "
    "(authorized academic research; sequential case retrieval)"
)

SOURCE_REQUIRED_FIELDS = {
    "search_rank",
    "case_title",
    "result_url",
    "canonical_url",
    "url_path_crime_type",
    "api_result_id",
    "unodc_case_number",
}

DOWNLOAD_FIELDS = [
    "search_rank",
    "case_title",
    "url_path_crime_type",
    "api_result_id",
    "unodc_case_number",
    "canonical_url",
    "requested_url",
    "final_url",
    "final_url_relation",
    "og_url",
    "og_url_relation",
    "http_status",
    "content_type",
    "content_encoding",
    "download_timestamp",
    "raw_filename",
    "byte_count",
    "sha256",
    "download_status",
    "attempts",
    "redirect_count",
    "redirect_chain",
    "elapsed_seconds",
    "pilot_selected",
    "pilot_selection_reasons",
    "last_action",
    "last_checked_at",
    "structural_markers",
    "warnings",
    "error",
    "source_manifest_discovered_at",
]

MUTABLE_DOWNLOAD_FIELDS = {
    "final_url",
    "final_url_relation",
    "og_url",
    "og_url_relation",
    "http_status",
    "content_type",
    "content_encoding",
    "download_timestamp",
    "raw_filename",
    "byte_count",
    "sha256",
    "download_status",
    "attempts",
    "redirect_count",
    "redirect_chain",
    "elapsed_seconds",
    "last_action",
    "last_checked_at",
    "structural_markers",
    "warnings",
    "error",
}

UNRESERVED = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~"
)
YEAR_SEGMENT = re.compile(r"/(19\d{2}|20\d{2})/")
PAGE_LOCALE_EN = re.compile(
    r"(?:const|var)\s+pageLocale\s*=\s*['\"]en['\"]", re.IGNORECASE
)


class DownloadError(RuntimeError):
    """Raised for invalid input/state or an unsafe download condition."""


@dataclass
class PageValidation:
    valid: bool
    og_url: str
    document_title: str
    case_title: str
    final_url_relation: str
    og_url_relation: str
    markers: Dict[str, Any]
    warnings: List[str]
    errors: List[str]


@dataclass
class HttpOutcome:
    body: bytes
    status: Optional[int]
    final_url: str
    headers: Dict[str, str]
    attempts: int
    redirect_chain: List[Dict[str, Any]]
    elapsed_seconds: float
    validation: Optional[PageValidation]
    error: str


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def file_timestamp(path: Path) -> str:
    return (
        datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def sha256_bytes(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_text(value: str) -> str:
    return unicodedata.normalize("NFKC", " ".join(html.unescape(value).split()))


def split_warnings(value: Any) -> List[str]:
    return [item for item in str(value or "").split("|") if item]


def join_warnings(values: Iterable[str]) -> str:
    return "|".join(sorted(set(item for item in values if item)))


def parse_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def extract_path_year(url: str) -> Optional[int]:
    match = YEAR_SEGMENT.search(urlsplit(url).path)
    return int(match.group(1)) if match else None


def normalize_percent_path(path: str) -> str:
    output: List[str] = []
    index = 0
    while index < len(path):
        if path[index] != "%" or index + 2 >= len(path):
            output.append(path[index])
            index += 1
            continue
        token = path[index + 1 : index + 3]
        try:
            byte_value = int(token, 16)
        except ValueError:
            output.append("%25")
            index += 1
            continue
        decoded = chr(byte_value)
        output.append(decoded if decoded in UNRESERVED else f"%{byte_value:02X}")
        index += 3
    return quote("".join(output), safe="/%:@!$&'()*+,;=-._~")


def canonical_case_identity(url: str, *, base_url: str = "") -> Optional[str]:
    if not url:
        return None
    absolute = urljoin(base_url, url) if base_url else url
    parsed = urlsplit(absolute)
    if parsed.scheme.lower() != "https" or not parsed.hostname:
        return None
    try:
        port = parsed.port
    except ValueError:
        return None
    if port not in {None, 443}:
        return None
    host = parsed.hostname.lower()
    if host == "sherloc.unodc.org":
        host = "www.unodc.org"
    if host not in ALLOWED_HOSTS:
        return None
    path = re.sub(r"/{2,}", "/", parsed.path.replace("\\", "/"))
    path = re.sub(
        r"^/cld/(?:ar|zh|en|fr|ru|es)(?=/case-law-doc/)",
        "/cld",
        path,
        flags=re.IGNORECASE,
    )
    path = normalize_percent_path(path).rstrip("/")
    return f"www.unodc.org{path}"


def url_relation(actual: str, canonical: str, *, base_url: str = "") -> str:
    if not actual:
        return "MISSING"
    if actual == canonical:
        return "EXACT_MATCH"
    actual_identity = canonical_case_identity(actual, base_url=base_url)
    canonical_identity = canonical_case_identity(canonical)
    if actual_identity and actual_identity == canonical_identity:
        return "CANONICAL_EQUIVALENT"
    return "MISMATCH"


class CaseHtmlProbe(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.seen_html = False
        self.html_lang = ""
        self.ids: Set[str] = set()
        self.classes: Set[str] = set()
        self.og_url = ""
        self.document_title_parts: List[str] = []
        self.case_title_parts: List[str] = []
        self.in_document_title = False
        self.h2_depth = 0
        self.in_case_title = False
        self.doctype_count = 0

    def handle_decl(self, decl: str) -> None:
        if decl.lower().startswith("doctype"):
            self.doctype_count += 1

    def handle_starttag(
        self, tag: str, attrs: List[Tuple[str, Optional[str]]]
    ) -> None:
        tag = tag.lower()
        attributes = {str(key).lower(): value or "" for key, value in attrs}
        if tag == "html":
            self.seen_html = True
            self.html_lang = attributes.get("lang", "")
        element_id = attributes.get("id", "")
        if element_id:
            self.ids.add(element_id)
        class_tokens = set(attributes.get("class", "").split())
        self.classes.update(class_tokens)
        if tag == "meta" and attributes.get("property", "").lower() == "og:url":
            self.og_url = attributes.get("content", "").strip()
        if tag == "title":
            self.in_document_title = True
        if tag == "h2":
            self.h2_depth += 1
        if tag == "span" and self.h2_depth and "title" in class_tokens:
            self.in_case_title = True

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag == "title":
            self.in_document_title = False
        if tag == "span" and self.in_case_title:
            self.in_case_title = False
        if tag == "h2" and self.h2_depth:
            self.h2_depth -= 1
            self.in_case_title = False

    def handle_data(self, data: str) -> None:
        if self.in_document_title:
            self.document_title_parts.append(data)
        if self.in_case_title:
            self.case_title_parts.append(data)

    @property
    def document_title(self) -> str:
        return normalize_text(" ".join(self.document_title_parts))

    @property
    def case_title(self) -> str:
        return normalize_text(" ".join(self.case_title_parts))


def validate_case_html(
    body: bytes,
    *,
    content_type: str,
    content_encoding: str,
    expected_canonical_url: str,
    final_url: str,
    expected_title: str,
) -> PageValidation:
    warnings: List[str] = []
    errors: List[str] = []

    html_content_type = (
        "text/html" in content_type.lower()
        or "application/xhtml+xml" in content_type.lower()
    )
    if not html_content_type:
        errors.append("NON_HTML_CONTENT_TYPE")
    if content_encoding and content_encoding.lower() not in {"identity", "none"}:
        errors.append("CONTENT_ENCODING_NOT_IDENTITY")
    if len(body) < HARD_MINIMUM_BYTES:
        errors.append("RESPONSE_BELOW_HARD_MINIMUM")
    elif len(body) < SMALL_PAGE_WARNING_BYTES:
        warnings.append("SUSPICIOUSLY_SMALL_PAGE")
    if len(body) > LARGE_PAGE_WARNING_BYTES:
        warnings.append("SUSPICIOUSLY_LARGE_PAGE")

    text = body.decode("utf-8", errors="replace")
    probe = CaseHtmlProbe()
    try:
        probe.feed(text)
        probe.close()
    except Exception as exc:
        errors.append(f"HTML_PROBE_ERROR:{type(exc).__name__}")

    page_locale_en = bool(PAGE_LOCALE_EN.search(text))
    html_lang_en = probe.html_lang.lower().startswith("en")
    has_case_content = "case-law-content" in probe.ids
    has_case_detail = "case-law-detail" in probe.classes
    has_db_header = "db-headder" in probe.ids
    has_case_title = bool(probe.case_title)
    has_fact_summary = "factSummary" in probe.classes
    has_trafficking_badge = (
        "traffickingPersonsCrimeType-details-badge" in probe.classes
    )

    if not probe.seen_html:
        errors.append("MISSING_HTML_ROOT")
    if not has_case_content:
        errors.append("MISSING_CASE_LAW_CONTENT")
    if not has_case_detail:
        errors.append("MISSING_CASE_LAW_DETAIL")
    if not has_db_header:
        errors.append("MISSING_SHERLOC_DB_HEADER")
    if not has_case_title:
        errors.append("MISSING_CASE_TITLE_MARKER")
    if not (page_locale_en or html_lang_en):
        errors.append("MISSING_ENGLISH_LOCALE_SIGNAL")

    error_title_tokens = (
        "service unavailable",
        "access denied",
        "too many requests",
        "just a moment",
        "internal server error",
        "not found",
    )
    lowered_title = probe.document_title.lower()
    if any(token in lowered_title for token in error_title_tokens) and not (
        has_case_content and has_case_detail
    ):
        errors.append("ERROR_OR_CHALLENGE_TITLE")

    if not has_fact_summary:
        warnings.append("MISSING_FACT_SUMMARY_MARKER")
    if not has_trafficking_badge:
        warnings.append("MISSING_TRAFFICKING_BADGE")
    if not probe.og_url:
        warnings.append("MISSING_OG_URL")
    if probe.doctype_count != 1:
        warnings.append(f"DOCTYPE_COUNT_{probe.doctype_count}")

    normalized_expected_title = normalize_text(expected_title)
    if (
        normalized_expected_title
        and probe.case_title
        and probe.case_title != normalized_expected_title
    ):
        warnings.append("CASE_TITLE_DIFFERS_FROM_MANIFEST")

    final_relation = url_relation(final_url, expected_canonical_url)
    if final_relation == "MISMATCH":
        errors.append("FINAL_URL_CANONICAL_MISMATCH")
    if canonical_case_identity(final_url) is None:
        errors.append("FINAL_URL_NOT_ALLOWED_HTTPS_SHERLOC")

    og_relation = url_relation(
        probe.og_url,
        expected_canonical_url,
        base_url=final_url,
    )
    if og_relation == "MISMATCH":
        errors.append("OG_URL_CANONICAL_MISMATCH")

    markers = {
        "seen_html": probe.seen_html,
        "html_lang": probe.html_lang,
        "page_locale_en": page_locale_en,
        "case_law_content": has_case_content,
        "case_law_detail": has_case_detail,
        "db_header": has_db_header,
        "case_title": has_case_title,
        "fact_summary": has_fact_summary,
        "trafficking_badge": has_trafficking_badge,
        "og_url": bool(probe.og_url),
        "doctype_count": probe.doctype_count,
    }
    return PageValidation(
        valid=not errors,
        og_url=probe.og_url,
        document_title=probe.document_title,
        case_title=probe.case_title,
        final_url_relation=final_relation,
        og_url_relation=og_relation,
        markers=markers,
        warnings=sorted(set(warnings)),
        errors=sorted(set(errors)),
    )


def resolve_ca_bundle(explicit: Optional[Path]) -> Optional[Path]:
    if explicit is not None:
        candidate = explicit.expanduser().resolve()
        if not candidate.is_file():
            raise DownloadError(f"CA bundle does not exist: {candidate}")
        return candidate
    candidates = [
        os.environ.get("SSL_CERT_FILE"),
        ssl.get_default_verify_paths().cafile,
        "/etc/ssl/cert.pem",
        "/etc/ssl/certs/ca-certificates.crt",
        "/etc/pki/tls/certs/ca-bundle.crt",
    ]
    for raw in candidates:
        if raw and Path(raw).is_file():
            return Path(raw)
    return None


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


class TrackingRedirectHandler(HTTPRedirectHandler):
    def __init__(self) -> None:
        super().__init__()
        self.chain: List[Dict[str, Any]] = []

    def reset(self) -> None:
        self.chain = []

    def redirect_request(
        self,
        req: Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> Optional[Request]:
        target = urljoin(req.full_url, newurl)
        self.chain.append({"status": code, "from": req.full_url, "to": target})
        parsed = urlsplit(target)
        try:
            port = parsed.port
        except ValueError:
            port = -1
        if (
            parsed.scheme.lower() != "https"
            or (parsed.hostname or "").lower() not in ALLOWED_HOSTS
            or port not in {None, 443}
        ):
            raise HTTPError(
                req.full_url,
                code,
                f"Off-domain or non-HTTPS redirect blocked: {target}",
                headers,
                fp,
            )
        return super().redirect_request(req, fp, code, msg, headers, target)


class PoliteHttpClient:
    def __init__(
        self,
        *,
        delay_seconds: float,
        timeout_seconds: float,
        max_retries: int,
        backoff_base_seconds: float,
        ca_bundle: Optional[Path],
    ) -> None:
        self.delay_seconds = max(0.0, delay_seconds)
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.backoff_base_seconds = max(0.0, backoff_base_seconds)
        self.ca_bundle = resolve_ca_bundle(ca_bundle)
        context = ssl.create_default_context(
            cafile=str(self.ca_bundle) if self.ca_bundle else None
        )
        self.redirect_handler = TrackingRedirectHandler()
        self.opener = build_opener(
            HTTPSHandler(context=context),
            HTTPCookieProcessor(CookieJar()),
            self.redirect_handler,
        )
        self.last_request_started_at: Optional[float] = None
        self.events: List[Dict[str, Any]] = []

    def _wait_for_request_slot(self) -> None:
        if self.last_request_started_at is not None:
            remaining = self.delay_seconds - (
                time.monotonic() - self.last_request_started_at
            )
            if remaining > 0:
                time.sleep(remaining)
        self.last_request_started_at = time.monotonic()

    def fetch_case(
        self,
        *,
        requested_url: str,
        canonical_url: str,
        case_title: str,
        search_rank: int,
    ) -> HttpOutcome:
        overall_started = time.monotonic()
        last_body = b""
        last_status: Optional[int] = None
        last_final_url = requested_url
        last_headers: Dict[str, str] = {}
        last_validation: Optional[PageValidation] = None
        last_error = ""
        last_chain: List[Dict[str, Any]] = []

        for attempt in range(1, self.max_retries + 2):
            self._wait_for_request_slot()
            self.redirect_handler.reset()
            attempt_started = time.monotonic()
            attempt_started_at = utc_now()
            request = Request(
                requested_url,
                headers={
                    "Accept": "text/html,application/xhtml+xml",
                    "Accept-Encoding": "identity",
                    "User-Agent": USER_AGENT,
                },
                method="GET",
            )
            retry_delay = 0.0
            should_retry = False
            try:
                with self.opener.open(request, timeout=self.timeout_seconds) as response:
                    body = response.read()
                    status = int(getattr(response, "status", response.getcode()))
                    headers = {
                        key.lower(): value for key, value in response.headers.items()
                    }
                    final_url = response.geturl()
                    validation = validate_case_html(
                        body,
                        content_type=headers.get("content-type", ""),
                        content_encoding=headers.get("content-encoding", ""),
                        expected_canonical_url=canonical_url,
                        final_url=final_url,
                        expected_title=case_title,
                    )
                    last_body = body
                    last_status = status
                    last_headers = headers
                    last_final_url = final_url
                    last_validation = validation
                    last_chain = list(self.redirect_handler.chain)
                    last_error = "" if validation.valid else ";".join(validation.errors)
                    should_retry = status in TRANSIENT_HTTP_STATUSES or (
                        status == 200 and not validation.valid
                    )
                    if should_retry:
                        retry_delay = max(
                            parse_retry_after(headers.get("retry-after")),
                            self.backoff_base_seconds * (2 ** (attempt - 1)),
                        )
                    event = {
                        "search_rank": search_rank,
                        "attempt": attempt,
                        "started_at": attempt_started_at,
                        "status": status,
                        "elapsed_seconds": round(
                            time.monotonic() - attempt_started, 3
                        ),
                        "response_bytes": len(body),
                        "content_type": headers.get("content-type", ""),
                        "content_encoding": headers.get("content-encoding", ""),
                        "final_url": final_url,
                        "redirect_chain": last_chain,
                        "content_valid": validation.valid,
                        "validation_errors": validation.errors,
                    }
                    self.events.append(event)
                    if not should_retry or attempt > self.max_retries:
                        return HttpOutcome(
                            body=body,
                            status=status,
                            final_url=final_url,
                            headers=headers,
                            attempts=attempt,
                            redirect_chain=last_chain,
                            elapsed_seconds=time.monotonic() - overall_started,
                            validation=validation,
                            error=last_error,
                        )

            except HTTPError as exc:
                body = exc.read()
                status = int(exc.code)
                headers = {
                    key.lower(): value for key, value in (exc.headers.items() if exc.headers else [])
                }
                last_body = body
                last_status = status
                last_headers = headers
                last_final_url = exc.geturl() or requested_url
                last_validation = None
                last_chain = list(self.redirect_handler.chain)
                last_error = str(exc)
                should_retry = status in TRANSIENT_HTTP_STATUSES
                retry_delay = max(
                    parse_retry_after(headers.get("retry-after")),
                    self.backoff_base_seconds * (2 ** (attempt - 1)),
                )
                self.events.append(
                    {
                        "search_rank": search_rank,
                        "attempt": attempt,
                        "started_at": attempt_started_at,
                        "status": status,
                        "elapsed_seconds": round(
                            time.monotonic() - attempt_started, 3
                        ),
                        "response_bytes": len(body),
                        "content_type": headers.get("content-type", ""),
                        "content_encoding": headers.get("content-encoding", ""),
                        "final_url": last_final_url,
                        "redirect_chain": last_chain,
                        "error": str(exc),
                        "retry_after": headers.get("retry-after"),
                    }
                )
                if not should_retry or attempt > self.max_retries:
                    return HttpOutcome(
                        body=body,
                        status=status,
                        final_url=last_final_url,
                        headers=headers,
                        attempts=attempt,
                        redirect_chain=last_chain,
                        elapsed_seconds=time.monotonic() - overall_started,
                        validation=None,
                        error=str(exc),
                    )

            except (URLError, TimeoutError, OSError, HTTPException) as exc:
                last_body = b""
                last_status = None
                last_headers = {}
                last_final_url = requested_url
                last_validation = None
                last_chain = list(self.redirect_handler.chain)
                last_error = repr(exc)
                should_retry = True
                retry_delay = self.backoff_base_seconds * (2 ** (attempt - 1))
                self.events.append(
                    {
                        "search_rank": search_rank,
                        "attempt": attempt,
                        "started_at": attempt_started_at,
                        "status": None,
                        "elapsed_seconds": round(
                            time.monotonic() - attempt_started, 3
                        ),
                        "response_bytes": 0,
                        "final_url": requested_url,
                        "redirect_chain": last_chain,
                        "error": repr(exc),
                    }
                )
                if attempt > self.max_retries:
                    return HttpOutcome(
                        body=b"",
                        status=None,
                        final_url=requested_url,
                        headers={},
                        attempts=attempt,
                        redirect_chain=last_chain,
                        elapsed_seconds=time.monotonic() - overall_started,
                        validation=None,
                        error=repr(exc),
                    )

            if should_retry:
                time.sleep(retry_delay)

        return HttpOutcome(
            body=last_body,
            status=last_status,
            final_url=last_final_url,
            headers=last_headers,
            attempts=self.max_retries + 1,
            redirect_chain=last_chain,
            elapsed_seconds=time.monotonic() - overall_started,
            validation=last_validation,
            error=last_error,
        )


def load_source_manifest(path: Path) -> Tuple[List[Dict[str, str]], str]:
    if not path.is_file():
        raise DownloadError(f"Source manifest not found: {path}")
    raw = path.read_bytes()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise DownloadError("Source manifest has no header")
        missing = SOURCE_REQUIRED_FIELDS.difference(reader.fieldnames)
        if missing:
            raise DownloadError(
                f"Source manifest missing required fields: {sorted(missing)}"
            )
        rows = list(reader)
    if not rows:
        raise DownloadError("Source manifest is empty")

    ranks = [parse_int(row.get("search_rank"), -1) for row in rows]
    if ranks != list(range(1, len(rows) + 1)):
        raise DownloadError("Source search ranks must be contiguous from 1")
    if len({row["canonical_url"] for row in rows}) != len(rows):
        raise DownloadError("Source manifest canonical URLs are not unique")
    if len({row["api_result_id"] for row in rows}) != len(rows):
        raise DownloadError("Source manifest API result IDs are not unique")
    for row in rows:
        for field in ("result_url", "canonical_url"):
            parsed = urlsplit(row[field])
            try:
                port = parsed.port
            except ValueError:
                port = -1
            if (
                parsed.scheme.lower() != "https"
                or (parsed.hostname or "").lower() not in ALLOWED_HOSTS
                or port not in {None, 443}
            ):
                raise DownloadError(
                    f"Rank {row['search_rank']} has unsafe {field}: {row[field]}"
                )
    return rows, hashlib.sha256(raw).hexdigest()


def quantile_select(rows: Sequence[Dict[str, str]], count: int) -> List[Dict[str, str]]:
    ordered = sorted(rows, key=lambda row: parse_int(row["search_rank"]))
    if count >= len(ordered):
        return list(ordered)
    if count <= 1:
        return [ordered[len(ordered) // 2]] if count else []
    denominator = count - 1
    indexes = [
        (index * (len(ordered) - 1) + denominator // 2) // denominator
        for index in range(count)
    ]
    return [ordered[index] for index in indexes]


def select_pilot(
    source_rows: Sequence[Dict[str, str]],
    *,
    pilot_size: int,
    rank_bands: int,
) -> Tuple[Set[int], Dict[int, List[str]], Dict[str, Any]]:
    if not 50 <= pilot_size <= 100:
        raise DownloadError("Pilot size must be between 50 and 100")
    if rank_bands < 2:
        raise DownloadError("Pilot rank bands must be at least 2")

    selected: Set[int] = set()
    reasons: Dict[int, List[str]] = defaultdict(list)
    total_rows = len(source_rows)

    # Five trafficking-path records from each rank band provide broad rank/time
    # coverage without relying on title or country assumptions.
    primary_target = min(pilot_size, rank_bands * 5)
    per_band = max(1, primary_target // rank_bands)
    for band in range(rank_bands):
        start_index = (band * total_rows) // rank_bands
        end_index = ((band + 1) * total_rows) // rank_bands
        candidates = [
            row
            for row in source_rows[start_index:end_index]
            if row["url_path_crime_type"] == PRIMARY_PATH_TYPE
        ]
        if not candidates:
            raise DownloadError(f"No primary-path pilot candidate in rank band {band + 1}")
        for row in quantile_select(candidates, min(per_band, len(candidates))):
            rank = parse_int(row["search_rank"])
            selected.add(rank)
            reasons[rank].append(f"primary_rank_band_{band + 1}")

    cross_groups: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in source_rows:
        path_type = row["url_path_crime_type"]
        if path_type != PRIMARY_PATH_TYPE:
            cross_groups[path_type].append(row)
    if not cross_groups:
        raise DownloadError("No cross-classified URL paths found for pilot")

    allocations = {
        path_type: min(5, len(rows)) for path_type, rows in cross_groups.items()
    }
    remaining = pilot_size - len(selected) - sum(allocations.values())
    for path_type in sorted(cross_groups, key=lambda key: (-len(cross_groups[key]), key)):
        if remaining <= 0:
            break
        capacity = len(cross_groups[path_type]) - allocations[path_type]
        extra = min(capacity, remaining)
        allocations[path_type] += extra
        remaining -= extra

    for path_type in sorted(cross_groups):
        for row in quantile_select(cross_groups[path_type], allocations[path_type]):
            rank = parse_int(row["search_rank"])
            selected.add(rank)
            reasons[rank].append(f"cross_path_{path_type}")

    # Mandatory earliest/latest URL-year anchors. Replace a yearless primary
    # record in the same band if needed so the configured pilot size is stable.
    year_rows = [
        (extract_path_year(row["canonical_url"]), row)
        for row in source_rows
        if extract_path_year(row["canonical_url"]) is not None
    ]
    anchors = [
        min(year_rows, key=lambda item: (int(item[0]), parse_int(item[1]["search_rank"]))),
        max(year_rows, key=lambda item: (int(item[0]), -parse_int(item[1]["search_rank"]))),
    ]
    for label, (_, anchor_row) in zip(("earliest_year", "latest_year"), anchors):
        anchor_rank = parse_int(anchor_row["search_rank"])
        if anchor_rank in selected:
            reasons[anchor_rank].append(f"temporal_anchor_{label}")
            continue
        band = min(rank_bands - 1, ((anchor_rank - 1) * rank_bands) // total_rows)
        band_start = (band * total_rows) // rank_bands + 1
        band_end = ((band + 1) * total_rows) // rank_bands
        removable = [
            rank
            for rank in sorted(
                selected,
                key=lambda candidate: (abs(candidate - anchor_rank), candidate),
            )
            if rank not in {1, total_rows}
            and band_start <= rank <= band_end
            and source_rows[rank - 1]["url_path_crime_type"] == PRIMARY_PATH_TYPE
            and extract_path_year(source_rows[rank - 1]["canonical_url"]) is None
            and all(not reason.startswith("cross_path_") for reason in reasons[rank])
        ]
        if not removable:
            removable = [
                rank
                for rank in sorted(
                    selected,
                    key=lambda candidate: (abs(candidate - anchor_rank), candidate),
                )
                if rank not in {1, total_rows}
                and band_start <= rank <= band_end
                and source_rows[rank - 1]["url_path_crime_type"] == PRIMARY_PATH_TYPE
                and all(not reason.startswith("cross_path_") for reason in reasons[rank])
            ]
        if not removable:
            raise DownloadError(f"Cannot make room for temporal anchor rank {anchor_rank}")
        removed = removable[0]
        selected.remove(removed)
        reasons.pop(removed, None)
        selected.add(anchor_rank)
        reasons[anchor_rank].append(f"temporal_anchor_{label}")

    # If overlap made the set short, fill by evenly spaced unselected ranks.
    if len(selected) < pilot_size:
        fill_candidates = [
            row
            for row in source_rows
            if parse_int(row["search_rank"]) not in selected
        ]
        for row in quantile_select(fill_candidates, pilot_size - len(selected)):
            rank = parse_int(row["search_rank"])
            selected.add(rank)
            reasons[rank].append("deterministic_fill")

    if len(selected) != pilot_size:
        raise DownloadError(
            f"Pilot selection produced {len(selected)} rows, expected {pilot_size}"
        )

    selected_rows = [source_rows[rank - 1] for rank in sorted(selected)]
    path_counts = Counter(row["url_path_crime_type"] for row in selected_rows)
    missing_path_types = set(row["url_path_crime_type"] for row in source_rows).difference(
        path_counts
    )
    if missing_path_types:
        raise DownloadError(
            f"Pilot does not cover path types: {sorted(missing_path_types)}"
        )

    rank_band_counts: Dict[str, int] = {}
    for band in range(rank_bands):
        start_rank = (band * total_rows) // rank_bands + 1
        end_rank = ((band + 1) * total_rows) // rank_bands
        rank_band_counts[f"{start_rank}-{end_rank}"] = sum(
            start_rank <= rank <= end_rank for rank in selected
        )

    year_counts = Counter(
        str(extract_path_year(row["canonical_url"]))
        if extract_path_year(row["canonical_url"]) is not None
        else "NO_YEAR_SEGMENT"
        for row in selected_rows
    )
    years = [
        year
        for row in selected_rows
        for year in [extract_path_year(row["canonical_url"])]
        if year is not None
    ]
    coverage = {
        "method": "deterministic_rank_and_path_quantiles_with_temporal_anchors",
        "pilot_size": pilot_size,
        "selected_ranks": sorted(selected),
        "path_type_counts": dict(sorted(path_counts.items())),
        "rank_band_counts": rank_band_counts,
        "year_segment_counts": dict(sorted(year_counts.items())),
        "minimum_year_segment": min(years) if years else None,
        "maximum_year_segment": max(years) if years else None,
        "yearless_selected": year_counts.get("NO_YEAR_SEGMENT", 0),
    }
    return selected, reasons, coverage


def stable_raw_name(search_rank: int, canonical_url: str, max_rank: int) -> str:
    width = max(6, len(str(max_rank)))
    short_hash = hashlib.sha256(canonical_url.encode("utf-8")).hexdigest()[:12]
    return f"{search_rank:0{width}d}_{short_hash}.html"


def load_previous_state(path: Path) -> Dict[int, Dict[str, str]]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return {}
        return {
            parse_int(row.get("search_rank")): row
            for row in reader
            if parse_int(row.get("search_rank")) > 0
        }


def initialize_download_rows(
    source_rows: Sequence[Dict[str, str]],
    *,
    previous: Dict[int, Dict[str, str]],
    pilot_selected: Set[int],
    pilot_reasons: Dict[int, List[str]],
    raw_dir: Path,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    max_rank = len(source_rows)
    for source in source_rows:
        rank = parse_int(source["search_rank"])
        filename = stable_raw_name(rank, source["canonical_url"], max_rank)
        relative_raw = str((raw_dir / filename).relative_to(REPO_ROOT))
        row: Dict[str, Any] = {
            "search_rank": rank,
            "case_title": source["case_title"],
            "url_path_crime_type": source["url_path_crime_type"],
            "api_result_id": source["api_result_id"],
            "unodc_case_number": source["unodc_case_number"],
            "canonical_url": source["canonical_url"],
            "requested_url": source["result_url"],
            "final_url": "",
            "final_url_relation": "",
            "og_url": "",
            "og_url_relation": "",
            "http_status": "",
            "content_type": "",
            "content_encoding": "",
            "download_timestamp": "",
            "raw_filename": relative_raw,
            "byte_count": "",
            "sha256": "",
            "download_status": "NOT_ATTEMPTED",
            "attempts": "",
            "redirect_count": "",
            "redirect_chain": "",
            "elapsed_seconds": "",
            "pilot_selected": "true" if rank in pilot_selected else "false",
            "pilot_selection_reasons": "|".join(pilot_reasons.get(rank, [])),
            "last_action": "",
            "last_checked_at": "",
            "structural_markers": "",
            "warnings": "",
            "error": "",
            "source_manifest_discovered_at": source.get("discovered_at", ""),
        }
        old = previous.get(rank)
        if old:
            if old.get("canonical_url") != source["canonical_url"]:
                row["warnings"] = "PREVIOUS_STATE_CANONICAL_MISMATCH_IGNORED"
            elif old.get("api_result_id") != source["api_result_id"]:
                row["warnings"] = "PREVIOUS_STATE_API_ID_MISMATCH_IGNORED"
            else:
                for field in MUTABLE_DOWNLOAD_FIELDS:
                    if field in old:
                        row[field] = old[field]
                # Fixed fields always come from the frozen source manifest.
                old_raw_filename = str(old.get("raw_filename") or "")
                expected_failed = str(
                    (
                        raw_dir
                        / "_failed"
                        / f"{Path(filename).stem}.failed.html"
                    ).relative_to(REPO_ROOT)
                )
                if (
                    old.get("download_status") != SUCCESS_STATUS
                    and old_raw_filename == expected_failed
                ):
                    row["raw_filename"] = old_raw_filename
                else:
                    row["raw_filename"] = relative_raw
        rows.append(row)

    names = [row["raw_filename"] for row in rows]
    if len(names) != len(set(names)):
        raise DownloadError("Stable raw filename collision detected")
    return rows


def atomic_write_bytes(path: Path, body: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.part-{os.getpid()}")
    with temporary.open("wb") as handle:
        handle.write(body)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DOWNLOAD_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def expected_raw_path(row: Dict[str, Any], raw_dir: Path) -> Path:
    rank = parse_int(row.get("search_rank"))
    if rank < 1:
        raise DownloadError("Download row is missing a valid search_rank")
    return raw_dir / stable_raw_name(
        rank,
        str(row.get("canonical_url", "")),
        rank,
    )


def validate_existing_file(
    row: Dict[str, Any], raw_dir: Path
) -> Tuple[bool, Optional[PageValidation], bytes, List[str]]:
    path = expected_raw_path(row, raw_dir)
    if not path.is_file():
        return False, None, b"", ["EXPECTED_RAW_FILE_MISSING"]
    body = path.read_bytes()
    warnings: List[str] = []
    recorded_hash = str(row.get("sha256") or "")
    current_hash = sha256_bytes(body)
    if recorded_hash and recorded_hash != current_hash:
        return False, None, body, ["EXISTING_FILE_SHA256_MISMATCH"]
    recorded_size = parse_int(row.get("byte_count"), -1)
    if recorded_size >= 0 and recorded_size != len(body):
        return False, None, body, ["EXISTING_FILE_BYTE_COUNT_MISMATCH"]

    validation = validate_case_html(
        body,
        content_type=str(row.get("content_type") or "text/html"),
        content_encoding=str(row.get("content_encoding") or ""),
        expected_canonical_url=str(row["canonical_url"]),
        final_url=str(row.get("final_url") or row["requested_url"]),
        expected_title=str(row["case_title"]),
    )
    if not validation.valid:
        warnings.extend(validation.errors)
        return False, validation, body, warnings
    warnings.extend(validation.warnings)
    return True, validation, body, warnings


def record_success(
    row: Dict[str, Any],
    *,
    outcome: HttpOutcome,
    raw_dir: Path,
) -> None:
    if outcome.validation is None or not outcome.validation.valid:
        raise DownloadError("record_success called for invalid outcome")
    raw_path = expected_raw_path(row, raw_dir)
    atomic_write_bytes(raw_path, outcome.body)
    validation = outcome.validation
    warnings = list(validation.warnings)
    if outcome.attempts > 1:
        warnings.append("RETRIED_BEFORE_SUCCESS")
    if outcome.redirect_chain:
        warnings.append("HTTP_REDIRECT")
    row.update(
        {
            "final_url": outcome.final_url,
            "final_url_relation": validation.final_url_relation,
            "og_url": validation.og_url,
            "og_url_relation": validation.og_url_relation,
            "http_status": outcome.status or "",
            "content_type": outcome.headers.get("content-type", ""),
            "content_encoding": outcome.headers.get("content-encoding", ""),
            "download_timestamp": utc_now(),
            "raw_filename": str(raw_path.relative_to(REPO_ROOT)),
            "byte_count": len(outcome.body),
            "sha256": sha256_bytes(outcome.body),
            "download_status": SUCCESS_STATUS,
            "attempts": outcome.attempts,
            "redirect_count": len(outcome.redirect_chain),
            "redirect_chain": compact_json(outcome.redirect_chain),
            "elapsed_seconds": f"{outcome.elapsed_seconds:.3f}",
            "last_action": "DOWNLOADED",
            "last_checked_at": utc_now(),
            "structural_markers": compact_json(validation.markers),
            "warnings": join_warnings(warnings),
            "error": "",
        }
    )


def record_skip_or_recovery(
    row: Dict[str, Any],
    *,
    validation: PageValidation,
    body: bytes,
    raw_dir: Path,
    recovered: bool,
) -> None:
    raw_path = expected_raw_path(row, raw_dir)
    warnings = list(validation.warnings)
    provenance_warnings = {
        "HTTP_REDIRECT",
        "RECOVERED_EXISTING_VALID_FILE",
        "RETRIED_BEFORE_SUCCESS",
    }
    warnings.extend(
        warning
        for warning in split_warnings(row.get("warnings"))
        if warning in provenance_warnings
    )
    if recovered:
        warnings.append("RECOVERED_EXISTING_VALID_FILE")
    row.update(
        {
            "final_url": row.get("final_url") or row["requested_url"],
            "final_url_relation": validation.final_url_relation,
            "og_url": validation.og_url,
            "og_url_relation": validation.og_url_relation,
            "http_status": row.get("http_status") or 200,
            "content_type": row.get("content_type") or "text/html (recovered)",
            "content_encoding": row.get("content_encoding") or "",
            "download_timestamp": row.get("download_timestamp") or file_timestamp(raw_path),
            "raw_filename": str(raw_path.relative_to(REPO_ROOT)),
            "byte_count": len(body),
            "sha256": sha256_bytes(body),
            "download_status": SUCCESS_STATUS,
            "attempts": row.get("attempts") or 0,
            "redirect_count": row.get("redirect_count") or 0,
            "redirect_chain": row.get("redirect_chain") or "[]",
            "elapsed_seconds": row.get("elapsed_seconds") or "0.000",
            "last_action": "RECOVERED_EXISTING" if recovered else "SKIPPED_VALID_EXISTING",
            "last_checked_at": utc_now(),
            "structural_markers": compact_json(validation.markers),
            "warnings": join_warnings(warnings),
            "error": "",
        }
    )


def quarantine_body(row: Dict[str, Any], body: bytes, raw_dir: Path) -> str:
    if not body:
        return ""
    expected = expected_raw_path(row, raw_dir)
    quarantine = raw_dir / "_failed" / f"{expected.stem}.failed.html"
    atomic_write_bytes(quarantine, body)
    return str(quarantine.relative_to(REPO_ROOT))


def record_failure(
    row: Dict[str, Any],
    *,
    outcome: HttpOutcome,
    raw_dir: Path,
    preexisting_warnings: Iterable[str],
) -> None:
    validation = outcome.validation
    warnings = list(preexisting_warnings)
    if validation is not None:
        warnings.extend(validation.warnings)
        warnings.extend(validation.errors)
    if outcome.redirect_chain:
        warnings.append("HTTP_REDIRECT")
    if outcome.attempts > 1:
        warnings.append("RETRIES_EXHAUSTED")

    if outcome.status is None:
        status = "NETWORK_ERROR"
    elif outcome.status != 200:
        status = "HTTP_ERROR"
    else:
        status = "VALIDATION_ERROR"
    quarantined = quarantine_body(row, outcome.body, raw_dir)
    row.update(
        {
            "final_url": outcome.final_url,
            "final_url_relation": (
                validation.final_url_relation
                if validation
                else url_relation(outcome.final_url, str(row["canonical_url"]))
            ),
            "og_url": validation.og_url if validation else "",
            "og_url_relation": validation.og_url_relation if validation else "",
            "http_status": outcome.status or "",
            "content_type": outcome.headers.get("content-type", ""),
            "content_encoding": outcome.headers.get("content-encoding", ""),
            "download_timestamp": utc_now(),
            "raw_filename": quarantined or str(row["raw_filename"]),
            "byte_count": len(outcome.body) if outcome.body else "",
            "sha256": sha256_bytes(outcome.body) if outcome.body else "",
            "download_status": status,
            "attempts": outcome.attempts,
            "redirect_count": len(outcome.redirect_chain),
            "redirect_chain": compact_json(outcome.redirect_chain),
            "elapsed_seconds": f"{outcome.elapsed_seconds:.3f}",
            "last_action": "FAILED",
            "last_checked_at": utc_now(),
            "structural_markers": compact_json(validation.markers) if validation else "",
            "warnings": join_warnings(warnings),
            "error": outcome.error,
        }
    )


def percentile(values: Sequence[int], fraction: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def build_diagnostics(
    *,
    args: argparse.Namespace,
    source_rows: Sequence[Dict[str, str]],
    source_sha256: str,
    rows: Sequence[Dict[str, Any]],
    targets: Set[int],
    pilot_coverage: Dict[str, Any],
    client: PoliteHttpClient,
    started_at: str,
    elapsed_seconds: float,
    run_status: str,
    circuit_breaker_reason: str,
    download_manifest_path: Path,
    prior_pilot_run: Optional[Dict[str, Any]],
    prior_full_runs: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    status_counts = Counter(str(row["download_status"]) for row in rows)
    action_counts = Counter(str(row["last_action"] or "NONE") for row in rows)
    target_rows = [row for row in rows if parse_int(row["search_rank"]) in targets]
    target_failures = [
        row for row in target_rows if row["download_status"] != SUCCESS_STATUS
    ]
    successful = [row for row in rows if row["download_status"] == SUCCESS_STATUS]

    warning_counts: Counter[str] = Counter()
    for row in rows:
        warning_counts.update(split_warnings(row.get("warnings")))

    hash_groups: Dict[str, List[int]] = defaultdict(list)
    for row in successful:
        if row.get("sha256"):
            hash_groups[str(row["sha256"])].append(parse_int(row["search_rank"]))
    duplicate_hashes = [
        {"sha256": digest, "search_ranks": ranks, "occurrences": len(ranks)}
        for digest, ranks in hash_groups.items()
        if len(ranks) > 1
    ]

    sizes = [parse_int(row.get("byte_count")) for row in successful]
    sizes = [size for size in sizes if size > 0]
    q1 = percentile(sizes, 0.25)
    q3 = percentile(sizes, 0.75)
    iqr = (q3 - q1) if q1 is not None and q3 is not None else None
    lower_fence = max(0.0, q1 - 1.5 * iqr) if iqr is not None else None
    upper_fence = q3 + 1.5 * iqr if iqr is not None else None

    size_outliers = [
        {
            "search_rank": parse_int(row["search_rank"]),
            "byte_count": parse_int(row["byte_count"]),
            "canonical_url": row["canonical_url"],
        }
        for row in successful
        if (
            lower_fence is not None
            and upper_fence is not None
            and (
                parse_int(row["byte_count"]) < lower_fence
                or parse_int(row["byte_count"]) > upper_fence
            )
        )
    ]

    event_statuses = Counter(
        str(event.get("status")) if event.get("status") is not None else "NETWORK_ERROR"
        for event in client.events
    )
    retry_attempts = sum(1 for event in client.events if parse_int(event.get("attempt"), 1) > 1)
    retry_after_events = sum(1 for event in client.events if event.get("retry_after"))
    redirect_rows = [row for row in successful if parse_int(row.get("redirect_count")) > 0]
    true_og_mismatches = [
        parse_int(row["search_rank"])
        for row in rows
        if row.get("og_url_relation") == "MISMATCH"
    ]
    true_final_mismatches = [
        parse_int(row["search_rank"])
        for row in rows
        if row.get("final_url_relation") == "MISMATCH"
    ]

    marker_warning_names = [
        "MISSING_FACT_SUMMARY_MARKER",
        "MISSING_TRAFFICKING_BADGE",
        "MISSING_OG_URL",
        "MISSING_CASE_LAW_CONTENT",
        "MISSING_CASE_LAW_DETAIL",
        "MISSING_CASE_TITLE_MARKER",
        "MISSING_ENGLISH_LOCALE_SIGNAL",
    ]
    marker_issues = {
        name: [
            parse_int(row["search_rank"])
            for row in rows
            if name in split_warnings(row.get("warnings"))
        ]
        for name in marker_warning_names
    }

    manifest_sha = (
        sha256_file(download_manifest_path)
        if download_manifest_path.is_file()
        else None
    )
    complete_target = not target_failures
    return {
        "schema_version": "1.0",
        "downloader_version": DOWNLOADER_VERSION,
        "run": {
            "mode": args.mode,
            "started_at": started_at,
            "finished_at": utc_now(),
            "elapsed_seconds": round(elapsed_seconds, 3),
            "status": run_status,
            "target_complete": complete_target,
            "pilot_passed": args.mode == "pilot" and complete_target,
            "full_download_complete": (
                args.mode == "full"
                and len(successful) == len(source_rows)
                and not target_failures
            ),
            "force": args.force,
            "delay_seconds": args.delay_seconds,
            "timeout_seconds": args.timeout_seconds,
            "max_retries": args.max_retries,
            "backoff_base_seconds": args.backoff_base_seconds,
            "circuit_breaker_reason": circuit_breaker_reason or None,
            "tls_ca_bundle": str(client.ca_bundle) if client.ca_bundle else None,
        },
        "source_manifest": {
            "path": str(args.source_manifest),
            "sha256": source_sha256,
            "expected_rows": len(source_rows),
            "sole_corpus_membership_source": True,
        },
        "pilot": pilot_coverage,
        "prior_pilot_run": prior_pilot_run,
        "prior_full_runs": list(prior_full_runs),
        "counts": {
            "expected_manifest_rows": len(source_rows),
            "download_manifest_rows": len(rows),
            "target_rows_this_mode": len(targets),
            "target_successes": len(target_rows) - len(target_failures),
            "target_failures": len(target_failures),
            "all_successful_downloads": len(successful),
            "all_remaining_not_successful": len(rows) - len(successful),
            "status_counts": dict(sorted(status_counts.items())),
            "last_action_counts": dict(sorted(action_counts.items())),
        },
        "failures": [
            {
                "search_rank": parse_int(row["search_rank"]),
                "download_status": row["download_status"],
                "http_status": row["http_status"],
                "requested_url": row["requested_url"],
                "warnings": split_warnings(row.get("warnings")),
                "error": row.get("error") or None,
            }
            for row in target_failures
        ],
        "http": {
            "request_attempts": len(client.events),
            "status_counts": dict(sorted(event_statuses.items())),
            "retry_attempts": retry_attempts,
            "retry_after_events": retry_after_events,
            "http_403_events": event_statuses.get("403", 0),
            "http_429_events": event_statuses.get("429", 0),
            "redirected_successful_pages": len(redirect_rows),
            "redirected_ranks": [parse_int(row["search_rank"]) for row in redirect_rows],
            "anti_bot_or_rate_limit_observed": bool(
                event_statuses.get("403", 0)
                or event_statuses.get("429", 0)
                or retry_after_events
            ),
        },
        "url_validation": {
            "og_url_relation_counts": dict(
                sorted(Counter(str(row.get("og_url_relation") or "UNSET") for row in successful).items())
            ),
            "final_url_relation_counts": dict(
                sorted(Counter(str(row.get("final_url_relation") or "UNSET") for row in successful).items())
            ),
            "true_og_url_mismatch_count": len(true_og_mismatches),
            "true_og_url_mismatch_ranks": true_og_mismatches,
            "true_final_url_mismatch_count": len(true_final_mismatches),
            "true_final_url_mismatch_ranks": true_final_mismatches,
        },
        "page_sizes": {
            "count": len(sizes),
            "total_bytes": sum(sizes),
            "minimum": min(sizes) if sizes else None,
            "p01": percentile(sizes, 0.01),
            "p05": percentile(sizes, 0.05),
            "q1": q1,
            "median": statistics.median(sizes) if sizes else None,
            "mean": statistics.mean(sizes) if sizes else None,
            "q3": q3,
            "p95": percentile(sizes, 0.95),
            "p99": percentile(sizes, 0.99),
            "maximum": max(sizes) if sizes else None,
            "tukey_lower_fence": lower_fence,
            "tukey_upper_fence": upper_fence,
            "tukey_outlier_count": len(size_outliers),
            "tukey_outliers": size_outliers,
            "fixed_small_warning_threshold": SMALL_PAGE_WARNING_BYTES,
            "fixed_large_warning_threshold": LARGE_PAGE_WARNING_BYTES,
        },
        "checksums": {
            "unique_successful_sha256": len(hash_groups),
            "duplicate_checksum_groups": len(duplicate_hashes),
            "duplicate_checksum_rows": sum(
                len(group["search_ranks"]) - 1 for group in duplicate_hashes
            ),
            "duplicates": duplicate_hashes,
        },
        "template_warnings": {
            "warning_counts": dict(sorted(warning_counts.items())),
            "marker_issue_ranks": marker_issues,
        },
        "request_events": client.events,
        "outputs": {
            "download_manifest": str(download_manifest_path),
            "download_manifest_sha256": manifest_sha,
            "diagnostics": str(args.diagnostics),
            "raw_directory": str(args.raw_dir),
        },
    }


def run(args: argparse.Namespace) -> Tuple[Dict[str, Any], int]:
    started_at = utc_now()
    monotonic_started = time.monotonic()
    prior_pilot_run: Optional[Dict[str, Any]] = None
    prior_full_runs: List[Dict[str, Any]] = []
    if args.mode == "full" and args.diagnostics.is_file():
        try:
            with args.diagnostics.open("r", encoding="utf-8") as handle:
                prior = json.load(handle)
            candidate = prior
            if prior.get("run", {}).get("mode") == "full":
                prior_full_runs = list(prior.get("prior_full_runs") or [])
                prior_full_runs.append(
                    {
                        key: prior.get(key)
                        for key in (
                            "run",
                            "counts",
                            "failures",
                            "http",
                            "url_validation",
                            "checksums",
                            "template_warnings",
                        )
                    }
                )
                candidate = prior.get("prior_pilot_run") or {}
            prior_run = candidate.get("run", {})
            if prior_run.get("mode") == "pilot" and prior_run.get("pilot_passed"):
                prior_pilot_run = {
                    key: candidate.get(key)
                    for key in (
                        "run",
                        "source_manifest",
                        "pilot",
                        "counts",
                        "failures",
                        "http",
                        "url_validation",
                        "page_sizes",
                        "checksums",
                        "template_warnings",
                    )
                }
        except (OSError, json.JSONDecodeError, AttributeError):
            prior_pilot_run = None
    source_rows, source_sha = load_source_manifest(args.source_manifest)
    pilot_selected, pilot_reasons, pilot_coverage = select_pilot(
        source_rows,
        pilot_size=args.pilot_size,
        rank_bands=args.rank_bands,
    )
    previous = load_previous_state(args.download_manifest)
    rows = initialize_download_rows(
        source_rows,
        previous=previous,
        pilot_selected=pilot_selected,
        pilot_reasons=pilot_reasons,
        raw_dir=args.raw_dir,
    )
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(args.download_manifest, rows)

    targets = (
        set(range(1, len(source_rows) + 1))
        if args.mode == "full"
        else set(pilot_selected)
    )
    client = PoliteHttpClient(
        delay_seconds=args.delay_seconds,
        timeout_seconds=args.timeout_seconds,
        max_retries=args.max_retries,
        backoff_base_seconds=args.backoff_base_seconds,
        ca_bundle=args.ca_bundle,
    )
    consecutive_blockers = 0
    circuit_breaker_reason = ""
    run_status = "RUNNING"
    processed = 0

    try:
        if args.mode == "full" and not args.allow_full_without_pilot:
            guard_errors: List[str] = []
            if prior_pilot_run is None:
                guard_errors.append("no successful pilot diagnostics found")
            else:
                prior_source = prior_pilot_run.get("source_manifest") or {}
                if prior_source.get("sha256") != source_sha:
                    guard_errors.append("pilot source-manifest SHA-256 differs")
                prior_ranks = (prior_pilot_run.get("pilot") or {}).get(
                    "selected_ranks"
                )
                if prior_ranks != sorted(pilot_selected):
                    guard_errors.append("pilot selection differs")

            incomplete_pilot: List[int] = []
            for rank in sorted(pilot_selected):
                row = rows[rank - 1]
                if row["download_status"] != SUCCESS_STATUS:
                    incomplete_pilot.append(rank)
                    continue
                existing_valid, _, _, _ = validate_existing_file(row, args.raw_dir)
                if not existing_valid:
                    incomplete_pilot.append(rank)
            if incomplete_pilot:
                guard_errors.append(
                    f"invalid or incomplete pilot ranks begin {incomplete_pilot[:10]}"
                )
            if guard_errors:
                raise DownloadError(
                    "Full mode requires the validated deterministic pilot: "
                    + "; ".join(guard_errors)
                )

        for rank in sorted(targets):
            row = rows[rank - 1]
            raw_path = expected_raw_path(row, args.raw_dir)
            existing_warnings: List[str] = []

            if not args.force and raw_path.is_file():
                existing_valid, validation, body, existing_warnings = validate_existing_file(
                    row, args.raw_dir
                )
                if existing_valid and validation is not None:
                    recovered = row["download_status"] != SUCCESS_STATUS
                    record_skip_or_recovery(
                        row,
                        validation=validation,
                        body=body,
                        raw_dir=args.raw_dir,
                        recovered=recovered,
                    )
                    processed += 1
                    consecutive_blockers = 0
                    if processed % args.checkpoint_every == 0:
                        atomic_write_csv(args.download_manifest, rows)
                    if processed % args.progress_every == 0 or processed == len(targets):
                        print(
                            f"[{args.mode}] {processed}/{len(targets)} targets; "
                            f"rank {rank}; skipped/recovered valid",
                            flush=True,
                        )
                    continue

            outcome = client.fetch_case(
                requested_url=str(row["requested_url"]),
                canonical_url=str(row["canonical_url"]),
                case_title=str(row["case_title"]),
                search_rank=rank,
            )
            if (
                outcome.status == 200
                and outcome.validation is not None
                and outcome.validation.valid
            ):
                record_success(row, outcome=outcome, raw_dir=args.raw_dir)
                consecutive_blockers = 0
            else:
                record_failure(
                    row,
                    outcome=outcome,
                    raw_dir=args.raw_dir,
                    preexisting_warnings=existing_warnings,
                )
                validation_errors = (
                    set(outcome.validation.errors) if outcome.validation else set()
                )
                blocker = (
                    outcome.status is None
                    or outcome.status in (TRANSIENT_HTTP_STATUSES | {403})
                    or bool(
                        validation_errors.intersection(
                            {"ERROR_OR_CHALLENGE_TITLE", "MISSING_SHERLOC_DB_HEADER"}
                        )
                    )
                )
                consecutive_blockers = consecutive_blockers + 1 if blocker else 0

            processed += 1
            if processed % args.checkpoint_every == 0:
                atomic_write_csv(args.download_manifest, rows)
            if processed % args.progress_every == 0 or processed == len(targets):
                success_so_far = sum(
                    rows[target_rank - 1]["download_status"] == SUCCESS_STATUS
                    for target_rank in targets
                )
                print(
                    f"[{args.mode}] {processed}/{len(targets)} targets; "
                    f"rank {rank}; target successes={success_so_far}",
                    flush=True,
                )

            if consecutive_blockers >= args.circuit_breaker_threshold:
                circuit_breaker_reason = (
                    f"{consecutive_blockers} consecutive network, transient HTTP, "
                    "or challenge-like failures"
                )
                run_status = "CIRCUIT_BREAKER"
                break

        if run_status == "RUNNING":
            run_status = "COMPLETE"
    except KeyboardInterrupt:
        run_status = "INTERRUPTED"
        circuit_breaker_reason = "KeyboardInterrupt"
    except Exception as exc:
        run_status = "FAILED"
        circuit_breaker_reason = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        atomic_write_csv(args.download_manifest, rows)
        diagnostics = build_diagnostics(
            args=args,
            source_rows=source_rows,
            source_sha256=source_sha,
            rows=rows,
            targets=targets,
            pilot_coverage=pilot_coverage,
            client=client,
            started_at=started_at,
            elapsed_seconds=time.monotonic() - monotonic_started,
            run_status=run_status,
            circuit_breaker_reason=circuit_breaker_reason,
            download_manifest_path=args.download_manifest,
            prior_pilot_run=prior_pilot_run,
            prior_full_runs=prior_full_runs,
        )
        atomic_write_json(args.diagnostics, diagnostics)

    target_failures = diagnostics["counts"]["target_failures"]
    exit_code = 0 if run_status == "COMPLETE" and target_failures == 0 else 2
    return diagnostics, exit_code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download original SHERLOC case HTML bytes from the frozen URL manifest."
        )
    )
    parser.add_argument("--mode", choices=("pilot", "full"), default="pilot")
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument(
        "--download-manifest", type=Path, default=DEFAULT_DOWNLOAD_MANIFEST
    )
    parser.add_argument("--diagnostics", type=Path, default=DEFAULT_DIAGNOSTICS)
    parser.add_argument("--pilot-size", type=int, default=DEFAULT_PILOT_SIZE)
    parser.add_argument("--rank-bands", type=int, default=DEFAULT_RANK_BANDS)
    parser.add_argument("--delay-seconds", type=float, default=1.0)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--backoff-base-seconds", type=float, default=2.0)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--circuit-breaker-threshold", type=int, default=3)
    parser.add_argument("--ca-bundle", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--allow-full-without-pilot",
        action="store_true",
        help="override the normal full-mode pilot completion guard",
    )
    return parser


def validate_arguments(args: argparse.Namespace) -> None:
    for attribute in (
        "source_manifest",
        "raw_dir",
        "download_manifest",
        "diagnostics",
    ):
        value = getattr(args, attribute)
        if not value.is_absolute():
            setattr(args, attribute, (REPO_ROOT / value).resolve())
    if args.delay_seconds < 0:
        raise SystemExit("--delay-seconds cannot be negative")
    if args.timeout_seconds <= 0:
        raise SystemExit("--timeout-seconds must be positive")
    if args.max_retries < 0:
        raise SystemExit("--max-retries cannot be negative")
    if args.backoff_base_seconds < 0:
        raise SystemExit("--backoff-base-seconds cannot be negative")
    if args.checkpoint_every < 1 or args.progress_every < 1:
        raise SystemExit("checkpoint/progress intervals must be positive")
    if args.circuit_breaker_threshold < 1:
        raise SystemExit("--circuit-breaker-threshold must be positive")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_arguments(args)
    try:
        diagnostics, exit_code = run(args)
    except (DownloadError, HTTPError, URLError, OSError) as exc:
        print(f"Page download failed: {exc}", file=sys.stderr)
        return 1

    counts = diagnostics["counts"]
    print(
        f"SHERLOC {args.mode}: targets={counts['target_rows_this_mode']}, "
        f"target_successes={counts['target_successes']}, "
        f"target_failures={counts['target_failures']}, "
        f"all_successful={counts['all_successful_downloads']}"
    )
    print(f"Download manifest: {args.download_manifest}")
    print(f"Diagnostics: {args.diagnostics}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
