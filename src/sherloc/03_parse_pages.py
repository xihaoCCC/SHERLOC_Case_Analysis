#!/usr/bin/env python3
"""Parse the frozen SHERLOC trafficking-in-persons corpus.

Parser v2 deliberately uses only the Python standard library so that the
frozen raw HTML can be reparsed without a browser or an undeclared HTML-parser
dependency.  It implements three gates in order: the 19 manual regression
fixtures, a deterministic corpus challenge set, and then the complete corpus.

The parser is an extraction layer, not a normalization layer.  SHERLOC labels
and values are retained as displayed, multilingual panes remain separate, and
the trafficking sidebar and legacy Keywords section are never reconciled.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import tempfile
import time
import traceback
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple
from urllib.parse import quote, urljoin, urlsplit


PARSER_VERSION = "2.0.0"
SCHEMA_VERSION = "sherloc-extraction-contract-v2"
CORPUS_SNAPSHOT_DATE = "2026-08-09"

STATUS_FOUND = "FOUND"
STATUS_SECTION_ABSENT = "SECTION_ABSENT"
STATUS_EMPTY = "EMPTY"
STATUS_PARTIAL = "PARTIAL"
STATUS_PARSE_ERROR = "PARSE_ERROR"

VALID_STATUSES = {
    STATUS_FOUND,
    STATUS_SECTION_ABSENT,
    STATUS_EMPTY,
    STATUS_PARTIAL,
    STATUS_PARSE_ERROR,
}

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS_MANIFEST = REPO_ROOT / "data/manifests/case_urls.csv"
DEFAULT_DOWNLOAD_MANIFEST = REPO_ROOT / "logs/page_download_manifest.csv"
DEFAULT_RAW_HTML_DIR = REPO_ROOT / "data/raw_html"
DEFAULT_FIXTURE_DIR = REPO_ROOT / "data/sample_html"
DEFAULT_JSONL = REPO_ROOT / "data/interim/sherloc_cases_raw.jsonl"
DEFAULT_COVERAGE = REPO_ROOT / "outputs/metrics/parser_coverage.csv"
DEFAULT_DIAGNOSTICS = REPO_ROOT / "logs/parser_diagnostics.json"
DEFAULT_REPORT = REPO_ROOT / "docs/parser_v2_report.md"

LANGUAGE_LABELS = {
    "english": "en",
    "français": "fr",
    "french": "fr",
    "español": "es",
    "spanish": "es",
    "português": "pt",
    "portuguese": "pt",
    "italiano": "it",
    "italian": "it",
    "deutsch": "de",
    "german": "de",
    "中文": "zh",
    "chinese": "zh",
    "русский": "ru",
    "russian": "ru",
    "عربي": "ar",
    "العربية": "ar",
    "arabic": "ar",
}
LANGUAGE_SUFFIXES = tuple(sorted(set(LANGUAGE_LABELS.values()), key=len, reverse=True))

SIDEBAR_CORE_CLASSES = {
    "offences": "crimeTypes_traffickingPersonsCrimeType_offences",
    "acts": "crimeTypes_traffickingPersonsCrimeType_actsInvolved",
    "means": "crimeTypes_traffickingPersonsCrimeType_meansUsed",
    "exploitative_purposes": "crimeTypes_traffickingPersonsCrimeType_exploitativePurposes",
    "form_of_trafficking": "crimeTypes_traffickingPersonsCrimeType_formOfTrafficking",
    "sector": "crimeTypes_traffickingPersonsCrimeType_sectorsInWhichExploitationTakesPlace",
    "keywords": "crimeTypes_traffickingPersonsCrimeType_keywords",
}

LEGACY_CORE_LABELS = {
    "acts": {"Acts"},
    "means": {"Means"},
    "exploitative_purposes": {"Purpose of Exploitation"},
    "form_of_trafficking": {"Form of Trafficking"},
    "sector": {"Sector in which exploitation takes place"},
}

KNOWN_MAIN_HEADINGS = {
    "Commentary and Significant Features": "commentary_significant_features",
    "Cross-Cutting Issues": "cross_cutting_issues",
    "Procedural Information": "procedural_information",
    "Sources / Citations": "sources_citations",
    "Sources/Citations": "sources_citations",
    "Attachments": "attachments",
    "Court": "court",
    "Jurisdiction": "jurisdiction",
    "Victims / Witnesses Summary": "victims_witnesses_summary",
}

PERSON_CONTAINER_CLASSES = {
    "victimsPlaintiffs": "person_role",
    "defendantsRespondents": "defendant_respondent",
}

VOID_ELEMENTS = {
    "area", "base", "br", "col", "embed", "hr", "img", "input", "link",
    "meta", "param", "source", "track", "wbr",
}
BLOCK_ELEMENTS = {
    "address", "article", "aside", "blockquote", "div", "dl", "dt", "dd",
    "fieldset", "figcaption", "figure", "footer", "form", "h1", "h2", "h3",
    "h4", "h5", "h6", "header", "hr", "li", "main", "nav", "ol", "p",
    "pre", "section", "table", "tbody", "td", "tfoot", "th", "thead", "tr", "ul",
}
TEXT_EXCLUDED_TAGS = {"script", "style", "noscript", "template"}

UNRESERVED = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~"
)
ALLOWED_HOSTS = {"www.unodc.org", "sherloc.unodc.org"}
YEAR_SEGMENT = re.compile(r"/(19\d{2}|20\d{2})/")

# Every missing-Fact rank is mandatory.  The additional anchors were selected
# deterministically from the frozen manifest and corpus-wide structural audit.
KNOWN_MISSING_FACT_RANKS = {
    84, 307, 318, 380, 431, 543, 575, 592, 593, 845, 871, 911, 923,
    952, 965, 1007, 1022, 1231, 1255, 1295, 1317, 1419, 1423, 1435, 1482,
}
CHALLENGE_ANCHOR_REASONS = {
    1: ["newest_url_year", "structured_only_labels", "direct_english_fact"],
    3: ["drug_url_path", "rare_sidebar_protection", "no_person_role_section"],
    4: ["empty_legacy_keywords", "criminal_group_url_path"],
    7: ["rare_sidebar_sector_other"],
    9: ["maximum_sidebar_field_variety", "rare_other_exploitative_purpose"],
    46: ["explicit_legal_entity_defendant"],
    63: ["malformed_nested_charge_subjects", "missing_closing_divs"],
    62: ["unlabeled_english_plus_french_fact", "no_active_fact_pane", "migrant_smuggling_url_path"],
    89: ["multilingual_procedural", "migrants_heading", "rare_jurisdiction"],
    91: ["arabic_multilingual_marker"],
    104: ["unlabeled_english_plus_spanish_fact", "no_active_fact_pane"],
    138: ["corruption_url_path", "large_multilingual_multi_defendant"],
    226: ["multilingual_person_section", "multi_defendant"],
    446: ["well_formed_multilingual_contrast"],
    574: ["three_pane_duplicate_id_group"],
    767: ["widest_multilingual_section_coverage"],
    937: ["largest_html", "corruption_url_path", "multi_party_stress"],
    1171: ["maximum_defendant_count"],
    1483: ["oldest_url_year"],
    1489: ["malformed_nested_charge_subjects", "missing_closing_divs"],
    1515: ["smallest_html", "yearless_url", "neither_label_source"],
}


class ParsePipelineError(RuntimeError):
    """Raised when a validation gate makes a full-corpus run unsafe."""


class Node:
    """Minimal loss-aware HTML node used by the dependency-free parser."""

    __slots__ = ("tag", "attrs", "parent", "children")

    def __init__(
        self,
        tag: str,
        attrs: Optional[Dict[str, str]] = None,
        parent: Optional["Node"] = None,
    ) -> None:
        self.tag = tag.lower()
        self.attrs = attrs or {}
        self.parent = parent
        self.children: List[Any] = []

    @property
    def classes(self) -> List[str]:
        return [part for part in self.attrs.get("class", "").split() if part]

    def has_class(self, name: str) -> bool:
        return name in self.classes

    def element_children(self) -> List["Node"]:
        return [child for child in self.children if isinstance(child, Node)]

    def ancestors(self) -> Iterator["Node"]:
        current = self.parent
        while current is not None:
            yield current
            current = current.parent

    def descendants(self, include_self: bool = False) -> Iterator["Node"]:
        if include_self:
            yield self
        for child in self.children:
            if isinstance(child, Node):
                yield child
                yield from child.descendants()


class TolerantHTMLTreeBuilder(HTMLParser):
    """Build a small DOM while tolerating SHERLOC's duplicated document shells."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.root = Node("document")
        self.stack: List[Node] = [self.root]

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        values = {str(key).lower(): (value or "") for key, value in attrs}
        node = Node(tag, values, self.stack[-1])
        self.stack[-1].children.append(node)
        if tag.lower() not in VOID_ELEMENTS:
            self.stack.append(node)

    def handle_startendtag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        values = {str(key).lower(): (value or "") for key, value in attrs}
        self.stack[-1].children.append(Node(tag, values, self.stack[-1]))

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        for index in range(len(self.stack) - 1, 0, -1):
            if self.stack[index].tag == tag:
                del self.stack[index:]
                return

    def handle_data(self, data: str) -> None:
        if data:
            self.stack[-1].children.append(data)

    def error(self, message: str) -> None:  # pragma: no cover - required on old Python
        return


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def warning(code: str, message: str, location: str, severity: str = "WARNING") -> Dict[str, str]:
    return {"code": code, "message": message, "location": location, "severity": severity}


def class_predicate(name: str) -> Callable[[Node], bool]:
    return lambda node: node.has_class(name)


def first_node(root: Node, predicate: Callable[[Node], bool], include_self: bool = False) -> Optional[Node]:
    return next((node for node in root.descendants(include_self=include_self) if predicate(node)), None)


def all_nodes(root: Node, predicate: Callable[[Node], bool], include_self: bool = False) -> List[Node]:
    return [node for node in root.descendants(include_self=include_self) if predicate(node)]


def closest_ancestor(node: Node, predicate: Callable[[Node], bool]) -> Optional[Node]:
    return next((ancestor for ancestor in node.ancestors() if predicate(ancestor)), None)


def nearest_owner(node: Node, owner_predicate: Callable[[Node], bool]) -> Optional[Node]:
    return closest_ancestor(node, owner_predicate)


def top_level_owned_nodes(
    root: Node,
    item_predicate: Callable[[Node], bool],
    boundary_predicate: Optional[Callable[[Node], bool]] = None,
) -> List[Node]:
    """Return matching descendants not nested in another matching item.

    If a boundary predicate is supplied, an item is retained only when the
    nearest boundary ancestor is ``root`` (or when root itself is the boundary).
    """

    output: List[Node] = []
    for node in root.descendants():
        if not item_predicate(node):
            continue
        ancestor_item = closest_ancestor(node, item_predicate)
        if ancestor_item is not None and ancestor_item is not root:
            continue
        if boundary_predicate is not None:
            owner = closest_ancestor(node, boundary_predicate)
            if owner is not root:
                continue
        output.append(node)
    return output


def normalize_inline_text(value: str) -> Optional[str]:
    value = value.replace("\u00a0", " ")
    value = re.sub(r"\s+", " ", value).strip()
    return value or None


def normalize_block_text(value: str) -> Optional[str]:
    value = value.replace("\r\n", "\n").replace("\r", "\n").replace("\u00a0", " ")
    lines: List[str] = []
    for raw_line in value.split("\n"):
        line = re.sub(r"[\t\f\v ]+", " ", raw_line).strip()
        if line:
            lines.append(line)
        elif lines and lines[-1] != "":
            lines.append("")
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines) or None


def node_text(
    node: Optional[Node],
    *,
    preserve_blocks: bool = True,
    excluded_nodes: Optional[Set[int]] = None,
) -> Optional[str]:
    if node is None:
        return None
    excluded = excluded_nodes or set()
    pieces: List[str] = []

    def visit(current: Node) -> None:
        if id(current) in excluded or current.tag in TEXT_EXCLUDED_TAGS:
            return
        is_block = current.tag in BLOCK_ELEMENTS
        if preserve_blocks and is_block:
            pieces.append("\n")
        for child in current.children:
            if isinstance(child, Node):
                if child.tag == "br":
                    pieces.append("\n")
                else:
                    visit(child)
            else:
                pieces.append(str(child))
        if preserve_blocks and is_block:
            pieces.append("\n")

    visit(node)
    joined = "".join(pieces)
    return normalize_block_text(joined) if preserve_blocks else normalize_inline_text(joined)


def direct_child(node: Node, predicate: Callable[[Node], bool]) -> Optional[Node]:
    return next((child for child in node.element_children() if predicate(child)), None)


def direct_children(node: Node, predicate: Callable[[Node], bool]) -> List[Node]:
    return [child for child in node.element_children() if predicate(child)]


def parse_html_bytes(body: bytes) -> Tuple[Node, int]:
    text = body.decode("utf-8", errors="replace")
    replacement_count = text.count("\ufffd")
    builder = TolerantHTMLTreeBuilder()
    builder.feed(text)
    builder.close()
    return builder.root, replacement_count


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


def canonical_case_identity(url: str, base_url: str = "") -> Optional[str]:
    if not url:
        return None
    absolute = urljoin(base_url, url) if base_url else url
    parsed = urlsplit(absolute)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        return None
    try:
        port = parsed.port
    except ValueError:
        return None
    if port not in {None, 80, 443}:
        return None
    host = parsed.hostname.lower()
    if host not in ALLOWED_HOSTS:
        return None
    host = "www.unodc.org"
    path = re.sub(r"/{2,}", "/", parsed.path.replace("\\", "/"))
    path = re.sub(
        r"^/cld/(?:ar|zh|en|fr|ru|es)(?=/case-law-doc/)",
        "/cld",
        path,
        flags=re.IGNORECASE,
    )
    path = normalize_percent_path(path).rstrip("/")
    return f"{host}{path}"


def canonical_url_relation(actual: str, expected: str) -> str:
    if not actual:
        return "MISSING"
    if actual == expected:
        return "EXACT_MATCH"
    actual_identity = canonical_case_identity(actual)
    expected_identity = canonical_case_identity(expected)
    if actual_identity and actual_identity == expected_identity:
        return "CANONICAL_EQUIVALENT"
    return "MISMATCH"


def decorative_value(source_text: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """Remove one SHERLOC list bullet while retaining its source representation."""
    if source_text is None:
        return None, None
    value = re.sub(r"^[\u2022]\s*", "", source_text, count=1).strip()
    removed = "\u2022" if value != source_text else None
    return value or None, removed


def parse_int(value: Any) -> Optional[int]:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def bool_string(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def safe_json_value(value: str) -> Any:
    if not value:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def semantic_class_key(node: Node) -> Optional[str]:
    ignored = {
        "field", "fieldFullWidth", "line", "plainValue", "pull-left", "clear",
        "col-md-12", "col-lg-12", "col-xs-12",
    }
    return next((name for name in node.classes if name not in ignored), None)


def is_case_law_detail(node: Node) -> bool:
    return node.has_class("case-law-detail")


def owning_case_detail(node: Node) -> Optional[Node]:
    return node if is_case_law_detail(node) else closest_ancestor(node, is_case_law_detail)


def owned_descendants(
    root: Node,
    predicate: Callable[[Node], bool],
    owner_predicate: Callable[[Node], bool],
) -> List[Node]:
    return [
        node
        for node in root.descendants()
        if predicate(node) and nearest_owner(node, owner_predicate) is root
    ]


def extract_fields(scope: Node, *, exclude_within_classes: Optional[Set[str]] = None) -> List[Dict[str, Any]]:
    """Extract all ordered field nodes without turning them into a lossy dict."""

    excluded_classes = exclude_within_classes or set()

    def is_field(node: Node) -> bool:
        return node.has_class("field") or node.has_class("fieldFullWidth")

    field_nodes: List[Node] = []
    scope_detail = owning_case_detail(scope)
    for node in scope.descendants():
        if not is_field(node):
            continue
        if scope_detail is not None and closest_ancestor(node, is_case_law_detail) is not scope_detail:
            continue
        if any(
            any(ancestor.has_class(name) for name in excluded_classes)
            for ancestor in node.ancestors()
            if ancestor is not scope
        ):
            continue
        field_nodes.append(node)

    return serialize_field_nodes(field_nodes)


def serialize_field_nodes(field_nodes: Sequence[Node]) -> List[Dict[str, Any]]:
    """Serialize already selected field nodes with their nesting retained."""

    def is_field(node: Node) -> bool:
        return node.has_class("field") or node.has_class("fieldFullWidth")

    node_to_ordinal = {id(node): index + 1 for index, node in enumerate(field_nodes)}
    fields: List[Dict[str, Any]] = []
    for ordinal, field_node in enumerate(field_nodes, 1):
        parent_field = closest_ancestor(field_node, is_field)
        label_nodes = owned_descendants(
            field_node,
            lambda node: node.has_class("label"),
            is_field,
        )
        value_nodes = owned_descendants(
            field_node,
            lambda node: node.has_class("value"),
            is_field,
        )
        label_raw = node_text(label_nodes[0], preserve_blocks=False) if label_nodes else None
        value_parts = [node_text(node, preserve_blocks=True) for node in value_nodes]
        value_parts = [value for value in value_parts if value]
        raw_text = node_text(field_node, preserve_blocks=True)
        fields.append(
            {
                "ordinal": ordinal,
                "dom_classes": list(field_node.classes),
                "class_key": semantic_class_key(field_node),
                "parent_field_ordinal": node_to_ordinal.get(id(parent_field)) if parent_field else None,
                "label_raw": label_raw,
                "value_raw": "\n".join(value_parts) if value_parts else None,
                "raw_text": raw_text,
                "status": STATUS_FOUND if raw_text else STATUS_EMPTY,
            }
        )
    return fields


def infer_language(
    pane_id: Optional[str],
    href_id: Optional[str],
    tab_label: Optional[str],
    page_locale: Optional[str],
    pane_index: int,
) -> Tuple[Optional[str], str]:
    label_key = normalize_inline_text(tab_label or "")
    if label_key and label_key.casefold() in LANGUAGE_LABELS:
        return LANGUAGE_LABELS[label_key.casefold()], "explicit_tab_label"

    for source_name, candidate in (("pane_id_suffix", pane_id), ("tab_href_suffix", href_id)):
        candidate_lower = (candidate or "").lower()
        for suffix in LANGUAGE_SUFFIXES:
            if re.search(rf"(?:^|[_-]){re.escape(suffix)}$", candidate_lower) or (
                candidate_lower.endswith(suffix)
                and candidate_lower[:-len(suffix)].isdigit()
            ):
                return suffix, source_name

    if page_locale:
        method = "page_locale_default_first_pane" if pane_index == 1 else "page_locale_fallback"
        return page_locale.lower(), method
    return None, "unknown"


def nearest_tab_group_nav(section: Node, tab_content: Node) -> Optional[Node]:
    """Find the nav list associated with a tab-content node without ID lookup."""

    def is_nav(node: Node) -> bool:
        return node.tag == "ul" and node.has_class("nav-tabs")

    parent = tab_content.parent
    if parent is not None:
        siblings = parent.element_children()
        try:
            position = siblings.index(tab_content)
        except ValueError:
            position = 0
        for sibling in reversed(siblings[:position]):
            if is_nav(sibling):
                return sibling

    candidates = [
        node
        for node in section.descendants()
        if is_nav(node) and not any(ancestor is tab_content for ancestor in node.ancestors())
    ]
    tab_nodes = list(section.descendants())
    positions = {id(node): index for index, node in enumerate(tab_nodes)}
    before = [node for node in candidates if positions.get(id(node), -1) < positions.get(id(tab_content), -1)]
    return before[-1] if before else None


def extract_tab_groups(
    section: Node,
    page_locale: Optional[str],
    location: str,
) -> Tuple[List[Dict[str, Any]], Dict[int, Dict[str, Any]], List[Dict[str, str]]]:
    """Extract every tab pane in DOM order, including duplicate IDs/hrefs."""

    warnings: List[Dict[str, str]] = []

    def is_tab_content(node: Node) -> bool:
        return node.has_class("tab-content")

    scope_detail = owning_case_detail(section)
    tab_contents = [
        node
        for node in top_level_owned_nodes(section, is_tab_content)
        if scope_detail is None or closest_ancestor(node, is_case_law_detail) is scope_detail
    ]
    groups: List[Dict[str, Any]] = []
    pane_context_by_node: Dict[int, Dict[str, Any]] = {}

    for group_index, tab_content in enumerate(tab_contents, 1):
        panes = [
            node
            for node in tab_content.descendants()
            if node.has_class("tab-pane") and nearest_owner(node, is_tab_content) is tab_content
        ]
        nav = nearest_tab_group_nav(section, tab_content)
        anchors: List[Node] = []
        if nav is not None:
            anchors = [
                node
                for node in nav.descendants()
                if node.tag == "a" and "href" in node.attrs
            ]

        pane_ids = [pane.attrs.get("id") or None for pane in panes]
        href_ids = [
            (anchor.attrs.get("href", "")[1:] if anchor.attrs.get("href", "").startswith("#") else anchor.attrs.get("href") or None)
            for anchor in anchors
        ]
        tab_labels = [node_text(anchor, preserve_blocks=False) for anchor in anchors]
        group_warnings: List[Dict[str, str]] = []

        if anchors and len(anchors) != len(panes):
            group_warnings.append(
                warning(
                    "TAB_PANE_COUNT_MISMATCH",
                    f"Found {len(anchors)} tab links and {len(panes)} panes; panes were paired by DOM ordinal.",
                    f"{location}.tab_groups[{group_index}]",
                )
            )
        duplicate_pane_ids = sorted({value for value, count in Counter(pane_ids).items() if value and count > 1})
        duplicate_hrefs = sorted({value for value, count in Counter(href_ids).items() if value and count > 1})
        if duplicate_pane_ids:
            group_warnings.append(
                warning(
                    "DUPLICATE_TAB_PANE_ID",
                    f"Duplicate pane IDs retained in DOM order: {duplicate_pane_ids}",
                    f"{location}.tab_groups[{group_index}]",
                )
            )
        if duplicate_hrefs:
            group_warnings.append(
                warning(
                    "DUPLICATE_TAB_HREF",
                    f"Duplicate tab hrefs retained in DOM order: {duplicate_hrefs}",
                    f"{location}.tab_groups[{group_index}]",
                )
            )

        active_count = sum(1 for pane in panes if "active" in pane.classes)
        if panes and active_count == 0:
            group_warnings.append(
                warning(
                    "TAB_GROUP_NO_ACTIVE_PANE",
                    "No pane is marked active; every pane was still extracted.",
                    f"{location}.tab_groups[{group_index}]",
                    severity="INFO",
                )
            )
        elif active_count > 1:
            group_warnings.append(
                warning(
                    "TAB_GROUP_MULTIPLE_ACTIVE_PANES",
                    f"{active_count} panes are marked active; every pane was retained.",
                    f"{location}.tab_groups[{group_index}]",
                )
            )

        pane_records: List[Dict[str, Any]] = []
        for pane_index, pane in enumerate(panes, 1):
            anchor = anchors[pane_index - 1] if pane_index <= len(anchors) else None
            pane_id = pane.attrs.get("id") or None
            href_raw = anchor.attrs.get("href") if anchor is not None else None
            href_id = href_raw[1:] if href_raw and href_raw.startswith("#") else href_raw
            tab_label = node_text(anchor, preserve_blocks=False) if anchor is not None else None
            language, language_method = infer_language(
                pane_id,
                href_id,
                tab_label,
                page_locale,
                pane_index,
            )
            raw_text = node_text(pane, preserve_blocks=True)
            pane_status = STATUS_FOUND if raw_text else STATUS_EMPTY
            record = {
                "group_index": group_index,
                "pane_index": pane_index,
                "language": language,
                "language_detection_method": language_method,
                "pane_id_raw": pane_id,
                "tab_href_raw": href_raw,
                "tab_label_raw": tab_label,
                "is_active_in_html": "active" in pane.classes,
                "status": pane_status,
                "text_raw": raw_text,
                "fields": extract_fields(pane),
            }
            pane_records.append(record)
            pane_context_by_node[id(pane)] = {
                key: record[key]
                for key in (
                    "group_index", "pane_index", "language", "language_detection_method",
                    "pane_id_raw", "tab_href_raw", "tab_label_raw", "is_active_in_html",
                )
            }

        group_status = STATUS_FOUND if any(pane["text_raw"] for pane in pane_records) else STATUS_EMPTY
        if any(item["severity"] == "WARNING" for item in group_warnings) and group_status == STATUS_FOUND:
            group_status = STATUS_PARTIAL
        groups.append(
            {
                "group_index": group_index,
                "status": group_status,
                "pane_count": len(pane_records),
                "panes": pane_records,
                "warnings": group_warnings,
            }
        )
        warnings.extend(group_warnings)

    return groups, pane_context_by_node, warnings


def pane_context_for(node: Node, pane_context_by_node: Dict[int, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    pane = next((ancestor for ancestor in node.ancestors() if ancestor.has_class("tab-pane")), None)
    return dict(pane_context_by_node[id(pane)]) if pane is not None and id(pane) in pane_context_by_node else None


def section_heading(section: Node) -> Optional[str]:
    heading = direct_child(section, lambda node: node.tag == "h3")
    return node_text(heading, preserve_blocks=False)


def section_content_text(section: Node, tab_groups_present: bool) -> Optional[str]:
    excluded: Set[int] = set()
    for child in section.element_children():
        if child.tag in {"h2", "h3"} or child.has_class("headericon"):
            excluded.add(id(child))
    for node in section.descendants():
        if is_case_law_detail(node) and closest_ancestor(node, is_case_law_detail) is section:
            excluded.add(id(node))
        if node.tag == "ul" and node.has_class("nav-tabs"):
            excluded.add(id(node))
        if tab_groups_present and node.has_class("tab-content"):
            excluded.add(id(node))
    return node_text(section, preserve_blocks=True, excluded_nodes=excluded)


def extract_generic_section(
    section: Optional[Node],
    page_locale: Optional[str],
    location: str,
) -> Dict[str, Any]:
    if section is None:
        return {
            "status": STATUS_SECTION_ABSENT,
            "heading_raw": None,
            "non_pane_text_raw": None,
            "fields": [],
            "tab_groups": [],
            "warnings": [],
        }

    heading = section_heading(section)
    tab_groups, _, tab_warnings = extract_tab_groups(section, page_locale, location)
    non_pane_text = section_content_text(section, bool(tab_groups))
    fields = extract_fields(section)
    has_content = bool(non_pane_text) or any(
        pane["text_raw"]
        for group in tab_groups
        for pane in group["panes"]
    )
    status = STATUS_FOUND if has_content else STATUS_EMPTY
    if any(item["severity"] == "WARNING" for item in tab_warnings) and has_content:
        status = STATUS_PARTIAL
    return {
        "status": status,
        "heading_raw": heading,
        "non_pane_text_raw": non_pane_text,
        "fields": fields,
        "tab_groups": tab_groups,
        "warnings": tab_warnings,
    }


def extract_fact_summary(
    section: Optional[Node],
    page_locale: Optional[str],
) -> Dict[str, Any]:
    absent = {
        "status": STATUS_SECTION_ABSENT,
        "heading_raw": None,
        "variants": [],
        "english_text_raw": None,
        "english_variant_indices": [],
        "warnings": [],
    }
    if section is None:
        return absent

    location = "narrative.fact_summary"
    container = first_node(section, class_predicate("factSummary"))
    if container is None:
        item = warning(
            "FACT_SUMMARY_CONTAINER_MISSING",
            "A Fact Summary section heading exists, but no .factSummary container was found.",
            location,
        )
        result = dict(absent)
        result.update(
            {
                "status": STATUS_PARTIAL,
                "heading_raw": section_heading(section),
                "warnings": [item],
            }
        )
        return result

    tab_groups, _, tab_warnings = extract_tab_groups(container, page_locale, location)
    variants: List[Dict[str, Any]] = []
    for group in tab_groups:
        variants.extend(group["panes"])
    if not tab_groups:
        text = section_content_text(container, False)
        variants.append(
            {
                "group_index": None,
                "pane_index": 1,
                "language": page_locale,
                "language_detection_method": "page_locale_single_block" if page_locale else "unknown",
                "pane_id_raw": container.attrs.get("id") or None,
                "tab_href_raw": None,
                "tab_label_raw": None,
                "is_active_in_html": True,
                "status": STATUS_FOUND if text else STATUS_EMPTY,
                "text_raw": text,
                "fields": extract_fields(container),
            }
        )

    usable = [variant for variant in variants if variant.get("text_raw")]
    english_indices = [
        index
        for index, variant in enumerate(variants, 1)
        if variant.get("language") == "en" and variant.get("text_raw")
    ]
    english_text = variants[english_indices[0] - 1]["text_raw"] if english_indices else None
    warnings = list(tab_warnings)
    if len(english_indices) > 1:
        warnings.append(
            warning(
                "MULTIPLE_ENGLISH_FACT_VARIANTS",
                f"{len(english_indices)} nonempty Fact Summary variants were inferred as English; all were retained.",
                location,
                severity="INFO",
            )
        )
    status = STATUS_FOUND if usable else STATUS_EMPTY
    if any(item["severity"] == "WARNING" for item in warnings) and usable:
        status = STATUS_PARTIAL
    return {
        "status": status,
        "heading_raw": section_heading(section),
        "variants": variants,
        "english_text_raw": english_text,
        "english_variant_indices": english_indices,
        "warnings": warnings,
    }


def next_element_sibling(node: Node, predicate: Optional[Callable[[Node], bool]] = None) -> Optional[Node]:
    if node.parent is None:
        return None
    siblings = node.parent.element_children()
    try:
        start = siblings.index(node) + 1
    except ValueError:
        return None
    if start >= len(siblings):
        return None
    sibling = siblings[start]
    return sibling if predicate is None or predicate(sibling) else None


def crime_badge_type(node: Node) -> Optional[str]:
    suffix = "CrimeType-details-badge"
    for name in node.classes:
        if name.endswith(suffix) and name != "crimeType-details-badge":
            return name[: -len("-details-badge")]
    return None


def extract_badge_fields(badge: Node, badge_index: int) -> List[Dict[str, Any]]:
    fields: List[Dict[str, Any]] = []
    heading_nodes = [
        node
        for node in badge.descendants()
        if node.tag in {"h3", "h4", "h5", "h6"}
        and any(name.startswith("crimeTypes_") for name in node.classes)
        and closest_ancestor(node, class_predicate("crimeType-details-badge")) is badge
    ]
    for field_index, heading in enumerate(heading_nodes, 1):
        container = next_element_sibling(heading, class_predicate("containerListElement"))
        value_nodes: List[Node] = []
        if container is not None:
            value_nodes = [
                node
                for node in container.descendants()
                if node.has_class("value")
                and closest_ancestor(node, class_predicate("containerListElement")) is container
            ]
        source_values = [node_text(node, preserve_blocks=True) for node in value_nodes]
        source_values = [value for value in source_values if value]
        value_records: List[Dict[str, Any]] = []
        for value_index, source_value in enumerate(source_values, 1):
            extracted, removed = decorative_value(source_value)
            value_records.append(
                {
                    "ordinal": value_index,
                    "value_raw": extracted,
                    "source_text_raw": source_value,
                    "decorative_prefix_removed": removed,
                }
            )
        heading_classes = list(heading.classes)
        structural_class = next(
            (name for name in heading_classes if name.startswith("crimeTypes_")),
            None,
        )
        fields.append(
            {
                "ordinal": field_index,
                "badge_ordinal": badge_index,
                "structural_class": structural_class,
                "heading_raw": node_text(heading, preserve_blocks=False),
                "heading_classes": heading_classes,
                "status": STATUS_FOUND if value_records else (STATUS_EMPTY if container is not None else STATUS_PARTIAL),
                "values_raw": [record["value_raw"] for record in value_records if record["value_raw"]],
                "value_records": value_records,
                "container_present": container is not None,
                "raw_text": node_text(container, preserve_blocks=True),
            }
        )
    return fields


def extract_crime_badges(root: Node) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, str]]]:
    def is_badge(node: Node) -> bool:
        return node.has_class("crimeType-details-badge")

    badge_nodes = top_level_owned_nodes(root, is_badge)
    badges: List[Dict[str, Any]] = []
    for badge_index, badge in enumerate(badge_nodes, 1):
        label_node = first_node(badge, class_predicate("media-heading"))
        badge_type = crime_badge_type(badge)
        badges.append(
            {
                "ordinal": badge_index,
                "badge_type": badge_type,
                "dom_classes": list(badge.classes),
                "label_raw": node_text(label_node, preserve_blocks=False),
                "fields": extract_badge_fields(badge, badge_index),
                "raw_text": node_text(badge, preserve_blocks=True),
            }
        )

    warnings: List[Dict[str, str]] = []
    trafficking_badges = [
        badge for badge in badges if badge["badge_type"] == "traffickingPersonsCrimeType"
    ]
    if not trafficking_badges:
        warnings.append(
            warning(
                "TRAFFICKING_BADGE_ABSENT",
                "No traffickingPersonsCrimeType badge was found even though corpus membership comes from the frozen trafficking result set.",
                "trafficking_sidebar",
            )
        )
    if len(trafficking_badges) > 1:
        warnings.append(
            warning(
                "MULTIPLE_TRAFFICKING_BADGES",
                f"{len(trafficking_badges)} trafficking badges were found; each source is retained separately.",
                "trafficking_sidebar",
            )
        )

    core_fields: Dict[str, Dict[str, Any]] = {}
    for core_name, structural_class in SIDEBAR_CORE_CLASSES.items():
        sources: List[Dict[str, Any]] = []
        for badge in trafficking_badges:
            for field in badge["fields"]:
                if field["structural_class"] == structural_class:
                    sources.append(
                        {
                            "badge_ordinal": badge["ordinal"],
                            "field_ordinal": field["ordinal"],
                            "heading_raw": field["heading_raw"],
                            "structural_class": structural_class,
                            "status": field["status"],
                            "values_raw": list(field["values_raw"]),
                            "value_records": list(field["value_records"]),
                        }
                    )
        values = [value for source in sources for value in source["values_raw"]]
        if not sources:
            status = STATUS_SECTION_ABSENT
        elif not values:
            status = STATUS_EMPTY
        elif len(trafficking_badges) > 1 or any(source["status"] == STATUS_PARTIAL for source in sources):
            status = STATUS_PARTIAL
        else:
            status = STATUS_FOUND
        core_fields[core_name] = {
            "status": status,
            "values_raw": values,
            "sources": sources,
        }

    known_classes = set(SIDEBAR_CORE_CLASSES.values())
    additional_fields = [
        {
            "badge_ordinal": badge["ordinal"],
            **field,
        }
        for badge in trafficking_badges
        for field in badge["fields"]
        if field["structural_class"] not in known_classes
    ]
    if not trafficking_badges:
        status = STATUS_SECTION_ABSENT
    elif not any(badge["fields"] for badge in trafficking_badges):
        status = STATUS_EMPTY
    elif warnings:
        status = STATUS_PARTIAL
    else:
        status = STATUS_FOUND
    trafficking = {
        "status": status,
        "badge_count": len(trafficking_badges),
        "badge_ordinals": [badge["ordinal"] for badge in trafficking_badges],
        "fields": core_fields,
        "additional_fields": additional_fields,
        "warnings": warnings,
    }
    return badges, trafficking, warnings


def extract_legacy_keywords(
    section: Optional[Node],
    page_locale: Optional[str],
) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
    core = {
        key: {"status": STATUS_SECTION_ABSENT, "values_raw": [], "category_ordinals": []}
        for key in LEGACY_CORE_LABELS
    }
    if section is None:
        return (
            {
                "status": STATUS_SECTION_ABSENT,
                "heading_raw": None,
                "non_pane_text_raw": None,
                "categories": [],
                "core_fields": core,
                "tab_groups": [],
                "warnings": [],
            },
            [],
        )

    location = "legacy_keywords"
    tab_groups, _, tab_warnings = extract_tab_groups(section, page_locale, location)
    category_nodes = [
        node
        for node in section.descendants()
        if node.has_class("keywordCategory") and node.has_class("field")
        and closest_ancestor(node, class_predicate("case-law-detail")) is section
    ]
    categories: List[Dict[str, Any]] = []
    for ordinal, category in enumerate(category_nodes, 1):
        label_node = first_node(
            category,
            lambda node: node.has_class("label")
            and closest_ancestor(node, class_predicate("keywordCategory")) is category,
        )
        label_raw = node_text(label_node, preserve_blocks=False)
        value_nodes = [
            node
            for node in category.descendants()
            if node.has_class("tag")
            and closest_ancestor(node, class_predicate("keywordCategory")) is category
        ]
        if not value_nodes:
            value_nodes = [
                node
                for node in category.descendants()
                if node.has_class("value")
                and closest_ancestor(node, class_predicate("keywordCategory")) is category
            ]
        values = [node_text(node, preserve_blocks=True) for node in value_nodes]
        values = [value for value in values if value]
        categories.append(
            {
                "ordinal": ordinal,
                "label_raw": label_raw,
                "values_raw": values,
                "status": STATUS_FOUND if values else STATUS_EMPTY,
                "raw_text": node_text(category, preserve_blocks=True),
            }
        )

    for key, accepted_labels in LEGACY_CORE_LABELS.items():
        matching = [
            category
            for category in categories
            if (category["label_raw"] or "").rstrip(":").strip() in accepted_labels
        ]
        values = [value for category in matching for value in category["values_raw"]]
        core[key] = {
            "status": STATUS_FOUND if values else (STATUS_EMPTY if matching else STATUS_SECTION_ABSENT),
            "values_raw": values,
            "category_ordinals": [category["ordinal"] for category in matching],
        }

    non_pane_text = section_content_text(section, bool(tab_groups))
    status = STATUS_FOUND if categories else STATUS_EMPTY
    if any(item["severity"] == "WARNING" for item in tab_warnings) and status == STATUS_FOUND:
        status = STATUS_PARTIAL
    record = {
        "status": status,
        "heading_raw": section_heading(section),
        "non_pane_text_raw": non_pane_text,
        "categories": categories,
        "core_fields": core,
        "tab_groups": tab_groups,
        "warnings": tab_warnings,
    }
    return record, tab_warnings


def extract_person_section(
    section: Node,
    container: Node,
    container_type: str,
    page_locale: Optional[str],
    section_index: int,
) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
    location = f"participants.sections[{section_index}]"
    tab_groups, pane_map, tab_warnings = extract_tab_groups(section, page_locale, location)

    def is_person(node: Node) -> bool:
        return node.has_class("person")

    persons = [
        node
        for node in container.descendants()
        if is_person(node)
        and closest_ancestor(node, is_person) is None
        and closest_ancestor(node, lambda item: item.has_class(container_type)) is container
    ]
    records: List[Dict[str, Any]] = []
    for ordinal, person in enumerate(persons, 1):
        raw_text = node_text(person, preserve_blocks=True)
        records.append(
            {
                "ordinal": ordinal,
                "status": STATUS_FOUND if raw_text else STATUS_EMPTY,
                "dom_classes": list(person.classes),
                "pane_provenance": pane_context_for(person, pane_map),
                "fields": extract_fields(person),
                "raw_text": raw_text,
            }
        )

    # Extract section/container metadata outside every repeated person subtree.
    person_ids = {id(person) for person in persons}
    metadata_nodes = [
        field_node
        for field_node in container.descendants()
        if (field_node.has_class("field") or field_node.has_class("fieldFullWidth"))
        and not any(id(ancestor) in person_ids for ancestor in field_node.ancestors())
    ]
    metadata_fields = serialize_field_nodes(metadata_nodes)

    has_text = bool(section_content_text(section, bool(tab_groups)))
    status = STATUS_FOUND if records or has_text else STATUS_EMPTY
    if any(item["severity"] == "WARNING" for item in tab_warnings) and status == STATUS_FOUND:
        status = STATUS_PARTIAL
    record = {
        "status": status,
        "dom_role_container_type": container_type,
        "role_family": PERSON_CONTAINER_CLASSES[container_type],
        "visible_section_heading_raw": section_heading(section),
        "container_dom_classes": list(container.classes),
        "non_pane_text_raw": section_content_text(section, bool(tab_groups)),
        "tab_groups": tab_groups,
        "container_metadata_fields": metadata_fields,
        "records": records,
        "record_count": len(records),
        "warnings": tab_warnings,
    }
    return record, tab_warnings


def fields_owned_by_record(record: Node, excluded_record_classes: Set[str]) -> List[Dict[str, Any]]:
    """Extract fields belonging to a record but outside nested charge blocks."""
    excluded_nodes = [
        node for node in record.descendants() if any(node.has_class(name) for name in excluded_record_classes)
    ]
    excluded_ids = {id(node) for node in excluded_nodes}

    def inside_excluded(node: Node) -> bool:
        return any(id(ancestor) in excluded_ids for ancestor in node.ancestors())

    field_nodes = [
        node
        for node in record.descendants()
        if (node.has_class("field") or node.has_class("fieldFullWidth"))
        and not inside_excluded(node)
    ]
    return serialize_field_nodes(field_nodes)


def extract_charges_section(
    section: Optional[Node],
    page_locale: Optional[str],
) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
    if section is None:
        return (
            {
                "status": STATUS_SECTION_ABSENT,
                "heading_raw": None,
                "non_pane_text_raw": None,
                "tab_groups": [],
                "section_metadata_fields": [],
                "subject_records": [],
                "orphan_charge_records": [],
                "warnings": [],
            },
            [],
        )

    location = "charges_claims_decisions"
    tab_groups, pane_map, tab_warnings = extract_tab_groups(section, page_locale, location)
    charges_container = first_node(
        section,
        lambda node: node.has_class("charges")
        and closest_ancestor(node, is_case_law_detail) is section,
    )
    warnings = list(tab_warnings)
    if charges_container is None:
        warnings.append(
            warning(
                "CHARGES_CONTAINER_MISSING",
                "A Charges / Claims / Decisions section exists without a .charges container; raw section text is retained.",
                location,
            )
        )

    scope = charges_container or section

    def is_person(node: Node) -> bool:
        return node.has_class("person")

    # A small number of pages omit closing divs between charge subjects.  The
    # browser DOM consequently nests later explicit `.person` source blocks
    # inside the first.  Every explicit source block remains a subject record;
    # nesting is diagnosed rather than used to discard identities.
    subject_nodes = [
        node
        for node in scope.descendants()
        if is_person(node) and closest_ancestor(node, is_case_law_detail) is section
    ]
    nested_subject_count = sum(closest_ancestor(node, is_person) is not None for node in subject_nodes)
    if nested_subject_count:
        warnings.append(
            warning(
                "NESTED_CHARGE_SUBJECT_RECORDS",
                f"{nested_subject_count} explicit charge-subject .person blocks are nested inside another .person; all were retained in DOM order.",
                location,
            )
        )
    subjects: List[Dict[str, Any]] = []
    all_subject_charge_ids: Set[int] = set()
    for subject_index, subject in enumerate(subject_nodes, 1):
        charge_nodes = [
            node
            for node in subject.descendants()
            if node.has_class("charge")
            and closest_ancestor(node, class_predicate("charge")) is None
            and closest_ancestor(node, is_person) is subject
        ]
        charges: List[Dict[str, Any]] = []
        for charge_index, charge in enumerate(charge_nodes, 1):
            all_subject_charge_ids.add(id(charge))
            charge_raw_text = node_text(charge, preserve_blocks=True)
            charges.append(
                {
                    "ordinal": charge_index,
                    "status": STATUS_FOUND if charge_raw_text else STATUS_EMPTY,
                    "dom_classes": list(charge.classes),
                    "pane_provenance": pane_context_for(charge, pane_map),
                    "fields": extract_fields(charge),
                    "raw_text": charge_raw_text,
                }
            )
        nested_subject_nodes = [node for node in subject.descendants() if is_person(node)]
        dom_subtree_text = node_text(subject, preserve_blocks=True)
        subject_raw_text = node_text(
            subject,
            preserve_blocks=True,
            excluded_nodes={id(node) for node in nested_subject_nodes},
        )
        subjects.append(
            {
                "ordinal": subject_index,
                "status": STATUS_FOUND if (subject_raw_text or dom_subtree_text) else STATUS_EMPTY,
                "dom_classes": list(subject.classes),
                "pane_provenance": pane_context_for(subject, pane_map),
                "subject_and_disposition_fields": fields_owned_by_record(subject, {"charge", "person"}),
                "charges": charges,
                "charge_count": len(charges),
                "raw_text": subject_raw_text,
                "raw_text_excluded_nested_subject_count": len(nested_subject_nodes),
                "dom_subtree_text_raw": dom_subtree_text if nested_subject_nodes else None,
            }
        )

    orphan_nodes = [
        node
        for node in scope.descendants()
        if node.has_class("charge")
        and closest_ancestor(node, class_predicate("charge")) is None
        and closest_ancestor(node, is_case_law_detail) is section
        and id(node) not in all_subject_charge_ids
    ]
    orphan_records = [
        {
            "ordinal": ordinal,
            "status": STATUS_FOUND if node_text(node, preserve_blocks=True) else STATUS_EMPTY,
            "dom_classes": list(node.classes),
            "pane_provenance": pane_context_for(node, pane_map),
            "fields": extract_fields(node),
            "raw_text": node_text(node, preserve_blocks=True),
        }
        for ordinal, node in enumerate(orphan_nodes, 1)
    ]
    if orphan_records:
        warnings.append(
            warning(
                "ORPHAN_CHARGE_RECORDS",
                f"{len(orphan_records)} charge blocks were outside a subject person record; no relationship was invented.",
                location,
            )
        )

    has_text = bool(section_content_text(section, bool(tab_groups)))
    status = STATUS_FOUND if subjects or orphan_records or has_text else STATUS_EMPTY
    if any(item["severity"] == "WARNING" for item in warnings) and status == STATUS_FOUND:
        status = STATUS_PARTIAL
    record = {
        "status": status,
        "heading_raw": section_heading(section),
        "non_pane_text_raw": section_content_text(section, bool(tab_groups)),
        "tab_groups": tab_groups,
        "section_metadata_fields": fields_owned_by_record(scope, {"person", "charge"}),
        "subject_records": subjects,
        "orphan_charge_records": orphan_records,
        "warnings": warnings,
    }
    return record, warnings


def raw_descendant_data(node: Node) -> str:
    pieces: List[str] = []

    def visit(current: Node) -> None:
        for child in current.children:
            if isinstance(child, Node):
                visit(child)
            else:
                pieces.append(str(child))

    visit(node)
    return "".join(pieces)


def extract_meta_content(root: Node, *, property_name: str = "", name: str = "") -> Optional[str]:
    for node in root.descendants():
        if node.tag != "meta":
            continue
        if property_name and node.attrs.get("property", "").casefold() != property_name.casefold():
            continue
        if name and node.attrs.get("name", "").casefold() != name.casefold():
            continue
        value = normalize_inline_text(node.attrs.get("content", ""))
        if value:
            return value
    return None


def extract_page_locale(root: Node, source_url: Optional[str]) -> Tuple[Optional[str], str, Optional[str]]:
    html_node = first_node(root, lambda node: node.tag == "html")
    html_lang = normalize_inline_text(html_node.attrs.get("lang", "")) if html_node else None
    for script in all_nodes(root, lambda node: node.tag == "script"):
        match = re.search(
            r"(?:const|var|let)\s+pageLocale\s*=\s*['\"]([^'\"]+)['\"]",
            raw_descendant_data(script),
            flags=re.IGNORECASE,
        )
        if match:
            return match.group(1).lower(), "pageLocale_script", html_lang
    if html_lang:
        return html_lang.lower(), "html_lang", html_lang
    if source_url:
        match = re.search(r"/cld/([a-z]{2})/", source_url, flags=re.IGNORECASE)
        if match:
            return match.group(1).lower(), "url_locale", html_lang
    return None, "unknown", html_lang


def main_detail_sections(case_root: Node) -> List[Node]:
    # Explicit case-law-detail blocks remain sections even when malformed
    # missing closing tags make later sibling blocks appear nested in the DOM.
    return [node for node in case_root.descendants() if is_case_law_detail(node)]


def extract_visible_title(sections: Sequence[Node]) -> Optional[str]:
    for section in sections:
        heading = direct_child(section, lambda node: node.tag == "h2")
        if heading is None:
            continue
        title_node = first_node(heading, class_predicate("title"))
        value = node_text(title_node or heading, preserve_blocks=False)
        if value:
            return value
    return None


def extract_country(sections: Sequence[Node]) -> Optional[str]:
    for section in sections:
        country_node = first_node(section, class_predicate("countryNoHighlight"))
        if country_node is None:
            continue
        heading = first_node(country_node, lambda node: node.tag == "h3")
        value = node_text(heading, preserve_blocks=False)
        if value:
            return value
    return None


def parse_case_root(
    root: Node,
    *,
    body: bytes,
    actual_path: Path,
    source_row: Dict[str, str],
    download_row: Dict[str, str],
    input_kind: str,
    replacement_count: int,
    parsed_at: Optional[str] = None,
) -> Dict[str, Any]:
    parsed_at = parsed_at or utc_now()
    parse_warnings: List[Dict[str, str]] = []
    expected_url = source_row.get("canonical_url", "")

    case_root = first_node(root, lambda node: node.attrs.get("id") == "case-law-content")
    if case_root is None:
        raise ValueError("#case-law-content was not found")
    sections = main_detail_sections(case_root)
    if not sections:
        raise ValueError("No top-level .case-law-detail containers were found under #case-law-content")

    nested_detail_sections = [
        section for section in sections if closest_ancestor(section, is_case_law_detail) is not None
    ]
    for section in nested_detail_sections:
        nested_heading = section_heading(section)
        parse_warnings.append(
            warning(
                "NESTED_CASE_LAW_DETAIL",
                f"An explicit .case-law-detail block{f' headed {nested_heading!r}' if nested_heading else ''} is nested by malformed HTML; it was recovered as an independent main-record section.",
                "main_record_sections",
            )
        )

    document_title_node = first_node(root, lambda node: node.tag == "title")
    document_title = node_text(document_title_node, preserve_blocks=False)
    visible_title = extract_visible_title(sections)
    og_url = extract_meta_content(root, property_name="og:url")
    page_locale, locale_method, html_lang = extract_page_locale(root, og_url or expected_url)
    country = extract_country(sections)

    if replacement_count:
        parse_warnings.append(
            warning(
                "UTF8_REPLACEMENT_CHARACTERS",
                f"UTF-8 decoding produced {replacement_count} replacement characters.",
                "source_input",
            )
        )
    if not visible_title:
        parse_warnings.append(warning("VISIBLE_TITLE_MISSING", "Visible h2 case title was not found.", "case_identity"))
    if not country:
        parse_warnings.append(warning("COUNTRY_MISSING", "Country heading was not found.", "case_identity"))
    if not page_locale:
        parse_warnings.append(warning("PAGE_LOCALE_UNKNOWN", "Page locale could not be determined.", "case_identity"))
    if canonical_url_relation(og_url or "", expected_url) == "MISMATCH":
        parse_warnings.append(
            warning(
                "OG_URL_CANONICAL_MISMATCH",
                "The parsed og:url is not canonically equivalent to the frozen corpus URL.",
                "case_identity.og_url",
            )
        )
    manifest_title = normalize_inline_text(source_row.get("case_title", ""))
    if visible_title and manifest_title and visible_title != manifest_title:
        parse_warnings.append(
            warning(
                "TITLE_MANIFEST_MISMATCH",
                f"Visible title {visible_title!r} differs from manifest title {manifest_title!r}.",
                "case_identity.title_raw",
            )
        )

    computed_sha256 = hashlib.sha256(body).hexdigest()
    computed_byte_count = len(body)
    if input_kind == "production_raw_html":
        expected_sha = download_row.get("sha256", "")
        expected_bytes = parse_int(download_row.get("byte_count"))
        if expected_sha and computed_sha256 != expected_sha:
            parse_warnings.append(
                warning(
                    "RAW_SHA256_MISMATCH",
                    f"Computed SHA-256 {computed_sha256} differs from download manifest {expected_sha}.",
                    "source_input",
                )
            )
        if expected_bytes is not None and computed_byte_count != expected_bytes:
            parse_warnings.append(
                warning(
                    "RAW_BYTE_COUNT_MISMATCH",
                    f"Computed byte count {computed_byte_count} differs from download manifest {expected_bytes}.",
                    "source_input",
                )
            )
    if download_row.get("download_status") != "HTTP_OK_VALID":
        parse_warnings.append(
            warning(
                "DOWNLOAD_NOT_VALIDATED",
                f"Download status is {download_row.get('download_status')!r}, not HTTP_OK_VALID.",
                "provenance.download",
            )
        )

    heading_to_sections: Dict[str, List[Node]] = defaultdict(list)
    identity_section_ids: Set[int] = set()
    for section in sections:
        if first_node(section, class_predicate("countryNoHighlight")) is not None:
            identity_section_ids.add(id(section))
        if direct_child(section, lambda node: node.tag == "h2") is not None:
            identity_section_ids.add(id(section))
        heading = section_heading(section)
        if heading:
            heading_to_sections[heading].append(section)

    fact_section = (heading_to_sections.get("Fact Summary") or [None])[0]
    keyword_section = (heading_to_sections.get("Keywords") or [None])[0]
    charges_section = (heading_to_sections.get("Charges / Claims / Decisions") or [None])[0]

    fact_summary = extract_fact_summary(fact_section, page_locale)
    parse_warnings.extend(fact_summary["warnings"])
    crime_badges, trafficking_sidebar, sidebar_warnings = extract_crime_badges(root)
    parse_warnings.extend(sidebar_warnings)
    legacy_keywords, legacy_warnings = extract_legacy_keywords(keyword_section, page_locale)
    parse_warnings.extend(legacy_warnings)

    participant_sections: List[Dict[str, Any]] = []
    participant_section_node_ids: Set[int] = set()
    for section in sections:
        for container_type in PERSON_CONTAINER_CLASSES:
            containers = [
                node
                for node in section.descendants()
                if node.has_class(container_type)
                and closest_ancestor(node, class_predicate("case-law-detail")) is section
            ]
            for container in containers:
                record, section_warnings = extract_person_section(
                    section,
                    container,
                    container_type,
                    page_locale,
                    len(participant_sections) + 1,
                )
                participant_sections.append(record)
                participant_section_node_ids.add(id(section))
                parse_warnings.extend(section_warnings)

    charges_claims, charge_warnings = extract_charges_section(charges_section, page_locale)
    parse_warnings.extend(charge_warnings)

    routed_node_ids: Set[int] = {
        id(node)
        for node in (fact_section, keyword_section, charges_section)
        if node is not None
    }
    routed_node_ids.update(participant_section_node_ids)
    main_record_sections: Dict[str, List[Dict[str, Any]]] = {
        key: [] for key in sorted(set(KNOWN_MAIN_HEADINGS.values()))
    }
    other_sections: List[Dict[str, Any]] = []
    unheaded_sections: List[Dict[str, Any]] = []

    for section in sections:
        if id(section) in identity_section_ids or id(section) in routed_node_ids:
            continue
        heading = section_heading(section)
        if heading in KNOWN_MAIN_HEADINGS:
            key = KNOWN_MAIN_HEADINGS[heading]
            record = extract_generic_section(
                section,
                page_locale,
                f"main_record_sections.{key}[{len(main_record_sections[key]) + 1}]",
            )
            main_record_sections[key].append(record)
            parse_warnings.extend(record["warnings"])
        elif heading:
            record = extract_generic_section(
                section,
                page_locale,
                f"main_record_sections.other[{len(other_sections) + 1}]",
            )
            other_sections.append(record)
            parse_warnings.extend(record["warnings"])
            parse_warnings.append(
                warning(
                    "UNRECOGNIZED_MAIN_SECTION_HEADING",
                    f"Unrecognized top-level main-record heading retained verbatim: {heading!r}.",
                    f"main_record_sections.other[{len(other_sections)}]",
                    severity="INFO",
                )
            )
        else:
            record = extract_generic_section(
                section,
                page_locale,
                f"main_record_sections.unheaded[{len(unheaded_sections) + 1}]",
            )
            if record["non_pane_text_raw"] or record["tab_groups"] or record["fields"]:
                unheaded_sections.append(record)
                parse_warnings.extend(record["warnings"])

    source_manifest_record = dict(source_row)
    download_manifest_record = dict(download_row)
    for field in ("is_canonical_duplicate",):
        if field in source_manifest_record:
            source_manifest_record[field] = bool_string(source_manifest_record[field])
    for field in ("search_rank", "result_page_or_offset", "result_page_number", "validation_http_status"):
        if field in source_manifest_record and source_manifest_record[field] != "":
            source_manifest_record[field] = parse_int(source_manifest_record[field])
    for field in ("search_rank", "http_status", "byte_count", "attempts", "redirect_count"):
        if field in download_manifest_record and download_manifest_record[field] != "":
            download_manifest_record[field] = parse_int(download_manifest_record[field])
    for field in ("pilot_selected",):
        if field in download_manifest_record:
            download_manifest_record[field] = bool_string(download_manifest_record[field])
    for field in ("redirect_chain", "structural_markers"):
        if field in download_manifest_record:
            download_manifest_record[field] = safe_json_value(str(download_manifest_record[field]))
    if "warnings" in download_manifest_record:
        download_manifest_record["warnings"] = [
            value for value in str(download_manifest_record["warnings"] or "").split("|") if value
        ]

    warning_errors = [item for item in parse_warnings if item["severity"] == "ERROR"]
    warning_warnings = [item for item in parse_warnings if item["severity"] == "WARNING"]
    overall_status = STATUS_PARSE_ERROR if warning_errors else (STATUS_PARTIAL if warning_warnings else STATUS_FOUND)

    record = {
        "schema_version": SCHEMA_VERSION,
        "corpus_membership": {
            "membership_rule": "Frozen SHERLOC search result set filtered by Crime Type = Trafficking in persons; URL-path crime type does not determine membership.",
            "snapshot_date": CORPUS_SNAPSHOT_DATE,
            "source_manifest_path": str(DEFAULT_CORPUS_MANIFEST.relative_to(REPO_ROOT)),
            "source_manifest_record": source_manifest_record,
        },
        "provenance": {
            "search_rank": parse_int(source_row.get("search_rank")),
            "api_result_id": source_row.get("api_result_id") or None,
            "api_result_uri": source_row.get("api_result_uri") or None,
            "unodc_case_number": source_row.get("unodc_case_number") or None,
            "canonical_url": expected_url or None,
            "requested_url": download_row.get("requested_url") or None,
            "resolved_url": download_row.get("final_url") or None,
            "download_manifest_raw_filename": download_row.get("raw_filename") or None,
            "download_manifest_sha256": download_row.get("sha256") or None,
            "download_timestamp": download_row.get("download_timestamp") or None,
            "download_validation": {
                "download_status": download_row.get("download_status") or None,
                "http_status": parse_int(download_row.get("http_status")),
                "content_type": download_row.get("content_type") or None,
                "final_url_relation": download_row.get("final_url_relation") or None,
                "og_url_relation": download_row.get("og_url_relation") or None,
                "structural_markers": safe_json_value(download_row.get("structural_markers", "")),
                "warnings": [value for value in download_row.get("warnings", "").split("|") if value],
            },
            "download_manifest_record": download_manifest_record,
        },
        "source_input": {
            "input_kind": input_kind,
            "actual_path": str(actual_path.relative_to(REPO_ROOT)) if actual_path.is_relative_to(REPO_ROOT) else str(actual_path),
            "computed_byte_count": computed_byte_count,
            "computed_sha256": computed_sha256,
            "utf8_replacement_character_count": replacement_count,
        },
        "case_identity": {
            "title_raw": visible_title,
            "document_title_raw": document_title,
            "manifest_title_raw": source_row.get("case_title") or None,
            "country_raw": country,
            "page_locale": page_locale,
            "page_locale_detection_method": locale_method,
            "html_lang_raw": html_lang,
            "og_url": og_url,
            "og_url_relation_to_canonical": canonical_url_relation(og_url or "", expected_url),
            "url_path_crime_type": source_row.get("url_path_crime_type") or None,
        },
        "narrative": {"fact_summary": fact_summary},
        "trafficking_sidebar": trafficking_sidebar,
        "legacy_keywords": legacy_keywords,
        "participants": {"sections": participant_sections},
        "charges_claims_decisions": charges_claims,
        "crime_type_badges": crime_badges,
        "main_record_sections": {
            **main_record_sections,
            "other": other_sections,
            "unheaded": unheaded_sections,
        },
        "parser_provenance": {
            "parser_version": PARSER_VERSION,
            "parsed_at": parsed_at,
            "whitespace_policy": "HTML entities decoded; NBSP mapped to space; intra-line whitespace collapsed; block and line-break boundaries retained as newlines; no Unicode, label, taxonomy, entity, or semantic normalization.",
            "parse_status": overall_status,
            "warning_count": len(parse_warnings),
            "warnings": parse_warnings,
        },
    }
    return record


def parse_case_file(
    path: Path,
    source_row: Dict[str, str],
    download_row: Dict[str, str],
    *,
    input_kind: str = "production_raw_html",
    parsed_at: Optional[str] = None,
) -> Dict[str, Any]:
    path = Path(path)
    body = path.read_bytes()
    root, replacement_count = parse_html_bytes(body)
    return parse_case_root(
        root,
        body=body,
        actual_path=path,
        source_row=source_row,
        download_row=download_row,
        input_kind=input_kind,
        replacement_count=replacement_count,
        parsed_at=parsed_at,
    )


def failed_case_record(
    source_row: Dict[str, str],
    download_row: Optional[Dict[str, str]],
    path: Optional[Path],
    exc: BaseException,
    parsed_at: str,
) -> Dict[str, Any]:
    trace = "".join(traceback.format_exception_only(type(exc), exc)).strip()
    item = warning("CASE_PARSE_ERROR", trace, "case", severity="ERROR")
    return {
        "schema_version": SCHEMA_VERSION,
        "corpus_membership": {
            "membership_rule": "Frozen SHERLOC search result set filtered by Crime Type = Trafficking in persons; URL-path crime type does not determine membership.",
            "snapshot_date": CORPUS_SNAPSHOT_DATE,
            "source_manifest_path": str(DEFAULT_CORPUS_MANIFEST.relative_to(REPO_ROOT)),
            "source_manifest_record": dict(source_row),
        },
        "provenance": {
            "search_rank": parse_int(source_row.get("search_rank")),
            "api_result_id": source_row.get("api_result_id") or None,
            "unodc_case_number": source_row.get("unodc_case_number") or None,
            "canonical_url": source_row.get("canonical_url") or None,
            "requested_url": (download_row or {}).get("requested_url") or None,
            "resolved_url": (download_row or {}).get("final_url") or None,
            "download_manifest_raw_filename": (download_row or {}).get("raw_filename") or None,
            "download_manifest_sha256": (download_row or {}).get("sha256") or None,
            "download_timestamp": (download_row or {}).get("download_timestamp") or None,
            "download_manifest_record": dict(download_row or {}),
        },
        "source_input": {"input_kind": "production_raw_html", "actual_path": str(path) if path else None},
        "case_identity": {
            "title_raw": None,
            "document_title_raw": None,
            "manifest_title_raw": source_row.get("case_title") or None,
            "country_raw": None,
            "page_locale": None,
            "og_url": None,
            "url_path_crime_type": source_row.get("url_path_crime_type") or None,
        },
        "narrative": {"fact_summary": {"status": STATUS_PARSE_ERROR, "variants": [], "english_text_raw": None}},
        "trafficking_sidebar": {"status": STATUS_PARSE_ERROR, "fields": {}},
        "legacy_keywords": {"status": STATUS_PARSE_ERROR, "categories": [], "core_fields": {}},
        "participants": {"sections": []},
        "charges_claims_decisions": {"status": STATUS_PARSE_ERROR, "subject_records": []},
        "crime_type_badges": [],
        "main_record_sections": {},
        "parser_provenance": {
            "parser_version": PARSER_VERSION,
            "parsed_at": parsed_at,
            "parse_status": STATUS_PARSE_ERROR,
            "warning_count": 1,
            "warnings": [item],
        },
    }


def load_csv_rows(path: Path) -> List[Dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def load_and_validate_manifests(
    corpus_manifest: Path,
    download_manifest: Path,
) -> Tuple[List[Dict[str, str]], Dict[int, Dict[str, str]], Dict[str, Any]]:
    source_rows = load_csv_rows(corpus_manifest)
    download_rows = load_csv_rows(download_manifest)
    if not source_rows:
        raise ParsePipelineError(f"Corpus manifest is empty: {corpus_manifest}")

    ranks = [parse_int(row.get("search_rank")) for row in source_rows]
    if any(rank is None for rank in ranks):
        raise ParsePipelineError("Corpus manifest contains a non-integer search_rank")
    expected_ranks = list(range(1, len(source_rows) + 1))
    if ranks != expected_ranks:
        raise ParsePipelineError("Corpus manifest search_rank must be ordered, unique, and contiguous from 1")

    canonical_urls = [row.get("canonical_url", "") for row in source_rows]
    if any(not value for value in canonical_urls):
        raise ParsePipelineError("Corpus manifest contains an empty canonical_url")
    if len(set(canonical_urls)) != len(canonical_urls):
        raise ParsePipelineError("Corpus manifest canonical_url values are not unique")

    download_by_rank: Dict[int, Dict[str, str]] = {}
    duplicate_download_ranks: List[int] = []
    for row in download_rows:
        rank = parse_int(row.get("search_rank"))
        if rank is None:
            raise ParsePipelineError("Download manifest contains a non-integer search_rank")
        if rank in download_by_rank:
            duplicate_download_ranks.append(rank)
        download_by_rank[rank] = row
    if duplicate_download_ranks:
        raise ParsePipelineError(f"Download manifest contains duplicate ranks: {duplicate_download_ranks[:20]}")

    join_issues: List[Dict[str, Any]] = []
    for row in source_rows:
        rank = int(row["search_rank"])
        download = download_by_rank.get(rank)
        if download is None:
            join_issues.append({"search_rank": rank, "issue": "MISSING_DOWNLOAD_ROW"})
            continue
        if download.get("canonical_url") != row.get("canonical_url"):
            join_issues.append({"search_rank": rank, "issue": "CANONICAL_URL_MISMATCH"})
        if download.get("api_result_id") != row.get("api_result_id"):
            join_issues.append({"search_rank": rank, "issue": "API_RESULT_ID_MISMATCH"})
        if download.get("download_status") != "HTTP_OK_VALID":
            join_issues.append(
                {
                    "search_rank": rank,
                    "issue": "DOWNLOAD_STATUS_NOT_VALID",
                    "value": download.get("download_status"),
                }
            )
    extra_download_ranks = sorted(set(download_by_rank) - set(expected_ranks))
    if extra_download_ranks:
        join_issues.extend(
            {"search_rank": rank, "issue": "EXTRA_DOWNLOAD_ROW"}
            for rank in extra_download_ranks
        )
    if join_issues:
        raise ParsePipelineError(
            f"Manifest/download provenance join failed with {len(join_issues)} issue(s); first: {join_issues[:3]}"
        )

    diagnostics = {
        "status": "PASS",
        "corpus_manifest_path": str(Path(corpus_manifest)),
        "download_manifest_path": str(Path(download_manifest)),
        "expected_manifest_rows": len(source_rows),
        "download_manifest_rows": len(download_rows),
        "unique_canonical_urls": len(set(canonical_urls)),
        "download_status_counts": dict(Counter(row.get("download_status", "") for row in download_rows)),
        "url_path_crime_type_counts": dict(Counter(row.get("url_path_crime_type", "") for row in source_rows)),
        "join_issue_count": 0,
    }
    return source_rows, download_by_rank, diagnostics


def resolve_raw_path(download_row: Dict[str, str], raw_html_dir: Path) -> Path:
    manifest_value = download_row.get("raw_filename", "")
    if not manifest_value:
        return Path(raw_html_dir) / "__MISSING_RAW_FILENAME__"
    path = Path(manifest_value)
    if path.is_absolute():
        return path
    repo_candidate = REPO_ROOT / path
    if repo_candidate.exists() or path.parts[:2] == ("data", "raw_html"):
        return repo_candidate
    return Path(raw_html_dir) / path.name


def fixture_source_rank(path: Path, source_identity_to_rank: Dict[str, int]) -> int:
    body = Path(path).read_bytes()
    root, _ = parse_html_bytes(body)
    og_url = extract_meta_content(root, property_name="og:url")
    identity = canonical_case_identity(og_url or "")
    if not identity:
        raise ParsePipelineError(f"Fixture has no usable og:url: {path.name}")
    rank = source_identity_to_rank.get(identity)
    if rank is None:
        raise ParsePipelineError(f"Fixture og:url does not join the frozen corpus manifest: {path.name}")
    return rank


def recursive_text_values(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from recursive_text_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from recursive_text_values(child)


def all_record_text(record: Dict[str, Any]) -> str:
    return "\n".join(recursive_text_values(record))


def validation_check(name: str, passed: bool, details: Any = None) -> Dict[str, Any]:
    return {"name": name, "status": "PASS" if passed else "FAIL", "details": details}


def fact_variants(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    return record.get("narrative", {}).get("fact_summary", {}).get("variants", [])


def participant_sections(record: Dict[str, Any], role_family: str) -> List[Dict[str, Any]]:
    return [
        section
        for section in record.get("participants", {}).get("sections", [])
        if section.get("role_family") == role_family
    ]


def participant_record_count(record: Dict[str, Any], role_family: str) -> int:
    return sum(section.get("record_count", 0) for section in participant_sections(record, role_family))


def charge_subject_count(record: Dict[str, Any]) -> int:
    return len(record.get("charges_claims_decisions", {}).get("subject_records", []))


def charge_record_count(record: Dict[str, Any]) -> int:
    charges = record.get("charges_claims_decisions", {})
    return sum(subject.get("charge_count", 0) for subject in charges.get("subject_records", [])) + len(
        charges.get("orphan_charge_records", [])
    )


def main_section_present(record: Dict[str, Any], key: str) -> bool:
    return any(
        section.get("status") in {STATUS_FOUND, STATUS_PARTIAL}
        for section in record.get("main_record_sections", {}).get(key, [])
    )


def section_panes(record: Dict[str, Any], key: str) -> List[Dict[str, Any]]:
    return [
        pane
        for section in record.get("main_record_sections", {}).get(key, [])
        for group in section.get("tab_groups", [])
        for pane in group.get("panes", [])
    ]


def validate_fixture_records(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    by_title = {record.get("case_identity", {}).get("title_raw"): record for record in records}
    checks: List[Dict[str, Any]] = []
    checks.append(validation_check("fixture_count_19", len(records) == 19, len(records)))
    checks.append(
        validation_check(
            "all_fixtures_join_identity_and_locale",
            all(
                record.get("provenance", {}).get("search_rank")
                and record.get("case_identity", {}).get("title_raw")
                and record.get("case_identity", {}).get("country_raw")
                and record.get("case_identity", {}).get("page_locale") == "en"
                for record in records
            ),
        )
    )
    checks.append(
        validation_check(
            "all_fixtures_have_fact_summary",
            all(any(variant.get("text_raw") for variant in fact_variants(record)) for record in records),
        )
    )

    b637 = by_title.get("B637.L6.961-X7-DF")
    b637_languages = [variant.get("language") for variant in fact_variants(b637 or {})]
    checks.append(
        validation_check(
            "b637_fact_english_french_separate",
            b637 is not None
            and b637_languages == ["en", "fr"]
            and len(fact_variants(b637)) == 2
            and bool(b637["narrative"]["fact_summary"]["english_text_raw"]),
            b637_languages,
        )
    )

    causa = by_title.get("Causa 2422")
    causa_fact_languages = [variant.get("language") for variant in fact_variants(causa or {})]
    causa_section_languages = {
        key: [pane.get("language") for pane in section_panes(causa or {}, key)]
        for key in ("commentary_significant_features", "procedural_information", "sources_citations")
    }
    checks.append(
        validation_check(
            "causa_2422_multilingual_sections_separate",
            causa is not None
            and causa_fact_languages == ["en", "es"]
            and all(set(values) == {"en", "es"} for values in causa_section_languages.values()),
            {"fact": causa_fact_languages, **causa_section_languages},
        )
    )

    twitter = next((record for title, record in by_title.items() if title and "Twitter" in title), None)
    checks.append(
        validation_check(
            "twitter_corporate_respondent_preserved",
            twitter is not None
            and participant_record_count(twitter, "defendant_respondent") == 1
            and "Twitter, INC." in all_record_text(twitter),
        )
    )

    cross_classified = [
        record
        for record in records
        if record.get("case_identity", {}).get("url_path_crime_type") != "traffickingpersonscrimetype"
    ]
    scoped_ok = True
    for record in cross_classified:
        traffic_ordinals = set(record.get("trafficking_sidebar", {}).get("badge_ordinals", []))
        for field in record.get("trafficking_sidebar", {}).get("fields", {}).values():
            if any(source.get("badge_ordinal") not in traffic_ordinals for source in field.get("sources", [])):
                scoped_ok = False
    checks.append(
        validation_check(
            "cross_classified_sidebar_strictly_scoped",
            bool(cross_classified) and scoped_ok,
            [record["provenance"]["search_rank"] for record in cross_classified],
        )
    )

    legacy_found = sum(
        record.get("legacy_keywords", {}).get("status") in {STATUS_FOUND, STATUS_PARTIAL}
        for record in records
    )
    checks.append(validation_check("fixture_legacy_keywords_count_17", legacy_found == 17, legacy_found))

    sentencia = by_title.get("Sentencia 298/2015")
    sentencia_roles = [
        (section.get("dom_role_container_type"), section.get("visible_section_heading_raw"), section.get("record_count"))
        for section in (sentencia or {}).get("participants", {}).get("sections", [])
    ]
    checks.append(
        validation_check(
            "sentencia_migrants_role_preserved",
            ("victimsPlaintiffs", "Migrants", 1) in sentencia_roles,
            sentencia_roles,
        )
    )

    totals = {
        "person_role_records": sum(participant_record_count(record, "person_role") for record in records),
        "defendant_records": sum(participant_record_count(record, "defendant_respondent") for record in records),
        "charge_subject_records": sum(charge_subject_count(record) for record in records),
        "charge_records": sum(charge_record_count(record) for record in records),
        "court_sections": sum(main_section_present(record, "court") for record in records),
    }
    checks.append(
        validation_check(
            "fixture_repeated_record_totals",
            totals == {
                "person_role_records": 54,
                "defendant_records": 45,
                "charge_subject_records": 45,
                "charge_records": 111,
                "court_sections": 16,
            },
            totals,
        )
    )

    robinson = by_title.get("United States v Robinson")
    checks.append(
        validation_check(
            "robinson_cardinality_not_truncated",
            robinson is not None
            and participant_record_count(robinson, "defendant_respondent") == 17
            and charge_subject_count(robinson) == 17
            and charge_record_count(robinson) == 63,
            {
                "defendants": participant_record_count(robinson or {}, "defendant_respondent"),
                "subjects": charge_subject_count(robinson or {}),
                "charges": charge_record_count(robinson or {}),
            },
        )
    )
    absent_court_titles = {
        title for title, record in by_title.items() if not main_section_present(record, "court")
    }
    checks.append(
        validation_check(
            "court_scoped_to_direct_main_section",
            len(absent_court_titles) == 3
            and "Sentencia 298/2015" in absent_court_titles
            and any("Twitter" in (title or "") for title in absent_court_titles)
            and any("Querela" in (title or "") for title in absent_court_titles),
            sorted(title for title in absent_court_titles if title),
        )
    )
    checks.append(
        validation_check(
            "no_fixture_parse_errors",
            all(record["parser_provenance"]["parse_status"] != STATUS_PARSE_ERROR for record in records),
        )
    )

    failures = [check for check in checks if check["status"] == "FAIL"]
    return {
        "status": "PASS" if not failures else "FAIL",
        "fixture_count": len(records),
        "checks": checks,
        "failure_count": len(failures),
        "failed_checks": [check["name"] for check in failures],
        "fixture_search_ranks": sorted(record["provenance"]["search_rank"] for record in records),
    }


def challenge_rank_reasons(source_rows: Sequence[Dict[str, str]]) -> Dict[int, List[str]]:
    reasons: Dict[int, List[str]] = {
        rank: ["fact_summary_structurally_absent"] for rank in sorted(KNOWN_MISSING_FACT_RANKS)
    }
    for rank, values in CHALLENGE_ANCHOR_REASONS.items():
        reasons.setdefault(rank, []).extend(values)

    # Re-derive path coverage from the frozen manifest.  Add a first-rank anchor
    # only if the fixed structural challenge does not already cover that path.
    first_by_path: Dict[str, int] = {}
    path_by_rank: Dict[int, str] = {}
    for row in source_rows:
        rank = int(row["search_rank"])
        path_type = row.get("url_path_crime_type", "")
        path_by_rank[rank] = path_type
        first_by_path.setdefault(path_type, rank)
    covered_paths = {path_by_rank[rank] for rank in reasons if rank in path_by_rank}
    for path_type, rank in first_by_path.items():
        if path_type not in covered_paths:
            reasons.setdefault(rank, []).append(f"first_rank_for_url_path:{path_type}")
    return {rank: sorted(set(values)) for rank, values in sorted(reasons.items())}


def all_warning_codes(record: Dict[str, Any]) -> List[str]:
    return [item.get("code", "") for item in record.get("parser_provenance", {}).get("warnings", [])]


def validate_challenge_records(
    records: Sequence[Dict[str, Any]],
    source_rows: Sequence[Dict[str, str]],
    reasons: Dict[int, List[str]],
) -> Dict[str, Any]:
    by_rank = {record["provenance"]["search_rank"]: record for record in records}
    checks: List[Dict[str, Any]] = []
    expected_ranks = set(reasons)
    checks.append(
        validation_check(
            "challenge_rank_set_complete",
            set(by_rank) == expected_ranks,
            {"expected_count": len(expected_ranks), "actual_count": len(by_rank)},
        )
    )
    checks.append(
        validation_check(
            "all_25_missing_fact_pages_are_section_absent",
            all(
                by_rank.get(rank, {}).get("narrative", {}).get("fact_summary", {}).get("status")
                == STATUS_SECTION_ABSENT
                for rank in KNOWN_MISSING_FACT_RANKS
            ),
            sorted(
                rank
                for rank in KNOWN_MISSING_FACT_RANKS
                if by_rank.get(rank, {}).get("narrative", {}).get("fact_summary", {}).get("status")
                != STATUS_SECTION_ABSENT
            ),
        )
    )

    corpus_paths = {row.get("url_path_crime_type") for row in source_rows}
    challenge_paths = {
        record.get("case_identity", {}).get("url_path_crime_type") for record in records
    }
    checks.append(
        validation_check(
            "all_url_path_crime_types_covered",
            challenge_paths == corpus_paths,
            {"corpus": sorted(corpus_paths), "challenge": sorted(challenge_paths)},
        )
    )

    b62 = by_rank.get(62, {})
    checks.append(
        validation_check(
            "rank_62_unlabeled_en_fr_fact_preserved",
            [item.get("language") for item in fact_variants(b62)] == ["en", "fr"]
            and len(fact_variants(b62)) == 2,
        )
    )
    checks.append(
        validation_check(
            "rank_307_duplicate_tab_identity_preserved",
            {"DUPLICATE_TAB_PANE_ID", "DUPLICATE_TAB_HREF"}.issubset(set(all_warning_codes(by_rank.get(307, {})))),
            all_warning_codes(by_rank.get(307, {})),
        )
    )
    checks.append(
        validation_check(
            "rank_4_legacy_keywords_empty_not_absent",
            by_rank.get(4, {}).get("legacy_keywords", {}).get("status") == STATUS_EMPTY,
            by_rank.get(4, {}).get("legacy_keywords", {}).get("status"),
        )
    )
    rank1423_roles = [
        (section.get("dom_role_container_type"), section.get("visible_section_heading_raw"))
        for section in by_rank.get(1423, {}).get("participants", {}).get("sections", [])
    ]
    checks.append(
        validation_check(
            "rank_1423_migrants_role_preserved",
            ("victimsPlaintiffs", "Migrants") in rank1423_roles,
            rank1423_roles,
        )
    )
    checks.append(
        validation_check(
            "rank_46_legal_entity_preserved",
            "LA SAP" in all_record_text(by_rank.get(46, {})),
        )
    )
    checks.append(
        validation_check(
            "rank_1171_maximum_defendants_preserved",
            participant_record_count(by_rank.get(1171, {}), "defendant_respondent") == 34
            and charge_subject_count(by_rank.get(1171, {})) == 34,
            {
                "defendants": participant_record_count(by_rank.get(1171, {}), "defendant_respondent"),
                "charge_subjects": charge_subject_count(by_rank.get(1171, {})),
            },
        )
    )
    malformed_subject_counts = {
        rank: charge_subject_count(by_rank.get(rank, {})) for rank in (63, 1489)
    }
    malformed_orphan_counts = {
        rank: len(by_rank.get(rank, {}).get("charges_claims_decisions", {}).get("orphan_charge_records", []))
        for rank in (63, 1489)
    }
    malformed_local_fallbacks = {
        rank: {
            "subjects_excluding_nested": sum(
                subject.get("raw_text_excluded_nested_subject_count", 0) > 0
                for subject in by_rank.get(rank, {}).get("charges_claims_decisions", {}).get("subject_records", [])
            ),
            "literal_subtrees_preserved": sum(
                bool(subject.get("dom_subtree_text_raw"))
                for subject in by_rank.get(rank, {}).get("charges_claims_decisions", {}).get("subject_records", [])
            ),
        }
        for rank in (63, 1489)
    }
    checks.append(
        validation_check(
            "malformed_nested_charge_subjects_retained",
            malformed_subject_counts == {63: 4, 1489: 3}
            and malformed_orphan_counts == {63: 0, 1489: 0}
            and all(
                value["subjects_excluding_nested"] > 0
                and value["subjects_excluding_nested"] == value["literal_subtrees_preserved"]
                for value in malformed_local_fallbacks.values()
            )
            and all(
                "NESTED_CHARGE_SUBJECT_RECORDS" in all_warning_codes(by_rank.get(rank, {}))
                for rank in (63, 1489)
            ),
            {
                "subject_counts": malformed_subject_counts,
                "orphan_counts": malformed_orphan_counts,
                "record_local_fallbacks": malformed_local_fallbacks,
            },
        )
    )
    nested_court_recovery = {
        rank: {
            "court_present": main_section_present(by_rank.get(rank, {}), "court"),
            "warning_present": "NESTED_CASE_LAW_DETAIL" in all_warning_codes(by_rank.get(rank, {})),
            "court_excluded_from_charges_fallback": all(
                not section.get("non_pane_text_raw")
                or section.get("non_pane_text_raw")
                not in (by_rank.get(rank, {}).get("charges_claims_decisions", {}).get("non_pane_text_raw") or "")
                for section in by_rank.get(rank, {}).get("main_record_sections", {}).get("court", [])
            ),
        }
        for rank in (1, 63)
    }
    checks.append(
        validation_check(
            "nested_main_record_court_sections_recovered",
            all(
                value["court_present"]
                and value["warning_present"]
                and value["court_excluded_from_charges_fallback"]
                for value in nested_court_recovery.values()
            ),
            nested_court_recovery,
        )
    )
    checks.append(
        validation_check(
            "size_extremes_present",
            {937, 1515}.issubset(set(by_rank)),
        )
    )
    checks.append(
        validation_check(
            "oldest_and_newest_url_year_present",
            {1, 1483}.issubset(set(by_rank)),
        )
    )
    checks.append(
        validation_check(
            "no_challenge_parse_errors",
            all(record["parser_provenance"]["parse_status"] != STATUS_PARSE_ERROR for record in records),
            [
                record["provenance"]["search_rank"]
                for record in records
                if record["parser_provenance"]["parse_status"] == STATUS_PARSE_ERROR
            ],
        )
    )
    checks.append(
        validation_check(
            "challenge_provenance_trace_complete",
            all(
                record.get("source_input", {}).get("computed_sha256")
                == record.get("provenance", {}).get("download_manifest_sha256")
                and record.get("case_identity", {}).get("og_url_relation_to_canonical")
                in {"EXACT_MATCH", "CANONICAL_EQUIVALENT"}
                for record in records
            ),
        )
    )
    failures = [check for check in checks if check["status"] == "FAIL"]
    return {
        "status": "PASS" if not failures else "FAIL",
        "challenge_count": len(records),
        "selection": [
            {
                "search_rank": rank,
                "case_title": by_rank.get(rank, {}).get("case_identity", {}).get("title_raw"),
                "reasons": reasons[rank],
            }
            for rank in sorted(reasons)
        ],
        "checks": checks,
        "failure_count": len(failures),
        "failed_checks": [check["name"] for check in failures],
    }


def truth(value: Any) -> int:
    return 1 if bool(value) else 0


def field_values(record: Dict[str, Any], source: str, field: str) -> List[str]:
    if source == "sidebar":
        return list(
            record.get("trafficking_sidebar", {})
            .get("fields", {})
            .get(field, {})
            .get("values_raw", [])
        )
    return list(
        record.get("legacy_keywords", {})
        .get("core_fields", {})
        .get(field, {})
        .get("values_raw", [])
    )


def iter_record_tab_groups(record: Dict[str, Any]) -> Iterator[Tuple[str, Dict[str, Any]]]:
    fact = record.get("narrative", {}).get("fact_summary", {})
    fact_variants_list = fact.get("variants", [])
    if fact_variants_list and any(variant.get("group_index") is not None for variant in fact_variants_list):
        grouped: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
        for variant in fact_variants_list:
            grouped[variant.get("group_index")].append(variant)
        for group_index, panes in grouped.items():
            yield "Fact Summary", {"group_index": group_index, "panes": panes}

    legacy = record.get("legacy_keywords", {})
    for group in legacy.get("tab_groups", []):
        yield "Keywords", group
    for section in record.get("participants", {}).get("sections", []):
        label = section.get("visible_section_heading_raw") or section.get("dom_role_container_type")
        for group in section.get("tab_groups", []):
            yield str(label), group
    charges = record.get("charges_claims_decisions", {})
    for group in charges.get("tab_groups", []):
        yield charges.get("heading_raw") or "Charges / Claims / Decisions", group
    for key, sections in record.get("main_record_sections", {}).items():
        for section in sections:
            label = section.get("heading_raw") or f"unheaded:{key}"
            for group in section.get("tab_groups", []):
                yield str(label), group


def multilingual_metrics(record: Dict[str, Any]) -> Dict[str, Any]:
    tabbed_labels: List[str] = []
    multi_pane_labels: List[str] = []
    multilingual_labels: List[str] = []
    pane_count = 0
    for label, group in iter_record_tab_groups(record):
        panes = group.get("panes", [])
        tabbed_labels.append(label)
        pane_count += len(panes)
        nonempty = [pane for pane in panes if pane.get("text_raw")]
        if len(nonempty) > 1:
            multi_pane_labels.append(label)
        languages = {pane.get("language") for pane in nonempty if pane.get("language")}
        if len(languages) > 1:
            multilingual_labels.append(label)
    return {
        "tabbed_section_count": len(tabbed_labels),
        "tabbed_pane_count": pane_count,
        "multi_pane_section_count": len(multi_pane_labels),
        "multilingual_section_count": len(multilingual_labels),
        "multilingual_section_labels": multilingual_labels,
    }


COVERAGE_FIELDS = [
    "search_rank",
    "case_title",
    "unodc_case_number",
    "canonical_url",
    "url_path_crime_type",
    "fact_summary_status",
    "has_english_fact_summary",
    "has_any_fact_summary",
    "fact_summary_variant_count",
    "fact_summary_language_count",
    "fact_summary_languages",
    "has_sidebar_acts",
    "sidebar_acts_count",
    "has_sidebar_means",
    "sidebar_means_count",
    "has_sidebar_purpose",
    "sidebar_purpose_count",
    "has_legacy_keyword_acts",
    "legacy_keyword_acts_count",
    "has_legacy_keyword_means",
    "legacy_keyword_means_count",
    "has_legacy_keyword_purpose",
    "legacy_keyword_purpose_count",
    "has_victim_or_person_role_section",
    "person_role_section_count",
    "person_role_record_count",
    "has_defendant_or_respondent_section",
    "defendant_respondent_section_count",
    "defendant_respondent_record_count",
    "has_charges_claims_decisions",
    "charge_subject_record_count",
    "charge_record_count",
    "has_court",
    "tabbed_section_count",
    "tabbed_pane_count",
    "multi_pane_section_count",
    "multilingual_section_count",
    "multilingual_section_labels",
    "crime_type_badge_count",
    "parse_warning_count",
    "parse_warning_severity_count",
    "parse_warning_codes",
    "overall_parse_status",
]


def coverage_row(record: Dict[str, Any]) -> Dict[str, Any]:
    fact = record.get("narrative", {}).get("fact_summary", {})
    variants = [variant for variant in fact.get("variants", []) if variant.get("text_raw")]
    languages = sorted({variant.get("language") for variant in variants if variant.get("language")})
    sidebar_acts = field_values(record, "sidebar", "acts")
    sidebar_means = field_values(record, "sidebar", "means")
    sidebar_purpose = field_values(record, "sidebar", "exploitative_purposes")
    legacy_acts = field_values(record, "legacy", "acts")
    legacy_means = field_values(record, "legacy", "means")
    legacy_purpose = field_values(record, "legacy", "exploitative_purposes")
    person_role = participant_sections(record, "person_role")
    defendants = participant_sections(record, "defendant_respondent")
    charges = record.get("charges_claims_decisions", {})
    charge_present = charges.get("status") in {STATUS_FOUND, STATUS_PARTIAL}
    warnings = record.get("parser_provenance", {}).get("warnings", [])
    warning_codes = [item.get("code") for item in warnings if item.get("code")]
    warning_severity_count = sum(item.get("severity") in {"WARNING", "ERROR"} for item in warnings)
    multilingual = multilingual_metrics(record)
    return {
        "search_rank": record.get("provenance", {}).get("search_rank"),
        "case_title": record.get("case_identity", {}).get("title_raw")
        or record.get("case_identity", {}).get("manifest_title_raw"),
        "unodc_case_number": record.get("provenance", {}).get("unodc_case_number"),
        "canonical_url": record.get("provenance", {}).get("canonical_url"),
        "url_path_crime_type": record.get("case_identity", {}).get("url_path_crime_type"),
        "fact_summary_status": fact.get("status"),
        "has_english_fact_summary": truth(bool(fact.get("english_text_raw"))),
        "has_any_fact_summary": truth(bool(variants)),
        "fact_summary_variant_count": len(variants),
        "fact_summary_language_count": len(languages),
        "fact_summary_languages": "|".join(languages),
        "has_sidebar_acts": truth(sidebar_acts),
        "sidebar_acts_count": len(sidebar_acts),
        "has_sidebar_means": truth(sidebar_means),
        "sidebar_means_count": len(sidebar_means),
        "has_sidebar_purpose": truth(sidebar_purpose),
        "sidebar_purpose_count": len(sidebar_purpose),
        "has_legacy_keyword_acts": truth(legacy_acts),
        "legacy_keyword_acts_count": len(legacy_acts),
        "has_legacy_keyword_means": truth(legacy_means),
        "legacy_keyword_means_count": len(legacy_means),
        "has_legacy_keyword_purpose": truth(legacy_purpose),
        "legacy_keyword_purpose_count": len(legacy_purpose),
        "has_victim_or_person_role_section": truth(person_role),
        "person_role_section_count": len(person_role),
        "person_role_record_count": sum(section.get("record_count", 0) for section in person_role),
        "has_defendant_or_respondent_section": truth(defendants),
        "defendant_respondent_section_count": len(defendants),
        "defendant_respondent_record_count": sum(section.get("record_count", 0) for section in defendants),
        "has_charges_claims_decisions": truth(charge_present),
        "charge_subject_record_count": charge_subject_count(record),
        "charge_record_count": charge_record_count(record),
        "has_court": truth(main_section_present(record, "court")),
        "tabbed_section_count": multilingual["tabbed_section_count"],
        "tabbed_pane_count": multilingual["tabbed_pane_count"],
        "multi_pane_section_count": multilingual["multi_pane_section_count"],
        "multilingual_section_count": multilingual["multilingual_section_count"],
        "multilingual_section_labels": "|".join(multilingual["multilingual_section_labels"]),
        "crime_type_badge_count": len(record.get("crime_type_badges", [])),
        "parse_warning_count": len(warnings),
        "parse_warning_severity_count": warning_severity_count,
        "parse_warning_codes": "|".join(warning_codes),
        "overall_parse_status": record.get("parser_provenance", {}).get("parse_status"),
    }


def corpus_summary(records: Sequence[Dict[str, Any]], coverage: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    status_counts = Counter(row["overall_parse_status"] for row in coverage)
    warning_codes = Counter(
        code
        for record in records
        for code in all_warning_codes(record)
        if code
    )
    warning_severities = Counter(
        item.get("severity", "")
        for record in records
        for item in record.get("parser_provenance", {}).get("warnings", [])
    )
    other_headings = Counter(
        section.get("heading_raw") or "<missing>"
        for record in records
        for section in record.get("main_record_sections", {}).get("other", [])
    )
    partial_sections = Counter()
    for record in records:
        if record.get("narrative", {}).get("fact_summary", {}).get("status") == STATUS_PARTIAL:
            partial_sections["fact_summary"] += 1
        if record.get("trafficking_sidebar", {}).get("status") == STATUS_PARTIAL:
            partial_sections["trafficking_sidebar"] += 1
        if record.get("legacy_keywords", {}).get("status") == STATUS_PARTIAL:
            partial_sections["legacy_keywords"] += 1
        if record.get("charges_claims_decisions", {}).get("status") == STATUS_PARTIAL:
            partial_sections["charges_claims_decisions"] += 1
        for key, sections in record.get("main_record_sections", {}).items():
            partial_sections[key] += sum(section.get("status") == STATUS_PARTIAL for section in sections)
        for section in record.get("participants", {}).get("sections", []):
            if section.get("status") == STATUS_PARTIAL:
                partial_sections[f"participants:{section.get('dom_role_container_type')}"] += 1

    def page_count(field_name: str) -> int:
        return sum(int(row[field_name]) for row in coverage)

    def value_count(field_name: str) -> int:
        return sum(int(row[field_name]) for row in coverage)

    result = {
        "total_cases_parsed": len(records),
        "overall_parse_status_counts": dict(sorted(status_counts.items())),
        "parse_error_count": status_counts.get(STATUS_PARSE_ERROR, 0),
        "parse_warning_total": sum(row["parse_warning_count"] for row in coverage),
        "parse_warning_severity_total": sum(row["parse_warning_severity_count"] for row in coverage),
        "cases_with_any_parse_warning": sum(row["parse_warning_count"] > 0 for row in coverage),
        "cases_with_warning_or_error_severity": sum(row["parse_warning_severity_count"] > 0 for row in coverage),
        "warning_severity_counts": dict(sorted(warning_severities.items())),
        "warning_code_counts": dict(sorted(warning_codes.items())),
        "fact_summary": {
            "with_english": page_count("has_english_fact_summary"),
            "with_any": page_count("has_any_fact_summary"),
            "without_usable": len(coverage) - page_count("has_any_fact_summary"),
            "status_counts": dict(Counter(row["fact_summary_status"] for row in coverage)),
        },
        "sidebar_coverage": {
            "acts_cases": page_count("has_sidebar_acts"),
            "acts_values": value_count("sidebar_acts_count"),
            "means_cases": page_count("has_sidebar_means"),
            "means_values": value_count("sidebar_means_count"),
            "purpose_cases": page_count("has_sidebar_purpose"),
            "purpose_values": value_count("sidebar_purpose_count"),
        },
        "legacy_keyword_coverage": {
            "acts_cases": page_count("has_legacy_keyword_acts"),
            "acts_values": value_count("legacy_keyword_acts_count"),
            "means_cases": page_count("has_legacy_keyword_means"),
            "means_values": value_count("legacy_keyword_means_count"),
            "purpose_cases": page_count("has_legacy_keyword_purpose"),
            "purpose_values": value_count("legacy_keyword_purpose_count"),
        },
        "multilingual_sections": {
            "cases_with_tabbed_sections": sum(row["tabbed_section_count"] > 0 for row in coverage),
            "tabbed_section_groups": value_count("tabbed_section_count"),
            "tabbed_panes": value_count("tabbed_pane_count"),
            "cases_with_multi_pane_sections": sum(row["multi_pane_section_count"] > 0 for row in coverage),
            "multi_pane_section_groups": value_count("multi_pane_section_count"),
            "cases_with_multilingual_sections": sum(row["multilingual_section_count"] > 0 for row in coverage),
            "multilingual_section_groups": value_count("multilingual_section_count"),
        },
        "participants": {
            "cases_with_person_role_section": page_count("has_victim_or_person_role_section"),
            "person_role_sections": value_count("person_role_section_count"),
            "person_role_records": value_count("person_role_record_count"),
            "cases_with_defendant_respondent_section": page_count("has_defendant_or_respondent_section"),
            "defendant_respondent_sections": value_count("defendant_respondent_section_count"),
            "defendant_respondent_records": value_count("defendant_respondent_record_count"),
        },
        "charges_claims_decisions": {
            "cases_with_section": page_count("has_charges_claims_decisions"),
            "subject_records": value_count("charge_subject_record_count"),
            "charge_records": value_count("charge_record_count"),
        },
        "court": {"cases_with_main_record_court": page_count("has_court")},
        "unfamiliar_main_section_headings": dict(sorted(other_headings.items())),
        "partial_section_counts": {key: value for key, value in sorted(partial_sections.items()) if value},
    }
    return result


def atomic_write_json(path: Path, value: Any, *, indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temp_path = Path(handle.name)
        json.dump(value, handle, ensure_ascii=False, indent=indent)
        handle.write("\n")
    temp_path.chmod(0o644)
    temp_path.replace(path)


def atomic_write_jsonl(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temp_path = Path(handle.name)
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
    temp_path.chmod(0o644)
    temp_path.replace(path)


def atomic_write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent, delete=False) as handle:
        temp_path = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=COVERAGE_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temp_path.chmod(0o644)
    temp_path.replace(path)


def atomic_write_text(path: Path, value: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temp_path = Path(handle.name)
        handle.write(value)
        if not value.endswith("\n"):
            handle.write("\n")
    temp_path.chmod(0o644)
    temp_path.replace(path)


def render_parser_report(diagnostics: Dict[str, Any]) -> str:
    fixture = diagnostics["fixture_validation"]
    challenge = diagnostics["challenge_validation"]
    summary = diagnostics["corpus_summary"]
    fact = summary["fact_summary"]
    sidebar = summary["sidebar_coverage"]
    legacy = summary["legacy_keyword_coverage"]
    multilingual = summary["multilingual_sections"]
    parties = summary["participants"]
    charges = summary["charges_claims_decisions"]
    court = summary["court"]
    warning_codes = summary["warning_code_counts"]
    unfamiliar = summary["unfamiliar_main_section_headings"]
    partial = summary["partial_section_counts"]
    elapsed = diagnostics["run"]["elapsed_seconds"]
    generated = diagnostics["run"]["completed_at"]

    lines = [
        "# SHERLOC parser v2 report",
        "",
        f"Generated: `{generated}`",
        "",
        "## Outcome",
        "",
        f"Parser v2 passed the 19-page regression-fixture gate and the {challenge['challenge_count']}-page deterministic challenge gate before parsing all {summary['total_cases_parsed']:,} frozen-manifest records. The run completed in {elapsed:.2f} seconds with {summary['parse_error_count']} case-level parse errors.",
        "",
        "Corpus membership remains defined only by `data/manifests/case_urls.csv` (the frozen 2026-08-09 SHERLOC `Crime Type = Trafficking in persons` result set). URL-path crime type and page badges are audit fields only.",
        "",
        "## Validation gates",
        "",
        f"- Manual fixtures: **{fixture['status']}** ({fixture['fixture_count']}/19 parsed; {fixture['failure_count']} failed checks).",
        f"- Deterministic challenge: **{challenge['status']}** ({challenge['challenge_count']} cases; {challenge['failure_count']} failed checks).",
        f"- Full corpus: **{summary['total_cases_parsed']:,}** records written; status counts `{json.dumps(summary['overall_parse_status_counts'], sort_keys=True)}`.",
        "",
        "The fixture checks cover B637 English/French separation, Causa 2422 multilingual sections, Twitter as a corporate respondent, strict trafficking-badge scoping, independent legacy Keywords, Sentencia 298/2015's visible `Migrants` role, repeated entities, conservative charges, and direct-main-section Court scoping.",
        "",
        "The challenge contains all 25 genuine Fact Summary absences plus every URL-path category, temporal and byte-size extremes, malformed/duplicate tab identities, rare sidebar fields, multilingual party sections, legal entities, `Jurisdiction`, and maximum-defendant stress cases.",
        "",
        "## Corpus extraction coverage",
        "",
        f"- Fact Summary: English {fact['with_english']:,}; any usable variant {fact['with_any']:,}; no usable Fact Summary {fact['without_usable']:,}; statuses `{json.dumps(fact['status_counts'], sort_keys=True)}`.",
        f"- Trafficking sidebar: Acts {sidebar['acts_cases']:,} cases / {sidebar['acts_values']:,} values; Means {sidebar['means_cases']:,} / {sidebar['means_values']:,}; Exploitative Purposes {sidebar['purpose_cases']:,} / {sidebar['purpose_values']:,}.",
        f"- Legacy Keywords: Acts {legacy['acts_cases']:,} cases / {legacy['acts_values']:,} values; Means {legacy['means_cases']:,} / {legacy['means_values']:,}; Purpose {legacy['purpose_cases']:,} / {legacy['purpose_values']:,}.",
        f"- Multilingual/tabbed markup: {multilingual['cases_with_tabbed_sections']:,} cases, {multilingual['tabbed_section_groups']:,} tab groups, {multilingual['tabbed_panes']:,} panes; {multilingual['cases_with_multilingual_sections']:,} cases have at least one group with multiple detected languages.",
        f"- Person-role sections: {parties['cases_with_person_role_section']:,} cases, {parties['person_role_sections']:,} sections, {parties['person_role_records']:,} source records.",
        f"- Defendant/respondent sections: {parties['cases_with_defendant_respondent_section']:,} cases, {parties['defendant_respondent_sections']:,} sections, {parties['defendant_respondent_records']:,} source records.",
        f"- Charges / Claims / Decisions: {charges['cases_with_section']:,} cases, {charges['subject_records']:,} subject records, {charges['charge_records']:,} charge blocks.",
        f"- Main-record Court: {court['cases_with_main_record_court']:,} cases.",
        "",
        "## Missing Fact Summary pages",
        "",
        "All 25 known pages without `.factSummary` were confirmed as `SECTION_ABSENT`, not parser failures. Twenty-three retain case-specific prose in commentary, procedural, legal-reasoning, proceeding, or appellate-decision structures. Ranks 431 and 923 are genuinely structured-only records. Parser v2 does not substitute those sections into the dedicated Fact Summary field.",
        "",
        "## Warnings and unfamiliar structures",
        "",
        f"There are {summary['parse_warning_total']:,} recorded diagnostics across {summary['cases_with_any_parse_warning']:,} cases; {summary['parse_warning_severity_total']:,} are warning/error severity (the remainder are informational). Warning-code counts: `{json.dumps(warning_codes, sort_keys=True)}`.",
        "",
        f"Unfamiliar direct main-section headings, all preserved under `main_record_sections.other`: `{json.dumps(unfamiliar, ensure_ascii=False, sort_keys=True)}`.",
        "",
        f"Sections marked `PARTIAL`: `{json.dumps(partial, sort_keys=True)}`. A partial status means usable source content was retained but the source markup was ambiguous (for example duplicate tab IDs/hrefs, multiple active panes, an orphan charge, or duplicate trafficking badges).",
        "",
        "Notable source structures retained for later audit include rank 205's two trafficking badges; rank 307's distinct panes sharing blank labels and the same ID/href; rank 574's three-pane duplicate-ID group; recovered nested main-record sections on ranks 1 and 63; malformed nested charge subjects on ranks 63 and 1489, whose record-local text and literal DOM-subtree fallback are stored separately; repeated defendant-level legal reasoning/statutes; companies and grouped legal persons encoded in ordinary `.person` nodes; and `victimsPlaintiffs` containers visibly headed `Migrants`.",
        "",
        "## Fields intentionally only conservatively parsed",
        "",
        "- Charges, claims, verdicts, statutes, sentences, and person-level dispositions are ordered within their source DOM containers, with complete per-record/per-pane raw text retained. No defendant-charge-verdict relationships are inferred beyond explicit containment.",
        "- Procedural, commentary, source, attachment, cross-cutting, jurisdiction, appellate, and unheaded metadata sections retain ordered fields, pane provenance, and raw fallback text. Their heterogeneous subtypes are not semantically normalized.",
        "- Entity type is not inferred. DOM role, visible heading, original labels, values, and source record text are preserved for people, groups, companies, authorities, and other organizations.",
        "- The decorative SHERLOC sidebar bullet is removed only in `value_raw`; its exact displayed form remains in `source_text_raw` with `decorative_prefix_removed` recorded.",
        "",
        "## Before label normalization",
        "",
        "The next audit should define explicit, versioned policies for label aliases, duplicate/repeated translations, grouped parties, entity typing, and any reconciliation of trafficking-sidebar versus legacy Keyword annotations. Those decisions must occur in a separate normalization layer; parser v2 deliberately leaves the two annotation sources independent.",
        "",
        "No Act/Means/Purpose normalization, exploitation-type derivation, source reconciliation, benchmark inclusion decision, LLM extraction, evidence-grounding experiment, abstention model, or scientific interpretation was performed here.",
        "",
        "## Outputs",
        "",
        "- `data/interim/sherloc_cases_raw.jsonl`: one nested record per frozen-manifest case.",
        "- `outputs/metrics/parser_coverage.csv`: one coverage row per case.",
        "- `logs/parser_diagnostics.json`: gates, challenge membership, aggregate diagnostics, and per-case warning/error records.",
        "- `docs/sherloc_extraction_contract_v2.md`: schema and extraction rules.",
        "- `tests/test_sherloc_parser_v2.py`: offline regression tests.",
    ]
    return "\n".join(lines) + "\n"


def run_pipeline(args: argparse.Namespace) -> Dict[str, Any]:
    started_monotonic = time.monotonic()
    started_at = utc_now()
    parsed_at = started_at
    source_rows, download_by_rank, manifest_diagnostics = load_and_validate_manifests(
        args.corpus_manifest,
        args.download_manifest,
    )
    source_by_rank = {int(row["search_rank"]): row for row in source_rows}
    source_identity_to_rank = {
        canonical_case_identity(row["canonical_url"]): int(row["search_rank"])
        for row in source_rows
    }

    fixture_paths = sorted(Path(args.fixture_dir).glob("*.html"), key=lambda path: path.name)
    fixture_records: List[Dict[str, Any]] = []
    fixture_inputs: List[Dict[str, Any]] = []
    for path in fixture_paths:
        rank = fixture_source_rank(path, source_identity_to_rank)
        fixture_inputs.append({"fixture": path.name, "search_rank": rank})
        fixture_records.append(
            parse_case_file(
                path,
                source_by_rank[rank],
                download_by_rank[rank],
                input_kind="manual_regression_fixture",
                parsed_at=parsed_at,
            )
        )
    fixture_validation = validate_fixture_records(fixture_records)
    fixture_validation["fixtures"] = fixture_inputs
    print(
        f"Fixture gate: {fixture_validation['status']} "
        f"({len(fixture_records)} fixtures, {fixture_validation['failure_count']} failed checks)",
        flush=True,
    )
    if fixture_validation["status"] != "PASS":
        raise ParsePipelineError(
            f"Fixture validation failed: {fixture_validation['failed_checks']}. Full-corpus parsing was not started."
        )
    if args.stage == "fixtures":
        return {"fixture_validation": fixture_validation, "manifest_validation": manifest_diagnostics}

    challenge_reasons = challenge_rank_reasons(source_rows)
    challenge_records: List[Dict[str, Any]] = []
    for rank in sorted(challenge_reasons):
        path = resolve_raw_path(download_by_rank[rank], args.raw_html_dir)
        challenge_records.append(
            parse_case_file(
                path,
                source_by_rank[rank],
                download_by_rank[rank],
                input_kind="production_raw_html",
                parsed_at=parsed_at,
            )
        )
    challenge_validation = validate_challenge_records(
        challenge_records,
        source_rows,
        challenge_reasons,
    )
    print(
        f"Challenge gate: {challenge_validation['status']} "
        f"({len(challenge_records)} cases, {challenge_validation['failure_count']} failed checks)",
        flush=True,
    )
    if challenge_validation["status"] != "PASS":
        raise ParsePipelineError(
            f"Challenge validation failed: {challenge_validation['failed_checks']}. Full-corpus parsing was not started."
        )
    if args.stage == "challenge":
        return {
            "manifest_validation": manifest_diagnostics,
            "fixture_validation": fixture_validation,
            "challenge_validation": challenge_validation,
        }

    records_by_rank = {
        record["provenance"]["search_rank"]: record for record in challenge_records
    }
    parse_failures: List[Dict[str, Any]] = []
    for index, source_row in enumerate(source_rows, 1):
        rank = int(source_row["search_rank"])
        if rank not in records_by_rank:
            download_row = download_by_rank.get(rank)
            path = resolve_raw_path(download_row or {}, args.raw_html_dir)
            try:
                records_by_rank[rank] = parse_case_file(
                    path,
                    source_row,
                    download_row or {},
                    input_kind="production_raw_html",
                    parsed_at=parsed_at,
                )
            except Exception as exc:  # preserve every manifest member and failure
                records_by_rank[rank] = failed_case_record(source_row, download_row, path, exc, parsed_at)
                parse_failures.append(
                    {"search_rank": rank, "raw_path": str(path), "error": repr(exc)}
                )
        if args.progress_every and (index % args.progress_every == 0 or index == len(source_rows)):
            print(f"Parsed {index}/{len(source_rows)} manifest records", flush=True)

    records = [records_by_rank[rank] for rank in range(1, len(source_rows) + 1)]
    coverage = [coverage_row(record) for record in records]
    summary = corpus_summary(records, coverage)
    if len(records) != len(source_rows) or len(coverage) != len(source_rows):
        raise ParsePipelineError("Output cardinality does not match the frozen corpus manifest")

    completed_at = utc_now()
    diagnostics = {
        "schema_version": SCHEMA_VERSION,
        "parser_version": PARSER_VERSION,
        "run": {
            "started_at": started_at,
            "completed_at": completed_at,
            "elapsed_seconds": round(time.monotonic() - started_monotonic, 3),
            "command_stage": args.stage,
            "python_version": sys.version.split()[0],
            "standard_library_only": True,
            "corpus_snapshot_date": CORPUS_SNAPSHOT_DATE,
        },
        "manifest_validation": manifest_diagnostics,
        "fixture_validation": fixture_validation,
        "challenge_validation": challenge_validation,
        "corpus_summary": summary,
        "case_parse_failures": parse_failures,
        "case_diagnostics": [
            {
                "search_rank": record["provenance"]["search_rank"],
                "case_title": record["case_identity"].get("title_raw")
                or record["case_identity"].get("manifest_title_raw"),
                "overall_parse_status": record["parser_provenance"]["parse_status"],
                "warnings": record["parser_provenance"].get("warnings", []),
            }
            for record in records
            if record["parser_provenance"].get("warnings")
            or record["parser_provenance"]["parse_status"] != STATUS_FOUND
        ],
    }

    atomic_write_jsonl(args.output_jsonl, records)
    atomic_write_csv(args.coverage_output, coverage)
    atomic_write_json(args.diagnostics_output, diagnostics)
    atomic_write_text(args.report_output, render_parser_report(diagnostics))
    print(
        f"Full corpus: {len(records)} records, {summary['parse_error_count']} parse errors, "
        f"{summary['parse_warning_total']} diagnostics -> {args.output_jsonl}",
        flush=True,
    )
    return diagnostics


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate parser v2 and parse the frozen SHERLOC trafficking corpus."
    )
    parser.add_argument("--corpus-manifest", type=Path, default=DEFAULT_CORPUS_MANIFEST)
    parser.add_argument("--download-manifest", type=Path, default=DEFAULT_DOWNLOAD_MANIFEST)
    parser.add_argument("--raw-html-dir", type=Path, default=DEFAULT_RAW_HTML_DIR)
    parser.add_argument("--fixture-dir", type=Path, default=DEFAULT_FIXTURE_DIR)
    parser.add_argument("--output-jsonl", type=Path, default=DEFAULT_JSONL)
    parser.add_argument("--coverage-output", type=Path, default=DEFAULT_COVERAGE)
    parser.add_argument("--diagnostics-output", type=Path, default=DEFAULT_DIAGNOSTICS)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--stage",
        choices=("fixtures", "challenge", "all"),
        default="all",
        help="Stop after a successful validation stage; later stages always require earlier gates.",
    )
    parser.add_argument("--progress-every", type=int, default=100)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        run_pipeline(args)
    except (ParsePipelineError, FileNotFoundError, PermissionError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
