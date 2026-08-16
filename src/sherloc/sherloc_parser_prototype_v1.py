from pathlib import Path
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import re, json

PARSER_VERSION = "0.1.0"

LANG_LABELS = {
    "English": "en",
    "Français": "fr",
    "French": "fr",
    "Español": "es",
    "Spanish": "es",
    "中文": "zh",
    "Chinese": "zh",
    "Русский": "ru",
    "Russian": "ru",
    "عربي": "ar",
    "Arabic": "ar",
}

SIDEBAR_CLASSES = {
    "offences_raw": "crimeTypes_traffickingPersonsCrimeType_offences",
    "acts_raw": "crimeTypes_traffickingPersonsCrimeType_actsInvolved",
    "means_raw": "crimeTypes_traffickingPersonsCrimeType_meansUsed",
    "purpose_raw": "crimeTypes_traffickingPersonsCrimeType_exploitativePurposes",
    "form_raw": "crimeTypes_traffickingPersonsCrimeType_formOfTrafficking",
    "sector_raw": "crimeTypes_traffickingPersonsCrimeType_sectorsInWhichExploitationTakesPlace",
    "keywords_raw": "crimeTypes_traffickingPersonsCrimeType_keywords",
}

LEGACY_ALIASES = {
    "acts": {"Acts", "Acts:"},
    "means": {"Means", "Means:"},
    "purpose": {"Purpose of Exploitation", "Purpose of Exploitation:"},
    "form": {"Form of Trafficking", "Form of Trafficking:"},
    "sector": {
        "Sector in which exploitation takes place",
        "Sector in which exploitation takes place:",
    },
}

def clean_text(node):
    if node is None:
        return None
    text = " ".join(node.stripped_strings)
    text = re.sub(r"\s+", " ", text).strip()
    return text or None

def strip_bullet(text):
    if text is None:
        return None
    return re.sub(r"^[•\-\u2022]\s*", "", text).strip()

def get_page_locale(soup, url=None):
    script = soup.find("script", string=lambda s: s and "pageLocale" in s)
    if script:
        m = re.search(r'pageLocale\s*=\s*["\']([^"\']+)["\']', script.string or "")
        if m:
            return m.group(1).lower()
    html = soup.find("html")
    if html and html.get("lang"):
        return html.get("lang").lower()
    if url:
        m = re.search(r"/cld/([a-z]{2})/", url)
        if m:
            return m.group(1)
    return None

def infer_pane_language(pane_id, tab_label, page_locale):
    label = (tab_label or "").strip()
    if label in LANG_LABELS:
        return LANG_LABELS[label], "explicit_tab_label"
    if pane_id:
        for suffix in ("en","fr","es","zh","ru","ar"):
            if pane_id.lower().endswith(suffix):
                return suffix, "pane_id_suffix"
    if page_locale:
        return page_locale, "page_locale_default"
    return None, "unknown"

def extract_fact_summaries(soup, page_locale):
    fs = soup.select_one("div.factSummary")
    if not fs:
        return [], None, "SECTION_ABSENT", ["Fact Summary container not found"]

    panes = fs.select(".tab-content .tab-pane")
    records = []

    if panes:
        tab_map = {}
        for a in fs.select("ul.nav-tabs a[href]"):
            href = a.get("href", "")
            if href.startswith("#"):
                tab_map[href[1:]] = clean_text(a) or ""

        for pane in panes:
            pane_id = pane.get("id")
            label = tab_map.get(pane_id, "")
            lang, method = infer_pane_language(pane_id, label, page_locale)
            text = clean_text(pane)
            if text:
                records.append({
                    "language": lang,
                    "text_raw": text,
                    "source_pane_id": pane_id,
                    "tab_label_raw": label,
                    "language_detection_method": method,
                    "is_active_in_html": "active" in (pane.get("class") or []),
                })
    else:
        # Work on a copy so navigation labels are not mixed with narrative.
        fs_copy = BeautifulSoup(str(fs), "html.parser")
        for nav in fs_copy.select("ul.nav-tabs"):
            nav.decompose()
        text = clean_text(fs_copy.select_one("div.factSummary") or fs_copy)
        if text:
            records.append({
                "language": page_locale,
                "text_raw": text,
                "source_pane_id": None,
                "tab_label_raw": None,
                "language_detection_method": "page_locale_single_block",
                "is_active_in_html": True,
            })

    en = next((r["text_raw"] for r in records if r.get("language") == "en"), None)
    status = "FOUND" if records else "EMPTY"
    warnings = []
    if not records:
        warnings.append("Fact Summary container exists but no text was recovered")
    return records, en, status, warnings

def extract_sidebar(soup):
    badge = soup.select_one("div.traffickingPersonsCrimeType-details-badge")
    out = {k: [] for k in SIDEBAR_CLASSES}
    status = {k: "SECTION_ABSENT" for k in SIDEBAR_CLASSES}
    if not badge:
        return out, status

    for out_key, cls in SIDEBAR_CLASSES.items():
        h = badge.find(["h3","h4","h5"], class_=lambda classes:
                       classes and cls in (classes if isinstance(classes, list) else [classes]))
        if not h:
            continue
        container = h.find_next_sibling("div", class_="containerListElement")
        vals = []
        if container:
            for v in container.select(".value"):
                t = strip_bullet(clean_text(v))
                if t:
                    vals.append(t)
        out[out_key] = vals
        status[out_key] = "FOUND" if vals else "EMPTY"
    return out, status

def find_case_section(soup, heading_text):
    for h in soup.find_all(["h2","h3","h4"]):
        if clean_text(h) == heading_text:
            # Main case sections are usually contained by case-law-detail.
            parent = h.find_parent("div", class_="case-law-detail")
            return parent or h.parent
    return None

def extract_legacy_keywords(soup):
    sec = find_case_section(soup, "Keywords")
    result = {"categories_raw": {}, "acts_raw": [], "means_raw": [],
              "purpose_raw": [], "form_raw": [], "sector_raw": []}
    if not sec:
        return result, "SECTION_ABSENT"

    categories = sec.select("div.keywordCategory.field")
    for cat in categories:
        label_node = cat.find("div", class_="label")
        label = clean_text(label_node)
        if not label:
            continue
        vals = []
        # Restrict to this category's own tag values.
        tags = cat.select(":scope > div.tags div.value")
        if not tags:
            tags = cat.select("div.value")
        for v in tags:
            t = clean_text(v)
            if t and t not in vals:
                vals.append(t)
        result["categories_raw"][label] = vals

        label_norm = label.rstrip(":").strip()
        if label in LEGACY_ALIASES["acts"] or label_norm == "Acts":
            result["acts_raw"] = vals
        elif label in LEGACY_ALIASES["means"] or label_norm == "Means":
            result["means_raw"] = vals
        elif label in LEGACY_ALIASES["purpose"] or label_norm == "Purpose of Exploitation":
            result["purpose_raw"] = vals
        elif label in LEGACY_ALIASES["form"] or label_norm == "Form of Trafficking":
            result["form_raw"] = vals
        elif label in LEGACY_ALIASES["sector"] or label_norm == "Sector in which exploitation takes place":
            result["sector_raw"] = vals

    return result, "FOUND" if categories else "EMPTY"

def extract_person_records(soup, heading_text, container_selector):
    sec = find_case_section(soup, heading_text)
    if not sec:
        return [], None, "SECTION_ABSENT"

    container = sec.select_one(container_selector)
    persons = []
    if container:
        for person in container.select(":scope > div.person"):
            fields = []
            for fld in person.select(".field, .fieldFullWidth"):
                label = clean_text(fld.find("div", class_="label"))
                # Prefer direct value child.
                val_node = fld.find("div", class_="value")
                value = clean_text(val_node)
                if label or value:
                    fields.append({"label_raw": label, "value_raw": value})
            persons.append({
                "fields_raw": fields,
                "raw_text": clean_text(person)
            })
    raw = clean_text(sec)
    return persons, raw, "FOUND" if (persons or raw) else "EMPTY"

def extract_labeled_section(soup, heading_text):
    sec = find_case_section(soup, heading_text)
    if not sec:
        return {"fields_raw": [], "raw_text": None}, "SECTION_ABSENT"
    fields = []
    for fld in sec.select(".field, .fieldFullWidth"):
        label = clean_text(fld.find("div", class_="label"))
        val_node = fld.find("div", class_="value")
        value = clean_text(val_node)
        if label or value:
            fields.append({"label_raw": label, "value_raw": value})
    raw = clean_text(sec)
    return {"fields_raw": fields, "raw_text": raw}, "FOUND" if raw else "EMPTY"

def extract_country(soup):
    node = soup.select_one("div.countryNoHighlight.field div.text h3")
    if node:
        return clean_text(node)
    return None

def url_path_crime_type(url):
    if not url:
        return None
    m = re.search(r"/case-law-doc/([^/]+)/", url)
    return m.group(1) if m else None

def parse_case(path):
    path = Path(path)
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "html.parser")
    warnings = []

    title = clean_text(soup.find("title"))
    og = soup.find("meta", attrs={"property": "og:url"})
    url = og.get("content") if og else None
    page_locale = get_page_locale(soup, url)

    summaries, fact_en, fact_status, fact_warn = extract_fact_summaries(soup, page_locale)
    warnings.extend(fact_warn)

    sidebar, sidebar_status = extract_sidebar(soup)
    legacy, legacy_status = extract_legacy_keywords(soup)

    commentary_sec = find_case_section(soup, "Commentary and Significant Features")
    commentary = clean_text(commentary_sec)
    cross_sec = find_case_section(soup, "Cross-Cutting Issues")
    cross = clean_text(cross_sec)

    victims, victims_raw, victims_status = extract_person_records(
        soup, "Victims / Plaintiffs in the first instance", "div.victimsPlaintiffs"
    )

    # Defendant structures vary more. Preserve raw section even if repeated entity blocks
    # are not yet fully normalized in v0.1.
    defendants_sec = find_case_section(soup, "Defendants / Respondents in the first instance")
    defendants_raw = clean_text(defendants_sec)

    procedural, procedural_status = extract_labeled_section(soup, "Procedural Information")
    charges_sec = find_case_section(soup, "Charges / Claims / Decisions")
    charges_raw = clean_text(charges_sec)
    court_sec = find_case_section(soup, "Court")
    court_raw = clean_text(court_sec)

    result = {
        "case_identity": {
            "source_file": path.name,
            "source_url": url,
            "title": title,
            "country_raw": extract_country(soup),
            "url_path_crime_type": url_path_crime_type(url),
            "page_locale": page_locale,
        },
        "narrative": {
            "fact_summaries": summaries,
            "fact_summary_en_raw": fact_en,
            "commentary_raw": commentary,
            "cross_cutting_issues_raw": cross,
        },
        "trafficking_sidebar": sidebar,
        "legacy_keywords": legacy,
        "victims": victims,
        "victims_section_raw": victims_raw,
        "defendants_section_raw": defendants_raw,
        "procedural_information": procedural,
        "charges_claims_decisions_raw": charges_raw,
        "court_raw": court_raw,
        "availability": {
            "fact_summary": fact_status,
            "commentary": "FOUND" if commentary else "SECTION_ABSENT",
            "acts_sidebar": sidebar_status["acts_raw"],
            "means_sidebar": sidebar_status["means_raw"],
            "purpose_sidebar": sidebar_status["purpose_raw"],
            "legacy_keywords": legacy_status,
            "victims": victims_status,
            "defendants": "FOUND" if defendants_raw else "SECTION_ABSENT",
            "charges": "FOUND" if charges_raw else "SECTION_ABSENT",
            "court": "FOUND" if court_raw else "SECTION_ABSENT",
        },
        "parser_provenance": {
            "parser_version": PARSER_VERSION,
            "parse_warnings": warnings,
        },
    }
    return result

def parse_directory(html_dir, out_json):
    html_dir = Path(html_dir)
    cases = [parse_case(p) for p in sorted(html_dir.glob("*.html"))]
    Path(out_json).write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
    return cases

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("html_dir")
    ap.add_argument("out_json")
    args = ap.parse_args()
    cases = parse_directory(args.html_dir, args.out_json)
    print(f"Parsed {len(cases)} HTML files -> {args.out_json}")
