"""Fetch public OpenReview submissions by venue or invitation."""

from __future__ import annotations

import re
import time
from datetime import datetime, timezone
from typing import Any

import requests


API2_BASE = "https://api2.openreview.net/notes"
API1_BASE = "https://api.openreview.net/notes"
FORUM_URL = "https://openreview.net/forum?id={id}"
HEADERS = {"User-Agent": "iDeer research digest/1.0"}


def _content_value(content: dict[str, Any], key: str, default: Any = "") -> Any:
    value = content.get(key, default)
    if isinstance(value, dict) and "value" in value:
        return value.get("value", default)
    return value


def _as_text(value: Any) -> str:
    if isinstance(value, list):
        return ", ".join(str(item).strip() for item in value if str(item).strip())
    return str(value or "").strip()


def _as_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value or "").strip()
    return [text] if text else []


def _year_from_timestamp(value: Any) -> str:
    try:
        millis = int(value)
    except (TypeError, ValueError):
        return ""
    if millis <= 0:
        return ""
    return datetime.fromtimestamp(millis / 1000, tz=timezone.utc).strftime("%Y")


def _iso_from_timestamp(value: Any) -> str:
    try:
        millis = int(value)
    except (TypeError, ValueError):
        return ""
    if millis <= 0:
        return ""
    return datetime.fromtimestamp(millis / 1000, tz=timezone.utc).isoformat()


def _year_from_text(text: str) -> str:
    match = re.search(r"\b(20\d{2}|19\d{2})\b", str(text or ""))
    return match.group(1) if match else ""


def _submission_invitation(venue: str) -> str:
    venue = str(venue or "").strip().rstrip("/")
    return f"{venue}/-/Submission" if venue else ""


def _request_notes(base_url: str, params: dict[str, Any], max_results: int) -> list[dict]:
    notes: list[dict] = []
    offset = 0
    page_size = min(max(max_results, 1), 100)

    while len(notes) < max_results:
        page_params = {**params, "limit": page_size, "offset": offset}
        try:
            response = requests.get(base_url, params=page_params, headers=HEADERS, timeout=30)
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            print(f"[openreview] Fetch failed from {base_url}: {exc}")
            break

        page_notes = payload.get("notes") or []
        if not isinstance(page_notes, list) or not page_notes:
            break

        notes.extend(note for note in page_notes if isinstance(note, dict))
        if len(page_notes) < page_size:
            break
        offset += page_size
        time.sleep(0.2)

    return notes[:max_results]


def _normalize_note(note: dict, venue_hint: str = "", api_version: str = "2") -> dict | None:
    content = note.get("content") if isinstance(note.get("content"), dict) else {}
    note_id = str(note.get("id") or "").strip()
    forum_id = str(note.get("forum") or note_id).strip()
    title = _as_text(_content_value(content, "title"))
    abstract = _as_text(_content_value(content, "abstract") or _content_value(content, "summary"))
    if not title or not abstract:
        return None

    authors = _as_list(_content_value(content, "authors"))
    keywords = _as_list(_content_value(content, "keywords"))
    venue = _as_text(
        _content_value(content, "venue")
        or _content_value(content, "venueid")
        or _content_value(content, "venue_id")
        or venue_hint
    )
    decision = _as_text(_content_value(content, "decision"))
    year = (
        _as_text(_content_value(content, "year"))
        or _year_from_text(venue)
        or _year_from_timestamp(note.get("pdate") or note.get("cdate") or note.get("tcdate"))
    )
    venue_or_status = venue or "OpenReview submission"
    if decision and decision.lower() not in venue_or_status.lower():
        venue_or_status = f"{venue_or_status} / {decision}"

    invitation = note.get("invitation") or ""
    if not invitation and isinstance(note.get("invitations"), list):
        invitation = ", ".join(str(item) for item in note.get("invitations", [])[:2])

    return {
        "id": note_id,
        "forum": forum_id,
        "title": title,
        "abstract": abstract,
        "authors": authors,
        "keywords": keywords,
        "year": year,
        "venue_or_status": venue_or_status,
        "venue": venue,
        "decision": decision,
        "invitation": invitation,
        "api_version": api_version,
        "cdate": note.get("cdate") or note.get("tcdate") or note.get("mdate") or "",
        "published_at": _iso_from_timestamp(note.get("pdate") or note.get("cdate") or note.get("tcdate")),
        "url": FORUM_URL.format(id=forum_id or note_id),
    }


def _within_days(item: dict, days: int) -> bool:
    if days <= 0:
        return True
    try:
        millis = int(item.get("cdate") or 0)
    except (TypeError, ValueError):
        return True
    if millis <= 0:
        return True
    cutoff = int(time.time() * 1000) - days * 24 * 60 * 60 * 1000
    return millis >= cutoff


def _matches_queries(item: dict, queries: list[str]) -> bool:
    if not queries:
        return True
    haystack = " ".join(
        [
            item.get("title", ""),
            item.get("abstract", ""),
            item.get("venue_or_status", ""),
            " ".join(item.get("keywords") or []),
        ]
    ).lower()
    return any(query.lower() in haystack for query in queries if query.strip())


def fetch_openreview_submissions(
    venues: list[str] | None = None,
    invitations: list[str] | None = None,
    queries: list[str] | None = None,
    max_results: int = 50,
    days: int = 0,
    api_versions: list[str] | None = None,
) -> list[dict]:
    venues = [v.strip() for v in (venues or []) if v and v.strip()]
    invitations = [v.strip() for v in (invitations or []) if v and v.strip()]
    queries = [q.strip() for q in (queries or []) if q and q.strip()]
    versions = [v.strip() for v in (api_versions or ["2", "1"]) if v and v.strip()]

    if not venues and not invitations:
        print("[openreview] No venues or invitations configured; skipping fetch.")
        return []

    candidates: list[tuple[str, str, dict[str, Any]]] = []
    for venue in venues:
        invitation = _submission_invitation(venue)
        if invitation:
            candidates.append((venue, "2", {"invitation": invitation}))
            candidates.append((venue, "1", {"invitation": invitation}))
        candidates.append((venue, "2", {"content.venueid": venue}))
        candidates.append((venue, "2", {"domain": venue}))
    for invitation in invitations:
        venue_hint = invitation.split("/-/")[0] if "/-/" in invitation else ""
        candidates.append((venue_hint, "2", {"invitation": invitation}))
        candidates.append((venue_hint, "1", {"invitation": invitation}))

    seen_candidates: set[tuple[str, str, tuple[tuple[str, Any], ...]]] = set()
    seen_items: set[str] = set()
    items: list[dict] = []
    per_request_limit = max(max_results * 2, max_results)

    for venue_hint, version, params in candidates:
        if version not in versions:
            continue
        key = (venue_hint, version, tuple(sorted(params.items())))
        if key in seen_candidates:
            continue
        seen_candidates.add(key)

        base_url = API2_BASE if version == "2" else API1_BASE
        notes = _request_notes(base_url, params, per_request_limit)
        print(f"[openreview] {len(notes)} notes fetched via API{version} params={params}")
        for note in notes:
            item = _normalize_note(note, venue_hint=venue_hint, api_version=version)
            if not item:
                continue
            dedupe_key = item.get("forum") or item.get("id") or item.get("title", "").lower()
            title_key = re.sub(r"\s+", " ", item.get("title", "").lower()).strip()
            if dedupe_key in seen_items or title_key in seen_items:
                continue
            if not _within_days(item, days):
                continue
            if not _matches_queries(item, queries):
                continue
            seen_items.add(dedupe_key)
            seen_items.add(title_key)
            items.append(item)
            if len(items) >= max_results:
                return items

    return items[:max_results]
