"""Fetch and periodically index pages from explicitly trusted medical sources."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Protocol
from urllib.error import URLError
from urllib.parse import urljoin, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

LOGGER = logging.getLogger(__name__)
TRUSTED_DOMAINS = ("cdc.gov", "medlineplus.gov", "nih.gov", "who.int")
MAX_SOURCE_BYTES = 2_000_000
MAX_SOURCE_TEXT = 100_000


class _KnowledgeBase(Protocol):
    def replace_source(
        self,
        source_url: str,
        source_name: str,
        title: str,
        text: str,
    ) -> int: ...


@dataclass(frozen=True)
class SourcePage:
    url: str
    title: str
    text: str


@dataclass(frozen=True)
class UpdateResult:
    updated_sources: int
    indexed_chunks: int
    errors: tuple[str, ...] = ()


class _PageTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title_parts: list[str] = []
        self.text_parts: list[str] = []
        self._title_depth = 0
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag == "title":
            self._title_depth += 1
        if tag in {"script", "style", "noscript", "svg", "nav", "footer", "header"}:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag == "title" and self._title_depth:
            self._title_depth -= 1
        if tag in {"script", "style", "noscript", "svg", "nav", "footer", "header"}:
            self._skip_depth = max(0, self._skip_depth - 1)

    def handle_data(self, data: str) -> None:
        cleaned = " ".join(data.split())
        if not cleaned:
            return
        if self._title_depth:
            self.title_parts.append(cleaned)
        elif not self._skip_depth:
            self.text_parts.append(cleaned)


def validate_source_url(url: str) -> str:
    parsed = urlparse(url.strip())
    hostname = (parsed.hostname or "").casefold().rstrip(".")
    if (
        parsed.scheme != "https"
        or parsed.username
        or parsed.password
        or parsed.port not in (None, 443)
        or not any(hostname == domain or hostname.endswith(f".{domain}") for domain in TRUSTED_DOMAINS)
    ):
        raise ValueError(
            "Source URLs must use HTTPS and a trusted host: "
            + ", ".join(TRUSTED_DOMAINS)
        )
    return url.strip()


class _TrustedRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        validate_source_url(urljoin(req.full_url, newurl))
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def load_source_urls(config_path: Path) -> list[str]:
    if not config_path.exists():
        return []
    urls: list[str] = []
    for line_number, line in enumerate(config_path.read_text(encoding="utf-8").splitlines(), 1):
        value = line.split("#", 1)[0].strip()
        if not value:
            continue
        try:
            urls.append(validate_source_url(value))
        except ValueError as error:
            raise ValueError(f"{config_path}:{line_number}: {error}") from error
    return list(dict.fromkeys(urls))


def fetch_source_page(url: str, timeout: int = 20) -> SourcePage:
    validated_url = validate_source_url(url)
    opener = build_opener(_TrustedRedirectHandler())
    request = Request(
        validated_url,
        headers={
            "User-Agent": "MedQuADKnowledgeUpdater/1.0 (educational medical information)",
            "Accept": "text/html,application/xhtml+xml",
        },
    )
    try:
        with opener.open(request, timeout=timeout) as response:
            final_url = validate_source_url(response.geturl())
            content_type = response.headers.get_content_type()
            if content_type not in {"text/html", "application/xhtml+xml"}:
                raise ValueError(f"Unsupported source content type {content_type}: {final_url}")
            raw = response.read(MAX_SOURCE_BYTES + 1)
            if len(raw) > MAX_SOURCE_BYTES:
                raise ValueError(f"Source page exceeds {MAX_SOURCE_BYTES} bytes: {final_url}")
            charset = response.headers.get_content_charset() or "utf-8"
    except (URLError, TimeoutError, OSError) as error:
        raise RuntimeError(f"Unable to fetch trusted source {validated_url}: {error}") from error

    parser = _PageTextParser()
    parser.feed(raw.decode(charset, errors="replace"))
    title = " ".join(parser.title_parts).strip() or final_url
    text = " ".join(parser.text_parts).strip()[:MAX_SOURCE_TEXT]
    if len(text) < 80:
        raise ValueError(f"Source page has too little readable text: {final_url}")
    return SourcePage(url=final_url, title=title, text=text)


class KnowledgeUpdater:
    def __init__(self, knowledge_base: _KnowledgeBase, source_urls: list[str]) -> None:
        self.knowledge_base = knowledge_base
        self.source_urls = [validate_source_url(url) for url in source_urls]
        self._refresh_lock = threading.Lock()

    def refresh(self) -> UpdateResult:
        with self._refresh_lock:
            return self._refresh_sources()

    def _refresh_sources(self) -> UpdateResult:
        updated_sources = 0
        indexed_chunks = 0
        errors: list[str] = []
        for url in self.source_urls:
            try:
                page = fetch_source_page(url)
                indexed_chunks += self.knowledge_base.replace_source(
                    page.url,
                    _source_name(page.url),
                    page.title,
                    page.text,
                )
                updated_sources += 1
            except (RuntimeError, ValueError, OSError) as error:
                message = f"{url}: {error}"
                LOGGER.error("Failed to refresh trusted source %s: %s", url, error)
                errors.append(message)
        return UpdateResult(updated_sources, indexed_chunks, tuple(errors))


def start_periodic_updates(
    updater: KnowledgeUpdater,
    interval_hours: int,
) -> threading.Thread:
    if interval_hours < 1:
        raise ValueError("Update interval must be at least one hour.")
    import schedule

    scheduler = schedule.Scheduler()
    scheduler.every(interval_hours).hours.do(_run_scheduled_update, updater)

    def run() -> None:
        while True:
            scheduler.run_pending()
            time.sleep(30)

    thread = threading.Thread(
        target=run,
        name="trusted-medical-source-updater",
        daemon=True,
    )
    thread.start()
    return thread


def _run_scheduled_update(updater: KnowledgeUpdater) -> None:
    try:
        result = updater.refresh()
    except Exception:
        LOGGER.exception("Scheduled trusted-source refresh failed")
        return
    if result.errors:
        LOGGER.error(
            "Trusted-source refresh completed with %d error(s): %s",
            len(result.errors),
            "; ".join(result.errors),
        )
    else:
        LOGGER.info(
            "Trusted-source refresh indexed %d chunks from %d source(s)",
            result.indexed_chunks,
            result.updated_sources,
        )


def _source_name(url: str) -> str:
    hostname = urlparse(url).hostname or url
    return hostname.removeprefix("www.")
