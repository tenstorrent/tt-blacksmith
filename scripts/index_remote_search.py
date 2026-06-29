#!/usr/bin/env python3
"""Crawl the built docs in output/ and push them to the OpenSearch-backed
docs search service so the in-page search modal has something to query.

Configured entirely through environment variables (see main()); intended to run
as a step in the Pages deploy workflow after `python build_docs.py`."""

from __future__ import annotations

import html
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable


# Keep batches small: the indexer Lambda fans each batch out to an OpenSearch
# _bulk call, and API Gateway caps the request at ~29s. 200 docs in one POST
# times out (HTTP 504) on the larger catalogs; 50 stays comfortably under it.
MAX_BATCH_SIZE = 50
TIMEOUT_SECONDS = 30


def _strip_html_to_text(content: str) -> str:
    # Remove script/style blocks first so they do not pollute search text.
    # \s* before > matches end tags with optional whitespace e.g. </script >.
    content = re.sub(r"<script\b[^>]*>.*?</script\s*>", " ", content, flags=re.IGNORECASE | re.DOTALL)
    content = re.sub(r"<style\b[^>]*>.*?</style\s*>", " ", content, flags=re.IGNORECASE | re.DOTALL)
    # Strip tags.
    content = re.sub(r"<[^>]+>", " ", content)
    # Decode entities and normalize whitespace.
    content = html.unescape(content)
    content = re.sub(r"\s+", " ", content).strip()
    return content


def _extract_title(content: str, fallback: str) -> str:
    match = re.search(r"<title[^>]*>(.*?)</title>", content, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return fallback
    return _strip_html_to_text(match.group(1)) or fallback


def _iter_html_files(output_root: Path) -> Iterable[Path]:
    for path in output_root.rglob("*.html"):
        rel = path.relative_to(output_root).as_posix()
        if rel.startswith("_static/") or rel.startswith("_sources/"):
            continue
        if "/_static/" in rel or "/_sources/" in rel:
            continue
        # Skip Sphinx's own search/genindex shell pages — they carry no content.
        name = path.name
        if name in {"search.html", "genindex.html"}:
            continue
        yield path


def _build_documents(output_root: Path, site_base_url: str, catalog: str, version: str, id_namespace: str) -> list[dict]:
    site_base_url = site_base_url.rstrip("/")
    docs: list[dict] = []

    for html_file in _iter_html_files(output_root):
        rel = html_file.relative_to(output_root).as_posix()
        raw = html_file.read_text(encoding="utf-8", errors="ignore")
        title = _extract_title(raw, rel)
        body = _strip_html_to_text(raw)
        doc_id = f"{catalog}:{id_namespace}:{version}:{rel}"
        url = f"{site_base_url}/{rel}"
        docs.append({"id": doc_id, "title": title, "body": body, "url": url})

    return docs


def _chunks(items: list[dict], size: int) -> Iterable[list[dict]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _post_json(url: str, payload: dict, api_key: str) -> tuple[int, str]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url=url,
        method="POST",
        data=data,
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS) as response:
            return response.getcode(), response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as err:
        return err.code, err.read().decode("utf-8", errors="replace")


def _read_config() -> tuple[dict[str, str], int]:
    """Read and validate required environment variables. Returns (config, exit_code)."""
    required = [
        ("SEARCH_API_BASE", "api_base"),
        ("DOC_CATALOG_SOURCE_ID", "source_id"),
        ("DOCS_SEARCH_INGEST_API_KEY", "api_key"),
        ("DOC_SITE_BASE_URL", "site_base"),
        ("DOCS_ID_NAMESPACE", "id_namespace"),
    ]
    cfg: dict[str, str] = {}
    for env_var, key in required:
        val = os.environ.get(env_var, "").strip()
        if not val:
            print(f"Missing {env_var}", file=sys.stderr)
            return {}, 2
        cfg[key] = val
    cfg["api_base"] = cfg["api_base"].rstrip("/")
    cfg["site_base"] = cfg["site_base"].rstrip("/")
    cfg["output_dir"] = os.environ.get("DOCS_OUTPUT_DIR", "output").strip()
    cfg["version"] = os.environ.get("DOCS_INDEX_VERSION", "latest").strip()
    return cfg, 0


def main() -> int:
    cfg, rc = _read_config()
    if rc != 0:
        return rc

    output_root = Path(cfg["output_dir"])
    if not output_root.exists():
        print(f"Output directory not found: {output_root}", file=sys.stderr)
        return 2

    docs = _build_documents(output_root, cfg["site_base"], cfg["source_id"], cfg["version"], cfg["id_namespace"])
    if not docs:
        print("No HTML docs found to index.", file=sys.stderr)
        return 1

    endpoint = f"{cfg['api_base']}/v1/index/{cfg['source_id']}"
    print(f"Indexing {len(docs)} documents to {endpoint}")

    indexed = 0
    for batch in _chunks(docs, MAX_BATCH_SIZE):
        payload = {"version": cfg["version"], "documents": batch}
        status, body = _post_json(endpoint, payload, cfg["api_key"])
        if status < 200 or status >= 300:
            print(f"Indexing failed: HTTP {status}", file=sys.stderr)
            print(body, file=sys.stderr)
            if status == 403:
                print(
                    (
                        "Hint: DOCS_SEARCH_INGEST_API_KEY must be the API key VALUE, "
                        "not the API key ID (for example, not '427rdpxpb6')."
                    ),
                    file=sys.stderr,
                )
            return 1
        indexed += len(batch)
        print(f"Indexed {indexed}/{len(docs)}")

    print("Indexing complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
