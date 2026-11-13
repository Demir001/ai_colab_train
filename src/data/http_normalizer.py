"""HTTP Request/Response normalizer to strip HTML/CSS/SVG/base64/nonce noise and preserve security signals."""

from __future__ import annotations

import re
from typing import Iterable, List, Set

CRITICAL_HEADERS: Set[str] = {
    "host",
    "authorization",
    "cookie",
    "set-cookie",
    "content-type",
    "accept",
    "origin",
    "referer",
    "location",
    "x-forwarded-for",
    "x-real-ip",
    "x-api-key",
    "x-csrf-token",
    "x-xsrf-token",
    "idempotency-key",
    "access-control-allow-origin",
    "access-control-allow-credentials",
    "access-control-allow-methods",
    "access-control-allow-headers",
    "content-security-policy",
    "strict-transport-security",
    "x-frame-options",
    "x-content-type-options",
    "cache-control",
    "www-authenticate",
}

_STYLE_TAG_RE = re.compile(r"<style\b[^>]*>.*?</style>", re.IGNORECASE | re.DOTALL)
_SVG_TAG_RE = re.compile(r"<svg\b[^>]*>.*?</svg>", re.IGNORECASE | re.DOTALL)
_SCRIPT_NOISE_RE = re.compile(r"<script\b[^>]*>.*?</script>", re.IGNORECASE | re.DOTALL)
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_DATA_URI_B64_RE = re.compile(r"data:[a-zA-Z0-9/+.-]+;base64,[A-Za-z0-9+/=]{32,}")
_LONG_B64_RE = re.compile(r"\b[A-Za-z0-9+/]{120,}={0,2}\b")
_NONCE_PARAM_RE = re.compile(
    r"(\b(?:nonce|csrf_token|_csrf|authenticity_token|state)\s*[=:]\s*['\"]?)([A-Za-z0-9._~+/=-]{16,})(['\"]?)",
    re.IGNORECASE,
)
_MULTI_BLANK_RE = re.compile(r"\n{3,}")


def _compact_jwt_signature(text: str) -> str:
    """Keep JWT header.payload intact for claim inspection while compacting long signatures."""
    jwt_re = re.compile(r"\b(eyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,})\.([A-Za-z0-9_-]{16,})\b")
    return jwt_re.sub(r"\1.<SIG_COMPACTED>", text)


def _filter_http_headers(block: str, keep_headers: Iterable[str] = CRITICAL_HEADERS) -> str:
    """Filter raw HTTP header lines to retain only request/status lines, critical headers, and body."""
    keep_set = {h.lower() for h in keep_headers}
    lines = block.splitlines()
    out_lines: List[str] = []
    in_headers = False
    passed_first_line = False

    for line in lines:
        stripped = line.strip()
        if not passed_first_line:
            out_lines.append(line)
            if re.match(r"^(GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS)\s+\S+\s+HTTP/|HTTP/\d\.\d\s+\d{3}", stripped):
                passed_first_line = True
                in_headers = True
            continue

        if in_headers:
            if stripped == "":
                in_headers = False
                out_lines.append(line)
                continue
            if ":" in line:
                hdr_name = line.split(":", 1)[0].strip().lower()
                if hdr_name in keep_set:
                    out_lines.append(line)
            else:
                out_lines.append(line)
        else:
            out_lines.append(line)

    return "\n".join(out_lines)


def strip_body_noise(text: str, max_body_chars: int = 2400) -> str:
    """Remove HTML/CSS/SVG/base64/nonce noise from HTTP body or mixed trace text."""
    cleaned = _STYLE_TAG_RE.sub("<STYLE_OMITTED/>", text)
    cleaned = _SVG_TAG_RE.sub("<SVG_OMITTED/>", cleaned)
    cleaned = _SCRIPT_NOISE_RE.sub("<SCRIPT_OMITTED/>", cleaned)
    cleaned = _HTML_COMMENT_RE.sub("", cleaned)
    cleaned = _DATA_URI_B64_RE.sub("<DATA_URI_BASE64_OMITTED>", cleaned)
    cleaned = _LONG_B64_RE.sub(lambda m: f"<BASE64_OMITTED len={len(m.group(0))}>", cleaned)
    cleaned = _NONCE_PARAM_RE.sub(r"\1<DYNAMIC_NONCE>\3", cleaned)
    cleaned = _compact_jwt_signature(cleaned)
    cleaned = _MULTI_BLANK_RE.sub("\n\n", cleaned).strip()

    if len(cleaned) > max_body_chars:
        cleaned = cleaned[:max_body_chars] + "\n...[TRUNCATED_FOR_TOKEN_EFFICIENCY]"
    return cleaned


def normalize_http_exchange(raw_http: str, max_chars: int = 3000) -> str:
    """Normalize a raw HTTP Request/Response pair or multi-turn trace into compact security-focused text."""
    if not raw_http or not raw_http.strip():
        return ""

    # Split only at line-start HTTP request/response boundaries so JSON bodies are not mis-segmented as headers.
    segments = re.split(
        r"(?m)^(?=HTTP/\d\.\d\s+\d{3}\b|(?:GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS)\s+/)",
        raw_http,
    )
    normalized_segments: List[str] = []
    for seg in segments:
        if not seg.strip():
            continue
        filtered = _filter_http_headers(seg)
        normalized_segments.append(strip_body_noise(filtered, max_body_chars=max_chars))

    joined = "\n\n".join(normalized_segments)
    joined = _MULTI_BLANK_RE.sub("\n\n", joined).strip()
    if len(joined) > max_chars:
        return joined[:max_chars] + "\n...[TRUNCATED_FOR_TOKEN_EFFICIENCY]"
    return joined
