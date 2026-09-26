"""Gated harness for a genuine page-image resolution study.

This module exists so that a real multimodal experiment can be run without
inventing numbers. The current retrieval study is a *structural* layout map over
text chunks: it embeds and indexes no page image, figure, or table region, so no
image-resolution result is claimed anywhere in the project.

A real resolution study needs two things this repository does not currently have:

1. A PDF rasteriser. ``pypdf`` extracts text and cannot render pages, and Pillow
   cannot open a PDF without Ghostscript. Install ``pypdfium2``, a pure wheel
   with no system dependencies, to satisfy this.
2. A vision encoder that maps a page image to a vector. There is no local option
   installed: no torch, no transformers, no CLIP. The project can reach a vision
   model through its existing Ollama or OpenAI-compatible providers, or you can
   add a local encoder.

Deliberately absent: any synthetic image "embedding". A hash of pixel statistics
would let this module emit numbers, but those numbers would not measure visual
retrieval and would be worse than no result at all, because they would look like
a finding. This module therefore refuses to fabricate and records why it stopped.

Run:
    python -m research.multimodal
    python -m research.multimodal --pdf path/to.pdf --dpi 150

The status file it writes is the honest outcome: ``ran`` with measurements, or
``skipped`` with the missing prerequisite named.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

RESOLUTION_DPI = (72, 150, 300)
DEFAULT_PAGES = 4
MAX_PAGES = 40
MAX_DPI = 600
MIN_DPI = 36
STATUS_FILE = "results/multimodal.json"

SKIP_NO_PDF = "no_pdf_supplied"
SKIP_NO_RASTERIZER = "no_pdf_rasterizer"
SKIP_NO_VISION_MODEL = "no_vision_encoder"
SKIP_NO_PDF_TEXT = "pdf_has_no_extractable_text"


class PrerequisiteMissing(RuntimeError):
    """Raised when a required external component is not available."""


def available_rasterizer() -> str | None:
    """Return the name of an importable PDF rasteriser, or None."""
    try:
        import pypdfium2  # noqa: F401
    except Exception:
        return None
    return "pypdfium2"


def available_vision_model() -> str | None:
    """Return a description of a usable vision encoder, or None.

    Only a real encoder counts. Image statistics, pixel hashes, and colour
    histograms are explicitly not accepted, because they cannot represent visual
    meaning and any retrieval number derived from them would be meaningless.
    """
    if os.getenv("DOCMIND_VISION_MODEL", "").strip():
        return f"ollama:{os.environ['DOCMIND_VISION_MODEL'].strip()}"
    if os.getenv("DOCMIND_VISION_BASE_URL", "").strip():
        return f"openai-compatible:{os.environ['DOCMIND_VISION_BASE_URL'].strip()}"
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except Exception:
        return None
    return "local-transformers"


def probe(pdf_path: str | None = None, dpi: int = MIN_DPI) -> dict:
    """Report what a resolution study would need, without faking any measurement."""
    resolution = max(MIN_DPI, min(int(dpi), MAX_DPI))
    rasterizer = available_rasterizer()
    vision = available_vision_model()
    pdf = Path(pdf_path) if pdf_path else None
    missing = []
    if pdf is None or not pdf.is_file():
        missing.append(SKIP_NO_PDF)
    if not rasterizer:
        missing.append(SKIP_NO_RASTERIZER)
    if not vision:
        missing.append(SKIP_NO_VISION_MODEL)
    return {
        "status": "skipped" if missing else "ready",
        "missing": missing,
        "rasterizer": rasterizer,
        "vision_model": vision,
        "pdf": str(pdf) if pdf else None,
        "requested_dpi": resolution,
        "planned_dpi_ladder": list(RESOLUTION_DPI),
        "note": (
            "No image-resolution measurement was taken. The retrieval study in "
            "research/REPORT.md measures structural section granularity only."
        ),
    }


def rasterize(pdf_path: str, dpi: int, max_pages: int = DEFAULT_PAGES) -> list:
    """Render PDF pages to images at the requested DPI.

    Raises PrerequisiteMissing when no rasteriser is installed, so a caller can
    never mistake an empty result for a completed experiment.
    """
    if not available_rasterizer():
        raise PrerequisiteMissing(SKIP_NO_RASTERIZER)
    import pypdfium2

    pages = max(1, min(int(max_pages), MAX_PAGES))
    scale = max(1, round(float(dpi) / 72.0))
    document = pypdfium2.PdfDocument(pdf_path)
    images = []
    try:
        for index in range(min(pages, len(document))):
            page = document[index]
            bitmap = page.render(scale=scale)
            images.append(bitmap.to_pil())
    finally:
        close = getattr(document, "close", None)
        if callable(close):
            close()
    if not images:
        raise PrerequisiteMissing(SKIP_NO_PDF_TEXT)
    return images


def embed_pages(images, model: str | None = None):
    """Embed page images with a real vision encoder.

    Not implemented on purpose: there is no local encoder installed, and the
    project's remote providers are not exercised from research code. This raises
    rather than returning a placeholder vector.
    """
    raise PrerequisiteMissing(SKIP_NO_VISION_MODEL)


def write_status(status: dict, out_dir: Path | str = "research") -> Path:
    out = Path(out_dir)
    (out / "results").mkdir(parents=True, exist_ok=True)
    payload = dict(status)
    payload["generated_at"] = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    path = out / STATUS_FILE
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pdf", default=None, help="Path to a PDF with real pages.")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--pages", type=int, default=DEFAULT_PAGES)
    parser.add_argument("--out-dir", default="research")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    status = probe(args.pdf, args.dpi)
    if status["status"] == "ready":
        try:
            images = rasterize(args.pdf, args.dpi, args.pages)
        except PrerequisiteMissing as error:
            status["status"] = "skipped"
            status["missing"] = [str(error)]
        else:
            status["status"] = "ready"
            status["rasterized_pages"] = len(images)
            status["note"] = (
                "Pages rasterised, but no vision encoder is wired up in research code, "
                "so no retrieval measurement was taken."
            )
    path = write_status(status, args.out_dir)
    if not args.quiet:
        print(json.dumps(status, indent=2, sort_keys=True))
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
