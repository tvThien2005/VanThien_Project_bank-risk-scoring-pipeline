"""Helpers for transparent, consistently labelled HTML reports."""

from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path

from matplotlib.figure import Figure

from src.utils.logger import get_logger

LOGGER = get_logger(__name__)
SYNTHETIC_BANNER = (
    "<div class=\"synthetic-banner\"><strong>Dữ liệu synthetic:</strong> "
    "Kết quả trong báo cáo này không đại diện cho dữ liệu khách hàng thực.</div>"
)


def figure_to_html(figure: Figure) -> str:
    """Return a Matplotlib figure as an embedded PNG image tag."""
    buffer = BytesIO()
    figure.savefig(buffer, format="png", bbox_inches="tight", dpi=150)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f'<img alt="Biểu đồ" src="data:image/png;base64,{encoded}">'


def build_html_report(title: str, sections: list[tuple[str, str]]) -> str:
    """Build an HTML document with the synthetic-data banner at its start."""
    section_html = "".join(
        f"<section><h2>{heading}</h2>{content}</section>" for heading, content in sections
    )
    return f"""<!doctype html>
<html lang=\"vi\">
<head>
  <meta charset=\"utf-8\">
  <title>{title}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 2rem; color: #1f2937; }}
    .synthetic-banner {{ background: #fef3c7; border-left: 4px solid #d97706; padding: 1rem; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 1rem; }}
    th, td {{ border: 1px solid #d1d5db; padding: 0.5rem; text-align: left; }}
    th {{ background: #f3f4f6; }}
    img {{ max-width: 100%; height: auto; }}
  </style>
</head>
<body>
  {SYNTHETIC_BANNER}
  <h1>{title}</h1>
  {section_html}
</body>
</html>"""


def write_html_report(html: str, path: str | Path) -> None:
    """Write an HTML report to its configured output path."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    LOGGER.info("Wrote HTML report: %s", output_path)
