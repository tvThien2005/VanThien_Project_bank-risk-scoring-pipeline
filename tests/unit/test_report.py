"""Tests for shared HTML report framing."""

from src.utils.report import SYNTHETIC_BANNER, build_html_report


def test_html_report_places_synthetic_banner_before_content() -> None:
    """Ensure every generated report begins its body with the synthetic-data note."""
    report = build_html_report("Báo cáo", [("Kết quả", "<p>OK</p>")])

    assert SYNTHETIC_BANNER in report
    assert report.index(SYNTHETIC_BANNER) < report.index("<h1>Báo cáo</h1>")
