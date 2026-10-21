"""
แปลง report/index.html เป็น PDF (A4, 1 สไลด์ต่อหน้า) ด้วย Chrome/Edge แบบ headless

Usage:
    python report/make_pdf.py
"""
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
BROWSERS = [r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"]

PRINT_CSS = """
<style>
@page { size: A4; margin: 12mm 12mm 14mm; }
/* บังคับธีมสว่างสำหรับพิมพ์ */
:root, :root[data-theme="dark"] {
  --bg: #ffffff; --surface: #ffffff; --fg: #1f2124; --muted: #5d6066; --rule: #dedcd6;
  --accent: #2a78d6; --accent-2: #c9551f; --good: #1b7f4b; --bad: #b3261e; --chip: #eaf1fb; color-scheme: light;
}
html, body { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body { font-size: 13px; line-height: 1.55; }
.wrap { max-width: none; padding: 0; }
nav.toc ol { columns: 2; }
section.slide { break-before: page; margin-top: 0; padding-top: 10px; gap: 10px; }
figure, .key, pre.mermaid, tr { break-inside: avoid; }
figure img { max-height: 84mm; width: auto; max-width: 100%; margin: 0 auto; }
table { font-size: 11px; } th, td { padding: 5px 7px; }
pre.mermaid { padding: 4px; } pre.mermaid svg { max-height: 120mm; }
.verdict { grid-template-columns: repeat(3, 1fr); }
.verdict div { break-inside: avoid; }
a { color: inherit; text-decoration: none; }
</style>
<script src="https://cdn.jsdelivr.net/npm/mermaid@10.9.1/dist/mermaid.min.js"></script>
<script>mermaid.initialize({ startOnLoad: true, theme: "neutral", flowchart: { useMaxWidth: true } });</script>
"""


def main():
    html = (HERE / "index.html").read_text(encoding="utf-8")
    page = ("<!doctype html><html lang=\"th\"><head><meta charset=\"utf-8\">"
            + html.split("<div class=\"wrap\">")[0] + PRINT_CSS + "</head><body><div class=\"wrap\">"
            + html.split("<div class=\"wrap\">", 1)[1] + "</body></html>")
    page = page.replace("flowchart LR", "flowchart TB")  # แนวตั้งอ่านง่ายกว่าบนกระดาษ A4
    src = HERE / "_print.html"
    src.write_text(page, encoding="utf-8")
    out = HERE / "CN340_parity_report.pdf"
    browser = next(b for b in BROWSERS if Path(b).exists())
    subprocess.run([browser, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    "--virtual-time-budget=15000", f"--print-to-pdf={out}", src.as_uri()],
                   check=True, capture_output=True, timeout=180)
    src.unlink()
    print(f"saved {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
