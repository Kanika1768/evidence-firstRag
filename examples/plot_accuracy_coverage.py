"""Accuracy vs. Coverage Curve Generator for EvidenceFirst RAG Baselines.

Generates the Accuracy-Coverage trade-off curve comparing:
- B0 Baseline (Conventional one-shot RAG, no abstention)
- B1 Sufficiency-Aware RAG (One-shot with selective abstention)
- E1 EvidenceFirst Agentic RAG (Iterative recovery + selective abstention)

Outputs:
- plots/accuracy_coverage_curve.png (raster image)
- plots/accuracy_coverage_curve.svg (crisp vector visualization)
"""

from __future__ import annotations

import math
import struct
import sys
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLOTS_DIR = ROOT / "plots"


# ---------------------------------------------------------------------------
# Pure-Python Standalone PNG Canvas Generator (Zero External Dependencies)
# ---------------------------------------------------------------------------

class Canvas:
    """Minimal, self-contained 24-bit RGB bitmap canvas with PNG encoding."""

    def __init__(self, width: int, height: int, bg_color: tuple[int, int, int] = (255, 255, 255)) -> None:
        self.width = width
        self.height = height
        self.pixels = bytearray(width * height * 3)
        for i in range(0, len(self.pixels), 3):
            self.pixels[i] = bg_color[0]
            self.pixels[i + 1] = bg_color[1]
            self.pixels[i + 2] = bg_color[2]

    def set_pixel(self, x: int, y: int, color: tuple[int, int, int]) -> None:
        if 0 <= x < self.width and 0 <= y < self.height:
            idx = (y * self.width + x) * 3
            self.pixels[idx] = color[0]
            self.pixels[idx + 1] = color[1]
            self.pixels[idx + 2] = color[2]

    def fill_rect(self, x: int, y: int, w: int, h: int, color: tuple[int, int, int]) -> None:
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(self.width, x + w), min(self.height, y + h)
        for cy in range(y0, y1):
            row_start = (cy * self.width + x0) * 3
            row_end = (cy * self.width + x1) * 3
            for idx in range(row_start, row_end, 3):
                self.pixels[idx] = color[0]
                self.pixels[idx + 1] = color[1]
                self.pixels[idx + 2] = color[2]

    def draw_line(self, x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int], width: int = 1) -> None:
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        cx, cy = x0, y0
        while True:
            for wx in range(-width // 2, (width + 1) // 2):
                for wy in range(-width // 2, (width + 1) // 2):
                    self.set_pixel(cx + wx, cy + wy, color)
            if cx == x1 and cy == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                cx += sx
            if e2 < dx:
                err += dx
                cy += sy

    def draw_circle(self, cx: int, cy: int, r: int, color: tuple[int, int, int]) -> None:
        for y in range(-r, r + 1):
            for x in range(-r, r + 1):
                if x * x + y * y <= r * r:
                    self.set_pixel(cx + x, cy + y, color)

    def to_png_bytes(self) -> bytes:
        raw_scanlines = bytearray()
        stride = self.width * 3
        for y in range(self.height):
            raw_scanlines.append(0)  # Filter type 0: None
            row_start = y * stride
            raw_scanlines.extend(self.pixels[row_start : row_start + stride])

        compressed = zlib.compress(bytes(raw_scanlines), level=9)

        def make_chunk(tag: bytes, data: bytes) -> bytes:
            payload = tag + data
            crc = zlib.crc32(payload) & 0xFFFFFFFF
            return struct.pack(">I", len(data)) + payload + struct.pack(">I", crc)

        header = b"\x89PNG\r\n\x1a\n"
        ihdr_data = struct.pack(">IIBBBBB", self.width, self.height, 8, 2, 0, 0, 0)
        ihdr = make_chunk(b"IHDR", ihdr_data)
        idat = make_chunk(b"IDAT", compressed)
        iend = make_chunk(b"IEND", b"")
        return header + ihdr + idat + iend

    def save_png(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.to_png_bytes())


# ---------------------------------------------------------------------------
# Accuracy vs. Coverage Data & Visualizer
# ---------------------------------------------------------------------------

CURVE_DATA = {
    "B0 Baseline": {
        "color_rgb": (219, 68, 55),      # Red (#DB4437)
        "color_hex": "#DB4437",
        "description": "Conventional 1-shot (always answers, 0% abstention)",
        # (Coverage %, Selective Accuracy %)
        "points": [
            (100.0, 28.6),
        ],
    },
    "B1 Sufficiency-Aware": {
        "color_rgb": (244, 180, 0),     # Amber/Yellow (#F4B400)
        "color_hex": "#F4B400",
        "description": "1-shot + Sufficiency Judge (abstains on missing facts, 0 recovery)",
        "points": [
            (42.9, 100.0),
            (100.0, 28.6),
        ],
    },
    "E1 EvidenceFirst": {
        "color_rgb": (15, 157, 88),      # Green (#0F9D58)
        "color_hex": "#0F9D58",
        "description": "Agentic RAG (iterative recovery + selective abstention)",
        "points": [
            (57.1, 100.0),
            (100.0, 28.6),
        ],
    },
}


def render_svg(output_path: Path) -> None:
    """Generate crisp, scalable SVG visualization."""
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 600" width="900" height="600" style="background:#ffffff; font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,sans-serif;">
    <style>
        .grid {{ stroke: #e0e0e0; stroke-width: 1; stroke-dasharray: 4 4; }}
        .axis {{ stroke: #333333; stroke-width: 2; }}
        .title {{ font-size: 20px; font-weight: bold; fill: #1a1a1a; }}
        .subtitle {{ font-size: 13px; fill: #666666; }}
        .axis-label {{ font-size: 14px; font-weight: 600; fill: #333333; }}
        .tick-label {{ font-size: 12px; fill: #555555; }}
        .legend-text {{ font-size: 13px; fill: #222222; }}
    </style>

    <!-- Header -->
    <text x="450" y="45" text-anchor="middle" class="title">Accuracy vs. Coverage Curve (Selective RAG Evaluation)</text>
    <text x="450" y="68" text-anchor="middle" class="subtitle">Trade-off between response coverage and selective answer correctness across baselines</text>

    <!-- Plot Area Background -->
    <rect x="120" y="90" width="700" height="400" fill="#fafafa" stroke="#cccccc" stroke-width="1"/>

    <!-- Grid lines & Y-ticks (Accuracy 0 to 100%) -->
    <line x1="120" y1="490" x2="820" y2="490" class="grid"/>
    <text x="105" y="495" text-anchor="end" class="tick-label">0%</text>

    <line x1="120" y1="410" x2="820" y2="410" class="grid"/>
    <text x="105" y="415" text-anchor="end" class="tick-label">20%</text>

    <line x1="120" y1="330" x2="820" y2="330" class="grid"/>
    <text x="105" y="335" text-anchor="end" class="tick-label">40%</text>

    <line x1="120" y1="250" x2="820" y2="250" class="grid"/>
    <text x="105" y="255" text-anchor="end" class="tick-label">60%</text>

    <line x1="120" y1="170" x2="820" y2="170" class="grid"/>
    <text x="105" y="175" text-anchor="end" class="tick-label">80%</text>

    <line x1="120" y1="90" x2="820" y2="90" class="grid"/>
    <text x="105" y="95" text-anchor="end" class="tick-label">100%</text>

    <!-- Grid lines & X-ticks (Coverage 0 to 100%) -->
    <line x1="120" y1="90" x2="120" y2="490" class="axis"/>
    <text x="120" y="515" text-anchor="middle" class="tick-label">0%</text>

    <line x1="260" y1="90" x2="260" y2="490" class="grid"/>
    <text x="260" y="515" text-anchor="middle" class="tick-label">20%</text>

    <line x1="400" y1="90" x2="400" y2="490" class="grid"/>
    <text x="400" y="515" text-anchor="middle" class="tick-label">40%</text>

    <line x1="540" y1="90" x2="540" y2="490" class="grid"/>
    <text x="540" y="515" text-anchor="middle" class="tick-label">60%</text>

    <line x1="680" y1="90" x2="680" y2="490" class="grid"/>
    <text x="680" y="515" text-anchor="middle" class="tick-label">80%</text>

    <line x1="820" y1="90" x2="820" y2="490" class="axis"/>
    <text x="820" y="515" text-anchor="middle" class="tick-label">100%</text>

    <!-- Axis Labels -->
    <text x="470" y="545" text-anchor="middle" class="axis-label">Coverage (% of Questions Attempted)</text>
    <text x="45" y="290" text-anchor="middle" transform="rotate(-90, 45, 290)" class="axis-label">Selective Accuracy (% Correct when Answering)</text>

    <!-- Curves & Points -->
    <!-- B0 Baseline Point at (100%, 28.6%): x = 120 + 700 = 820, y = 490 - (28.6 * 4) = 375.6 -->
    <circle cx="820" cy="375.6" r="8" fill="#DB4437" stroke="#ffffff" stroke-width="2"/>
    <text x="805" y="370" text-anchor="end" font-size="12" font-weight="bold" fill="#DB4437">B0 (100% Cov, 28.6% Acc)</text>

    <!-- B1 Sufficiency-Aware Line: (42.9%, 100%) to (100%, 28.6%) -->
    <!-- (42.9%, 100%): x = 120 + 42.9*7 = 420.3, y = 490 - 400 = 90 -->
    <line x1="420.3" y1="90" x2="820" y2="375.6" stroke="#F4B400" stroke-width="3" stroke-dasharray="6 3"/>
    <circle cx="420.3" cy="90" r="8" fill="#F4B400" stroke="#ffffff" stroke-width="2"/>
    <text x="410" y="115" text-anchor="end" font-size="12" font-weight="bold" fill="#B78103">B1 Operating Point (42.9% Cov, 100% Acc)</text>

    <!-- E1 EvidenceFirst Line: (57.1%, 100%) to (100%, 28.6%) -->
    <!-- (57.1%, 100%): x = 120 + 57.1*7 = 519.7, y = 90 -->
    <line x1="519.7" y1="90" x2="820" y2="375.6" stroke="#0F9D58" stroke-width="4"/>
    <circle cx="519.7" cy="90" r="9" fill="#0F9D58" stroke="#ffffff" stroke-width="2"/>
    <text x="535" y="115" text-anchor="start" font-size="12" font-weight="bold" fill="#0F9D58">E1 Operating Point (57.1% Cov, 100% Acc)</text>

    <!-- Recovery Gain Annotation -->
    <path d="M 425 80 L 515 80" stroke="#0F9D58" stroke-width="2" marker-end="url(#arrow)"/>
    <text x="470" y="72" text-anchor="middle" font-size="11" font-weight="bold" fill="#0F9D58">+14.2% Recovery Gain (Q7 recovered)</text>

    <!-- Legend -->
    <rect x="140" y="110" width="310" height="95" rx="6" fill="#ffffff" stroke="#dddddd" stroke-width="1" opacity="0.95"/>
    <circle cx="160" cy="130" r="6" fill="#0F9D58"/>
    <text x="175" y="134" class="legend-text" font-weight="bold">E1 EvidenceFirst (Iterative Recovery)</text>

    <circle cx="160" cy="155" r="6" fill="#F4B400"/>
    <text x="175" y="159" class="legend-text" font-weight="bold">B1 Sufficiency-Aware (1-Shot + Abstain)</text>

    <circle cx="160" cy="180" r="6" fill="#DB4437"/>
    <text x="175" y="184" class="legend-text" font-weight="bold">B0 Baseline (1-Shot, No Abstain)</text>
</svg>"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(svg, encoding="utf-8")


def render_png(output_path: Path) -> None:
    """Generate pixel-rendered PNG visualization."""
    W, H = 900, 600
    canvas = Canvas(W, H, bg_color=(255, 255, 255))

    # Plot boundaries
    px0, py0, pw, ph = 120, 90, 700, 400
    canvas.fill_rect(px0, py0, pw, ph, (250, 250, 250))

    # Grid lines (X & Y)
    grid_color = (220, 220, 220)
    for y_val in [0, 20, 40, 60, 80, 100]:
        y_pos = int(py0 + ph - (y_val / 100.0) * ph)
        canvas.draw_line(px0, y_pos, px0 + pw, y_pos, grid_color, width=1)

    for x_val in [0, 20, 40, 60, 80, 100]:
        x_pos = int(px0 + (x_val / 100.0) * pw)
        canvas.draw_line(x_pos, py0, x_pos, py0 + ph, grid_color, width=1)

    # Outer border
    border_color = (180, 180, 180)
    canvas.draw_line(px0, py0, px0 + pw, py0, border_color, width=2)
    canvas.draw_line(px0, py0 + ph, px0 + pw, py0 + ph, (50, 50, 50), width=2)
    canvas.draw_line(px0, py0, px0, py0 + ph, (50, 50, 50), width=2)
    canvas.draw_line(px0 + pw, py0, px0 + pw, py0 + ph, border_color, width=2)

    # Plot lines:
    # E1: (57.1%, 100%) to (100%, 28.6%)
    e1_x0 = int(px0 + 0.571 * pw)
    e1_y0 = int(py0 + ph - 1.0 * ph)
    b0_x = int(px0 + 1.0 * pw)
    b0_y = int(py0 + ph - 0.286 * ph)
    canvas.draw_line(e1_x0, e1_y0, b0_x, b0_y, (15, 157, 88), width=4)

    # B1: (42.9%, 100%) to (100%, 28.6%)
    b1_x0 = int(px0 + 0.429 * pw)
    b1_y0 = int(py0 + ph - 1.0 * ph)
    canvas.draw_line(b1_x0, b1_y0, b0_x, b0_y, (244, 180, 0), width=3)

    # Draw data points (circles)
    canvas.draw_circle(b0_x, b0_y, 7, (219, 68, 55))
    canvas.draw_circle(b1_x0, b1_y0, 8, (244, 180, 0))
    canvas.draw_circle(e1_x0, e1_y0, 9, (15, 157, 88))

    # Recovery Gain bar
    canvas.draw_line(b1_x0, e1_y0 - 15, e1_x0, e1_y0 - 15, (15, 157, 88), width=3)

    # Legend box
    lx0, ly0, lw, lh = 140, 110, 310, 95
    canvas.fill_rect(lx0, ly0, lw, lh, (255, 255, 255))
    canvas.draw_line(lx0, ly0, lx0 + lw, ly0, (200, 200, 200), width=1)
    canvas.draw_line(lx0, ly0 + lh, lx0 + lw, ly0 + lh, (200, 200, 200), width=1)
    canvas.draw_line(lx0, ly0, lx0, ly0 + lh, (200, 200, 200), width=1)
    canvas.draw_line(lx0 + lw, ly0, lx0 + lw, ly0 + lh, (200, 200, 200), width=1)

    canvas.draw_circle(lx0 + 20, ly0 + 25, 6, (15, 157, 88))
    canvas.draw_circle(lx0 + 20, ly0 + 50, 6, (244, 180, 0))
    canvas.draw_circle(lx0 + 20, ly0 + 75, 6, (219, 68, 55))

    canvas.save_png(output_path)


def generate_accuracy_coverage_plots(output_dir: Path = PLOTS_DIR) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "accuracy_coverage_curve.png"
    svg_path = output_dir / "accuracy_coverage_curve.svg"

    render_svg(svg_path)
    render_png(png_path)

    print("\nACCURACY VS COVERAGE TABLE")
    print("-" * 75)
    print(f"{'System':<22} | {'Coverage (%)':<15} | {'Selective Accuracy (%)':<22}")
    print("-" * 75)
    for name, data in CURVE_DATA.items():
        for cov, acc in data["points"]:
            print(f"{name:<22} | {cov:<15.1f} | {acc:<22.1f}")
    print("-" * 75)
    print(f"✓ Saved PNG visualization: {png_path}")
    print(f"✓ Saved SVG visualization: {svg_path}")

    return png_path, svg_path


def main() -> None:
    generate_accuracy_coverage_plots()


if __name__ == "__main__":
    main()
