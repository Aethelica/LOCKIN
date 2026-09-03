"""Generate the extension's toolbar icons. Run once; commit the PNGs.

    python scripts/make_icons.py

Kept in the repo so the icons are reproducible rather than being four binary
files of unknown origin. Uses cv2/numpy, which are already dependencies for the
vision system, so this adds nothing to requirements.txt.

The mark is a ring with a filled centre -- an eye, or a target, depending on how
you look at it. Chosen because it survives being 16 pixels wide, which most
marks do not: at that size anything with fine detail turns to mush in the
browser toolbar. Drawn at 512px and downsampled with INTER_AREA for antialiasing.
"""

from pathlib import Path

import cv2
import numpy as np

OUT = Path("extension/icons")
SIZES = (16, 32, 48, 128)
MASTER = 512

# BGRA, to match OpenCV's channel order. Same palette as popup.css.
BG = (31, 28, 28, 255)        # #1c1c1f  near-black
FG = (246, 246, 246, 255)     # #f6f6f6  near-white
ACCENT = (246, 109, 47, 255)  # #2f6df6  the popup's accent blue


def build() -> np.ndarray:
    img = np.zeros((MASTER, MASTER, 4), dtype=np.uint8)

    # Rounded square: a filled rect plus four circles is enough at this scale,
    # and avoids depending on any drawing library beyond cv2.
    r = int(MASTER * 0.22)
    cv2.rectangle(img, (r, 0), (MASTER - r, MASTER), BG, -1)
    cv2.rectangle(img, (0, r), (MASTER, MASTER - r), BG, -1)
    for cx, cy in ((r, r), (MASTER - r, r), (r, MASTER - r), (MASTER - r, MASTER - r)):
        cv2.circle(img, (cx, cy), r, BG, -1)

    centre = (MASTER // 2, MASTER // 2)
    cv2.circle(img, centre, int(MASTER * 0.30), FG, int(MASTER * 0.075), cv2.LINE_AA)
    cv2.circle(img, centre, int(MASTER * 0.115), ACCENT, -1, cv2.LINE_AA)
    return img


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    master = build()
    for size in SIZES:
        icon = cv2.resize(master, (size, size), interpolation=cv2.INTER_AREA)
        path = OUT / f"icon{size}.png"
        cv2.imwrite(str(path), icon)
        print(f"wrote {path} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
