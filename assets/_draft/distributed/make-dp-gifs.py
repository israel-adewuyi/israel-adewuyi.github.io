"""Assemble DP mirror forward/backward browser captures into looping GIFs.

This keeps the export step editable without retaining generated preview files.
Place captures in ``previews/`` using these names:

- dp-draft-a-mirror-forward.png
- dp-draft-a-mirror-backward.png
"""

from pathlib import Path

from PIL import Image, ImageEnhance


ROOT = Path(__file__).resolve().parent
PREVIEWS = ROOT / "previews"


def fit_frame(image: Image.Image) -> Image.Image:
    frame = image.convert("RGB")
    return frame.crop((0, 0, frame.width, min(frame.height, 660)))


def tween(first: Image.Image, second: Image.Image, count: int) -> list[Image.Image]:
    return [Image.blend(first, second, index / count) for index in range(1, count)]


name = "dp-draft-a-mirror"
forward = fit_frame(Image.open(PREVIEWS / f"{name}-forward.png"))
backward = fit_frame(Image.open(PREVIEWS / f"{name}-backward.png"))

forward_lift = ImageEnhance.Brightness(forward).enhance(1.025)
backward_lift = ImageEnhance.Brightness(backward).enhance(1.025)

frames = (
    [forward] * 12
    + [forward_lift] * 2
    + tween(forward_lift, backward_lift, 7)
    + [backward] * 16
    + [backward_lift] * 2
    + tween(backward_lift, forward_lift, 7)
)
frames[0].save(
    ROOT / f"{name}.gif",
    save_all=True,
    append_images=frames[1:],
    duration=110,
    loop=0,
    optimize=True,
    disposal=2,
)
