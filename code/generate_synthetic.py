"""Reproducible R^2 classification data. No third-party dependencies required."""
import argparse
import csv
import json
import math
from pathlib import Path
import random


CASES = {
    "linear": "Uniform [-2,2]^2; y=sign(x1+x2); reject |x1+x2|<0.4 for a clear margin.",
    "linear_noisy": "Uniform [-2,2]^2; y=sign(x1+x2); independently flip each label with probability 0.10.",
    "circles": "Uniform [-2,2]^2; y=+1 inside radius sqrt(8/pi), otherwise -1; equal class area in expectation.",
    "xor": "Uniform [-2,2]^2; y=+1 if x1*x2>0, otherwise -1.",
    "linear_imbalanced": "Same clear-margin rule as linear; retain exactly about 10% positives per split by rejection sampling.",
    "quadratic": "Uniform [-2,2]^2; y=+1 when x2 >= x1^2-0.5, otherwise -1; degree-2 parabolic boundary.",
    "polynomial_degree4": "Uniform [-2,2]^2; y=+1 when x2 >= 0.3*x1^4-x1^2+0.5, otherwise -1; degree-4 boundary.",
    "mixed_geometry": "Uniform [-2,2]^2; y=+1 above x2=0.5*x1 for x1<0 and above x2=x1^2-x1 for x1>=0, otherwise -1; continuous piecewise linear/quadratic boundary.",
}


def boundary_height(name, x1):
    """Known graph boundary x2=g(x1) for the added geometry cases."""
    if name == "quadratic":
        return x1 ** 2 - 0.5
    if name == "polynomial_degree4":
        return 0.3 * x1 ** 4 - x1 ** 2 + 0.5
    if name == "mixed_geometry":
        return 0.5 * x1 if x1 < 0 else x1 ** 2 - x1
    raise ValueError(f"No graph boundary defined for {name}")


def sample_dataset(name, n, seed):
    """Return raw (x1,x2,y) rows using a local RNG; no global random state."""
    if name not in CASES:
        raise ValueError(f"Unknown dataset: {name}")
    if n < 20:
        raise ValueError("Use at least 20 rows per split to represent both classes.")
    rng = random.Random(seed)
    rows = []
    positive_target = max(1, round(n * 0.1))
    positives = negatives = 0
    while len(rows) < n:
        x1, x2 = rng.uniform(-2, 2), rng.uniform(-2, 2)
        if name in ("linear", "linear_imbalanced") and abs(x1 + x2) < 0.4:
            continue
        score = x1 * x2 if name == "xor" else x1 + x2
        if name == "circles":
            score = 8 / math.pi - x1 * x1 - x2 * x2
        elif name in ("quadratic", "polynomial_degree4", "mixed_geometry"):
            score = x2 - boundary_height(name, x1)
        y = 1 if score >= 0 else -1
        if name == "linear_noisy" and rng.random() < 0.1:
            y = -y
        if name == "linear_imbalanced":
            if y == 1 and positives >= positive_target:
                continue
            if y == -1 and negatives >= n - positive_target:
                continue
        rows.append((x1, x2, y))
        positives += y == 1
        negatives += y == -1
    rng.shuffle(rows)
    return rows


def load_csv(path):
    """Return X,y as lists; DualSVM.fit converts them to NumPy arrays."""
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [[float(r["x1"]), float(r["x2"])] for r in rows], [int(r["y"]) for r in rows]


def write_preview(path, datasets):
    """SVG shows raw training points and the known generating boundaries."""
    height = 100 + 340 * math.ceil(len(datasets) / 3)
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="1080" height="{height}" viewBox="0 0 1080 {height}">',
           f'<rect width="1080" height="{height}" fill="white"/>',
           '<g font-family="sans-serif" fill="#17202a">',
           '<text x="30" y="30" font-size="22">Synthetic SVM datasets in R²</text>',
           '<text x="30" y="54" font-size="14">Training points: blue +1, orange −1. Dashed lines: generating boundaries.</text>']
    for i, (name, rows) in enumerate(datasets.items()):
        left, top = 45 + (i % 3) * 355, 110 + (i // 3) * 340
        # Map raw [-2,2] coordinates to a square with equal axis scales.
        def point(x, y):
            return left + (x + 2) * 65, top + (2 - y) * 65
        svg.append(f'<text x="{left}" y="{top-15}" font-size="17">{name}</text>')
        svg.append(f'<rect x="{left}" y="{top}" width="260" height="260" fill="#f7f9fb" stroke="#bbc3ca"/>')
        for v in (-2, 0, 2):
            px, py = point(v, v)
            svg.append(f'<text x="{px-5}" y="{top+278}" font-size="12">{v}</text>')
            svg.append(f'<text x="{left-22}" y="{py+4}" font-size="12">{v}</text>')
        svg.append(f'<text x="{left+125}" y="{top+298}" font-size="13">x1</text>')
        svg.append(f'<text x="{left-30}" y="{top+125}" font-size="13">x2</text>')
        for x1, x2, y in rows:
            px, py = point(x1, x2)
            color = "#2463b4" if y == 1 else "#d76a25"
            svg.append(f'<circle cx="{px:.3f}" cy="{py:.3f}" r="2.7" fill="{color}" opacity="0.75"/>')
        style = 'stroke="#333" stroke-width="1.5" stroke-dasharray="6 4" fill="none"'
        if name == "circles":
            svg.append(f'<circle cx="{left+130}" cy="{top+130}" r="{math.sqrt(8/math.pi)*65}" {style}/>')
        elif name == "xor":
            svg.append(f'<path d="M {left+130} {top} v 260 M {left} {top+130} h 260" {style}/>')
        elif name in ("quadratic", "polynomial_degree4", "mixed_geometry"):
            # Clip the displayed curve to the plotting square.
            commands = []
            drawing = False
            for j in range(801):
                x1 = -2 + j / 200
                x2 = boundary_height(name, x1)
                if -2 <= x2 <= 2:
                    px, py = point(x1, x2)
                    commands.append(f'{"L" if drawing else "M"} {px:.3f} {py:.3f}')
                    drawing = True
                else:
                    drawing = False
            svg.append(f'<path d="{" ".join(commands)}" {style}/>')
        else:
            svg.append(f'<path d="M {left} {top} L {left+260} {top+260}" {style}/>')
    svg.append('</g></svg>')
    path.write_text("\n".join(svg), encoding="utf-8")


def generate(output, seed=582, train=300, validation=100, test=100):
    output = Path(output)
    sizes = {"train": train, "validation": validation, "test": test}
    if any(n < 20 for n in sizes.values()):
        raise ValueError("Use at least 20 rows per split.")
    output.mkdir(parents=True, exist_ok=True)
    metadata = {"seed": seed, "features": ["x1", "x2"], "labels": [-1, 1],
                "preprocessing": "Raw coordinates; fit any scaling on training rows only.",
                "split_method": "Independent samples; distinct deterministic RNG seeds per dataset/split.",
                "datasets": {}}
    previews = {}
    for index, (name, description) in enumerate(CASES.items()):
        folder = output / name
        folder.mkdir(exist_ok=True)
        info = {"description": description, "splits": {}}
        for split_index, (split, n) in enumerate(sizes.items()):
            split_seed = seed + index * 100 + split_index
            rows = sample_dataset(name, n, split_seed)
            if len({r[2] for r in rows}) != 2:
                raise ValueError(f"{name}/{split} has only one class; increase sample size or change seed.")
            with (folder / f"{split}.csv").open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["x1", "x2", "y"])
                writer.writerows(rows)
            info["splits"][split] = {"seed": split_seed, "rows": n,
                "positive": sum(r[2] == 1 for r in rows), "negative": sum(r[2] == -1 for r in rows)}
            if split == "train":
                previews[name] = rows
        metadata["datasets"][name] = info
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    write_preview(output / "preview.svg", previews)
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "synthetic")
    parser.add_argument("--seed", type=int, default=582)
    parser.add_argument("--train", type=int, default=300)
    parser.add_argument("--validation", type=int, default=100)
    parser.add_argument("--test", type=int, default=100)
    args = parser.parse_args()
    generate(args.output, args.seed, args.train, args.validation, args.test)
    print(f"Saved {len(CASES)} datasets, metadata.json, and preview.svg to {args.output.resolve()}")
