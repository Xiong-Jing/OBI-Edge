"""Calculate and plot the six manuscript diagnostics for eight methods."""

import csv
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Edit these paths before running this file.
BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "test_images"
OUTPUT_DIR = BASE_DIR / "results" / "metrics"
MAX_IMAGES = None  # Set an integer for a quick run, or None for all images.
FIGURE_DPI = 300
METHODS = ("Our_Method", "Canny", "Sobel", "Laplacian",
           "Roberts", "Prewitt", "Scharr", "Kirsch")
METRICS = ("Contour_SNR", "Solidity", "Smoothness", "Edge_Continuity",
           "Entropy", "Gradient_Consistency")


def get_diamond_kernel(size):
    """Return the (2 * size + 1)-square diamond used in the manuscript."""
    yy, xx = np.ogrid[-size:size + 1, -size:size + 1]
    return (np.abs(xx) + np.abs(yy) <= size).astype(np.uint8)


def calculate_adaptive_threshold(image):
    """SWBT from manuscript equations 11-15."""
    median = float(np.median(image))
    p80 = float(np.percentile(image, 80))
    contrast = float(np.std(image)) / 128.0
    if contrast < 0.8:
        threshold = 0.3 * p80 + 0.7 * median
    elif contrast > 1.2:
        # Preserved from the paper. For uint8, std <= 127.5, so this
        # branch is unreachable; the paper's contrast rule needs revision.
        threshold = 0.5 * p80 + 0.5 * median
    else:
        threshold = 0.7 * p80 + 0.3 * median
    return int(np.clip(threshold, 50, 120))


def read_gray_image(path):
    image = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    return image


def write_image(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(path.suffix, image)
    if not success:
        raise OSError(f"Cannot encode image: {path}")
    encoded.tofile(str(path))


def get_image_paths(input_dir, output_dir):
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    if not input_dir.is_dir():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    if input_dir.resolve() == output_dir.resolve():
        raise ValueError("Input and output directories must be different")
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    paths = sorted(p for p in input_dir.iterdir()
                   if p.is_file() and p.suffix.lower() in extensions)
    if MAX_IMAGES is not None:
        if not isinstance(MAX_IMAGES, int) or MAX_IMAGES < 1:
            raise ValueError("MAX_IMAGES must be a positive integer or None")
        paths = paths[:MAX_IMAGES]
    if not paths:
        raise ValueError(f"No supported images found in: {input_dir}")
    stems = [p.stem.casefold() for p in paths]
    if len(stems) != len(set(stems)):
        raise ValueError("Images with the same stem would overwrite output files")
    output_dir.mkdir(parents=True, exist_ok=True)
    return paths


def extract_contours(image):
    """Apply the complete method in manuscript sections 3.3-3.6."""
    if image.ndim != 2 or image.dtype != np.uint8 or image.size == 0:
        raise ValueError("Expected a nonempty uint8 grayscale image")

    # Gaussian filtering, strong erosion, and mild dilation (equations 1-5).
    denoised = cv2.GaussianBlur(image, (3, 3), 1.0)
    eroded = cv2.erode(denoised, get_diamond_kernel(10))
    enhanced = cv2.dilate(eroded, get_diamond_kernel(1))
    inverted = cv2.bitwise_not(enhanced)
    threshold = calculate_adaptive_threshold(inverted)
    _, binary = cv2.threshold(inverted, threshold, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    max_area = max((cv2.contourArea(c) for c in contours), default=0)
    valid_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 800:
            continue
        _, y, width, height = cv2.boundingRect(contour)
        is_serial_number = (
            y > image.shape[0] - 120
            and area < 0.1 * max_area
            and 0.6 < width / height < 1.6
        )
        if is_serial_number or width < 8 or height < 8:
            continue
        valid_contours.append(contour)

    filtered_mask = np.zeros_like(image)
    if valid_contours:
        cv2.drawContours(filtered_mask, valid_contours, -1, 255, cv2.FILLED)

    # Both closing passes are retained as printed in the manuscript.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    first_closed = cv2.morphologyEx(filtered_mask, cv2.MORPH_CLOSE, kernel)
    final_mask = cv2.morphologyEx(first_closed, cv2.MORPH_CLOSE, kernel)
    final_contours, _ = cv2.findContours(
        final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    edge = np.zeros_like(image)
    if final_contours:
        cv2.drawContours(edge, final_contours, -1, 255, 1)

    return {
        "original": image, "denoised": denoised, "eroded": eroded,
        "enhanced": enhanced, "inverted": inverted, "binary": binary,
        "raw_contours": contours, "filtered_mask": filtered_mask,
        "first_closed": first_closed, "mask": final_mask, "edge": edge,
        "contours": final_contours, "threshold": threshold,
        "raw_fragments": len(contours),
    }


def save_csv(path, rows):
    if not rows:
        return
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def get_traditional_result(image_gray, method):
    blurred = cv2.GaussianBlur(image_gray, (5, 5), 1.5)
    if method == 'Canny':
        edges_raw = cv2.Canny(blurred, 10, 30)
    elif method == 'Sobel':
        gx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        _, edges_raw = cv2.threshold(mag, 40, 255, cv2.THRESH_BINARY)
    elif method == 'Laplacian':
        dst = cv2.Laplacian(blurred, cv2.CV_64F, ksize=3)
        abs_dst = cv2.convertScaleAbs(dst)
        _, edges_raw = cv2.threshold(abs_dst, 20, 255, cv2.THRESH_BINARY)
    elif method == 'Roberts':
        kx = np.array([[1, 0], [0, -1]], dtype=np.float32)
        ky = np.array([[0, 1], [-1, 0]], dtype=np.float32)
        rx = cv2.filter2D(blurred, cv2.CV_32F, kx)
        ry = cv2.filter2D(blurred, cv2.CV_32F, ky)
        mag = cv2.magnitude(rx, ry)
        mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        _, edges_raw = cv2.threshold(mag, 30, 255, cv2.THRESH_BINARY)
    elif method == 'Prewitt':
        kx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
        ky = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
        px = cv2.filter2D(blurred, cv2.CV_32F, kx)
        py = cv2.filter2D(blurred, cv2.CV_32F, ky)
        mag = cv2.magnitude(px, py)
        mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        _, edges_raw = cv2.threshold(mag, 35, 255, cv2.THRESH_BINARY)
    elif method == 'Scharr':
        sx = cv2.Scharr(blurred, cv2.CV_64F, 1, 0)
        sy = cv2.Scharr(blurred, cv2.CV_64F, 0, 1)
        mag = cv2.magnitude(sx, sy)
        mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        _, edges_raw = cv2.threshold(mag, 45, 255, cv2.THRESH_BINARY)
    elif method == 'Kirsch':
        kernels = [
            np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]], dtype=np.float32),
            np.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]], dtype=np.float32),
            np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]], dtype=np.float32),
            np.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]], dtype=np.float32),
            np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]], dtype=np.float32),
            np.array([[-3, -3, -3], [5, 0, -3], [5, 5, -3]], dtype=np.float32),
            np.array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]], dtype=np.float32),
            np.array([[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]], dtype=np.float32),
        ]
        responses = []
        for kernel in kernels:
            response = cv2.filter2D(blurred, cv2.CV_32F, kernel)
            responses.append(response)
        kirsch_edges = np.max(np.abs(responses), axis=0)
        kirsch_edges = cv2.normalize(kirsch_edges, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        _, edges_raw = cv2.threshold(kirsch_edges, 50, 255, cv2.THRESH_BINARY)
    else:
        raise ValueError(f"Unknown method: {method}")
    contours, _ = cv2.findContours(edges_raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    raw_fragments = len(contours)
    if not contours:
        empty = np.zeros_like(image_gray)
        return (empty, empty, raw_fragments)
    main_contour = max(contours, key=cv2.contourArea)
    solid_mask = np.zeros_like(image_gray)
    cv2.drawContours(solid_mask, [main_contour], -1, 255, -1)
    edge_image = np.zeros_like(image_gray)
    cv2.drawContours(edge_image, [main_contour], -1, 255, 1)
    return (solid_mask, edge_image, raw_fragments)


def simple_contour_generation(image):
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    result = extract_contours(image)
    return result["mask"], result["edge"], result["raw_fragments"]


def calculate_final_metrics(solid_mask, edge_image, raw_fragments, original_gray):
    """Return the six manuscript diagnostics; these are not accuracy metrics."""
    contours, _ = cv2.findContours(solid_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return (0.0,) * 6
    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    perimeter = cv2.arcLength(contour, True)
    hull_perimeter = cv2.arcLength(hull, True)
    snr = area / raw_fragments if raw_fragments else 0.0
    solidity = area / hull_area if hull_area else 0.0
    smoothness = hull_perimeter / perimeter if perimeter else 0.0

    # Equation 24 uses components of the edge image, not of the filled mask.
    _, _, component_stats, _ = cv2.connectedComponentsWithStats(
        (edge_image > 0).astype(np.uint8), connectivity=8)
    edge_areas = component_stats[1:, cv2.CC_STAT_AREA]
    continuity = float(edge_areas.max() / edge_areas.sum()) if edge_areas.size else 0.0

    gx = cv2.Sobel(original_gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(original_gray, cv2.CV_64F, 0, 1, ksize=3)
    edge_pixels = edge_image > 0
    magnitudes = cv2.magnitude(gx, gy)[edge_pixels]
    entropy = consistency = 0.0
    if magnitudes.size and magnitudes.max() > 0:
        span = float(np.ptp(magnitudes))
        normalized = ((magnitudes - magnitudes.min()) / span
                      if span > 0 else np.zeros_like(magnitudes))
        histogram, _ = np.histogram(normalized, bins=32, range=(0, 1))
        probabilities = histogram[histogram > 0] / histogram.sum()
        entropy = float(-np.sum(probabilities * np.log2(probabilities)))
        directions = np.arctan2(gy, gx)[edge_pixels]
        if directions.size > 10:
            # Equation 26 uses ordinary angle variance; retain it only as a
            # legacy diagnostic. A local fixed seed makes results repeatable.
            indices = np.random.default_rng(0).choice(
                directions.size, min(100, directions.size), replace=False)
            consistency = float(1.0 / (np.var(directions[indices]) + 1e-6))
    return snr, solidity, smoothness, continuity, entropy, consistency


def summarize_metrics(rows):
    summaries = []
    for method in METHODS:
        group = [row for row in rows if row["Method"] == method]
        processed = [row for row in group if row["Status"] in ("ok", "empty")]
        summary = {"Method": method, "Image_Count": len(group),
                   "Processed": len(processed),
                   "Nonempty": sum(row["Status"] == "ok" for row in group)}
        # Empty predictions contribute zero rather than disappearing from the mean.
        for metric in METRICS:
            summary[metric] = float(np.mean([row[metric] for row in processed])) if processed else None
        summaries.append(summary)
    return summaries


def plot_simple_charts(summaries, output_dir):
    available = [row for row in summaries if row["Processed"] > 0]
    if not available:
        return
    for metric in METRICS:
        figure, axis = plt.subplots(figsize=(11, 6))
        values = [row[metric] for row in available]
        colors = ["#D62728" if row["Method"] == "Our_Method" else "#7F8C8D"
                  for row in available]
        bars = axis.bar([row["Method"] for row in available], values, color=colors)
        axis.tick_params(axis="x", labelrotation=40)
        axis.set_title(metric)
        axis.set_ylabel(metric)
        axis.grid(axis="y", linestyle="--", alpha=0.3)
        if metric == "Contour_SNR":
            axis.set_yscale("symlog", linthresh=1)
        for bar, value in zip(bars, values):
            axis.annotate(f"{value:.3g}", (bar.get_x() + bar.get_width() / 2, value),
                          xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9)
        figure.tight_layout()
        figure.savefig(Path(output_dir) / f"{metric}.png", dpi=FIGURE_DPI)
        plt.close(figure)


def generate_summary_report(summaries, output_dir):
    lines = ["Oracle-bone contour diagnostics", "",
             "These six diagnostics do not measure expert-reference accuracy.", ""]
    for row in summaries:
        lines.append(f"{row['Method']}: {row['Processed']}/{row['Image_Count']} processed, "
                     f"{row['Nonempty']} nonempty masks")
        for metric in METRICS:
            value = row[metric]
            lines.append(f"  {metric}: {value:.6g}" if value is not None else f"  {metric}: unavailable")
        lines.append("")
    (Path(output_dir) / "evaluation_summary.txt").write_text("\n".join(lines), encoding="utf-8")

def main(input_dir=INPUT_DIR, output_dir=OUTPUT_DIR):
    cv2.setNumThreads(1)
    paths = get_image_paths(input_dir, output_dir)
    output_dir = Path(output_dir)
    rows = []
    for index, path in enumerate(paths):
        try:
            image = read_gray_image(path)
        except (OSError, ValueError, cv2.error) as error:
            for method in METHODS:
                rows.append({"Image_Name": path.name, "Method": method,
                             "Status": "error", "Error": str(error)})
            print(f"Failed: {path.name}: {error}")
            continue
        for method in METHODS:
            try:
                if method == "Our_Method":
                    mask, edge, fragments = simple_contour_generation(image)
                else:
                    mask, edge, fragments = get_traditional_result(image, method)
                values = calculate_final_metrics(mask, edge, fragments, image)
                row = {"Image_Name": path.name, "Method": method,
                       "Status": "ok" if np.any(mask) else "empty"}
                row.update(zip(METRICS, values))
                rows.append(row)
            except (OSError, ValueError, cv2.error) as error:
                rows.append({"Image_Name": path.name, "Method": method,
                             "Status": "error", "Error": str(error)})
                print(f"Failed: {path.name}, {method}: {error}")

        if (index + 1) % 50 == 0 or index + 1 == len(paths):
            print(f"Processed {index + 1}/{len(paths)} images")
    summaries = summarize_metrics(rows)
    save_csv(output_dir / "detailed_metrics.csv", rows)
    save_csv(output_dir / "final_metrics.csv", summaries)
    plot_simple_charts(summaries, output_dir)
    generate_summary_report(summaries, output_dir)
    errors = sum(row["Status"] == "error" for row in rows)
    print(f"Completed {len(rows)} method-image runs; {errors} errors")
    print(f"Saved results to: {output_dir}")
    return rows


if __name__ == "__main__":
    main(INPUT_DIR, OUTPUT_DIR)
