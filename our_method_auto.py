"""Batch outer-contour extraction using the manuscript method."""

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
OUTPUT_DIR = BASE_DIR / "results" / "our_method"
MAX_IMAGES = None  # Set an integer for a quick run, or None for all images.
FIGURE_DPI = 300
FLOW_IMAGE_LIMIT = 20  # None saves a process-flow figure for every image.


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

def load_and_preprocess(image_path):
    return extract_contours(read_gray_image(image_path))


def visualize_process_flow_combined(result, save_dir, image_name):
    """Show actual processing stages instead of unused enhancement steps."""
    original = result["original"]
    raw_edges = np.zeros_like(original)
    if result["raw_contours"]:
        cv2.drawContours(raw_edges, result["raw_contours"], -1, 255, 1)
    # Thicken preview lines only; saved boundary data remains one pixel wide.
    preview_kernel = np.ones((3, 3), np.uint8)
    raw_preview = cv2.dilate(raw_edges, preview_kernel)
    edge_preview = cv2.dilate(result["edge"], preview_kernel)
    overlay = cv2.cvtColor(original, cv2.COLOR_GRAY2RGB)
    cv2.drawContours(overlay, result["contours"], -1, (0, 255, 0), 2)
    panels = [
        (original, "1. Original"),
        (result["denoised"], "2. Gaussian filtering"),
        (result["eroded"], "3. Strong erosion"),
        (result["enhanced"], "4. Mild dilation"),
        (result["inverted"], "5. Inversion"),
        (result["binary"], f"6. SWBT binary (T={result['threshold']})"),
        (raw_preview, "7. External contours"),
        (result["filtered_mask"], "8. Contour filtering"),
        (result["first_closed"], "9. First closing"),
        (result["mask"], "10. Second closing"),
        (edge_preview, "11. Final boundary"),
        (overlay, "12. Boundary overlay"),
    ]
    figure, axes = plt.subplots(3, 4, figsize=(16, 12))
    for axis, (image, title) in zip(axes.flat, panels):
        axis.imshow(image, cmap="gray" if image.ndim == 2 else None, vmin=0, vmax=255)
        axis.set_title(title, fontsize=11)
        axis.axis("off")
    figure.tight_layout()
    directory = Path(save_dir) / "process_flow"
    directory.mkdir(parents=True, exist_ok=True)
    figure.savefig(directory / f"{image_name}_flow.png", dpi=FIGURE_DPI)
    plt.close(figure)


def batch_process_images(input_dir, output_dir):
    paths = get_image_paths(input_dir, output_dir)
    output_dir = Path(output_dir)
    rows = []
    for index, path in enumerate(paths):
        try:
            result = load_and_preprocess(path)
            write_image(output_dir / "masks" / f"{path.stem}.png", result["mask"])
            write_image(output_dir / "boundaries" / f"{path.stem}.png", result["edge"])
            overlay = cv2.cvtColor(result["original"], cv2.COLOR_GRAY2BGR)
            cv2.drawContours(overlay, result["contours"], -1, (0, 255, 0), 2)
            write_image(output_dir / "overlays" / f"{path.stem}.png", overlay)
            if FLOW_IMAGE_LIMIT is None or index < FLOW_IMAGE_LIMIT:
                visualize_process_flow_combined(result, output_dir, path.stem)
            rows.append({"Image_Name": path.name,
                         "Status": "ok" if np.any(result["mask"]) else "empty",
                         "Threshold": result["threshold"],
                         "Raw_Fragments": result["raw_fragments"],
                         "Final_Contours": len(result["contours"])})
        except (OSError, ValueError, cv2.error) as error:
            rows.append({"Image_Name": path.name, "Status": "error", "Error": str(error)})
            print(f"Failed: {path.name}: {error}")
        if (index + 1) % 50 == 0 or index + 1 == len(paths):
            print(f"Processed {index + 1}/{len(paths)} images")
    save_csv(output_dir / "processing_log.csv", rows)
    print(f"Saved results to: {output_dir}")
    return rows


def main():
    cv2.setNumThreads(1)
    batch_process_images(INPUT_DIR, OUTPUT_DIR)


if __name__ == "__main__":
    main()
