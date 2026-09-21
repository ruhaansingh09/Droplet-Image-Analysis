import numpy as np
import cv2
import matplotlib.pyplot as plt
import alphashape

from pathlib import Path


IMAGE_DIR = Path(r"PATH_TO_YOUR_IMAGE_FOLDER")

image_files = sorted(
    list(IMAGE_DIR.glob("*.tif")) +
    list(IMAGE_DIR.glob("*.tiff"))
)

print(f"Found {len(image_files)} images.")

dia = []

for i, image_file in enumerate(image_files):

    img = cv2.imread(str(image_file))

    if img is None:
        print(f"Could not read: {image_file.name}")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    grad = cv2.magnitude(gx, gy)

    grad = cv2.normalize(
        grad, None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)

    _, grad_bin = cv2.threshold(
        grad,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5)
    )

    edges_closed = cv2.morphologyEx(
        grad_bin,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    edges_closed = cv2.dilate(
        edges_closed,
        kernel,
        iterations=1
    )

    ys, xs = np.where(edges_closed > 0)
    points = np.column_stack((xs, ys))

    alpha = 0.03

    hull = alphashape.alphashape(
        points,
        alpha
    )

    filled = np.zeros_like(
        edges_closed,
        dtype=np.uint8
    )

    if hull.geom_type == "Polygon":

        coords = np.array(
            hull.exterior.coords,
            dtype=np.int32
        )

        cv2.fillPoly(
            filled,
            [coords],
            255
        )

    else:

        for geom in hull.geoms:

            coords = np.array(
                geom.exterior.coords,
                dtype=np.int32
            )

            cv2.fillPoly(
                filled,
                [coords],
                255
            )

    ys, xs = np.where(filled == 255)

    h, w = filled.shape

    max_w = 0

    for y in range(h):

        row = filled[y, :]
        xs = np.where(row == 255)[0]

        if xs.size > 0:

            width = xs[-1] - xs[0]

            if width > max_w:
                max_w = width

    D_h_px = max_w

    max_h = 0

    for x in range(w):

        col = filled[:, x]
        ys = np.where(col == 255)[0]

        if ys.size > 0:

            height = ys[-1] - ys[0]

            if height > max_h:
                max_h = height

    D_v_px = max_h

    D_eff_px = (
        D_h_px**2 * D_v_px
    ) ** (1 / 3)

    dia.append(D_eff_px)

    print(
        f"{i + 1}/{len(image_files)}: "
        f"D_h = {D_h_px:.2f}, "
        f"D_v = {D_v_px:.2f}, "
        f"D_eff = {D_eff_px:.3f} px"
    )


x = np.arange(1, len(dia) + 1)

plt.figure(figsize=(6, 10))

plt.plot(
    x,
    dia,
    "-r"
)

plt.yticks(
    dia,
    [f"{d:.2f}" for d in dia]
)

plt.tick_params(
    axis="y",
    labelsize=8
)

plt.xlabel("Image number")
plt.ylabel("Effective diameter (pixels)")

plt.grid(
    True,
    linestyle="--",
    alpha=0.5
)

plt.tight_layout()
plt.show()