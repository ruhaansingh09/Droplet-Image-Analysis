# Droplet-Image-Analysis

Python-based image processing pipeline developed for extracting
geometric properties of a burning droplet 
from time-resolved experimental images.

## Overview

The pipeline processes individual droplet images and extracts
geometric descriptors of the droplet, including:

- Droplet area
- Horizontal diameter
- Vertical diameter
- Effective diameter
- Droplet boundary

## Image Processing

The image-processing pipeline consists of:

1. Grayscale conversion
2. Gaussian filtering
3. Sobel edge detection
4. Otsu thresholding
5. Morphological processing
6. Connected-component analysis
7. Contour extraction
8. Droplet geometric measurement

The effective diameter is calculated from the horizontal and
vertical droplet dimensions.

## Repository Structure

```text
ME209-Droplet-Image-Analysis/
│
├── droplet_analysis.py
├── requirements.txt
├── README.md
└── .gitignore
