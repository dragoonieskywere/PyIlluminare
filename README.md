# 🌟 PyIlluminare

**PyIlluminare** is a professional, open-source RAW photo editor built entirely in Python. Leveraging the power of OpenCV and matrix processing, it offers a lightweight, fast, and accessible alternative for image editing, focusing on color accuracy, detail recovery, and creative local adjustments.

---

## 🚀 Key Features

* **Native RAW Processing:** Support for `.CR2`, `.NEF`, `.ARW`, `.DNG`, `.ORF`, and `.RAF` files via the `rawpy` library.
* **Basic Adjustments:** Precise control over Exposure, Contrast, Highlights, Shadows, Color Temperature, and Saturation.
* **Advanced Adjustments:** RAW and Color Noise Reduction, Sharpness, and Texture recovery.
* **Lens Correction:** Manual correction of lens distortion (Barrel and Pincushion).
* **Tone Curve:** Interactive parametric curve tool (requires `scipy`) for contrast and luma manipulation.
* **Local Editing (Brush):** Paint non-destructive masks on the image to apply Exposure, Contrast, and Saturation adjustments to specific areas.
* **EXIF Preservation:** The engine reads, displays, and rewrites EXIF metadata when exporting final images.
* **Professional Export:** Save your results as optimized `JPEG` or 16-bit `TIFF` for maximum data retention.
* **Multilingual Interface:** Available in English, Portuguese, and Spanish.

---

## 📦 How to Install and Run

The easiest way to use PyIlluminare is to download the pre-compiled installer from the **Releases** tab of this repository. No Python installation is required.

1. Download the `PyIlluminare.exe` file.
2. Run the installer and follow the on-screen steps.
3. Open PyIlluminare from your Start Menu or Desktop.
---

## 🛠️ Compiling Your Own Executable
You can compile your own version of PyIlluminare using PyInstaller. Run the following command in the project root:

`python -m PyInstaller --noconsole --onefile source.py`

*(The `.exe` file will be generated in the `dist` folder).*

---

## 📚 System Requirements (Python Dependencies)
For developers wishing to contribute or modify the code, the project relies on the following core libraries:

* `tkinter` (Graphical Interface)
* `opencv-python` (Image Processing Engine)
* `numpy` (Matrix Manipulation)
* `Pillow` (Interface Display and TIFF Management)
* `rawpy` (RAW Decoding)
* `scipy` (Tone Curve Interpolation)
* `exifread` & `piexif` (Metadata Reading and Writing)

---

## 👨‍💻 Author

Developed by **Gregório Severiano (Dragoonie)**. 

---

## 📄 License
This project is open-source (MIT License). Feel free to fork, modify, and improve it!
