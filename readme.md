# MontserratSeismicDataClassification

## Background

Seismographs record ground vibrations caused by various geological events. One key goal in seismology is to **automatically determine the cause of these tremors**, enabling distinction between different types of earthquakes and other ground motion sources.\
**Rockfalls**, for example, are a frequent cause of seismograph deflections, especially in **mountainous and volcanic regions**.



A typical case is the island of **Montserrat in the Caribbean**, where three seismic monitoring stations continuously record ground motion. The island is home to the **Soufrière Hills volcano**, whose eruptions in the 1990s **forced the relocation of much of the island’s population** due to volcanic hazards.

---

## Dataset Overview

A **pre-classified seismic dataset** from Montserrat is available, containing several hundred labeled events from a short, discrete observation period.

### Data Structure:

```
data/
├── hy_jan/
├── hy_may/
├── rf_jan/
└── rf_may/
```

Where:

- `hy_*` = Hybrid earthquakes
- `rf_*` = Rockfalls

Each subfolder contains **hundreds of events**, with each event stored as **three **``** files** (one for each station component: E, N, Z).

In geoscience, **data availability is often limited**, posing challenges for model building and validation. This dataset offers a valuable opportunity to test **machine learning models for seismic event classification**.

---

## Feature Extraction

Using `generate_dataset.py`, various **statistical features** (e.g., **kurtosis**, **skewness**, **variance**, **mean**, **range**) are extracted from the waveform amplitudes of each station component.

You can choose between:

- **Full feature set (33 features)**: Using all three components (E, N, Z)
- **Z-component only (11 features)**: Focused on the vertical component, shown to be most discriminative in this dataset



---

## Installation

Requirements:

- **Python 3.9**
- Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Workflow

### 1. Data Preparation

Place your raw Montserrat data folders inside the provided `/data` directory.

### 2. Generate Dataset

Run:

```bash
python generate_dataset.py
```

In `generate_dataset.py`, set:

```python
use_z_only = True  # or False for full feature set
```

This will generate a large `.json` dataset for training.

---

### 3. Data Loading and Capping

In `load_data.py`, control how many events to load for training (RAM management):

```python
maxcap = 50000  # Example cap for training
```

Even when using Z-components only, the dataset may exceed 3GB.

---

### 4. Model Training

You can train models using:

- **Support Vector Machine (SVM)**:

```bash
python train_svm_model.py
```

- **Gradient Descent Model (GDM)**:

```bash
python train_gdm_model.py
```

**Note:**\
When working with the Z-only dataset, remember that **feature dimensions change (11 vs. 33 features)**—this must be adjusted in your training scripts.

---

### 5. Results and Evaluation

Here’s an example confusion matrix from an SVM trained on the reduced Z-only dataset:

```
Confusion Matrix:
[[3208  1193]
 [  384 5215]]
Accuracy: 0.8423
```

And here’s a **confusion matrix after scaling up to 50,000 files with Gradient Descent** (performance improves significantly):



---

## Research Questions and Outlook

Some open questions for future research:

- **Are rockfalls and hybrid events inherently this separable?**
- Could these high accuracies reflect **overfitting or sampling bias**?
- **How transferable are models trained on volcanic regions (e.g., Montserrat) to non-volcanic regions (e.g., California)?**

A promising line of research is to **train models on volcanic datasets and test them on tectonic earthquake datasets (and vice versa)** to assess generalization across different geophysical settings.

I see this project as a **starting point for machine learning in geoscience**, and hope to contribute to further development and collaboration in this field.

---

## Project Structure

| File/Folder           | Purpose                                    |
| --------------------- | ------------------------------------------ |
| `data/`               | Raw seismic data folders                   |
| `generate_dataset.py` | Feature extraction script                  |
| `load_data.py`        | Dataset loading and sample size control    |
| `train_svm_model.py`  | SVM training                               |
| `train_gdm_model.py`  | Gradient descent model training            |
| `utils.py`            | Utilities (e.g., reading `.ASC` files)     |
| `requirements.txt`    | Python dependencies                        |
| `images/`             | Contains visuals for README and evaluation |

---

## Contact

For questions, collaborations, or data requests:

📧 [**oliverehlbeck23@gmail.com**](mailto\:oliverehlbeck23@gmail.com)

---

## License

*(You can specify a license here, e.g., MIT, GPL, or Creative Commons. Let me know if you'd like to add one.)*

