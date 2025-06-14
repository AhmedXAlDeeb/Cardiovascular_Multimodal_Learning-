# Multimodal Analysis of ECG and PCG Signals for Activity and Heart Function Classification

**Authors:**
- Ahmed Mohamed (9220084)
- Mostafa Ali (9220842)
- Shahd Mahmoud (92200396)
- Nada Khaled (9220905)

**Affiliation:** Systems and Biomedical Engineering Department, Cairo University
**Supervisors:** Dr. Inas Ahmed, Eng. Samar Alaa

---

## 📖 Table of Contents
- [Project Overview](#-project-overview)
- [Problem Definition](#-problem-definition)
- [Dataset](#-dataset)
- [Methodology](#-methodology)
  - [Data Cleaning and Preparation](#data-cleaning-and-preparation)
  - [PCG Signal Processing](#pcg-signal-processing)
  - [ECG Signal Processing](#ecg-signal-processing)
  - [Feature Selection](#feature-selection)
- [Model Training and Evaluation](#-model-training-and-evaluation)
- [Results](#-results)
  - [Unimodal Performance: PCG vs. ECG](#unimodal-performance-pcg-vs-ecg)
  - [Multimodal Performance](#multimodal-performance)
  - [Key Findings](#key-findings)
- [Discussion](#-discussion)
- [How to Use](#-how-to-use)
- [References](#-references)

---

## 🚀 Project Overview

This project explores the use of machine learning for cardiac stress classification using electrocardiogram (ECG) and phonocardiogram (PCG) signals. Cardiovascular stress is a critical biomarker for cardiac health, and its early detection can significantly improve patient outcomes. While ECG is the standard, PCG offers a non-invasive and cost-effective alternative.

The primary goal is to compare the predictive power of models trained on:
1.  **ECG signals only**
2.  **PCG signals only**
3.  **A multimodal approach combining both ECG and PCG**

We employ a range of feature extraction techniques (MFCC, STFT, LPC, etc.) and evaluate seven traditional machine learning classifiers: Logistic Regression, Decision Tree, Random Forest, Naive Bayes, AdaBoost, XGBoost, and Support Vector Machine (SVM).

---

## 🎯 Problem Definition

Athletes and individuals undergoing physical exertion often struggle to identify when they've reached a level of fatigue that could be harmful. This project addresses this by classifying a subject's activity and stress level using synchronized ECG and PCG data.

The core challenges are:
-   **Feature Evaluation:** Identifying the most informative features from each signal type that correlate with stress.
-   **Modality Comparison:** Assessing the standalone predictive power of ECG versus PCG for stress classification.
-   **Multimodal Fusion:** Determining if combining both signals enhances classification accuracy.

---

## 📊 Dataset

We utilized the **EPHNOGRAM dataset** [1], which contains synchronous ECG and PCG recordings from 24 male subjects (ages 23–29) performing various physical tasks.

-   **Signals:** 69 ECG and PCG recordings.
-   **Recording Device:** 3-lead ECG at 8 kHz sampling frequency.
-   **Activity Levels (Recategorized into Stress Levels):**
    | Recording Scenario | Stress Level |
    | :--- | :--- |
    | Laying on Bed | Rest |
    | Sitting on Armchair | Rest |
    | Slow Walk | Medium |
    | Walking at Constant Speed | Medium |
    -   | Sit Down and Stand Up | Medium |
    | Pedaling a Stationary Bike | Medium |
    | Bruce Protocol | High |
    | Bicycle Stress Test | High |

#### Sample Signals
*30-Minute ECG and PCG Record*
![30 Minute ECG and PCG Record](https://i.imgur.com/your_image_link_here.png) *First 5 Seconds of a Record*
![First 5 Seconds of a Record](https://i.imgur.com/your_image_link_here.png) ---

## 🛠️ Methodology

### Data Cleaning and Preparation
1.  **Record Information Extraction:** A helper function was used to parse metadata for each recording.
2.  **Label Processing:** The labels CSV was cleaned, retaining only relevant columns.
3.  **Duration Standardization:** Long recordings were segmented, and the middle 4 minutes of each segment were extracted to ensure uniform duration.
4.  **Scenario Mapping:** The detailed recording scenarios were mapped to three stress levels: **Rest**, **Medium**, and **High**.

### PCG Signal Processing
1.  **Preprocessing (Denoising):**
    -   **Wavelet Denoising (DWT)** was chosen over digital filters due to its superior performance in preserving signal details while removing noise (SNR: 48.67 dB vs. 3.56 dB).
2.  **Feature Extraction:**
    -   **Time-Domain:** Descriptive stats, energy, power, amplitude envelope, RMS energy, and zero-crossing rate.
    -   **Frequency-Domain:** Peak frequency, band energy ratio (BER), spectral centroid, and spectral bandwidth.
    -   **Time-Frequency:** Mel-Frequency Cepstral Coefficients (MFCCs) and Discrete Wavelet Transform (DWT) coefficients.
3.  **Feature Preparation:**
    -   Highly skewed features were transformed using logarithmic, square root, or cube root functions.
    -   Features were normalized using `MinMaxScaler`.
    -   The training set was balanced using **SMOTE**.

### ECG Signal Processing
1.  **Preprocessing:**
    -   **Baseline Wander Removal:** A high-pass Butterworth filter (0.5 Hz) was applied.
    -   **Powerline Interference Removal:** A 50 Hz notch filter was used.
    -   **Normalization:** Signals were standardized to have zero mean and unit variance.
2.  **Feature Extraction:**
    -   *(Details on the specific ECG feature extraction methods are in the full paper.)*
3.  **Feature Preparation:**
    -   Similar to PCG, features were checked for skewness, transformed, normalized, and the training data was balanced using SMOTE.

### Feature Selection
A multi-step feature selection process was used to identify the most informative features and reduce redundancy.
1.  **Statistical Tests:** Hypothesis tests (e.g., t-test, p-value < 0.05) and Mutual Information were used to assess feature relevance.
2.  **Correlation Analysis:** Features with a Pearson correlation > 0.9 were reviewed, and the more skewed feature was dropped.
3.  **Wrapper-Based Selection:** A `GradientBoostingClassifier` was used with `SelectFromModel` to select features with importance above the median.

#### Final Selected Features
-   **ECG:** `{mean_rr, median_rr, min_rr, mean_hr, BER_MEAN, SC_MEAN, SB_MIN, MFCC_MEAN}`
-   **PCG:** `{MEAN, RM_MIN, ZCR_GLOBAL, PEAK_AMP, SB_MEAN, CA_MEAN, CD_MEDIAN}`

---

## 📈 Model Training and Evaluation

Models were trained and evaluated on three distinct feature sets: **PCG-only**, **ECG-only**, and **Multimodal (PCG + ECG)**. The dataset was split into training and testing sets, and hyperparameters for key models were tuned using grid search.

---

## 📊 Results

### Unimodal Performance: PCG vs. ECG

| Modality | Best Model | Accuracy |
| :--- | :--- | :--- |
| **PCG-Only** | **Random Forest** | **80%** |
| **ECG-Only** | **Gradient Boosting** | **77%** |

-   **PCG Analysis:** The Random Forest classifier performed exceptionally well, demonstrating that PCG features, especially non-linear ones, are highly effective for stress classification. The SVM model performed poorly (50%), suggesting the features are not linearly separable.
-   **ECG Analysis:** The Gradient Boosting classifier achieved the highest accuracy for ECG. The strong performance of Logistic Regression (73%) indicates that some ECG features have near-linear discriminative power.

### Multimodal Performance

When the selected features from both PCG and ECG were combined, the models' performance was unexpectedly lower.

| Modality | Model | Accuracy |
| :--- | :--- | :--- |
| **Multimodal** | Gradient Boosting | **64%** |
| **Multimodal** | Random Forest | **64%** |
| **Multimodal** | XGBoost | **64%** |

### Key Findings
1.  **PCG is a strong contender:** PCG-based models, particularly Random Forest, achieved an accuracy of **80%**, rivaling the best ECG-based model (77%). This highlights the underutilized potential of PCG signals for stress detection.
2.  **Simple fusion is not enough:** The simple concatenation of features for the multimodal model resulted in a performance drop. This suggests issues with feature redundancy or the need for more advanced fusion techniques (e.g., attention mechanisms, hierarchical models).
3.  **Ensemble methods excel:** Tree-based ensemble models (Random Forest, Gradient Boosting) consistently outperformed other models like SVM, indicating they are better suited for capturing the complex, non-linear relationships in both ECG and PCG data.

---

## 💬 Discussion

Our results challenge the conventional dominance of ECG in cardiac monitoring. The high accuracy of the PCG-only model suggests it is a viable and powerful tool for stress detection, especially in low-resource settings where ECG equipment may be unavailable.

The "multimodal paradox" we observed—where combining modalities decreased performance—is a critical finding. It suggests that a naive feature fusion approach can introduce noise and redundancy. Future work should focus on advanced fusion architectures that can intelligently integrate information from both signals, potentially leveraging cross-modal attention or deep learning-based autoencoders.

**Clinical Implications:**
-   **Accessibility:** PCG provides a cost-effective and accessible method for cardiac stress monitoring.
-   **Real-Time Systems:** Lightweight models like KNN showed promise for deployment on edge devices, though further optimization is needed.

---

## 🚀 How to Use

To replicate this project, you will need a Python environment with standard data science libraries.

**1. Clone the repository:**
```bash
git clone [https://github.com/your-username/multimodal-cardiac-analysis.git](https://github.com/your-username/multimodal-cardiac-analysis.git)
cd multimodal-cardiac-analysis
