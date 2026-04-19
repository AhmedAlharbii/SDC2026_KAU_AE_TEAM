# SDC2026 KAU AE Team

**Learning Conjunction Dynamics: A Self-Supervised Approach to Satellite Collision Risk Assessment**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.13](https://img.shields.io/badge/TensorFlow-2.13-orange.svg)](https://www.tensorflow.org/)

Official implementation of our research paper submitted to the **DebriSolver Space Data Challenge 2026**.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Results](#-results)
- [Repository Structure](#-repository-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Dataset](#-dataset)
- [Model Architecture](#-model-architecture)
- [Citation](#-citation)
- [Team](#-team)
- [License](#-license)
- [Contributing](#-contributing)
- [Contact](#-contact)
- [Acknowledgments](#-acknowledgments)
- [Related Work](#-related-work)
- [Future Work](#-future-work)

---

## 🌌 Overview

This repository contains the complete implementation of a **self-supervised deep learning system** for satellite conjunction risk assessment. Our approach addresses two critical challenges in space traffic management:

1. **Alert Overload**: Traditional threshold-based systems generate excessive false alarms
2. **No Collision Labels**: Actual collisions are too rare (~1/year globally) for supervised learning

### The Solution

We developed a **Bidirectional GRU with Monte Carlo Dropout** that:
- ✅ Learns from CDM trajectory evolution (self-supervised)
- ✅ Quantifies prediction uncertainty (epistemic confidence)
- ✅ Reduces urgent alerts by **96%** (from 2,003 to 81 events)
- ✅ Provides confidence-weighted risk classification
- ✅ Runs in real-time on CPU hardware

---

## 🎯 Key Features

### Self-Supervised Learning
- No collision labels required
- Learns from natural CDM sequence patterns
- Predicts next CDM state from historical trajectory
- Avoids labeling bias and threshold dependency

### Uncertainty Quantification
- Monte Carlo Dropout for epistemic uncertainty
- Confidence scores for each prediction
- Identifies when more observations are needed

### Operational Classification
Four-quadrant risk framework:
- **ACT NOW (4%)**: High threat + High confidence → Immediate action
- **WATCH CLOSELY (7.4%)**: High threat + Low confidence → More observations
- **SAFELY IGNORE (56.5%)**: Low threat + High confidence → Deprioritize
- **NOT PRIORITY (32%)**: Low threat + Low confidence → Routine monitoring

### Performance
- **Collision Probability MAE**: 0.403
- **log₁₀(Pc) R²**: 0.649 (explains 65% of variance)
- **Relative Speed R²**: 0.891
- **Training Time**: 24.8 minutes (CPU)
- **Inference**: Milliseconds per CDM

---

## 📊 Results

| Metric | Value |
|--------|-------|
| Test Pc MAE | 0.403 |
| Test Overall MAE | 0.464 |
| log₁₀(Pc) R² | 0.649 |
| Relative Speed R² | 0.891 |
| Alert Reduction | 96% |
| Training Time | 24.8 min (CPU) |
| Parameters | 244,171 |

### Model Performance
- Trained for 48 epochs with early stopping at epoch 28
- Smooth convergence with no overfitting
- Test performance aligns with validation (proper generalization)
- Handles 8 orders of magnitude in collision probability (10⁻¹⁰ to 10⁻²)

---

## 📁 Repository Structure
```
SDC2026_KAU_AE_TEAM/
│
├── Scripts/                          # Main implementation
│   ├── step1_parse_kvn.py            # Custom KVN parser (99.95% recovery)
│   ├── step2_prepare_sequences.py    # Self-supervised sequence creation
│   ├── step3_train_model.py          # BiGRU training with MC Dropout
│   ├── step4_inference_dashboard.py  # Production-safe inference dashboard
│   ├── step3b_evaluate_proxy_confidence.py  # Offline confidence-vs-truth diagnostics
|   ├── step5_visualize.py
│   ├── step5b_detailed_reports.py    # Event trajectory reports
|   ├── train_val_test_graph.py
|   ├── visualize_model_architecture.py
|   ├── calculate_R2.py
|   ├── requirements.txt
│   └── figures/                      # All paper figures
│        ├── Figure_1_Architecture.png
│        ├── Figure_2_Training_Loss.png
│        ├── Figure_3_MAE_Curves.png
│        ├── Figure_4_Pc_MAE_Curves.png
|        ├── Figure_5_Metrics_Comparison.png
│        ├── Figure_6_Threat_Correlation.png
│        ├── Figure_7_Quadrant_Dashboard.png
│        ├── Figure_8_Confidence_Calibration.png
│        ├── Model_Architecture_BiGRU.png
│        └── Table_1_High_Priority_Events.png
|
├── README.md                    # This file
└── LICENSE                      # MIT License
```

---

## 🔧 Installation

### Prerequisites
- Python 3.9 or higher
- TensorFlow 2.13+
- 8GB RAM minimum
- CPU sufficient (GPU optional)

### Setup
```bash
# Clone the repository
git clone https://github.com/AhmedAlharbii/SDC2026_KAU_AE_TEAM.git
cd SDC2026_KAU_AE_TEAM

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r Scripts/requirements.txt
```

### Requirements.txt Contents
```txt
# Core ML/Data Science
numpy==1.24.3
pandas==2.0.3
tensorflow==2.13.0
scikit-learn==1.3.0

# Visualization
matplotlib==3.7.2
seaborn==0.12.2

# Scientific Computing
scipy==1.11.1

# Utilities
tqdm==4.65.0
python-dateutil==2.8.2

# Additional dependencies for your scripts
openpyxl==3.1.2          # For Excel file handling (if you save .xlsx)
Pillow==10.0.0           # For image processing in visualizations
h5py==3.9.0              # For saving Keras models (.h5 format)

# Optional but recommended
jupyterlab==4.0.5        # For Jupyter notebooks (if you use them)
ipykernel==6.25.0        # Jupyter kernel
```

---

## 🚀 Usage

### Quick Start
```bash
# Step 1: Parse CDM files from KVN format
python Scripts/step1_parse_kvn.py

# Step 2: Create self-supervised sequences
python Scripts/step2_prepare_sequences.py

# Step 3: Train BiGRU model
python Scripts/step3_train_model.py

# Step 3B: Offline confidence gate (truth used only for validation)
python Scripts/step3b_evaluate_proxy_confidence.py

# Step 4: Production-safe inference (NO truth labels in scoring)
python Scripts/step4_inference_dashboard.py

# Step 5: Generate figures and reports
python Scripts/step5_visualize.py
python Scripts/step5b_detailed_reports.py
```

## 📦 Dataset

### DebriSolver Space Data Challenge Dataset

**Source**: DebriSolver Competition (January - June 2024)

**Statistics**:
- Total CDMs: 185,511
- Valid CDMs (after parsing): 185,415 (99.95% recovery)
- Unique Conjunction Events: 64,109
- Time Period: January 1 - June 30, 2024
- Coverage: Global conjunction screenings

**Features Extracted** (11 per CDM):
1. `COLLISION_PROBABILITY` (Pc)
2. `log10_pc` (log-scaled Pc)
3. `MISS_DISTANCE` (meters)
4. `time_to_tca_hours` (hours until TCA)
5. `RELATIVE_SPEED` (m/s)
6. `RELATIVE_POSITION_R` (RTN frame)
7. `RELATIVE_POSITION_T` (RTN frame)
8. `RELATIVE_POSITION_N` (RTN frame)
9. `combined_cr_r` (covariance RTN)
10. `combined_ct_t` (covariance RTN)
11. `combined_cn_n` (covariance RTN)

### Data Splits
- **Training**: 80% of events (51,287 events)
- **Validation**: 10% of events (6,411 events)
- **Test**: 10% of events (6,411 events)

**Important**: Events split by unique NORAD ID pairs (no event appears in multiple splits)

### Obtaining the Dataset

Due to licensing restrictions, the raw CDM dataset is **not included** in this repository.

**To obtain the data**:
1. Test on (https://www.space-track.org/)
2. Download the CDM dataset (KVN format)
3. Place files in `data/raw/cdms/`
4. Run our custom parser: `python Scripts/step1_parse_kvn.py`

---

## 🏗️ Model Architecture

### Bidirectional GRU with Monte Carlo Dropout
```
Input: Variable-length CDM sequences (max 20 timesteps × 11 features)
    ↓
Masking Layer (handles variable lengths)
    ↓
Bidirectional GRU Layer 1 (128 units, dropout=0.3)
    ↓
Bidirectional GRU Layer 2 (64 units, dropout=0.3)
    ↓
Dense Layer 1 (64 units, ReLU, dropout=0.3)
    ↓
Dense Layer 2 (32 units, ReLU, dropout=0.3)
    ↓
Output Layer (11 units, linear activation)
    ↓
Output: Predicted next CDM (11 features)
```

### Key Design Choices

**Bidirectional GRU**:
- Captures both forward and backward temporal context
- Simpler than LSTM (fewer parameters, faster training)
- Effective for conjunction trajectory patterns

**Monte Carlo Dropout**:
- Dropout active during inference (`training=True`)
- Multiple forward passes (50 samples) for uncertainty
- Provides epistemic confidence estimates

**Self-Supervised Task**:
- Input: CDM sequence [CDM₁, CDM₂, ..., CDMₙ₋₁]
- Target: CDMₙ
- No collision labels needed!

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| GRU Units (Layer 1) | 128 |
| GRU Units (Layer 2) | 64 |
| Dropout Rate | 0.3 |
| Dense Units | 64, 32 |
| Learning Rate | 0.001 → 0.00025 (decay) |
| Batch Size | 64 |
| Max Sequence Length | 20 |
| MC Dropout Samples | 50 |
| Total Parameters | 244,171 |

---



## 📝 Citation

If you use this code or model in your research, please cite our paper:
```bibtex
@inproceedings{alharbi2026selfsupervised,
  title={Learning Conjunction Dynamics: A Self-Supervised Approach to Satellite Collision Risk Assessment},
  author={Alharbi, Ahmad and Mojelad, Abdulelah and Alharbi, Hamzah and Alsadoon, Khalid and Hassan, Mohamedhakim},
  booktitle={DebriSolver Space Data Challenge},
  year={2025},
  organization={Saudi Space Agency}
}
```

**Paper**: [Link will be added after publication]

---

## 👥 Team

**SDC2026 KAU AE Team**  
King Abdulaziz University, Jeddah, Saudi Arabia

- **Ahmad Alharbi** - Lead Developer & Researcher   | Linkedin: https://www.linkedin.com/in/ahmed-alharbi-973b63246/
- **Abdulelah Mojelad**   - Team Member             | Linkedin: https://www.linkedin.com/in/abdulellah-mojalled/
- **Hamzah Alharbi**      - Team Member             | Linkedin: https://www.linkedin.com/in/hamzah-alharbi-00b18133a/
- **Khalid Alsadoon**     - Team Member             | Linkedin: https://www.linkedin.com/in/khalid-alsadoon-a95802242/
- **Mohamedhakim Hassan** - Team Member             | Linkedin: https://www.linkedin.com/in/mohamed-hassan-aero/

**Competition**: DebriSolver Space Data Challenge 2026  
**Organizer**: Saudi Space Agency

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

**Summary**: You are free to use, modify, and distribute this code for any purpose, including commercial use, as long as you include the original copyright notice.

---

## 🤝 Contributing

We welcome contributions! If you'd like to improve this work:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Open a Pull Request

### Areas for Contribution
- Incorporate object metadata (maneuverability, operational status)
- Integrate physical environment models (atmospheric density, space weather)
- Ensemble methods (BiLSTM, Transformers)
- Active learning frameworks
- Extended evaluation on additional datasets

---

## 📧 Contact

For questions, feedback, or collaboration opportunities:

- **Email**: ahmadharbi157@hotmail.com
- **GitHub Issues**: [Create an issue](https://github.com/AhmedAlharbii/SDC2026_KAU_AE_TEAM/issues)
- **Institution**: King Abdulaziz University

---

## 🙏 Acknowledgments

- **Saudi Space Agency** for organizing the DebriSolver Competition
- **King Abdulaziz University** for institutional support
- **DebriSolver Team** for providing the CDM dataset
- **Open-source community** for TensorFlow, scikit-learn, and other tools

---

## 📚 Related Work

### Key References

1. **Self-Supervised Learning in Space Domain**:
   - Sanchez et al. (2020) - Machine Learning for Collision Risk Management
   
2. **Uncertainty Quantification**:
   - Gal & Ghahramani (2016) - Dropout as Bayesian Approximation
   
3. **GRU Architecture**:
   - Cho et al. (2014) - Learning Phrase Representations using RNN Encoder-Decoder

4. **Space Traffic Management**:
   - ESA Space Debris Office Reports
   - NASA ODPO Conjunction Assessment Guidelines

---

## 🔮 Future Work

Planned enhancements:
- [ ] Multi-modal ensemble (BiGRU + BiLSTM + Transformer)
- [ ] Active learning for low-confidence events
- [ ] Real-time deployment API
- [ ] Integration with operational SSA systems
- [ ] Extended evaluation on 2024 H2 data
- [ ] Comparison with commercial CDM screening tools

---

## ⭐ Star History

If you find this work useful, please consider giving it a star! ⭐

It helps others discover this research and supports our work.

---

**Last Updated**: December 2025  
**Status**: Competition Submission Complete ✅  
**Code Status**: Fully Reproducible ✅

---

Made with ❤️ by the KAU AE Team for safer space operations 🛰️
