
# Endometriosis Federated Learning Framework

A privacy-preserving federated learning framework for endometriosis diagnosis using medical imaging, with differential privacy, secure aggregation, and Grad-CAM explainability.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Federated Learning Server                      │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────────────┐  │
│  │ Aggregation │  │  Early Stop  │  │  Global Model (TF)    │  │
│  │ FedAvg/Prox │  │  Convergence │  │  ResNet/EfficientNet  │  │
│  └──────┬──────┘  └──────────────┘  └───────────────────────┘  │
│         │                                                        │
│  ┌──────┴──────────────────────────────────────────────────┐    │
│  │              Privacy Layer                                │    │
│  │  ┌─────────────────────┐  ┌────────────────────────┐    │    │
│  │  │ Differential Privacy│  │ Secure Aggregation     │    │    │
│  │  │ Gaussian Mechanism  │  │ Additive Masking       │    │    │
│  │  │ L2 Norm Clipping    │  │ Zero-Sum Masks         │    │    │
│  │  └─────────────────────┘  └────────────────────────┘    │    │
│  └──────────────────────────────────────────────────────────┘    │
└──────────┬──────────┬──────────┬──────────┬──────────┬───────────┘
           │          │          │          │          │
    ┌──────┴──┐┌─────┴───┐┌────┴────┐┌────┴────┐┌───┴─────┐
    │Client 1 ││Client 2 ││Client 3 ││Client 4 ││Client 5 │
    │  30%    ││  25%    ││  20%    ││  15%    ││  10%    │
    │Hospital ││Hospital ││Hospital ││Hospital ││Hospital │
    └─────────┘└─────────┘└─────────┘└─────────┘└─────────┘
```

## Project Structure

```
endometriosis-fl-framework/
├── config/
│   └── config.yaml              # Complete hyperparameter configuration
├── data/
│   └── raw/                     # Dataset storage (gitignored)
├── experiments/
│   ├── __init__.py
│   ├── run_federated.py         # Federated learning experiments
│   ├── run_centralised.py       # Centralised baseline
│   ├── run_comparison.py        # Comparative analysis
│   ├── run_explainability.py    # Grad-CAM analysis
│   └── run_quick_test.py        # Integration tests
├── notebooks/
│   └── 01_sagemaker_experiment.ipynb
├── results/
│   ├── models/                  # Saved model weights
│   ├── figures/                 # Generated plots
│   └── logs/                    # Training logs
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── preprocessing.py     # Image preprocessing pipeline
│   │   ├── dataset_loader.py    # GLENDA + UT Endo MRI loaders
│   │   └── partition.py         # IID/non-IID partitioning
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base_model.py        # Classification head, compilation
│   │   ├── resnet50v2.py        # ResNet50V2 architecture
│   │   ├── efficientnet_b0.py   # EfficientNetB0 architecture
│   │   ├── mobilenetv2.py       # MobileNetV2 architecture
│   │   └── model_factory.py     # Model registry and factory
│   ├── federated/
│   │   ├── __init__.py
│   │   ├── client.py            # Federated client
│   │   ├── server.py            # Federated server + orchestration
│   │   └── aggregation.py       # FedAvg, FedProx aggregation
│   ├── privacy/
│   │   ├── __init__.py
│   │   ├── differential_privacy.py  # DP with Gaussian mechanism
│   │   └── secure_aggregation.py    # Additive masking protocol
│   ├── explainability/
│   │   ├── __init__.py
│   │   └── gradcam.py           # Grad-CAM implementation
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py           # Classification metrics
│   │   ├── comparison.py        # Statistical tests (paired t-test)
│   │   └── visualisation.py     # Publication-quality plots
│   └── utils/
│       ├── __init__.py
│       ├── seed.py              # Reproducibility
│       ├── config.py            # Configuration loading
│       └── logger.py            # Logging setup
├── main.py                      # CLI entry point
├── sagemaker_setup.py           # SageMaker environment setup
├── requirements.txt             # Python dependencies
├── .gitignore
└── README.md
```

## Quick Start (Amazon SageMaker)

### 1. Create SageMaker Instance

- Instance type: `ml.g4dn.xlarge` (NVIDIA T4, 16GB GPU, 4 vCPUs, 16GB RAM)
- Storage: 50 GB EBS
- Platform: TensorFlow 2.13 (Python 3.10)

### 2. Setup Environment

```bash
# Clone or upload the framework
cd endometriosis-fl-framework

# Run automated setup
python sagemaker_setup.py
```

### 3. Verify Installation

```bash
python main.py test
```

### 4. Run Experiments

```bash
# Quick test with synthetic data
python main.py federated --synthetic --architecture ResNet50V2

# Full federated experiment
python main.py federated --architecture ResNet50V2 --epsilon 1.0 --non_iid

# Centralised baseline
python main.py centralised --architecture ResNet50V2

# Complete comparative analysis
python main.py compare --architecture ResNet50V2

# Grad-CAM explainability
python main.py explain --architecture ResNet50V2 --num_samples 50
```

## Experimental Conditions

| Condition | Data Distribution | Privacy | Epsilon | Architecture |
|-----------|------------------|---------|---------|--------------|
| Centralised Baseline | All data | None | - | ResNet50V2 |
| FL-IID (no DP) | IID | None | - | ResNet50V2 |
| FL-NonIID (no DP) | Dirichlet(0.5) | None | - | ResNet50V2 |
| FL-NonIID-DP (strong) | Dirichlet(0.5) | DP + SA | 0.5 | ResNet50V2 |
| FL-NonIID-DP (moderate) | Dirichlet(0.5) | DP + SA | 1.0 | ResNet50V2 |
| FL-NonIID-DP (relaxed) | Dirichlet(0.5) | DP + SA | 5.0 | ResNet50V2 |

## Configuration Parameters

All hyperparameters are centralised in `config/config.yaml`:

| Category | Parameter | Value |
|----------|-----------|-------|
| Data | Image size | 224x224x3 |
| Data | Train/Val/Test split | 70/15/15 |
| Data | Augmentation | Rotation 30, Zoom 0.2, Brightness 0.2 |
| Model | Architectures | ResNet50V2, EfficientNetB0, MobileNetV2 |
| Model | Pretrained | ImageNet |
| Model | Dense units | 256, Dropout 0.3 |
| Federated | Clients | 5 |
| Federated | Rounds | 20 |
| Federated | Local epochs | 10 |
| Federated | Aggregation | FedProx (mu=0.01) |
| Federated | Non-IID | Dirichlet alpha=0.5 |
| Privacy | DP mechanism | Gaussian |
| Privacy | L2 norm clip | 1.0 |
| Privacy | Epsilon values | [0.5, 1.0, 5.0] |
| Privacy | Delta | 1e-5 |
| Privacy | SA protocol | Additive masking |
| Training | Optimiser | Adam (lr=0.001) |
| Training | LR schedule | Cosine annealing |
| Evaluation | K-folds | 5 |
| Evaluation | Statistical test | Paired t-test + Cohen's d |

## Evaluation Metrics

- **Classification**: Accuracy, Precision, Recall, F1-score, AUC-ROC
- **Explainability**: IoU, Dice coefficient (Grad-CAM vs ground truth masks)
- **Statistical**: Paired t-test (alpha=0.05), Cohen's d effect size
- **Convergence**: Rounds to target performance, improvement rate
- **Privacy**: Epsilon spent, noise magnitude, utility loss

## Technologies

| Component | Technology | Version |
|-----------|------------|---------|
| Deep Learning | TensorFlow | 2.13.0 |
| Privacy | TensorFlow Privacy | 0.9.0 |
| Federated Learning | Flower (FL simulation) | 1.5.0 |
| Medical Imaging | pydicom, nibabel, SimpleITK | - |
| Computer Vision | OpenCV | 4.8.1 |
| Statistical Analysis | SciPy, statsmodels | - |
| Visualisation | Matplotlib, Seaborn | - |
| Cloud Platform | Amazon SageMaker | - |

## SageMaker Instance Recommendations

| Workload | Instance | GPU | Cost Tier |
|----------|----------|-----|-----------|
| Development/Testing | ml.g4dn.xlarge | 1x T4 (16GB) | $ |
| Standard Training | ml.g4dn.2xlarge | 1x T4 (16GB) | $$ |
| Large-scale | ml.g4dn.12xlarge | 4x T4 (64GB) | $$$ |
| Maximum Performance | ml.p3.2xlarge | 1x V100 (16GB) | $$$$ |

## Datasets

- **GLENDA v1.5**: Gynaecological Laparoscopy ENdometriosis DAtaset - laparoscopic images
- **UT Endo MRI**: University pelvic MRI dataset - volumetric MRI scans

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{endometriosis_fl_framework,
  title={Endometriosis Federated Learning Framework},
  year={2024},
  description={Privacy-preserving federated learning for endometriosis diagnosis}
}
```

## License

This project is for research purposes. Please ensure compliance with medical data regulations (HIPAA, GDPR) when using real patient data.

# Endo-Fed-Framework

