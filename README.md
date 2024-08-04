# Dilated Convolution with Learnable Spacing (DCLS) - Enhanced Interpretability Evaluation with GradCAM

## Overview
Dilated Convolution with Learnable Spacing (DCLS) is a cutting-edge convolution method that expands the receptive fields (RF) without increasing the number of parameters, similar to dilated convolution but without enforcing a regular grid. DCLS has demonstrated superior performance compared to standard and dilated convolutions across several computer vision benchmarks. This project aims to showcase that DCLS also enhances model interpretability, measured by the alignment with human visual strategies.

## Key Features
- **Enlarged Receptive Fields**: DCLS increases RF without additional parameters.
- **Enhanced Interpretability**: Improves alignment with human visual attention.
- **Spearman Correlation Metric**: Uses Spearman correlation between models’ Grad-CAM heatmaps and ClickMe dataset heatmaps to quantify interpretability.
- **Threshold-Grad-CAM**: Introduces a modification to Grad-CAM to address interpretability issues in specific models.

## Models Evaluated
Eight reference models were evaluated in this study:
1. ResNet50
2. ConvNeXt (T, S, and B variants)
3. CAFormer
4. ConvFormer
5. FastViT (sa_24 and sa_36 variants)

## Results
- **Interpretability Improvement**: Seven out of eight models showed improved interpretability with DCLS.
- **Grad-CAM Issue**: CAFormer and ConvFormer models generated random heatmaps, resulting in low interpretability scores.
- **Threshold-Grad-CAM Solution**: Enhanced interpretability across nearly all models when implemented.

## Repository Contents
- **Code**: Implementation of DCLS and Threshold-Grad-CAM.
- **Checkpoints**: Pre-trained model checkpoints to reproduce the study results.
- **Scripts**: Scripts to evaluate models and compute interpretability scores.

## Getting Started

### Prerequisites
- Python 3.8+
- PyTorch 1.8+
- CUDA 10.2+ (for GPU acceleration)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/rabihchamas/DCLS-GradCAM-Eval.git
   cd DCLS-GradCAM-Eval
   
2. Install required packages:
   ```bash
   pip install -r requirements.txt
   
### Prerequisites
- Python 3.8+
- PyTorch 1.8+
- CUDA 10.2+ (for GPU acceleration)

### Usage
1. **Evaluate Models**: Run the evaluation script to compute interpretability scores.
   ```bash
   python main.py

## Contributions
We welcome contributions to improve the project. Please submit pull requests or report issues to the GitHub repository.

## License
This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Contact
For any questions or inquiries, please contact Rabih Chamas at [rabih.chamas@lis-lab.fr](mailto:rabih.chamas@lis-lab.fr).

## References
- [DCLS: Dilated Convolution with Learnable Spacing](https://arxiv.org/abs/2112.03740)
- [Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization](https://arxiv.org/abs/1610.02391)
- [ClickMe Dataset](https://clickme.clps.brown.edu/tutorial)
