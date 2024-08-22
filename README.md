# An End-to-End Framework to Classify and Generate Privacy-Preserving Explanations in Pornography Detection

## About 

This repository contains the source code for the paper "An End-to-End Framework to Classify and Generate Privacy-Preserving Explanations in Pornography Detection", by Margarida Vieira, Tiago Gonçalves, Wilson Silva, and Ana F. Sequeira.

## Abstract

The proliferation of explicit material online, particularly pornography, has emerged as a paramount concern in our society. While state-of-the-art pornography detection models already show some promising results, their decision-making processes are often opaque, raising ethical issues. This study focuses on uncovering the decision-making process of such models, specifically fine-tuned convolutional neural networks and transformer architectures. We compare various explainability techniques to illuminate the limitations, potential improvements, and ethical implications of using these algorithms. Results show that models trained on diverse and dynamic datasets tend to have more robustness and generalisability when compared to models trained on static datasets. Additionally, transformer models demonstrate superior performance and generalisation compared to convolutional ones. Furthermore, we implemented a privacy-preserving framework during explanation retrieval, which contributes to developing secure and ethically sound biometric applications.

**Keywords:** Biometrics, Computer Vision, Deep Learning, Explainable Artificial Intelligence, Pornography Detection, Privacy Preservation

## Clone this repository

To clone this repository, open a Terminal window and type:

```
git clone https://github.com/margaridav27/end-to-end-framework-pornography-detection.git
```

Then go to the repository's main directory:

```
cd end-to-end-framework-pornography-detection
```

## Dependencies

We advise you to create a virtual Python environment first (Python 3.10). To install the necessary Python packages run:

```
pip install -r requirements.txt
```

If using Conda, you can create the virtual Python environment with all the requirements already installed by simply running:

```
conda create --name <env> --file requirements.txt
```

## Authors

- Margarida Vieira
- Tiago Gonçalves
- Wilson Silva
- Ana F. Sequeira
