# Synthetic Brain MRI Generation with StyleGAN2

## Project Overview

This project investigates the generation of synthetic brain MRI images using StyleGAN2 on the ADNI dataset.

The egineering dilemma is the fidelity-memorization-privacy trade-off in synthetic medical image generation. A useful generative model should produce produce genuinely novel, diverse, and anatomically coherent scans while avoiding memorisation of individual training samples.

The proposed model is evaluated against a simpler baseline generative model using Structural Similarity Index (SSIM) for image fidelity, distributional metrics such as FID or feature-space distance, t-SNE/UMAP analysis of real and synthetic samples, qualitative failure-case inspection, and nearest-neighbour analysis for potential training-data memorisation.

## 1. Engineering Problem and Motivation


## 2. Feasibility Review


## 3. Dataset


## 4. Method


## 5. Implementation

### 5.1 Project Structure

```text
recognition/
└── StyleGAN2_s4909628/
    ├── modules.py
    ├── dataset.py
    ├── train.py
    ├── predict.py
    └── README.md
```


## 6. Experimental Setup


## 7. Quantitative Evaluation


## 8. Baseline vs Proposed Model

### 8.1 Quantitative Comparison

| Metric | DCGAN Baseline | StyleGAN2 |
|---|---:|---:|
| SSIM | TODO | TODO |
| FID | TODO | TODO |
| Diversity Metric | TODO | TODO |
| Parameter Count | TODO | TODO |
| Peak GPU VRAM | TODO | TODO |
| Training Time | TODO | TODO |
| Generation Time per Image | TODO | TODO |


## 9. Real vs Synthetic Distribution Analysis


## 10. Memorisation and Privacy Audit

### 10.1 Motivation
A synthetic image that closely reproduces a training patient scan may indicate memorisation rather than genuine generation. Therefore, visual realism alone is not sufficient.

### 10.2 Nearest-Neighbour Analysis
For each selected synthetic image:
1. extract its representation;
2. compare it against training images;
3. identify its nearest training neighbour;
4. report similarity / distance;
5. visually inspect the image pair.

### 10.3 Example Nearest-Neighbour Results

| Synthetic Image | Nearest Training Image | Distance / Similarity | Observation |
|---|---|---:|---|
| Sample 1 | TODO | TODO | TODO |
| Sample 2 | TODO | TODO | TODO |
| Sample 3 | TODO | TODO | TODO |

#### Sample 1

**Synthetic Image**

![Synthetic Sample 1](images/synthetic_1.png)

**Nearest Training Image**

![Nearest Training Sample 1](images/nearest_train_1.png)

**Distance / Similarity:**  
TODO

**Observation:**  
TODO


## 11. Latent Space Investigation


## 12. Qualitative Results


## 13. Failure Case Autopsy

Analyse at least 3–5 representative failure cases from the generated results.

### Failure Case 1 — TODO

**Observed Failure:**  
TODO

**Possible Trigger:**  
TODO

**Likely Model Behaviour:**  
TODO

**Engineering Significance:**  
TODO

---

### Failure Case 2 — TODO

**Observed Failure:**  
TODO

**Possible Trigger:**  
TODO

**Likely Model Behaviour:**  
TODO

**Engineering Significance:**  
TODO

---

### Failure Case 3 — TODO

**Observed Failure:**  
TODO

**Possible Trigger:**  
TODO

**Likely Model Behaviour:**  
TODO

**Engineering Significance:**  
TODO


## 14. Fidelity–Memorization–Privacy Trade-off
Summarise the central engineering dilemma.


## 15. Engineering Recommendation


## 16. Limitations


## 17. Conclusion


## 18. Usage


## 19. Artificial Intelligence Usage Disclosure
Generative AI tools were used during the development of this project.


## 20. Referneces