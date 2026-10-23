# Synthetic Brain MRI Generation with StyleGAN2

## Project Overview

This project investigates the generation of synthetic brain MRI images using StyleGAN2 on the ADNI dataset.

The egineering dilemma is the fidelity-memorization-privacy trade-off in synthetic medical image generation. A useful generative model should produce diverse synthetic MRI slices while avoiding memorisation of individual training samples.

The proposed model is evaluated against a simpler baseline generative model using Structural Similarity Index (SSIM) for image fidelity, distributional metrics such as FID or feature-space distance, t-SNE/UMAP analysis of real and synthetic samples, qualitative failure-case inspection, and nearest-neighbour analysis for potential training-data memorisation.

## 1. Engineering Problem and Motivation
Medical-imaging and machine-learning researchers need varied image data to develop and evaluate 
computational methods, but access to patient images is constrained by data availability and privacy 
requirements. Synthetic brain MRI generation offers a potential source of supplementary research images. 
This project investigates whether a generative model can capture structural variation in the available data and produce additional samples that warrant further investigation for research use.

The engineering challenge extends beyond generating visually convincing images. 
The model may reproduce the overall brain outline while losing structural details, 
generate similar outputs, or produce images that closely resemble individual training samples. 
Furthermore, generative models can also expose information about their training data, 
as demonstrated by the membership-inference attacks in LOGAN [1]. 
These concerns motivate a joint assessment of fidelity, diversity, memorisation risk and computational cost. 
The central question is whether StyleGAN2 [2][3] improves image quality and diversity sufficiently over DCGAN [4] 
to justify its additional computational demands, while avoiding obvious memorization of training images.


## 2. Feasibility Review

### 1. User Need, Scope and Acceptance Criteria

The intended users of this prototype are medical-imaging and machine-learning researchers 
who require additional brain MRI data for model development but face limited clinical scan 
availability and patient privacy.

This project is to investigate whether StyleGAN2 can produce genuinely novel, diverse, and anatomically
coherent scans without simply memorising training patients. Structural similarity is assessed 
using SSIM [5], while distributional similarity is assessed using Fréchet Inception Distance (FID) [6]

A DCGAN is used as a simpler baseline to assess whether StyleGAN2's additional architectural 
and computational complexity provides a meaningful engineering advantage.

The prototype will be evaluated against the following criteria:

1. Reference-paired SSIM will be reported using the predefined nearest-neighbour matching protocol, including the proportion of generated images with SSIM > 0.6. This threshold is treated as a project evaluation target rather than evidence of clinical validity.
2. StyleGAN2 should provide measurable improvement over the DCGAN baseline in structural fidelity and distribution fidelity.
3. Generated samples and real-versus-synthetic embedding analysis should not show obvious mode collapse or severely restricted diversity.
4. Nearest-neighbour analysis should not reveal obvious near-duplicate training images that would indicate potential memorisation.
5. Training must remain feasible on the NVIDIA A100 40 GB GPU, with peak VRAM usage and training/generation runtime explicitly profiled.

### 2. Model Choice and Course Concepts

The project uses the ADNI brain MRI dataset and a StyleGAN2 generator as the proposed Hard-Difficulty model. A DCGAN trained using the same preprocessing workflow is used as the baseline.

StyleGAN2 was selected because its mapping network, intermediate latent space, learned affine style transformations, style-modulated convolutions and weight demodulation provide substantially greater control over image synthesis than a standard DCGAN. These components directly relate to generate high-quality and diverse new MRI samples, thus potentially reducing the need for directly sharing real patient scans.

Compared with the original StyleGAN, StyleGAN2 improves the synthesis architecture to reduce characteristic image artefacts and improve image fidelity. This is particularly relevant to brain MRI generation, where fine anatomical structures and tissue boundaries must be preserved to produce clinically plausible synthetic images.

### 3. Preliminary Feasibility Evidence

An initial audit of the ADNI dataset will document the number of patients, usable slices, original image dimensions, preprocessing resolution and normalisation strategy. The validation set is separated from the training data by patient, while the supplied test partition is retained. The DCGAN baseline is trained with the preprocessed ADNI data to establish initial quality and resource benchmarks. And StyleGAN2 will be trained using the same preprocessed datasets.

An initial StyleGAN2 smoke test has successfully run on an NVIDIA A100 40GB GPU in the cluster rangpur. At batch size 8, the first training epoch processed approximately 17.48 kimg and required 1191 s. Peak allocated GPU memory was 36.24 GiB and peak reserved memory was 36.46 GiB. The higher first-epoch memory peak is consistent with additional startup and convolution algorithm-selection overhead. Once the convolution algorithms and memory buffers had been initialised, subsequent epochs stabilised at below approximately 7 GiB of reserved VRAM.

These results indicate that the current model configuration fits within the available 40 GB GPU memory, although training cost is significant.

### 4. Risks, Compute Budget and Fallback

The main technical risks are adversarial training instability, mode collapse, memorisation of individual ADNI training scans, anatomically implausible generated structures, and imbalance between the generator and discriminator of StyleGAN2.

Also, the current measured training cost is approximately 20 minutes per epoch on rangpur with a-100. Longer training runs will therefore be guided by intermediate sample quality and training stability. Early stopping can be used to reduce the overfitting.

The DCGAN baseline under will be trained on the preprocessed datasets and compared initial sample quality, SSIM/FID, runtime and GPU memory against StyleGAN2.

Training will initially use established StyleGAN2 hyperparameter settings. If overfitting, unstable convergence, or insufficient improvements over the DCGAN baseline are observed, targeted hyperparameter adjustments will be made based on the identified issues, including learning rates, regularisation settings, and data augmentation strategies.

If these adjustments are insufficient, a reduced StyleGAN2 configuration, such as lower image resolution or fewer feature channels, will be considered to reduce computational requirements while retaining the core architecture.

If the model still fails to converge or produces anatomically implausible images, the best available checkpoint will be retained and evaluated against the DCGAN baseline. Representative failure cases will be documented and analysed to identify possible causes and limitations. The final report will transparently discuss unsuccessful experiments and whether StyleGAN2's additional complexity is justified by its observed performance.

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
stylegan2 000131

**Possible Trigger:**  
TODO

**Likely Model Behaviour:**  
TODO

**Engineering Significance:**  
TODO

---

### Failure Case 2 — TODO

**Observed Failure:**  
stylegan2 000158

**Possible Trigger:**  
TODO

**Likely Model Behaviour:**  
TODO

**Engineering Significance:**  
TODO

---

### Failure Case 3 — TODO

**Observed Failure:**  
stylegan2 000400
stylegan2 001444
stylegan2 001800
stylegan2 003184

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
ChatGPT was used during the development of this project.
Used to assist in learning, building network architectures and evaluation metrics, and debugging code.


## 20. References
[1] J. Hayes et al., "LOGAN: Membership Inference Attacks Against Generative Models," *PoPETs*, 2019.
[Paper](https://arxiv.org/abs/1705.07663)

[2] T. Karras, S. Laine, and T. Aila, "A Style-Based Generator Architecture for Generative Adversarial Networks," in *Proc. CVPR*, 2019. [Paper](https://arxiv.org/abs/1812.04948)

[3] T. Karras et al., "Analyzing and Improving the Image Quality of StyleGAN," in *Proc. CVPR*, 2020. [Paper](https://arxiv.org/abs/1912.04958)

[4] A. Radford, L. Metz, and S. Chintala, "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks," in *Proc. ICLR*, 2016. [Paper](https://arxiv.org/abs/1511.06434)

[5] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, "Image Quality Assessment: From Error Visibility to Structural Similarity," *IEEE Transactions on Image Processing*, vol. 13, no. 4, pp. 600–612, 2004. [Paper](https://doi.org/10.1109/TIP.2003.819861)

[6] M. Heusel et al., "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium," in *Proc. NeurIPS*, 2017. [Paper](https://arxiv.org/abs/1706.08500)

[7] NVIDIA Research (NVlabs), "StyleGAN2-ADA — Official PyTorch Implementation," GitHub Repository. [Online]. Available: https://github.com/NVlabs/stylegan2-ada-pytorch

[8] T. Karras et al., "Training Generative Adversarial Networks with Limited Data," in *Proc. NeurIPS*, 2020. [Paper](https://arxiv.org/abs/2006.06676)

[9] S. Pertuz, D. Puig, and M. A. Garcia, "Analysis of Focus Measure Operators for Shape-from-Focus," *Pattern Recognition*, vol. 46, no. 5, pp. 1415–1432, 2013. [Paper](https://doi.org/10.1016/j.patcog.2012.11.011)

