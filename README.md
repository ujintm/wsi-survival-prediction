# Title
Comparative Study of MIL Aggregation Methods for WSI-based Survival Prediction on TCGA-BRCA

# Overview
This project compares four Multiple Instance Learning (MIL) aggregation methods — Mean Pooling, ABMIL (Attention-Based MIL), TransMIL, and MambaMIL — for patient survival prediction from Whole Slide Images (WSIs).

Unlike prior work, which mostly compares MIL aggregators on classification tasks using ResNet-based patch features, this study asks a different question: once patch features come from a strong pretrained foundation model, does aggregation complexity still matter for survival prediction? Here, "complexity" refers to structural sophistication in modeling inter-instance interactions — not computational cost.

Experiments are conducted on the TCGA-BRCA (Breast Invasive Carcinoma) cohort. After matching WSI data with clinical records, 752 patients (151 death events, avg. ~2.2 WSIs/patient) were included, with patient-level train/test splitting to prevent data leakage.

# Architecture
<img width="5931" height="1402" alt="그림4" src="https://github.com/user-attachments/assets/b7711d19-6e75-40ae-b6f4-d145af08cd84" />


The pipeline has two aggregation levels. Level 1 (patch → slide) applies the compared aggregator — Mean Pooling, ABMIL, TransMIL, or MambaMIL — to a slide's patch features to produce a slide-level representation; this is the only stage that changes across experiments. Level 2 (slide → patient) applies a fixed attention aggregator (identical across all experiments) to combine a patient's multiple slide representations into a patient-level representation. This is concatenated with clinical features, passed through the survival prediction network to output a risk score, and trained with the Cox partial log-likelihood loss.

# Problem
- A single WSI contains tens of thousands of patches, and patients often have multiple slides
- Simple pooling (mean/max) can't capture which patches — or which slides — matter most for prognosis
- It's unclear whether increasingly complex aggregators (Attention → Transformer → Mamba) still help once patch features already come from a rich, pretrained foundation model
- Real clinical prediction should combine imaging information with clinical variables (age, stage, etc.), a setting under-explored in prior aggregation comparisons

# Method
- **Feature extraction:** Patch-level features extracted with **UNI**, a pathology foundation model, used frozen (no fine-tuning)
- **Hierarchical two-level aggregation** (patch → slide → patient):
  - **Level 1 (patch → slide):** the compared aggregator — Mean Pooling / ABMIL / TransMIL / MambaMIL — swapped in while everything else is held fixed
  - **Level 2 (slide → patient):** a fixed attention aggregator combining multiple slide-level representations per patient, kept identical across all experiments
- Patient-level representation is concatenated with clinical features and passed through a fully connected head to output a **risk score**
- Trained with a **Cox proportional hazards** partial log-likelihood loss
- Evaluated with **C-index** (appropriate for right-censored survival data — accuracy-style metrics don't apply)
- Stratified 5-fold cross-validation (stratified by event) with patient-level splitting
- MambaMIL trained with `use_fast_path=False` due to CUDA kernel incompatibility with the current GPU architecture (numerically equivalent, but slower)
- Gradient accumulation (batch size 1, 16-bag accumulation) used to stabilize Cox loss given the low absolute number of events

# Results (C-index, TCGA-BRCA, 752 patients, 151 events, stratified 5-fold CV)

| Aggregation Method | Trainable Params | C-index (mean ± std) |
|---|---|---|
| **ABMIL** | 1.05 M | **0.75 ± 0.03** |
| Mean Pooling | 0.79 M | 0.73 ± 0.04 |
| TransMIL | 2.94 M | 0.73 ± 0.05 |
| MambaMIL | 4.75 M | 0.73 ± 0.06 |


**Key finding:** aggregation complexity did **not** translate into better performance. With strong UNI-based features, the relatively simple ABMIL achieved the highest mean C-index *and* the lowest fold-to-fold variance, while the more complex TransMIL and MambaMIL did not improve over Mean Pooling. MambaMIL, the largest model, showed the widest spread (min 0.63, std 0.06). Two likely explanations:
1. High-quality foundation-model features may already encode enough patch-level semantic information that additional structural modeling (spatial correlation, long-range sequence modeling) has less left to contribute.
2. TCGA-BRCA's low event rate (151/752 patients) may disadvantage higher-capacity models, which need more learning signal than a sparse event set can reliably provide — consistent with the larger models' greater fold-to-fold variance.

# Limitations
- Single dataset (TCGA-BRCA); generalization to other cohorts/cancer types is not yet verified
- Low event rate (151/752) may disadvantage higher-capacity aggregators
- Clinical features are concatenated identically across all aggregators; the contribution of clinical information vs. WSI alone is not isolated
- A single feature extractor (UNI) is used, so conclusions are specific to this feature regime
# Future Work
- Ablate clinical-feature concatenation (WSI-only vs. WSI + clinical) across all four aggregators
- Compare additional feature extractors (ResNet, PLIP, UNI) under the same aggregation setup
- Bootstrapped confidence intervals / significance testing across aggregators
- Resolve MambaMIL's CUDA fast-path incompatibility and re-evaluate
- Multi-modal extension incorporating genomic/omics data alongside WSI and clinical features
# References
[1] Ilse et al., "Attention-based Deep Multiple Instance Learning," ICML 2018.
[2] Lu et al., "Data-efficient and weakly supervised computational pathology on whole-slide images," Nature Biomedical Engineering, 2021.
[3] Shao et al., "TransMIL: Transformer based correlated multiple instance learning for whole slide image classification," NeurIPS 2021.
[4] Yang et al., "MambaMIL: Enhancing long sequence modeling with sequence reordering in computational pathology," MICCAI 2024.
[5] Chen et al., "Towards a general-purpose foundation model for computational pathology," Nature Medicine, 2024.
