---
layout: default
title: CS6476 Computer Vision Final Project
description: Leveraging Contrastive Learning to Improve Editing Performance of Diffusion Models
toc: true
---
## Team Members:
- Archana Kutumbaka
- Siddharth Sriraman

## Introduction / Problem Definition: 
Diffusion models operate by gradually transforming random noise into a coherent output (like an image, audio, or video) through a process that iteratively refines the output by predicting and removing noise at each step, guided by learned data patterns. This unique approach allows them to scale up to mega-resolutions, enabling the creation of highly detailed and imaginative outputs across various forms of media, showcasing their remarkable ability to synthesize complex and nuanced content.

One significant challenge that diffusion models face is their struggle with compositionality, specifically their difficulty in understanding how different attributes are interconnected and in learning the rules for combining these attributes from language descriptions. For instance, diffusion models often fail to generate distinctly different outputs for descriptions like "a sheep to the right of a goat" versus "a goat to the right of a sheep." This limitation hampers their performance in visio-linguistic reasoning tasks, such as accurately matching images with text descriptions or vice versa. We believe this issue arises primarily due to the diffusion models' training approach, which lacks a mechanism for penalizing incorrect examples, unlike contrastive learning methods that specifically reinforce correct associations and penalize incorrect ones. Additionally, the reliance on CLIP-based text encoders, which have inherent weaknesses in handling complex visio-linguistic compositions, exacerbates this problem. To address these challenges, our strategy involves implementing a soft negative training approach inspired by Krojer et al., without resorting to hard negatives as in Zou et al. Orthogonally, we plan to enhance the dataset by introducing more compositionally challenging negative captions, generated using a LLaMa model, to improve the models' understanding and handling of compositional visio-linguistic reasoning.

### Related Works: 

#### _Methods_

The following are two motivating approaches that we build upon:

Zou et al. [2] show that state-of-the-art Vision and Language Models (VLMs) perform poorly on image generation tasks that involve compositionality. They curate a benchmark dataset of image-text pairs that cover various types of Attributes, Relations and Order (ARO) information. They generate misordered hard-negatives through perturbation of words in the captions. They show that state-of-the-art VLMs are not sensitive to ordering of objects, and instead behave like a bag-of-words model. They conduct experiments to show that composition-aware hard negative mining (of both images and captions) significantly improves compositionality and ordering performance.

Reddy et al. [1] attempts to solve this problem by fine-tuning a pre-trained diffusion model using a loss that maximizes error predicted for negative samples and minimizes error for the positive sample. They also propose DiffusionITM, a scheme that converts a generative model to a discriminative zero-shot image-text matching model, which allows generative models to now be tested on discriminative vision-and-language benchmarks. They also introduce GDBench, a benchmark dataset consisting of image-text pairs that capture a wide range of compositions.

We also found alternate approaches that focus on compositionality performance:

Chua et al. [3] aim to solve the text-to-image misalignment problem by improving the intrinsic compositional reasoning of generative models. To do this, they propose a two-stage scheme called Discriminative Probing and Tuning (DPT). The probing stage involves passing the latent space of the U-Net of a generative model like Stable Diffusion [6] (SD) through a discriminative adapter model that probes into its local (referring expression comprehension) and global grounding (image-text matching) abilities. The tuning stage involves parameter-efficient fine-tuning using LoRA [7]. They use this adapter in inference to improve compositionality performance in denoising-based text-to-image generation.

Feizi et al. [4] improve visuo-linguistic reasoning in CLIP [5] by modifying its standard loss to include a distillation loss component from a text-to-image model such as Stable Diffusion. They linearly map CLIP’s image encoder output into the SD’s U-Net input space, and learn this map by adding the denoising diffusion score to the CLIP loss. While the compositionality performance still remains bounded by SD’s abilities, they show that CLIP achieves significantly better visuo-linguistic reasoning by integrating knowledge from diffusion models.

#### _Benchmarks_

In addition to the ARO and GDBench benchmarks discussed above, we found other benchmarks specific to compositionality. Ross et al. [8] introduce Winoground, a carefully hand-crafted benchmark dataset of 1600 image-text pairs (800 being correct and 800 being incorrect) that differ in word ordering. Similar to [1], they show that VLMs show poor performance (close to random chance) when it comes to compositionality. 

### Methods / Approach: 

Following [1], we hypothesize that unconditional (no text) error prediction for a given image marginalizes the probability over the text dimension. They leverage this finding, to normalize the error predicted conditionally by this unconditional value and use the residue for the image retrieval task. In addition, they use hard negatives to contrastively finetune the diffusion backbone using the loss in equation (1).

$$ \mathcal{L}\_{\text{hard-neg}} = \mathcal{L}\_{\text{pos}} + \text{clip}(\mathcal{L}\_{\text{neg}}, |\lambda\mathcal{L}\_{\text{pos}}|) \tag{1} $$

where 

$$ \mathcal{L}\_{\text{pos}} = {\mathbb{E}}\_{x,t} [\|\mathbf{e} - \mathbf{e}\_{\theta}(x, t, w_{\text{pos}})\|_2^2] $$

$$ \mathcal{L}\_{\text{neg}} = -\mathbb{E}\_{x,t} [\|\mathbf{e} - \mathbf{e}\_{\theta}(x, t, w_{\text{neg}})\|_2^2] $$


In contrast, we follow a soft negative training policy that directly finetunes the diffusion model to minimize a new loss function in equation (2) that ensures that the correct caption is preferred over all other caption possibilities without the use of explicit negatives.

$$
\mathcal{L}\_{\text{soft-neg}} = \mathbb{E}\_{x,t} \left[ \left( \| \mathbf{e} - \mathbf{e}\_{\theta}(x, t, w) \|_2^2 - \| \mathbf{e} - \mathbf{e}\_{\theta} (x, t) \|_2^2 \right) \right]  \tag{2}
$$


We believe that this eliminates the need for creating hard-negatives and generating them by using swapping nouns similar to [2]. It would also allow for more stable training without the need for clipping and regularizing for gatekeeping potentially infinite gains. 

### Experiments / Results: 

**Experiment 1**: We are fine-tuning the Stable Diffusion-1.5 using our soft negative loss in equation (2) on the COCO-Order dataset [2] that was used for hard negative training for fair comparison. 

| Method | ImageCode (Image) | Winoground (Image) | Winoground (Text)
|-------|--------|---------|--------|
| Vanilla SD            | 30.1 | 9.0 | 32.3|
| + MS-COCO NoNeg       | 29.7 | 10.3 | 35.0 |
| + MS-COCO HardNeg     | 31.9 | 9.8 | 30.8 |
| + **MS-COCO SoftNeg** | | | |

Table 1: Results on GDBench components for two image retrieval tasks (ImageCode and Winoground), and one text retrieval task (Winoground). Our method **MS-COCO SoftNeg** is in bold, the results for the other methods were borrowed from [1]. The ImageCode dataset variant used is the image one, and the metric shown is R@1. The Winoground metric reported here is accuracy.

### What’s next:

Dataset Augmentation: We plan to generate a synthetic dataset that uses more compositionally confusing captions for the original image dataset. We will leverage LLaMA [9] for generating these new text captions and qualitatively evaluate the comparisons against the existing compositionality testing datasets in the ARO benchmark [2]. 


### Team Member Contributions: 

- Archana Kutumbaka: Tune the parameters for soft negative finetuning and conduct evaluations on GDBench. 

- Siddharth Sriraman: Generate the augmented COCO-Order dataset using LLaMA and compare against existing dataset for compositionality testing. 

### References:

[1] Krojer, B., et al, "Are Diffusion Models Vision-And-Language Reasoners?," in _Thirty-seventh Conference on Neural Information Processing Systems_, 2023.

[2] Yuksekgonul, M., et al, "When and why vision-language models behave like bags-of-words, and what to do about it?," in _The Eleventh International Conference on Learning Representations_, 2022.

[3] Qu, L., et al. "Discriminative Probing and Tuning for Text-to-Image Generation," in _arXiv preprint_ arXiv:2403.04321, 2024.

[4] Basu, S., et al. "Augmenting clip with improved visio-linguistic reasoning," in _arXiv preprint_ arXiv:2307.09233, 2023.

[5] Radford, A., et al, "Learning transferable visual models from natural language supervision," in _International conference on machine learning_, 2021, pp. 8748–8763.

[6] Rombach, R., et al, "High-resolution image synthesis with latent diffusion models," in Proceedings of the _IEEE/CVF conference on computer vision and pattern recognition_, 2022, pp. 10684–10695.

[7] Hu, E., et al. "Lora: Low-rank adaptation of large language models," in _arXiv preprint_ arXiv:2106.09685, 2021.

[8] Thrush, T., et al, "Winoground: Probing vision and language models for visio-linguistic compositionality," in _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_, 2022, pp. 5238–5248.

[9] Touvron, H., et al. "Llama: Open and efficient foundation language models," in _arXiv preprint_ arXiv:2302.13971, 2023.
