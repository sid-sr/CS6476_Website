---
layout: default
title: CS6476 Computer Vision Final Project
description: Leveraging Contrastive Learning to Improve Editing Performance of Diffusion Models
toc: true
---
## Team Members:
- Archana Kutumbaka
- Siddharth Sriraman

[GitHub Repo](https://github.com/archana53/diffusion-itm/) (training code is in the main branch, and benchmarking code is in the [inference](https://github.com/archana53/diffusion-itm/tree/inference) branch)
[GitHub Website Repo](https://github.com/sid-sr/CS6476_Website/)

## Introduction / Problem Definition: 
**High-level Description and Motivation**: Diffusion models operate by gradually transforming random noise into a coherent output (like an image, audio, or video) through a process that iteratively refines the output by predicting and removing noise at each step, guided by learned data patterns. This unique approach allows them to scale up to mega-resolutions, enabling the creation of highly detailed and imaginative outputs across various forms of media, showcasing their remarkable ability to synthesize complex and nuanced content.

**Specific Problem Definition**: One significant challenge that diffusion models face is their struggle with compositionality, specifically their difficulty in understanding how different attributes are interconnected and in learning the rules for combining these attributes from language descriptions. For instance, diffusion models often fail to generate distinctly different outputs for descriptions like "a sheep to the right of a goat" versus "a goat to the right of a sheep." This limitation hampers their performance in visio-linguistic reasoning tasks, such as accurately matching images with text descriptions or vice versa. We believe this issue arises primarily due to the diffusion models' training approach, which lacks a mechanism for penalizing incorrect examples, unlike contrastive learning methods that specifically reinforce correct associations and penalize incorrect ones. Additionally, the reliance on CLIP-based text encoders, which have inherent weaknesses in handling complex visio-linguistic compositions, exacerbates this problem. 

|![SD-bad-example](assets/images/comp_example.png)|
|:-:|
|*Figure 1: Example of Stable Diffusion text-to-image generation from the DrawBench prompts [10], image shown in [1].*|

Figure 1 is an example of poor compositional reasoning. The prompt given to Stable Diffusion [6] is "A stack of 3 books. A green book is on the top, sitting on a red book. The blue book is on the bottom." It does generate a stack of books but the relative positioning is not as expected. 

### Related Works: 

#### _Methods_

The following are two motivating approaches that we build upon:

Yuksekgonul et al. [2] show that state-of-the-art Vision and Language Models (VLMs) perform poorly on image generation tasks that involve compositionality. They curate a benchmark dataset of image-text pairs that cover various types of Attributes, Relations and Order (ARO) information. They generate misordered hard-negatives through perturbation of words in the captions. They show that state-of-the-art VLMs are not sensitive to ordering of objects, and instead behave like a bag-of-words model. They conduct experiments to show that composition-aware hard negative mining (of both images and captions) significantly improves compositionality and ordering performance.

Krojer et al. [1] attempts to solve this problem by fine-tuning a pre-trained diffusion model using a loss that maximizes error predicted for negative samples and minimizes error for the positive sample. They also propose DiffusionITM, a scheme that converts a generative model to a discriminative zero-shot image-text matching model, which allows generative models to now be tested on discriminative vision-and-language benchmarks. They also introduce GDBench, a benchmark dataset consisting of image-text pairs that capture a wide range of compositions.

We also found alternate approaches that focus on compositionality performance:

Qu et al. [3] aim to solve the text-to-image misalignment problem by improving the intrinsic compositional reasoning of generative models. To do this, they propose a two-stage scheme called Discriminative Probing and Tuning (DPT). The probing stage involves passing the latent space of the U-Net of a generative model like Stable Diffusion [6] (SD) through a discriminative adapter model that probes into its local (referring expression comprehension) and global grounding (image-text matching) abilities. The tuning stage involves parameter-efficient fine-tuning using LoRA [7]. They use this adapter in inference to improve compositionality performance in denoising-based text-to-image generation.

Basu et al. [4] improve visuo-linguistic reasoning in CLIP [5] by modifying its standard loss to include a distillation loss component from a text-to-image model such as Stable Diffusion. They linearly map CLIP’s image encoder output into the SD’s U-Net input space, and learn this map by adding the denoising diffusion score to the CLIP loss. While the compositionality performance still remains bounded by SD’s abilities, they show that CLIP achieves significantly better visuo-linguistic reasoning by integrating knowledge from diffusion models.

#### _Benchmarks_

In addition to the ARO and GDBench benchmarks discussed above, we found other benchmarks specific to compositionality. Thrush et al. [8] introduce Winoground, a carefully hand-crafted benchmark dataset of 1600 image-text pairs (800 being correct and 800 being incorrect) that differ in word ordering. Similar to [1], they show that VLMs show poor performance (close to random chance) when it comes to compositionality. 

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

In the second orthogonal approach, we aim to improve the quality of text hard-negatives used in [2], where they use rule-based noun and verb reordering to generate compositionally confusing captions for images from MS-COCO. 

For this, we first inspected MS-COCO to understand which aspects of compositionality it captures and which it does not, and look at how those are represented in different components of GDBench.

We saw that while it captures compositional information about a scene, it does not capture fine-grained information about where each object is located. For example, the following image has a caption of "a living room with a couch and a coffee table", but does not describe the relative position of the couch with respect to the coffee table.

|![COCO Chair image](assets/images/mscoco/chair.png)|
|:-:|
|*Figure 2: Example from MS-COCO*|

Similar to the above case, we also noted that ordering of items in the captions are not consistent with the ordering in the images. Different captions have different orderings. For example, in the following image, one caption captures an ordering that is inconsistent with the others.

|![COCO Fruit image](assets/images/mscoco/fruit.png)|
|:-:|
|*Figure 3: Example from MS-COCO*|

But we are bounded by the MS-COCO dataset, since datasets with high-quality compositional information like Winoground are difficult to manually curate and are hence very small (800 samples). So instead, we aim to improve the hard-negative mining method used by Yuksukgonul et al. [2]. The generated captions in their method currently do not semantically and gramatically make sense. For example, their COCO-Order component of ARO cites an example where they perturb the caption "A brown cat is looking at a gray dog sitting in a white bathtub" to "at brown cat a in looking a gray dog sitting is and a white bathtub" through shuffing all but adjectives and nouns.

We aim to use a large language model like LLaMA [9] to generate semantically and gramatically valid hard-negatives and train SD as in [2] to analyse if this method improves visuo-linguistic reasoning in diffusion models.

### Experiment Setup:

**Experiment Purpose**

**Input Description**:

**Desired Output Description**:

**Metric for Success**:

### Results: 

**Experiment 1**: We are fine-tuning the Stable Diffusion-1.5 using our soft negative loss in equation (2) on the COCO-Order dataset [2] that was used for hard negative training for fair comparison. 

| Method | ImageCode (Image) | Winoground (Image) | Winoground (Text)
|-------|--------|---------|--------|
| Vanilla SD            | 30.1 | 9.0 | 32.3|
| + MS-COCO NoNeg       | 29.7 | 10.3 | 35.0 |
| + MS-COCO HardNeg     | 31.9 | 9.8 | 30.8 |
| + **MS-COCO SoftNeg** | 29.6 | 8.8 | 30.9 |

Table 1: Results on GDBench components for two image retrieval tasks (ImageCode and Winoground), and one text retrieval task (Winoground). Our method **MS-COCO SoftNeg** is in bold, the results for the other methods were borrowed from [1]. The ImageCode dataset variant used is the image one, and the metric shown is R@1. The Winoground metric reported here is accuracy.

For fine-tuning, we performed distributed training on 4 NVIDIA A40s for 8 epochs, which took ~32 hours. For benchmark evaluation, ImageCode took ~45 minutes, and Winoground took ~20 minutes. At this scale, hyperparameters like batch size make a significant difference in results, so these results are very initial, we are currently experimenting with these parameters. For running benchmarks on fine-tuned models, we are also utilising an NVIDIA V100 on Colab Pro.

For the hard-negative generation experiment, we are experimenting with zero- and few-shot prompting, and are yet to start fine-tuning SD on these captions.

### Discussion:

### Challenges Encountered:


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

[10] Saharia, C., et al, "Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding," in _Advances in Neural Information Processing Systems_, 2022, pp. 36479–36494.
