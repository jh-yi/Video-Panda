# <img src="figs/logo.png" style="vertical-align: -10px;" :height="45px" width="45px"> Video-Panda: Parameter-efficient Alignment for Encoder-free Video-Language Models

[Jinhui Yi*](https://scholar.google.com/citations?user=kLZxzzUAAAAJ&hl=en),
[Syed Talal Wasim*](https://talalwasim.github.io),
[Yanan Luo*](https://scholar.google.com/citations?user=yuDQY0YAAAAJ&hl=en),
[Muzammal Naseer](https://muzammal-naseer.netlify.app/),
[Juergen Gall](https://pages.iai.uni-bonn.de/gall_juergen/)

*Joint first authors

[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2412.18609)
<hr />

> **Abstract:**
>*We present an efficient encoder-free approach for video-language understanding that achieves competitive performance while significantly reducing computational overhead. Current video-language models typically rely on heavyweight image encoders (300M-1.1B parameters) or video encoders (1B-1.4B parameters), creating a substantial computational burden when processing multi-frame videos. Our method introduces a novel Spatio-Temporal Alignment Block (STAB) that directly processes video inputs without requiring pre-trained encoders while using only 45M parameters for visual processing - at least a 6.5$\times$ reduction compared to traditional approaches. The STAB architecture combines Local Spatio-Temporal Encoding for fine-grained feature extraction, efficient spatial downsampling through learned attention and separate mechanisms for modeling frame-level and video-level relationships. Our model achieves comparable or superior performance to encoder-based approaches for open-ended video question answering on standard benchmarks. The fine-grained video question-answering evaluation demonstrates our model's effectiveness, outperforming the encoder-based approaches Video-ChatGPT and Video-LLaVA in key aspects like correctness and temporal understanding. Extensive ablation studies validate our architectural choices and demonstrate the effectiveness of our spatio-temporal modeling approach while achieving 3-4$\times$ faster processing speeds than previous methods.*

## Table of Contents
<!--ts-->
   * [News](#-news)
   * [Overview](#-overview)
   * [Visualization](#-visualization)
   * [Codes](#codes)
   * [Acknowledgements](#️-acknowledgements)
   * [Citation](#️-citation)
<!--te-->

## 🚀 News
* **(December 25, 2024)** 
  * Paper is released. Codes and models will be released soon.
<hr />


## 💡 Overview

<p align="center">
  <img alt="Overall Architecture" src="figs/main_arch.png" width="1200"/>
  <p align="center"><b>Detailed architecture of our Spatio-Temporal Alignment Block (STAB):</b> The input video is first converted into patches. The Local Spatio-Temporal Encoding (LSTE) uses 3D convolutions to model spatio-temporal relations and adds a 3D convolution dynamic position encoding (DPE) to encode position with respect to the local spatio-temporal window. As a result, we obtain per-frame tokens with positional encoding. The tokens are then processed in two ways. While the Global Spatio-Temporal Relationship Aggregator (GSTRA) at the top captures video-level context, the Frame-wise Spatial Relationship Aggregator (FSRA) at the bottom captures spatial context within each frame. To reduce the cost, we perform a Local Spatial Downsampling (LSD) to reduce the spatial dimension for each token. The video-level context tokens and the frame-wise spatial tokens are then linearly combined through learnable weighted fusion ($\alpha$), producing a frame-specific context token. These context tokens are then prepended to their respective frame's flattened spatial tokens, with $\texttt{<row>}$ split tokens inserted to demarcate row boundaries in the spatial layout. This combination of global context and preserved spatial structure enables effective video understanding while maintaining computational efficiency.</p>
</p>

<hr />


<p align="center">
  <img alt="Performance Overview" src="figs/intro_fig.png" width="600"/>
  <p align="center">Model performance on MSVD-QA versus the model size of the visual component in logarithmic scale. The bubble size indicates the amount of finetuning data (in thousands). Models using the same training dataset as ours (100K samples) are shown in dark green, while those using different datasets are in blue.</p>
</p>


## 🔍 Visualization

<p align="center">
  <img alt="Visualization" src="figs/vis_figure.png" width="1200"/>
  <p align="center">Qualitative examples showing the impact of removing Frame-wise Spatial Relationship Aggregator (FSRA) and Global Spatio-Temporal Relationship Aggregator (GSTRA).</p>
</p>

## Codes

Codes and models will be released soon.

## ❤️ Acknowledgements
Our code is based on [Video-LLaVA](https://github.com/PKU-YuanGroup/Video-LLaVA) and [EVE](https://github.com/baaivision/EVE) repositories. We thank the authors for releasing their code. If you use our model, please consider citing these works as well.

## ✏️ Citation
If you find our work, this repository, or pretrained models useful, please consider giving a star ⭐ and citation 📝.
```bibtex
@article{yi2024video-panda,
    author    = {Jinhui Yi* and Syed Talal Wasim* and Yanan Luo* and Muzammal Naseer and Juergen Gall},
    title     = {Video-Panda: Parameter-efficient Alignment for Encoder-free Video-Language Models},
    journal   = {arXiv preprint, arXiv:2412.18609},
    year      = {2024},
}
```

## 🔒 License
The content of this project is released under the MIT license license as found in the [LICENSE](https://github.com/jh-yi/MV-Match/blob/main/LICENSE) file.

---
If you have any questions, please create an issue on this repository or contact at jinhui.yi@uni-bonn.de and swasim@uni-bonn.de.
