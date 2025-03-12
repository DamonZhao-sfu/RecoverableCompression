# Recoverable-Compression-A-Multimodal-Vision-Token-Recovery-Mechanism-Guided-by-Text-Information
The code of  "Recoverable-Compression: A Multimodal Vision Token Recovery Mechanism Guided by Text Information"

## This work has been accepted by AAAI2025

# Abstract
With the advancement of large-scale language modeling techniques, large multimodal models combining visual encoders with large language models have demonstrated exceptional performance in various visual tasks. Most of the current large multimodal models achieve this by mapping visual features obtained from the visual encoder into a large language model and using them as inputs alongside text for downstream tasks. Therefore, the number of visual tokens directly affects the training and inference speed of the model. There has been significant work on token pruning for visual transformers, but for large multimodal models, only relying on visual information for token pruning or compression may lead to significant loss of important information. On the other hand, the textual input in the form of a question may contain valuable information that can aid in answering the question, providing additional knowledge to the model. To address the potential oversimplification and excessive pruning that can occur with most purely visual token pruning methods, we propose a text information-guided dynamic visual token recovery mechanism that does not require training. This mechanism leverages the similarity between the question text and visual tokens to recover visually meaningful tokens with important text information while merging other less important tokens, to achieve efficient computation for large multimodal models. Experimental results demonstrate that our proposed method achieves comparable performance to the original approach while compressing the visual tokens to an average of **10%** of the original quantity.

# Method
## Overview
To minimize the loss of important information during the token compression process, we propose a text information-guided dynamic visual token recovery mechanism. The framework of this method is illustrated in Figure \ref{fig: overview}. Firstly, the image and the question are separately encoded by visual and text encoders, resulting in visual tokens and text embeddings. Then, these outputs are fed into the token recovery module, which consists of four steps:

<font color=red> **Visual Filter** </font>: Calculate the similarity between the visual class token and other visual tokens, generating visual scores. A dynamic scale filter algorithm is used to determine the threshold for the visual scores, and the top-k tokens based on the threshold are selected as the visual tokens with high scores.

<font color=red> **Text Information Recovery** </font>: Calculate the similarity between the remaining tokens and the text embedding, generating text scores. Similarly, use a dynamic scale filter algorithm to determine the threshold for the text scores, and select the top-k tokens based on the threshold as the text tokens with high scores. This completes the first round of semantic-guided dynamic recovery.

<font color=red> **Secondary Recovery** </font>: For the remaining tokens, apply the KNN to perform clustering and merge each cluster into a single token.

<font color=red> **Token Merger** </font>: Concatenate all the tokens obtained from Steps 1, 2, and 3. It is worth noting that during the training phase, LLMs are trained on input sequences arranged according to the original token order. As a result, the input to LLM is highly sensitive to the sequence order. It is important to keep the original order of tokens when merging them from Steps 1 and 2.

<div align="center">
  <img src="https://github.com/banjiuyufen/images/blob/8eb158b8a4af6ca3eb5dde2819a83fd1a608ed29/Multimodal%20Vision%20Token%20Recycling%20Mechanism%20Guided%20by%20Text%20Information/Overview.png" alt="The approach" width="100%">
</div>

# Experiments
## Performance comparison with other multimodal models and pruning methods.
| Method             | ScienceQA | TextVQA | MME      | VQAv2   | POPE   | MMBench |
|--------------------|-----------|---------|----------|---------|--------|---------|
| BLIP-2             | 61.00     | 42.50   | 1293.80  | 41.00   | 85.30  | -       |
| InstrucBILP        | 60.50     | 50.10   | -        | -       | -      | 36.00   |
| InstrucBILP        | 63.10     | 50.70   | 1212.80  | -       | 78.90  | -       |
| Shikra             | -         | -       | -        | 77.40   | -      | 58.80   |
| IDEFICS-9B         | -         | 25.90   | -        | 50.90   | -      | 48.20   |
| IDEFICS-80B        | -         | 30.90   | -        | 60.00   | -      | 54.50   |
| Qwen-VL            | 67.10     | **63.80** | -      | 78.80   | -      | 38.20   |
| LLaVA-1.5          | **68.40** | 58.20   | **1476.90** | **79.10** | **86.40** | **66.10** |
| *Fine-tuning Method* |           |         |          |         |        |         |
| LLaVA-PruMerge     | 68.50     | 56.00   | 1350.30  | 72.00   | 76.30  | 60.90   |
| LLaVA-PruMerge+    | 68.30     | **57.10** | 1462.40 | 76.80   | **84.00** | **64.90** |
| CrossGET           | 66.70     | 54.90   | **1510.20** | **77.30** | 83.90  | 64.70   |
| Chat-UniVi         | 59.96     | -       | -        | -       | 73.10  | -       |
| Ours               | **68.72** | 56.16   | 1323.54  | 71.18   | 79.50  | 59.20   |
| *Training-Free Method* |         |         |          |         |        |         |
| ToMe               | 50.00     | 45.30   | 1138.00  | 57.10   | 52.50  | 43.70   |
| LLaVA-PruMerge     | 68.52     | 53.51   | 1191.50  | 65.90   | 70.70  | 56.78   |
| Ours               | **69.01** | **55.51** | **1284.90** | **70.41** | **72.00** | **57.90** |

## Comparison of computational costs on NVIDIA A100 GPU.
| Method   | LLM Backbone | Quantization | FLOPs (T) | Prefill Time (ms) | Total Memory (G) | Storing Activation (G) |
|----------|--------------|--------------|-----------|--------------------|-------------------|------------------------|
| LLaVA1.5 | Vicuna-7B   | FP16         | 8.5       | 30.3               | 22.2              | 4.1                    |
| Ours     | Vicuna-7B   | FP16         | **1.5**   | **9.2**            | **14.4**          | **0.49**               |
| LLaVA1.5 | Vicuna-7B   | INT8         | 4.3       | 15.2               | 11.1              | 2.0                    |
| Ours     | Vicuna-7B   | INT8         | **0.8**   | **4.6**            | **7.2**           | **0.24**               |
| LLaVA1.5 | Vicuna-7B   | INT4         | 2.1       | 14.2               | 5.56              | 1.0                    |
| Ours     | Vicuna-7B   | INT4         | **0.4**   | **2.6**            | **3.6**           | **0.12**               |

[banjiuyufen](https://github.com/banjiuyufen) [[Project Page](https://github.com/banjiuyufen/Recoverable-Compression)]





