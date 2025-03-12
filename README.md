# Recoverable-Compression-A-Multimodal-Vision-Token-Recovery-Mechanism-Guided-by-Text-Information
The code of  "Recoverable-Compression: A Multimodal Vision Token Recovery Mechanism Guided by Text Information"

# Abstract
With the advancement of large-scale language modeling techniques, large multimodal models combining visual encoders with large language models have demonstrated exceptional performance in various visual tasks. Most of the current large multimodal models achieve this by mapping visual features obtained from the visual encoder into a large language model and using them as inputs alongside text for downstream tasks. Therefore, the number of visual tokens directly affects the training and inference speed of the model. There has been significant work on token pruning for visual transformers, but for large multimodal models, only relying on visual information for token pruning or compression may lead to significant loss of important information. On the other hand, the textual input in the form of a question may contain valuable information that can aid in answering the question, providing additional knowledge to the model. To address the potential oversimplification and excessive pruning that can occur with most purely visual token pruning methods, we propose a text information-guided dynamic visual token recovery mechanism that does not require training. This mechanism leverages the similarity between the question text and visual tokens to recover visually meaningful tokens with important text information while merging other less important tokens, to achieve efficient computation for large multimodal models. Experimental results demonstrate that our proposed method achieves comparable performance to the original approach while compressing the visual tokens to an average of **10%** of the original quantity.

# Method
## Overview
To minimize the loss of important information during the token compression process, we propose a text information-guided dynamic visual token recovery mechanism. The framework of this method is illustrated in Figure \ref{fig: overview}. Firstly, the image and the question are separately encoded by visual and text encoders, resulting in visual tokens and text embeddings. Then, these outputs are fed into the token recovery module, which consists of four steps:

**Visual Filter** Calculate the similarity between the visual class token and other visual tokens, generating visual scores. A dynamic scale filter algorithm is used to determine the threshold for the visual scores, and the top-k tokens based on the threshold are selected as the visual tokens with high scores.

**Text Information Recovery** Calculate the similarity between the remaining tokens and the text embedding, generating text scores. Similarly, use a dynamic scale filter algorithm to determine the threshold for the text scores, and select the top-k tokens based on the threshold as the text tokens with high scores. This completes the first round of semantic-guided dynamic recovery.

**Secondary Recovery** For the remaining tokens, apply the KNN to perform clustering and merge each cluster into a single token.

**Token Merger** Concatenate all the tokens obtained from Steps 1, 2, and 3. It is worth noting that during the training phase, LLMs are trained on input sequences arranged according to the original token order. As a result, the input to LLM is highly sensitive to the sequence order. It is important to keep the original order of tokens when merging them from Steps 1 and 2.


<div align="center">
  <img src="https://github.com/banjiuyufen/images/blob/8eb158b8a4af6ca3eb5dde2819a83fd1a608ed29/Multimodal%20Vision%20Token%20Recycling%20Mechanism%20Guided%20by%20Text%20Information/Overview.png" alt="The approach" width="100%">
</div>

[banjiuyufen](https://github.com/banjiuyufen) [[Project Page](https://github.com/banjiuyufen/Recoverable-Compression)]





