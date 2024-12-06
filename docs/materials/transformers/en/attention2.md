
<div class="content-2columns" markdown>
![](../assets/imgs/2engines.png){: .rounded-title-img}

# The Complete Transformer Architecture
</div>

In the previous section, we studied the transformer as a system that progressively generates a sequence of tokens. However, the first use of the architecture was for a different case: transforming one sequence of tokens into another to enable machine translation between languages. For this use case, a two-phase system was designed: encoding and decoding, where the transformer from the previous section corresponds to the second phase. In this chapter, we will examine the encoder and its interaction with the decoder. Additionally, it is possible to use only the encoder for other types of tasks, which we will also explore. In all three options, the term *transformer* is used to describe the underlying architecture.

{%
   include-markdown "./assets/mds/texts.md"
   start="<!--nota-inicial-start-->"
   end="<!--nota-inicial-end-->"
%}

## The Encoder-Decoder Model

In the previous section, we studied all the elements of the transformer. However, in some applications where a sequence of tokens is transformed into another sequence of tokens, the transformer layers are associated with two distinct sub-models connected through attention mechanisms: the encoder and the decoder. The chapter [:octicons-book-24:][mt] "[Machine translation][mt]" focuses on the complete architecture. In this chapter, you only need to study sections 10.2 and 10.6 for now.

[mt]: https://web.archive.org/web/20221104204739/https://web.stanford.edu/~jurafsky/slp3/10.pdf

The chapter examines the complete transformer architecture from the perspective of machine translation (the first application for which transformers were used). However, the sections we will focus on are fairly general and do not assume a specific application within the task of transforming an input sequence into an output sequence.

Note that we already saw in the previous section that tasks such as machine translation or summarization can also be tackled using models based solely on the decoder.

## Book Annotations

{%
   include-markdown "./assets/mds/texts.md"
   start="<!--recomendable-start-->"
   end="<!--recomendable-end-->"
%}

Section 10.2
{: .section}

This section is very brief and simply introduces the idea of the encoder-decoder architecture, which is not exclusive to transformers. Notice how the encoder generates an intermediate representation of the input sequence that the decoder uses to guide the generation of the output sequence.

Section 10.6
{: .section}

Another brief section that highlights the only new aspects the complete architecture introduces to what you already know:

- The encoder is not autoregressive (see Figure 10.15), so there is no need to apply masks that block attention to future tokens. Later, we will look at the complete model equations under this approach, but the absence of a mask is the only significant change.
- In *cross-attention*, the keys and values are derived from the embeddings of the encoder's last layer, while the queries are generated in the decoder.

## Search and Subwords

There are additional elements beyond the definition of different neural models that are relevant when applying them in the field of natural language processing.

- Study the beam search mechanism described in [:octicons-book-24:][mt] [Section 10.5][mt].
- Read about subword segmentation in sections [:octicons-book-24:][mt] [10.7.1][mt] and [:octicons-book-24:][cap2] [2.4.3][cap2].

[cap2]: https://web.archive.org/web/20221026071429/https://web.stanford.edu/~jurafsky/slp3/2.pdf

Section 10.5
{: .section}

When generating the output sequence from the successive probability vectors emitted by the decoder, there are many possible approaches. One of the simplest is to select the token with the highest probability at each step. However, as discussed in the book, this strategy is often very inefficient. A slightly more sophisticated approach is beam search, which involves maintaining a set of possible output sequences at each step and selecting the best ones to continue the process.

Sections 2.4.3 and 10.7.1
{: .section}

Most current neural text processing models do not consider each word in the text as a separate input. Instead, they typically divide (*tokenize*) the text into smaller units before processing it through the neural network. Two of the most commonly used algorithms for this task are *BPE* and *SentencePiece*. The first requires the text to be divided into words, while the second processes the text character by character without treating potential separators (such as whitespace) as special. Both ultimately represent very frequent words with a single subword, while less frequent words are represented as a sequence of subwords. When determining the segmentation to use in our model, the developer specifies the maximum number of subwords to include in the problem's vocabulary.

## Pretrained Models

The chapter [:octicons-book-24:][bert] "[Transfer Learning with Contextual Embeddings and Pre-trained language models][bert]" explores pretrained models based on encoders and how to adapt them to our needs. For our study, the introduction and sections 11.1, 11.2, and 11.3 (excluding subsection 11.3.4) are relevant.

[bert]: https://web.archive.org/web/20221026071419/https://web.stanford.edu/~jurafsky/slp3/11.pdf

Introduction
{: .section}

Briefly, the introduction presents the concept of pretrained models and their adjustment (*fine-tuning*) for a specific task.

Section 11.1
{: .section}

This section reviews all the concepts we have seen regarding the transformer's encoder, which can help you fully understand the attention mechanism and transformer architecture. The only difference from the decoder in language models is that the encoder is not autoregressive, so there is no need to apply masks that block attention to future tokens. All input tokens are processed in parallel, and the encoder's output is a set of embedding vectors, one for each input token.

Section 11.2
{: .section}

This section addresses the task of training a transformer-based encoder so that its representations are as general as possible, meaning they are not adapted to a specific problem. To achieve this, the model is trained on a *neutral* task whose resolution nonetheless requires the model to learn to generate deep representations of tokens. This task is *self-supervised* in the sense that no labeled text (e.g., annotated with topic or sentiment) is needed; the plain token sequence is the key ingredient for training. One of the most commonly used tasks is predicting tokens deliberately removed from the input sequence. Once a deep model is trained this way, it can be distributed for other developers to fine-tune for their specific tasks. In the fine-tuning process, only the classifier weights that process the embeddings generated by the pretrained model are updated. One of the first pretrained transformer-based models was BERT, introduced in 2018, which quickly became one of the most widely used pretrained models.

Section 11.3
{: .section}

This section demonstrates how to adapt a pretrained model capable of producing *neutral* embeddings to specific classification tasks. The adaptation process, called *fine-tuning*, involves replacing the predictors in the pretrained model's final layer with a task-specific predictor. The pretrained model remains fixed during fine-tuning, and only the new predictor's weights are updated. If the classification task is at the sentence or sentence-pair level (e.g., determining sentiment or entailment), a predictor can be trained on the [CLS] token. If the task is token-level (e.g., identifying the grammatical category of each token), a predictor can be trained to act on all tokens in the sentence.

## Multilingual Models

So far, in our exploration of transformer-based natural language processing models, we have primarily considered systems that work on a single language or, at least, we have not explicitly considered the possibility of using a single model for multiple languages. However, as we will see in this section, models can handle multiple languages simultaneously, and there are significant advantages to doing so. One of these advantages stems from a phenomenon known as *transfer learning*, where knowledge gained about one language in the representations learned by the neural network can be applied to other languages. This is particularly valuable for *low-resource* languages, that is, languages with fewer training resources, as they may end up benefiting from the knowledge gained by the model from other languages.

Training Multilingual Models
{: .section}

Training a multilingual transformer model is not very different from training a monolingual model. The main difference lies in the training data, which is a mixture of texts in different languages. The tokenizer is created using the entire multilingual corpus with systems such as SentencePiece or byte pair encoding (BPE). If the languages are related, some tokens may be shared, which helps the neural network learn *joint representations*, also known as *cross-lingual embeddings*. For example, consider *centro*, *centru*, *centre*, and *center* in Spanish, Asturian, Catalan, and English, respectively; if these words are tokenized as *centr* and the remaining suffix, the model can learn a representation for *centr* from Spanish, Catalan, or English texts and apply it to Asturian inputs, even if the word *centru* has not been seen in the Asturian training data. Names of people or places can also be easily shared across languages. 

In the common case of data imbalance, where there are significantly more texts in some languages than in others, it is common to use a rebalancing formula to upsample languages with less data, ensuring that their texts are overrepresented in the training or evaluation datasets, as well as in the corpus used to train the tokenizer.

Interestingly, even without shared tokens, words like *atristuráu* (Asturian) and *sad* (English) might end up with similar embeddings in certain layers of the model. This is another example of the language-independent nature of some of the representations learned.

Additionally, depending on the task, language-specific tokens can be used as the first token to indicate the language of the text. This practice is more common in encoder-decoder models (e.g., to indicate the target language) than in decoder-only models. These special tokens are added to the vocabulary, prepended to every sentence during both training and inference, and learned in the same way as the *MASK* token in BERT-like pretraining.

Decoder-like language models are usually trained with multilingual data from the beginning but do not require language-specific tokens. The language used in the prompt is sufficient to guide the generation process toward the desired language.

Pre-trained Multilingual Models
{: .section}

Early pre-trained models, such as BERT, were English-centric. Over time, variants emerged for other languages, like CamemBERT for French, GilBERTo for Italian, or BERTo for Spanish. These models, however, were still essentially monolingual. A turning point came with pre-trained multilingual models like mBERT or XLM-R, which support around a hundred languages, and more recently, Glot500, which extends to 500 languages. These models are self-supervisedly trained with neutral tasks, such as masked language modeling, making them general-purpose models that can be fine-tuned for specific tasks in any of the supported languages. There are also encoder-decoder multilingual models like mBART, mT5 or NLLB-200. More recently, multilingual decoder-only models trained as large language models and covering a wide range of languages have started to emerge, such as EMMA-500, EuroLLM or Aya.

A notable phenomenon here is the *zero-shot generalization* ability of the model. For example, in a named entity recognition task, a model fine-tuned only with English texts can be applied to texts in other languages without requiring labeled data in those languages, thanks to the multilingual representations learned during pretraining. This is particularly useful for low-resource languages.

Challenges
{: .section}

Multilingual models often face the challenge known as *the curse of multilinguality* (a term inspired by the *curse of dimensionality* in statistics and machine learning). This phenomenon refers to a decline in performance for individual languages as the model expands to accommodate a larger number of languages. Languages with more resources often benefit the least from the multilingual model and may even exhibit lower performance compared to a monolingual model trained with the same data.

Several techniques have been proposed to mitigate the curse of multilinguality. One approach is the use of *adapters*, small trainable modules (e.g., a feedforward network), one for each language, inserted at specific points within the transformer layers. During training, common parameters are learned as usual, while adapter parameters are updated only for their associated language. This allows the model to learn a mix of cross-lingual and language-specific parameters. Adapters are also used to fine-tune pre-trained models for new tasks or languages without retraining the entire model. This preserves the original model's weights, updating only the adapter parameters.

Other alternatives for using monolingual models in multilingual scenarios, such as *translate-train* or *translate-test*, are also possible. In the first case, training data available in one language is translated into other languages, and a multilingual model is trained with both the source and translated data. In the second case, test data is translated into the language of a monolingual model at inference time. In both cases, machine translation systems perform the translation.

Finally, several multilingual datasets are available for different tasks, such as the universal dependencies treebanks, the XNLI dataset for natural language inference (determining whether a sentence entails, contradicts, or is neutral toward a hypothesis sentence), or the Seed or FLORES+ corpora for machine translation.

## The Many Faces of Attention

This content is optional. You can skip it unless explicitly instructed to study it. The following analysis is based on the [discussion][dontloo] by *dontloo* on Cross Validated. In original *seq2seq* systems based on recurrent networks, attention is computed as a weighted average of the keys:

[dontloo]: https://stats.stackexchange.com/a/424127/240809

$$
\mathbf{c} = \sum_{j} \alpha_j \mathbf{h}_j   \qquad \mathrm{with} \,\, \sum_j \alpha_j = 1
$$

If $\alpha$ were a one-hot vector, attention would reduce to retrieving a specific element among the different $\mathbf{h}_j$ based on its index. However, $\alpha$ is unlikely to be a one-hot vector, so it represents a weighted retrieval. In this case, $\mathbf{c}$ can be considered the resulting value.

There is a significant difference in how this weight vector (summing to 1) is obtained in *seq2seq* architectures compared to transformers. In *seq2seq*, a feedforward network, represented by the function $a$, determines the *compatibility* between the $i$-th token's representation in the decoder, $\mathbf{s}_i$, and the $j$-th token's representation in the encoder, $\mathbf{h}_j$:

$$
e_{ij} = a(\mathbf{s}_i,\mathbf{h}_j)
$$

From this:

$$
\alpha_{ij} = \frac{\mathrm{exp}(e_{ij})}{\sum_k \mathrm{exp}(e_{ik})}
$$

Suppose the input sequence length is $m$ and the output sequence generated so far is $n$. A drawback of this approach is that at every decoder step, the feedforward network $a$ must be executed $mn$ times to compute all $e_{ij}$ values.

A more efficient strategy involves projecting $\mathbf{s}_i$ and $\mathbf{h}_j$ into a common space (e.g., via linear transformations $f$ and $g$) and using a similarity measure (such as the dot product) to compute $e_{ij}$:

$$
e_{ij} = f(\mathbf{s}_i) \cdot g(\mathbf{h}_j)^T
$$

Here, $f(\mathbf{s}_i)$ is the decoder's query, and $g(\mathbf{h}_j)$ is the encoder's key. This reduces the complexity to $m+n$, requiring only $n$ calls to $f$ and $m$ calls to $g$. Additionally, $e_{ij}$ can now be efficiently computed via matrix multiplication.

The transformer's attention mechanism establishes the conditions for query and key projections and similarity calculations. When attention operates within vectors of the same origin (e.g., within the encoder), it is called *self-attention*. The transformer combines separate self-attention in the encoder and decoder with another *heterogeneous* attention mechanism, where $Q$ comes from the decoder, and $K$ and $V$ come from the encoder.

The basic self-attention equation for the transformer is:

$$
\mathrm{SelfAttn}(Q,K,V) = \mathrm{softmax} \left( \frac{QK^T}{\sqrt{d_k}} \right) V
$$

In practice, the transformer's final equations are slightly more complex due to the inclusion of multiple attention heads.

## Review Exercises

1. Explain how a mask $M$, similar to the one used to block attention to future tokens, could be applied to mask padding tokens in a mini-batch of training sentences.

2. An attention matrix represents, in each row, the level of attention given to the embeddings calculated for each token of the sentence in the previous layer (represented in the columns) when calculating the embedding of the token in that row for a specific head of a transformer's encoder. A darker color represents higher attention. Now, consider a head of a transformer's encoder that applies approximately monotonic attention to the embeddings from the previous layer. Specifically, for each token, a high degree of attention is given to the embedding of that same token in the previous layer, with much less attention to the token immediately to its right; the rest of the embeddings from the previous layer receive no attention. Draw the resulting attention matrix approximately.
