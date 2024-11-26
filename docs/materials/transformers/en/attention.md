
<div class="content-2columns" markdown>
![](../assets/imgs/engine-words.png){: .rounded-title-img}

# Transformers and Attention Models: Understanding Attention Mechanisms and Their Application in the Transformer Model
</div>

In 2017, recurrent neural networks based on LSTM units were the standard architecture for neural sequence processing, in general, and natural language processing, in particular. Some researchers were also achieving good results in this area with convolutional neural networks, which were traditionally used for image processing. On the other hand, attention mechanisms introduced a few years earlier in recurrent networks had enhanced their ability to solve certain tasks and expanded the range of possibilities for these models. Additionally, the encoder-decoder model became the cornerstone for systems that transformed one sequence into another (seq2seq systems, such as machine translation or summarization systems). However, in mid-2017, a paper titled "[Attention Is All You Need](https://arxiv.org/abs/1706.03762)" proposed removing recurrence from the encoder-decoder model and replacing it with what is known as self-attention. Although the paper focused on machine translation, it quickly became evident that this architecture, dubbed the *transformer*, was highly effective in many other fields, relegating recurrent architectures to a secondary role. Furthermore, the transformer became one of the foundational elements of pre-trained models, which we will study later and which began to emerge in the subsequent months or years. Nonetheless, there are still researchers working with recurrent networks, so it cannot be ruled out that they may regain prominence in the future.

{%
   include-markdown "../assets/mds/texts.md"
   start="<!--nota-inicial-start-->"
   end="<!--nota-inicial-end-->"
%}

## Transformer Fundamentals

We will study the basic elements of the transformer architecture following Chapter [:octicons-book-24:][basictransformer] "[Deep Learning Architectures for Sequence Processing][basictransformer]". Here, you will understand one of the most important equations in recent years in machine learning:

$$
\text{Attention}(Q,K,V) = \text{softmax} \left( \frac{Q K^T}{\sqrt{d_k}} \right) \, V
$$

Skip sections 9.2 to 9.6, which focus on an alternative model for sequence processing, recurrent neural networks, which have been used less in natural language processing after the advent of the transformer.

[basictransformer]: https://web.archive.org/web/20221218211150/https://web.stanford.edu/~jurafsky/slp3/9.pdf

## Book Annotations

{%
   include-markdown "../assets/mds/texts.md"
   start="<!--recomendable-start-->"
   end="<!--recomendable-end-->"
%}

Essentially, the transformer is an architecture that enables working with sequences of different natures. When we use it to analyze or generate sentences in natural language, the sequences will be sentences formed by word tokens. As we already know, to have "word calculators," these tokens must be represented numerically. Deep representations in the form of embedding vectors that capture certain underlying properties of words are particularly useful. In fact, the transformer is nothing more than a machine for calculating contextual word embeddings, and this contextuality differentiates them from algorithms like skip-grams, which provided a unique representation for each word. Transformers begin with a non-contextual representation of the words at their input and refine it layer by layer. From the output of the first layer, the representations become contextual, so the representation of *station* in "Summer is my favorite station" is different from *station* in "The train station is around the corner," and even different from "Winter is my favorite station." As we move through the layers, the representations not only become more contextual but also adapt to the specific task to be solved with the model's output in its final layer. The final layer is usually a classifier, often implemented through a dense layer (instrumented, as we’ve seen, via a matrix) that generates a logits vector, which is then transformed into probabilities using a softmax activation function. Some possible operations with the transformer are:

- Generate as many probability vectors at the output as there are tokens in the input, where each corresponds to the probability that the respective input token belongs to a certain class. This way, for example, the model output can determine if each token is a proper noun, a task useful for preventing machine translation systems from translating names when inappropriate.
- Combine all the output embeddings into a single vector (e.g., by calculating their mean) and use the resulting embedding as a representation of the entire sentence. Passing this embedding through a dense layer and a softmax activation function, we can obtain the probability that the sentence belongs to a certain class. For example, this can determine whether a sentence expresses a positive, negative, or neutral sentiment about a topic.
- An alternative approach for sentence-level classification tasks is to add a *special* token (traditionally represented as `CLS`) to the beginning or end of the sentence and use it as an additional input to the transformer. The transformer's output for this token will be the representation of the entire sentence, which we provide to the classifier.
- Train the model so that the probability vector output for a specific token indicates the likelihood of each vocabulary word being the next token in the sentence. For example, after training, the model output can generate text from a given prefix, resulting in a generative language model capable of engaging in dialogue, answering questions, translating text, or summarizing documents.

In its most general form, the transformer consists of two main modules: an encoder and a decoder. The encoder generates deep representations of the input tokens, which the decoder uses to generate the output sequence. When classifying input text or labeling its tokens, using the encoder alone may suffice. For tasks like transforming one text into another (e.g., machine translation or summarization), both modules have traditionally been used. For generative language models, the decoder is used, generating the output sequence token by token in an *autoregressive* manner. However, there has been a convergence towards models integrating both the encoder and decoder into a single module. Many people refer to all these options as "transformers," while others reserve the term "transformer" for the model integrating both modules and use terms like "encoder-like transformer" or "decoder-like transformer" for models that only use one module.

One of the significant advantages of transformers compared to other models, such as feedforward neural networks, is their ability to process sequences of variable length. They do not learn different parameters for processing the first word, the second, the third, and so on. Instead, transformers learn a single transformation in each layer that applies to all input tokens. Since each token is represented by a different embedding at the model's input, the transformation yields different results (query, key, and value) for each token.

Section 9.7 of this chapter focuses on the transformer as a generative language model and, therefore, on the decoder. However, the encoder functions in a largely similar manner, as we will see later.

Section 9.7
{: .section}

The section begins by introducing the concept of attention: the embedding of a token is refined by *mixing* it with the embeddings of other tokens in the input. In this way, a sequence of $n$ embeddings $\mathbf{x}_1,\ldots,\mathbf{x}_n$ is transformed into a sequence $\mathbf{y}_1,\ldots,\mathbf{y}_n$, where each $\mathbf{y}_i$ is a different *cocktail* of the input embeddings. Intuitively, an embedding should mix more with those that represent words it is related to. For instance, in the sentence "The woman started the journey tired," the embedding for "tired" should mix more with "woman" than with "journey," as the word describes the woman, not the journey. From our earlier study of non-contextual embeddings, we know that one way to measure the similarity of two vectors is through the dot product. Thus, as a first approach, we could calculate the dot product between the embedding of a token and that of each input token, and use the result as the argument for the softmax function, which will indicate the degree of mixing of the embeddings. Note that if an embedding is allowed to mix with itself, the softmax function will assign a significant portion of the mix to itself.

Although the previous way of combining embeddings can be useful in certain contexts, the transformer does so in a slightly more complex manner. Each input embedding is transformed into three vectors:

- The query defines *what* the token seeks in other tokens.
- The key indicates *how* each token defines itself.
- The value specifies *what* each token offers to other tokens.

A common analogy to explain this form of attention is a dating app, where people indicate what they are looking for (*query*) and how they define themselves (*key*). Once the affinity between each individual and the others is determined (via the dot product), the genes (*values*) of each person are combined to create a new embedding. This process is repeated for each token, resulting in $n^2$ dot products. Here, unlike dating apps, a token can relate to itself to a certain degree, and the new embeddings usually result from combining the genes of more than two individuals. Another analogy that helps explain attention is based on accessing a programming language dictionary, which you can consult [below][dictionary].

[dictionary]: #an-analogy-for-the-self-attention-mechanism

Interestingly, the transformation of a vector $\mathbf{x}_i$ into three $d$-dimensional vectors is performed using three linear transformations $\mathbf{W}^Q$, $\mathbf{W}^K$, and $\mathbf{W}^V$, applied to all input tokens. Thus, the number of parameters to learn is much smaller than if we applied a separate linear transformation to each input token. Additionally, since these linear transformations are independent of each other, they can be parallelized, thereby accelerating computation. Furthermore, as shown in the book, the different matrix products of the embeddings representing all the sentence's words can also be computed in parallel if the input embeddings are arranged *row-wise* in a matrix $X$. In fact, as we’ll see in the implementation, we can group the embeddings of all sentences in a mini-batch into a single tensor (equivalent to a vector of $X$ matrices) to further parallelize the calculations. GPUs are optimized to efficiently perform these types of matrix operations.

A noteworthy detail is the division by $\sqrt{d_k}$ in the denominator of the attention formula. This division avoids excessively high dot product values from dominating the probability distribution generated by the softmax. A [special section][division] below examines this issue in more detail.

[division]: #scaled-attention

A key focus of this section is the autoregressive approach of the transformer, i.e., its use as a generative model. During the inference stage (also called *usage*), the model generates the probability vector for the next token based on the current token. In this case, we can choose the token with the highest probability and feed it back into the input. It’s as if we were asking the system to respond to this request: given the prefix you’ve seen so far, if I add this new token, give me the probabilities for the next tokens. Choosing the token with the highest probability is just one way to use the probability vector, but there are many others. Later, we’ll explore the technique known as *beam search*, but a simpler option is to sample the next token based on the probabilities (the more likely a token, the higher its probability of being selected), so the model may generate a different sequence in each run. This explains why generative models sometimes produce different sequences for the same prefix (*prompt*).

The autoregressive use of the transformer implies that during training, the model must operate in a manner similar to its intended usage during inference. To emulate this behavior, the attention mechanism cannot consider information from subsequent tokens. Otherwise, the learning algorithm might easily focus on the next token when generating the probability vector, which would have severe negative effects during inference when this information is unavailable. This explains the mask shown in Figure 9.17.

Later, we’ll see scenarios where it’s beneficial for all tokens in a sentence to mix with each other, not just with preceding ones. For instance, when classifying the topic of input texts, all tokens are available from the start, so it’s unnecessary to disregard information from subsequent tokens when preparing the embedding cocktail. The encoder-based transformer version, which we’ll discuss later, enables this. However, this section focuses on the transformer as a decoder, used to autoregressively generate output sequences.

### Subsection 9.7.1
{: .section}

After examining the fundamentals of the self-attention mechanism, this subsection analyzes other complementary elements present in each transformer layer: the feedforward network, residual connections, and layer normalization.

The feedforward network typically has a single hidden layer with a nonlinear activation function.

Residual connections help partially circumvent the *vanishing gradient* problem in multi-layer architectures. This problem arises because the lower layers of deep networks receive very weak feedback signals during training, hindering error propagation and model convergence. To better understand this, think of a transformer as a successive composition of functions from the input layer to the output (each layer’s output depends on the operations performed on the outputs of the previous layer). As a result, the error function’s derivative with respect to parameters in the initial layers comprises numerous products (via the chain rule), which can easily become zero or extremely large, leading to very small or very large updates in stochastic gradient descent. With residual connections, the error function’s derivative splits into two paths (the derivative of a sum is the sum of the derivatives): one following the conventional route and the other following the residual connection route. Theoretically, part of the error now has the ability to reach any point in the network, no matter how far it is from the output layer, by bypassing entire modules through the residual connections.

Layer normalization also helps prevent excessively large or small intermediate outputs, which often negatively affect training (e.g., pushing activation functions into flat regions without gradients or causing most of the probability to concentrate on one or a few elements after the softmax function). You can find a more detailed analysis of this [normalization][normalization] in this guide.

[normalization]: #layer-normalization

### Subsection 9.7.2
{: .section}

Since the tokens in a text are related in different ways, the transformer replicates the attention mechanism in each layer by applying multiple attention heads. For example, some heads may focus on nearby tokens, while others may concentrate on tokens with specific syntactic or semantic relationships, regardless of their distance. Conceptually, multiple heads add little novelty compared to the attention mechanism already described. In any case, since subsequent elements expect to receive a vector of a given size, the outputs of the different heads are concatenated to form a single output vector, which is then passed through a linear projection to both standardize the representations from each head and obtain a vector of the desired size, if necessary.

Note that now each head has its own parameter matrices $\mathbf{W}^Q_i$, $\mathbf{W}^K_i$, and $\mathbf{W}^V_i$.

Subsection 9.7.3
{: .section}

Until now, we’ve presented as a positive feature the fact that the matrices used in each layer are the same for all input tokens (knowing, of course, that if there’s more than one head, these matrices are different). This reduces the number of parameters compared to, for instance, a feedforward network, which would require different parameters for each input token, likely necessitating a small token window at each step. However, this poses a problem since (ensure you understand why) the same embeddings would be computed for sentences like "Dog bites man" and "Man bites dog." Nevertheless, it’s easy to deduce that the role and semantic properties of each word differ significantly in both sentences. Therefore, before the first layer, a *positional embedding* vector is added to each token’s non-contextual embedding. Ideally, these positional embeddings meet several criteria:

- They do not repeat in different positions.
- They are more similar for tokens in closer positions.

Various approaches to assigning positional embedding values have been proposed. The original transformer paper used a fixed encoding for each position based on a series of sinusoidal functions with different frequencies at each position of the vector. However, it soon became apparent that the learning algorithm could also determine appropriate values for them. If training inputs have a bounded maximum length but inference sequences may grow arbitrarily, a combination of learned embeddings (for initial positions) and fixed embeddings (for subsequent ones) can be used.

Section 9.8
{: .section}

This section provides an example of using the transformer as a language model. In each step, one more token is processed at the input, and a probability vector is generated for the next token. A greedy strategy, for example, can be used to select the winning token as the next token for the input.

Section 9.9
{: .section}

Notice that the previous language model example is quite limited because the model is asked to generate text without providing any *seed*. If, as mentioned, we sample possible words from the probability distribution, we can generate more than one sequence. If we simply select the token with the highest probability, the generated sequence would be unique.

This section demonstrates how a prefix (*prompt*) can be provided to the model, asking it to continue the sequence. This is the underlying idea of recent language models (GPT, PaLM, LLaMA, etc.) that have shown the ability to generate text, sustain conversations, and present arguments of surprising quality. These models have been trained to generate the next token in a sequence but include additional training phases that generally consist of:

- *Fine-tuning* the weights with text datasets formed by questions and answers, enabling the system to handle dialogues effectively;
- Fine-tuning based on human feedback about the quality of the generated responses, where evaluators rank different responses to the same question by quality (truthful and polite answers receive the highest scores). Since there is no differentiable loss function in this case, reinforcement learning techniques are used to adjust the weights according to alternative policies.

Observe how the autoregressive architecture we have studied can also be used for summarizing or translating text into other languages. In these cases, the prefix consists of the text to be summarized or translated, respectively. A special token indicates to the model that the prefix has ended and that it should start generating the output text.

Finally, the representations learned by a transformer during training for each of its layers on new input phrases can be considered contextual embeddings of the different input tokens. These embeddings can be very useful in various natural language processing tasks. In principle, any layer can be suitable for obtaining these representations, but some studies have shown that certain layers are better suited for specific tasks. Layers closer to the input seem to capture more morphological information, while final layers are more related to semantics.

## A Self-Attention Analogy

The self-attention mechanism can be introduced for educational purposes based on a hypothetical version of Python that allows accessing dictionary values using *approximate* keys. Consider the following Python dictionary stored in the variable `d`; like any Python dictionary, it contains a set of keys (e.g., `apple`) and associated values (e.g., `8` is the value associated with the key `apple`):

```python
d = {"apple":8, "apricot":4, "orange":3}
```

In *conventional* Python, we can now perform a *query* on the dictionary using syntax like `d["apple"]` to retrieve the value `8`. The Python interpreter uses our query term (`apple`) to search among all the dictionary keys for an *exact* match and returns its value (`8` in this case).

Notice how in the discussion above, we used the terms "query," "key," and "value," which also appear in discussions of the transformer’s self-attention mechanism.

Now, imagine querying `d["orangicot"]`. A real Python interpreter would throw an exception for the above query, but an *imaginary* interpreter could traverse the dictionary, compare the query term to each dictionary key, and weight the values based on the similarity found. Consider a function `similarity` that takes two strings and returns a number, not necessarily bounded, that is larger for more similar strings (the exact values are not relevant here):

```
similarity("orangicot","apple") → 0
similarity("orangicot","apricot") → 20
similarity("orangicot","orange") → 30
```

These results, normalized so their sum is 1, are `0`, `0.4`, and `0.6`. Our imaginary Python interpreter could now return for the query `d["orangicot"]` the value 0 x 8 + 0.4 x 4 + 0.6 x 3 = 3.4.

In the case of the transformer, queries, keys, and values are vectors of a certain dimension, and the similarity function used is the dot product of the query and the different keys. The similarity scores are normalized using the softmax function and then used to weight the different values:

$$
\text{SelfAtt}(Q,K,V) = \text{softmax}\left( \frac{Q K^T}{\sqrt{d_k}} \right) V
$$

## Visual Guides

Jay Alammar published a well-known series of articles illustrating the functioning of transformers in a highly visual and educational way. These resources can help solidify concepts:

- "[The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/)"
- "[Visualizing A Neural Machine Translation Model](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)"
- "[The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)"

## Scaled Attention

One potentially arbitrary factor in the attention equation is the division by the square root of the key dimension. To understand the motivation for this operation, note that as the embedding size increases, the result of each dot product $q_i k_j$ also increases. The problem is that when the softmax function is applied to very large values, its exponential nature assigns very small values to all elements except the largest one. That is, when the softmax function saturates, it tends toward a *one-hot* vector. This causes attention to focus on a single token while ignoring the rest, which is not a desirable behavior.

Assume $Q$ and $K$ have size $B \times T \times C$, where $C$ is the size of the queries and keys. For simplicity, if we assume the elements of $Q$ and $K$ have variance around 1, the variance of the product will be approximately $C$. As it is true that, given a scalar $m$, $\mathrm{var}(mX) = m^2 \mathrm{var}(x)$, multiplying each element by $1 / \sqrt{C}$ reduces the product's variance by $\left(1 / \sqrt{C}\right)^2 = 1 / C$. Therefore, if the variance of $Q$ and $K$ elements is 1, the variance of the product will also be around 1.

The following code illustrates the above points:

```python
import torch

B, T, C = 10, 10, 5000
m = 1
Q = torch.randn(B, T, C)*m
K = torch.randn(B, T, C)*m

print(f'Variance of Q: {Q.var().item():.2f}')
print(f'Variance of K: {K.var().item():.2f}')
# variances close to m**2

QK = Q.matmul(K.transpose(-2, -1))

print(f'Variance of QK: {QK.var().item():.2f}') 
# very high variance close to C*(m**4)!

s = torch.softmax(QK, dim=-1)
torch.set_printoptions(precision=2)
print(f'Mean value of highest softmax: {s.max(dim=-1)[0].mean().item():.2f}') 
# max value of each channel close to 1

QK = QK / (C ** 0.5)

print(f'Variance of QK after normalization: {QK.var().item():.2f}')
# variance close to m**4

s = torch.softmax(QK, dim=-1)
print(f'Mean value of highest softmax: {s.max(dim=-1)[0].mean().item():.2f}') 
# max value of each channel smaller than 1
```

In general, if the variance of $Q$ and $K$ elements is $m$, the variance of the product will be approximately $m^4 C$. If $m=2$, for example, normalization doesn’t reduce the variance to 1, but it does reduce it to around $m^4 = 16$.

## Layer Normalization

Let $\hat{\mu}$ and $\hat{\sigma}^2$ be the mean and variance, respectively, of all inputs, represented by $\boldsymbol{x}$, to the neurons in a layer with $H$ neurons:

$$
\begin{align}
\hat{\mu} &= \frac{1}{H} \sum_{i=1}^H x_i \\
\hat{\sigma}^2 &= \frac{1}{H} \sum_{i=1}^H \left(x_i - \hat{\mu} \right)^2 + \epsilon
\end{align}
$$

where $\epsilon$ is a very small value to avoid division by zero in the next equation. The LN normalization function for each input to the layer is defined as standardization:

$$
\text{LN}(x_i) = \gamma_i \frac{x_i - \hat{\mu}}{\hat{\sigma}^2} + \beta
$$

The fraction ensures that all inputs to the layer at a given moment have a mean of zero and a variance of 1. Since these values are arbitrary, learnable parameters $\boldsymbol{\gamma}$ and $\boldsymbol{\beta}$ are added to rescale them. The normalized values become the new inputs to each neuron, and the corresponding activation function is applied to them. In the case of the transformer, there is no additional activation function.

The PyTorch code for layer normalization is quite straightforward:

```python
class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta
```
