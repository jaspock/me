
<div class="content-2columns" markdown>
![](../assets/imgs/robot-eureka.png){: .rounded-title-img}

# Problems
</div>

## Regressor Problems

Problem
{: .problema}

The arrangement of elements in matrices and vectors may differ from that used in section 5.2.3. The key point is that the scalar product of each of the $m$ samples with the weight vector $\mathbf{w}$ is computed. Indicate the sizes that the matrices and vectors should have if, instead of an equation like (5.14), we use one of the form $\mathbf{y} = \mathbf{w} \mathbf{X} + \mathbf{b}$.

Problem
{: .problema}

Calculate the derivative of the cost function with respect to the threshold $b$. If you base it on the derivative of the cost function with respect to the weights $w$, which is calculated in the book, you will quickly arrive at the solution.

Problem
{: .problema}

After training a logistic regressor, we apply an input $\mathbf{x}$ and calculate the derivative $\partial \hat{y} / \partial \mathbf{x}_i$ for a certain $i$. What does this derivative measure? Think about the basic concept of a derivative and how it measures the *sensitivity* of the value of a function concerning a change in one of its variables.

## Embedding Problems

Problem
{: .problema}

When searching for analogy relationships between word embeddings, we attempt, given words A and B (related to each other), to find in the embedding space two words, C and D (also related to each other and to A and B) for which it makes sense to state that "A is to B as C is to D." Mathematically, this involves finding four words whose embeddings satisfy certain geometric properties based on the distances between their vectors. For example, it is common to observe that these properties hold if A=man, B=woman, C=king, and D=queen (we say "man is to woman as king is to queen"). Suppose we have obtained word embeddings for English using the skip-gram algorithm, allowing such analogies. Consider the list of words L = {planes, France, Louvre, UK, Italy, plane, people, man, woman, pilot, cloud}. Indicate which word from list L makes the most sense to replace @1, @2, @3, @4, @5, and @6 in the following expressions to satisfy the analogies:

- A=Paris, B=@1, C=London, D=@2
- A=plane, B=@3, C=person, D=@4
- A=mother, B=father, C=@5, D=@6

## Transformer Problems

Problem
{: .problema}

Argue the truth or falsehood of the following statement: if the query of a token $q_i$ equals its key $k_i$, then the embedding computed by the self-attention mechanism for that token matches its value $v_i$.

Problem
{: .problema}

The following formula for calculating self-attention in a transformer's decoder is slightly different from the usual one, as it explicitly includes the mask that prevents the attention computed for a token during training from considering tokens that appear later in the sentence:

$$
\text{Attention}(Q,K,V,M) = \text{softmax} \left( \frac{Q K^T + M}{\sqrt{d_k}} \right) \, V
$$

Describe the shape of the mask matrix $M$. Find the PyTorch operations that allow you to initialize this matrix. If you need to use infinite values in the code, argue whether it is sufficient to use a large number like $10^9$.

Problem
{: .problema}

Given a sequence of input tokens, the transformer provides a set of embeddings for each input token. These embeddings are contextual in all layers except one. In which layer are the embeddings not contextual? How are the final values of these embeddings obtained? And what about the initial values?

Problem
{: .problema}

Suppose the non-contextual embedding stored in a transformer's embedding table for a given token is defined by the vector $\mathbf{e} = \mathbf{e}_1, \mathbf{e}_2, \ldots, \mathbf{e}_n$. Consider the case where this token appears in two different positions of an input sentence. If positional embeddings were not used, what would be the cosine value between the vectors of the two non-contextual embeddings used for the token? Let $\mathbf{p}$ and $\mathbf{p}’$ be the positional embeddings used in each of the two token occurrences. The cosine of the angle between the vectors resulting from adding the positional embeddings to the token embedding would be:

$$
\cos(\alpha)=\frac{\sum_{i=1}^d(\mathbf{e}_i+\mathbf{p}_i)(\mathbf{e}_i+\mathbf{p}'_i)}{\sqrt{\left(\sum_{j=1}^d(\mathbf{e}_j+\mathbf{p}_j)^2\right)\left(\sum_{j=1}^d(\mathbf{e}_j+\mathbf{p}_j')^2\right)}}
$$

What value will the above cosine approach as the distance between the two token occurrences in the sentence increases? Explain your reasoning.
