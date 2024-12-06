
<div class="content-2columns" markdown>
![](../assets/imgs/math-triangle.png){: .rounded-title-img}

# Feedforward Networks: The Basic Neural Computation Model
</div>

Feedforward neural networks can be seen as a simple machine learning model. Although we will not use them independently, they are a necessary component of the more advanced models we will explore later.

{%
   include-markdown "./assets/mds/texts.md"
   start="<!--nota-inicial-start-->"
   end="<!--nota-inicial-end-->"
%}

## Feedforward Networks

After the previous steps, we are now ready to study the basic neural network model known as the feedforward neural network. They are discussed in the chapter [:octicons-book-24:][neural] "[Neural Networks and Neural Language Models][neural]" of the book. While nowadays such simple architectures alone are not used in many natural language processing tasks, feedforward networks appear as components in advanced models based on the transformer architecture, for example, which we will explore later.

The entire chapter is relevant, although it might cover topics you already know. You may choose to either skip or skim through sections 7.6.3 ("Computation Graphs") and 7.6.4 ("Backward differentiation on computation graphs").

[neural]: https://web.archive.org/web/20221218211150/https://web.stanford.edu/~jurafsky/slp3/7.pdf


## Book Annotations

This chapter should not pose major challenges if you have understood the concepts from previous chapters well, especially if you have already studied neural networks in another course. In any case, here are some notes that may help you better understand the chapter.

The basic ideas of regressors are maintained in feedforward networks studied in this chapter. A neural network receives a series of numbers as input and produces one or more numbers as output. The inputs are transformed into outputs through a series of mathematical operations that can be expressed as matrix products, along with the additional application of nonlinear functions like sigmoid or softmax. If the inputs represent words, we know they can be numerically represented using embeddings. Thus, one way to enable the neural network to process, for instance, three words at a time would be to represent each word using its embedding (learned through an algorithm like skip-grams, as we saw before) and concatenate the embeddings to form a single vector, which would serve as the neural network's input. At the network's output, we could use a sigmoid-like function if we are interested in a binary classifier, e.g., to output a value between zero and one representing the positivity of a text. If we want the network to classify among several categories, we can use a softmax function to obtain a probability distribution over them.

This last interpretation of the output opens up an interesting possibility: using neural networks as a *language model*, i.e., a probabilistic model that, given a particular text, can predict the probability of each word from a list being the next in the sequence. Once trained, such a network would be very useful for generating a continuation to a given prefix. Suppose I have the prefix "When I look out the window, I can see the" and concatenate the embeddings of the last three words (*can*, *see*, *the*) to provide as input to the neural network. The network would output a probability vector with as many elements as we decided to have in the vocabulary. Suppose the highest-probability output corresponds to the word "sea." Now I can input the embeddings for *see*, *the*, and *sea* into the network to predict the next word, and so on. This way, the given prefix might generate a sequence like "When I look out the window, I can see the sea in front of me." Ideally, if the prefix is a question, the neural network should be able to answer it, as current large language models do. These types of *emergent* behaviors can be achieved with neural models, but we will need to go beyond the neural networks in this chapter to achieve them.

Why are feedforward networks not suitable for such tasks? You may have noticed that a major limitation of the above approach is the choice of three words as input to the network. While in the context of the full prefix, it might make sense for the word "sea" to have a high probability of appearing after "can see the," feedforward networks lack any memory to *recall* the rest of the prefix, so they are more likely to assign higher probabilities to unrelated words like *problem* or *case* rather than to *sea* or other contextually relevant words. We could imagine increasing the number of concatenated embeddings at the input to expand the context the system can consider when making predictions. And this is true to some extent. However, increasing the input size means increasing the number of weights to learn. In machine learning, the so-called *curse of dimensionality* is well-known, which implies that as the number of parameters to learn grows, the amount of data needed to do so grows exponentially. Additionally, there are other issues we will not delve into here, such as the parameters learned for a word in one position not being reusable when the word appears in other positions or having to decide a maximum length for sequences to process. The transformer, which we will see later, overcomes many of these problems. But before we get there, let's explore how a simpler neural network works.

Note that in all the above discussion there is nothing new compared to the regressors we already know. Where does the computational superiority of feedforward networks come from? The answer lies in the fact that these networks introduce additional layers of processing. While a multinomial regressor simply multiplies the input by a matrix and applies softmax to the result, a neural network multiplies the input by a weight matrix, applies a typically nonlinear function (e.g., a sigmoid) to the resulting vector, multiplies the new vector by another weight matrix, applies a new function (called an *activation function*), and so on. This process continues through several layers until producing the final output vector.

The remaining elements related to feedforward networks are also inherited from models like the regressor. Thus, the goal is to maximize the likelihood of the data using stochastic gradient descent. For this, cross-entropy is used as a loss function to determine the magnitude of the *adjustment* applied to the weights at each training step. Similarly, samples do not need to be processed one at a time; they can be grouped into batches (or *mini-batches*). In this case, the error is calculated as the average error for each sample in the batch.

Finally, while we mentioned that embeddings for each word could have been pre-learned before training the neural network, in practice, these are usually initialized randomly and learned along with the network's other parameters. The mechanism for this is already known: the derivative of the error function with respect to each component of the embeddings is computed, allowing them to be updated at each training step.

Section 7.1
{: .section}

This section may seem different, but it merely reiterates what you already know about binary logistic regressors. Since these processing units will generally be used not only in the final layer of the network but also in the intermediate layers, other activation functions (such as ReLU) are presented in addition to the sigmoid. The choice of one function over another is typically empirical and based on the specific problem or neural architecture.

Section 7.2
{: .section}

This section justifies the need for multiple processing layers from a computational perspective, demonstrating that it is impossible to classify data as simple as that of the exclusive-or function with a linear regressor.

Section 7.3
{: .section}

Here, we finally have a model that could be used for the next-word prediction task discussed above. The model takes an input vector and produces a probability vector through the following equations, which define the two layers of the model:

$$
  \begin{align} 
  \mathbf{h} &= \sigma(\mathbf{W} \mathbf{x} + \mathbf{b}) \nonumber \\ 
  \mathbf{y} &= \mathrm{softmax}(\mathbf{U} \mathbf{h}) \nonumber
  \end{align}
$$

Note that the input to the softmax function could also include the addition of a second bias vector, but this has been omitted in equation (7.12) for simplicity, though it reappears later. It is quite interesting to understand the reasoning presented at the end of this section about the necessity of nonlinear activation functions, as without them, the network would "collapse" into a less expressive regressor. In fact, a well-known theorem (the *universal approximation theorem*) demonstrates that any continuous function can, in principle, be approximated by a neural network with a single hidden layer and nonlinear activation functions. However, the existence of such a network does not imply that it is easy to find for a given problem (i.e., determining the number of parameters and their values). In practice, multiple layers are commonly used as this simplifies learning.

Section 7.4
{: .section}

This section introduces hardly any new elements. If we are to use a neural network for classification tasks involving sentences, we can avoid dealing with a variable number of word embeddings by representing the sentence with a single embedding obtained as the mean of the embeddings of its constituent words. Since the averaging operation is differentiable, we can continue using the stochastic gradient descent algorithm to learn the embeddings of each word.

The section also explains how to manage a mini-batch of inputs. Note that, apart from minor adjustments to the involved matrices, the process is the same as for a single example and consists essentially of multiplying the input (a matrix in this case, instead of a single vector) by the weight matrix. You may verify that the result is the same as if each input vector were multiplied by the weight matrix, and the results concatenated into another matrix.

Study the figures in this and the other sections of the chapter carefully, as they are very useful for reinforcing the discussed concepts.

Section 7.5
{: .section}

Again, this section simply consolidates earlier ideas by proposing a language model based on feedforward networks.

Although it conceptually makes sense to show the selection of an embedding from all those available in the embedding table as the product of a one-hot vector by the embedding matrix, in practical implementations, a more efficient representation is typically used to directly access the desired embedding via an index. While this operation is not strictly differentiable, the stochastic gradient descent algorithm can still be used: it suffices to update only the parameters of the selected embedding, leaving the others unchanged.

Section 7.6
{: .section}

Once more, this section elaborates on concepts already covered in the regressors topic, particularly in the multinomial regressors section. The cross-entropy to be minimized is obtained as the negative logarithm of the probability of the correct word. The analytical form of this loss function allows the gradient with respect to each model parameter to be computed, which is used to update each parameter at each training step.

With multiple layers, manually computing the error derivative becomes more tedious than in the case of a logistic regressor. Fortunately, PyTorch and other modern frameworks allow these derivatives to be computed automatically (through the technique known as *automatic differentiation*). The intuitive ideas behind this calculation are presented in sections 7.6.3 and 7.6.4, which you can skip unless you have a special interest in the topic.

Section 7.7
{: .section}

As already mentioned, all parameters, including embedding values, are updated at each training step using the stochastic gradient descent algorithm. This section serves as a culmination by summarizing these ideas.
