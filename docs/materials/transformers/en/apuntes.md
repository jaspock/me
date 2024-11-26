
<div class="content-2columns" markdown>
![](../assets/imgs/logo-mask.png){: .rounded-title-img}

# PyTorch Notes
</div>

Here are some notes on specific aspects of PyTorch that may be useful for understanding our model implementations. Each section of this page has been referenced from other pages, so you don't need to read it from start to finish.

{%
   include-markdown "../assets/mds/texts.md"
   start="<!--nota-inicial-start-->"
   end="<!--nota-inicial-end-->"
%}

## Broadcasting in PyTorch

Notice that in equation (5.12) of the Jurafsky and Martin book, the vector $\mathbf{b}$ is obtained by repeatedly copying the scalar value $b$. When implementing equations like this in PyTorch, it is not necessary to explicitly perform this copying, thanks to the *broadcasting* mechanism that is automatically activated in certain cases when combining tensors of initially incompatible sizes:

```python
import torch 
b = 10
X = torch.tensor([[1,2,3],[4,5,6]])
w = torch.tensor([-1,0,1])
yp = torch.matmul(X,w) + b
```

## Einstein Notation

Consider the case where we have a mini-batch of target words represented by their embeddings $\mathbf{w}_1,\mathbf{w}_2,\ldots,\mathbf{w}_E$. For each of these target words, we have an associated contextual word in the set $\mathbf{c}_1,\mathbf{c}_2,\ldots,\mathbf{c}_E$. For simplicity, we do not consider negative samples here, but the analysis is fully extensible to cases where they are included.

Let $N$ be the size of the embeddings. We want to calculate the dot product of each $\mathbf{w}_i$ with each $\mathbf{c}_i$, a calculation that is fundamental in training and using skip-gram models. To compute these dot products using PyTorch and benefit from the efficiency of GPU-calculated matrix operations, we can pack the target word embeddings row-wise in a matrix $A$ of size $E \times N$ and the contextual word embeddings column-wise in a matrix $B$ of size $N \times E$. If we compute the product $A \cdot B$, we get a matrix of size $E \times E$ where each element $i,j$ is the dot product of $\mathbf{w}_i$ with $\mathbf{c}_j$.

However, we are only interested in a subset of these dot products: those that lie on the diagonal of the result, which are of the form $\mathbf{w}_i \cdot \mathbf{c}_i$. Matrix multiplication is inefficient in this case for our purposes, but if we consult the PyTorch documentation, we won't initially find an operation that exactly matches our needs.

There is, however, an efficient and compact way in PyTorch to define matrix operations using Einstein notation. You can learn more by reading up to section 2.8 of the tutorial "[Einsum is all you need](https://rockt.github.io/2018/04/30/einsum)". Specifically, we are interested in obtaining a vector $\mathbf{d}$ such that:

$$
\mathbf{d}_i = \mathbf{w}_i \cdot \mathbf{c}_i = \sum_{j} \mathbf{w}_{i,j} \, \mathbf{c}_{j,i}
$$

Using Einstein notation with PyTorch's `einsum` function, we can write the above matrix operation and obtain the one-dimensional tensor we want as follows:

```python
d = torch.einsum('ij,ji->i', A, B)
```

## Unsqueezing Tensors

A frequent operation in PyTorch is the *unsqueezing*[^1] of a tensor using the `unsqueeze` operation. This operation adds a dimension of size 1 at the specified position. For example, if we have a tensor with shape `(2,3,4)` and apply `unsqueeze(1)`, the result will be a tensor with shape `(2,1,3,4)`. If we apply `unsqueeze(0)`, the result will have shape `(1,2,3,4)`. If we apply `unsqueeze(-1)`, the result will have shape `(2,3,4,1)`. One common use of `unsqueeze` is converting a single datum into a minibatch. For example, if we have a lexical category assignment model (e.g., verb, noun, adjective, etc.) that receives a minibatch of word embeddings and returns a probability vector for each category for each word, we can use `unsqueeze(0)` to turn a single word embedding into a minibatch with one element. Assuming there are 10 categories, running the model will yield a tensor with shape `(1,10)`, which we can convert to a tensor with shape `(10)` using `squeeze(0)`. The `squeeze` operation complements `unsqueeze`: by default, it removes all dimensions of size 1, but it allows specifying the position of the dimension to remove.

[^1]: It is challenging to find an appropriate translation for the terms *squeeze* and *unsqueeze* from English, but you can associate them with the shape alteration that occurs when opening or closing instruments (called *squeezeboxes*) like an accordion or concertina.

Adding a dimension of size 1 as done by `unsqueeze` does not affect the number of elements in the tensor but does change its shape. The underlying data block in memory remains unchanged. The following example demonstrates the results of unsqueezing operations at various positions:

```python
import torch 
a = torch.tensor([[1,2],[3,4]])  #   [[1, 2], [3, 4]]  2x2
a.unsqueeze(0)                  # [[[1, 2], [3, 4]]]  1x2x2
a.unsqueeze(1)                  # [[[1, 2]], [[3, 4]]]  2x1x2
a.unsqueeze(2)                  # [[[1], [2]], [[3], [4]]]  2x2x1
a.unsqueeze(3)                  # exception: dimension out of range
```

As is common in PyTorch, dimensions can be negative, allowing you to specify their position by counting from the end. In the example above, `a.unsqueeze(-1)` is equivalent to `a.unsqueeze(3)`. In terms of the `view` function, `t.unsqueeze(i)` is equivalent to `view(*t.shape[:i], 1, *t.shape[i:])`.

Viewing an $n$-dimensional tensor as a list of $(n-1)$-dimensional tensors often simplifies understanding tensor representations in PyTorch. For example, it is likely easier to visualize a 5-dimensional tensor as a list of 4-dimensional tensors (and so on) than as a matrix of cubes.

## Row and Column Vectors

The `squeeze` operation also helps clarify the difference between the representation of vectors, row vectors, and column vectors in PyTorch. To begin, consider these two tensors:

```python
a=torch.tensor([[1,2],[3,5]])
b=torch.tensor([2,3])
```

The tensor `a` corresponds to a 2x2 matrix, and `b` to a 2-element vector. The operation `torch.mm(a,b)` raises an error because the sizes are incompatible; this operation does not perform *broadcasting* and only works on two matrices. We can transform `b` into a column vector `[[2],[3]]` of size 2x1 using `unsqueeze`, so that `torch.mm(a,b.unsqueeze(1))` works correctly. Similarly, we can transform `b` into a row vector `[[2,3]]` of size 1x2 using `unsqueeze`, so that `torch.mm(b.unsqueeze(0),a)` works correctly. Note that the results of these two products are evidently different (the resulting tensors, in fact, have different shapes). We can now use `squeeze` on the result to obtain a 2-element vector.

The `torch.matmul` operation not only supports *broadcasting*, but it is also designed to operate with both bidimensional and unidimensional tensors. The result, in this case, is a unidimensional tensor. Therefore, the following two assertions do not fail:

```python
assert torch.equal(torch.mm(b.unsqueeze(0),a).squeeze(), torch.matmul(b,a))
assert torch.equal(torch.mm(a,b.unsqueeze(1)).squeeze(), torch.matmul(a,b))
```

## Tensor Memory Representation

To simplify, let us consider a 4x3 matrix initialized as follows:

```python
a = torch.tensor([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
```

In memory, the elements of a tensor like the one above are stored in consecutive positions in a row-major order, so the elements are arranged as 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12. The storage order of a tensor's elements is characterized by a concept called *stride*, which can be queried using the `stride` method:

```python
print(a.stride())  # (3, 1)
```

The tuple `(3,1)` indicates that to move along the first dimension (rows) to the next element, 3 memory positions must be skipped, and to move along the second dimension (columns) to the next element, 1 memory position must be skipped.

Some PyTorch operations (e.g., transpose or the `view` function) modify the strides of tensors without rearranging the elements in memory, making the operation very efficient as it avoids creating new values in memory or reorganizing existing ones:

```python
t = a.t()
print(t.stride())  # (1, 3)
```

Verify that the strides `(1, 3)` are correct if the data in memory has not been modified. Many PyTorch operations are implemented to iterate over the data from the last dimension to the first (e.g., first through columns and then through rows), assuming this means starting with the smallest strides (columns, in this case) and moving to dimensions with larger strides. This way, when the algorithm accesses the next data point, it is often a neighbor of the current one and likely already in cache. If the elements were arranged differently in memory, the algorithm would need to jump more positions in memory to access the data, making it slower or potentially non-functional. Therefore, some operations (e.g., `t.view(-1)`) raise an exception, and we must explicitly reorder the tensor's data in memory before using such operations:

```python
print(a.is_contiguous())  # True
print(t.is_contiguous())  # False
print(a.data_ptr()==t.data_ptr())  # True
t = t.contiguous()
print(t.stride())  # (4, 1)
print(a.data_ptr()==t.data_ptr())  # False
```

The `contiguous` operation returns the input tensor (`self`) if it is already contiguous, or a copy with reorganized data otherwise. For contiguous tensors of any shape, the stride is always greater in one dimension than in the next:

```python
x = torch.ones((5, 4, 3, 2))
print(x.stride())  # (24, 6, 2, 1)
```

## Ways to Use Matplotlib

There are two different ways to interact with the Matplotlib library. You will likely encounter both styles in code you find online, so it is important to know them. The first way (implicit) is the simplest and involves importing the library and calling its functions directly. The second way is more comprehensive and involves creating a `Figure` object and calling its methods using the returned objects to interact with the library. In both cases, you are working internally with a figure and one or more associated axes (in English, *axes*, but do not confuse with the axes or *axis* of a plot). However, in the first case, a global state is maintained, so it is not necessary to explicitly use the different objects; simply calling the functions directly is sufficient.

Here is an example of code using the implicit style:

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 2 * np.pi, 400)
y = np.sin(x ** 2)

plt.figure()
plt.subplots()
plt.suptitle('Sinusoidal function')
plt.plot(x, y)
plt.show()
```

And here is an example using the explicit style:

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 2 * np.pi, 400)
y = np.sin(x ** 2)

fig = plt.figure()
ax = fig.subplots()
fig.suptitle('Sinusoidal function')
ax.plot(x, y)
fig.show()
```
