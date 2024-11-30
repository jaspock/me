
<div class="content-2columns" markdown>
![](../assets/imgs/book-robot.png){: .rounded-title-img}

# Learning to Program with PyTorch: Exploring the Library for Neural Network Models
</div>

The [PyTorch][pytorch] library is, along with [TensorFlow][tf] and [JAX][jax], one of the most popular libraries for programming neural network models at a low level. Since our goal is to understand and modify code that implements the transformer, it is essential that you learn how to use PyTorch. Since the library is written in Python, you also need to learn this programming language.

[tf]: https://www.tensorflow.org/
[jax]: https://github.com/google/jax
[pytorch]: https://pytorch.org/


{%
   include-markdown "./assets/mds/texts.md"
   start="<!--nota-inicial-start-->"
   end="<!--nota-inicial-end-->"
%}

## Python

Python is currently the most widely used programming language in the field of natural language processing. To use PyTorch, you will need to know the fundamental elements of Python. It is a dynamic language, but it might not resemble other dynamic languages you know. However, it does not make sense to learn Python from scratch, as many basic elements of the language (loops, functions, classes, etc.) are not far from what you already know in other languages.

Machine learning courses often include an introduction to Python for programmers experienced in other languages. In this field, it is also very common to use scientific computing libraries such as NumPy. Although we will use more specific libraries like PyTorch, it shares many design principles with NumPy, so it is recommended that you also learn some NumPy.

Follow these tutorials. Use multiple sources to broaden your learning:

- The tutorial "[Python Numpy Tutorial (with Jupyter and Colab)][cs231]" [<i class="fas fa-file"></i>][cs231] from the course "CS231n: Deep Learning for Computer Vision" by Stanford University. Note that at the top there is a badge saying "Open in Colab". Clicking it allows you to open a cloud-based Python environment where you can execute example code as a Python notebook.
- The slides "[Python Review Session][review]" [<i class="fas fa-file"></i>][review] from the course "CS224n: Natural Language Processing with Deep Learning" by Stanford University. There is also a [Python notebook][cuaderno] [<i class="fas fa-file"></i>][cuaderno] you can download and upload to Google Colab.

[cs231]: https://cs231n.github.io/python-numpy-tutorial/
[review]: https://web.stanford.edu/class/cs224n/readings/cs224n-python-review.pdf
[cuaderno]: https://web.stanford.edu/class/cs224n/readings/python_tutorial.ipynb

When you need to delve deeper into or understand specific Python structures, consult the [official Python documentation][oficial].

[oficial]: https://docs.python.org/3/tutorial/index.html


## Notebooks

The use of Jupyter and Google Colab notebooks is explained in sections 20.1 ("[Using Jupyter Notebooks][notebooks]" [<i class="fas fa-file"></i>)][notebooks] and 20.4 ("[Using Google Colab][colab]" [<i class="fas fa-file"></i>][colab]) of the book "Dive into Deep Learning."

[notebooks]: https://d2l.ai/chapter_appendix-tools-for-deep-learning/jupyter.html
[colab]: https://d2l.ai/chapter_appendix-tools-for-deep-learning/colab.html


## PyTorch

PyTorch is not initially an easy library to grasp. When first studying the code of a neural model programmed in PyTorch, you might not understand certain parts or deduce all the underlying implicit behavior. This is why an initial effort to study the library is necessary.

You can get a basic understanding of PyTorch fundamentals by following the brief [PyTorch introduction][intro] [<i class="fas fa-file"></i>][intro] included in the book "Dive into Deep Learning." Note that you can view the PyTorch examples on the website, or open a notebook in Colab or SageMaker Studio Lab. However, as mentioned, you will need to dig deeper into the library: for this, follow the over 2-hour video tutorial in this [official PyTorch playlist][playlist]. [<i class="fas fa-file"></i>][playlist] At a minimum, watch the first four videos ("Introduction to PyTorch," "Introduction to PyTorch Tensors," "The Fundamentals of Autograd," and "Building Models with PyTorch").

As a complement, you can also consult the [official PyTorch tutorial][tutoficial]. Be sure to select your version of PyTorch in the upper-left corner. This concise [tutorial with simple examples] on fitting the function $a +bx + cx^2 + dx^3$ is particularly educational. Finally, when you need to delve deeper into a specific aspect of PyTorch, refer to the [official PyTorch documentation][docutorch].

[intro]: https://d2l.ai/chapter_preliminaries/ndarray.html
[tutoficial]: https://pytorch.org/tutorials/beginner/basics/intro.html
[docutorch]: https://pytorch.org/docs/stable/index.html
[tutorial with simple examples]: https://pytorch.org/tutorials/beginner/pytorch_with_examples.html
[playlist]: https://www.youtube.com/playlist?list=PL_lsbAsL_o2CTlGHgMxNrKhzP97BaG9ZN


## Debugging {#debug}

Debugging is one of the best ways to uncover all the details of the code you are studying. It is important to feel comfortable with the debugging process. Below are some tips for debugging Python code using the VS Code environment, but similar instructions apply to other development environments.

To debug the current file, place one or more breakpoints in the code by clicking the small column to the left of the line numbers until a red mark appears. Then ensure the correct Python environment is selected in the bottom bar and run `Run/Start Debugging` (`F5`) from the menu. The code will run until it hits a breakpoint. You can step through the code using `Run/Step Over` or `Run/Step Into`, but you will save a lot of time by learning the corresponding keyboard shortcuts (`F10` and `F11`, respectively). In addition to inspecting variables via the interface, you will find it very helpful to access the command palette (`Help/Show All Commands` or `Ctrl+Shift+P`) and show the debug terminal by typing `Debug Console`. In this terminal, you can run any Python code, for example:

- View the type of a variable: ```type(learning_rate)```
- View the contents of a tensor: ```x```
- View the shape of a tensor: ```x.shape```
- View the output of a given function by writing lines such as ```sigmoid(torch.tensor([-10.2,0.1,10.2]))``` or ```torch.logical_not(mask[:10])```

If the tensor is large, it is often enough to view just a few of its elements, which you can do with indexing, like ```x[:10]```. To systematically have PyTorch display only the first elements of each dimension and use a limited number of decimals, you can run this once:

```python
torch.set_printoptions(precision=4, edgeitems=4, threshold=30)
```

## More Details on PyTorch

The [notes page](apuntes.md) on PyTorch includes additional details about the library that are necessary to understand the implementations of the models we’ll be studying.
