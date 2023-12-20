
<div class="content-2columns" markdown>
![](assets/imgs/implementation-heart.png){: .rounded-title-img}

# Implementación de los modelos en PyTorch
</div>

La implementación en forma de código de los diferentes modelos es un enfoque complementario a su estudio desde un punto de vista matemático. En esta página se presentan implementaciones en PyTorch de cada uno de los modelos estudiados. La idea es que abordes el estudio de estas implementaciones después de haber estudiado conceptualmente el modelo en cuestión.

{%
   include-markdown "./assets/mds/texts.md"
   start="<!--nota-inicial-start-->"
   end="<!--nota-inicial-end-->"
%}


## Código para un regresor logístico y uno multinomial

Estas son dos implementaciones en PyTorch de los regresores que se estudian [en esta página](regresor.md) en unas pocas decenas de líneas de código. Asegúrate de que terminas entendiendo el código lo suficiente como para sentirte con ánimo de poder modificarlo para adaptarlo a otras necesidades. 

Repasa antes de abordar el código cómo puedes [depurar](pytorch.md#depuración) programas escritos en Python. Repasa también cómo funciona el mecanismo de [broadcasting](apuntes.md#broadcasting-en-pytorch) en PyTorch.

Los dos programas de este apartado son:

- Un  [:octicons-book-24:][pylog] [regresor logístico][pylog] que clasifica muestras bidimensionales sintéticas en dos clases. Se usan solo los elementos más básicos de PyTorch para poder tener una implementación lo más detallada posible. Como ejercicio, puedes hacer una traza y analizar qué tamaños tienen los tensores. Puedes jugar también con el número de pasos de entrenamiento  y la tasa de aprendizaje para ver cómo evoluciona el entrenamiento. Explora diversas posiciones de los centros de las clases y de la dispersión de los datos alrededor de estos y observa cómo cambia la frontera de decisión. Elimina el sesgo (*bias*) de las ecuaciones y observa cómo se restringe la forma de la frontera de decisión al obligar a esta a pasar por el origen de coordenadas.
- Un  [:octicons-book-24:][pysoft] [regresor softmax para clasificar imágenes de dígitos][pysoft]. Las imágenes y etiquetas de los dígitos se toman de un conjunto de datos muy conocido llamado MNIST. Como ejercicio, puedes simplificar este código para que realice una tarea de clasificación de sentimiento sobre un conjunto de datos sintéticos muy pequeño que se defina explícitamente en el propio programa.

Si no lo has hecho ya, puedes empezar a aprender Python y PyTorch siguiendo el [capítulo][cappy] correspondiente de esta serie.

[cappy]: pytorch.md
[pylog]: https://github.com/jaspock/me/blob/master/assets/code/transformers/logistic-regressor.py
[pysoft]: https://github.com/jaspock/me/blob/master/assets/code/transformers/softmax-regressor.py


## Código para skip-grams

Estudia una implementación del algoritmo [:octicons-book-24:][pyskip] [skip-gram][pyskip] para la obtención de embeddings incontextuales que sigue las pautas marcadas en el libro de Jurafsky y Martin. La implementación anterior se basa en [otra](https://github.com/jaspock/me/blob/master/assets/code/guia-transformers/skipgrams-original.py) que sigue el enfoque del trabajo "[Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546)" de 2013, que no se ajusta totalmente al que hemos estudiado nosotros.

Antes de abordar el código, lee sobre la [notación de Einstein](apuntes.md#notación-de-einstein) y su aplicación en PyTorch. Estudia también los apartados correspondientes al [estrujamiento de tensores](apuntes.md#desestrujando-tensores) y a los [vectores fila y columna](apuntes.md#vectores-fila-y-columna) en PyTorch.

[pyskip]: https://github.com/jaspock/me/blob/master/assets/code/transformers/skipgrams-jurafsky.py


## Código para un modelo de lengua con redes feedforward

Estudia una implementación de un [:octicons-book-24:][pylm] [modelo de lengua][pylm] con redes *feedforward* como el que se ha visto en el capítulo de redes *feedforward*. La implementación coincide con la del artículo "[A neural probabilistic language model](https://dl.acm.org/doi/10.5555/944919.944966)" de 2003, que puedes consultar si necesitas más información.

[pylm]: https://github.com/jaspock/me/blob/master/assets/code/transformers/ff-neural-lm.py


## Código para un transformer del proyecto minGPT

Una buena implementación en PyTorch de un modelo de lengua basado en transformer es la de [minGPT][minGPT] de Andrej Karpathy. Aunque no se centra exclusivamente en este código, Andrej tiene un [vídeo][vidkarpathy] donde explica las ideas generales de la implementación. El código permite entrenar y usar modelos de lengua, además de permitir la carga de los pesos del modelo GPT-2.

Esta guía tiene una [copia][copia] del código de minGPT con algunas pequeñas modificaciones. A continuación, se comenta qué ficheros son relevantes para nuestros intereses. Los ficheros de los que no se diga nada no tienes que mirarlos. Para usar el código y poder modificarlo, puedes instalarlo con:

```bash
pip install --editable .
```

[minGPT]: https://github.com/karpathy/minGPT
[vidkarpathy]: https://youtu.be/kCc8FmEb1nY
[copia]: https://github.com/jaspock/me/tree/master/assets/code/transformers/minGPT-20230108

### Fichero mingpt/bpe.py

Este fichero contiene la implementación necesaria para usar el modelo de subpalabras de tipo BPE usado por GPT-2. De su función se habla más adelante. El código principal del fichero muestra un ejemplo de tokenización paso a paso de una cadena de entrada que puedes ver haciendo `python bpe.py`. La primera vez que se llama a los métodos `encode` o `decode` se descargan los ficheros `encoder.json` y `vocab.bpe`, que contienen el vocabulario y las reglas de unión de subpalabras, respectivamente, usados por GPT-2. Estos ficheros se almacenan en el directorio `~/.cache/mingpt`.

No es necesario que estudies el código de este fichero. Basta con que sepas que nos permite obtener la lista de índices de los tokens de un texto de entrada, así como obtener el texto asociado a una lista de índices de tokens obtenida a la salida del modelo:

```python
bpe = BPETokenizer()
tokens = bpe("A relaxing cup of café con leche in Plaza Mayor") # encode
# tokens is a tensor of shape (1, 9)
print(bpe.decode(tokens[0]))  
# "A relaxing cup of café con leche in Plaza Mayor"
print(tokens[0].tolist()) 
# [32, 28175, 6508, 286, 40304, 369, 443, 2395, 287, 23280, 10106]
for token in tokens[0]:
   print(bpe.decode(torch.tensor([token])), end='/')
# A/ relaxing/ cup/ of/ café/ con/ le/che/ in/ Plaza/ Mayor/
```

### Fichero mingpt/utils.py

No es necesario que estudies con detalle el código de este fichero. Simplemente, ábrelo y observa que define dos funciones auxiliares (`set_seed` y `setup_logging`) y una clase (`CfgNode`) para gestionar los parámetros de configuración del modelo.

### Fichero mingpt/trainer.py

Estudia este fichero, que contiene el código general que se encarga de entrenar un modelo. El código no tiene nada específico de la arquitectura transformer, por lo que podría aplicarse con pocas modificaciones a cualquier otro modelo.

### Fichero mingpt/model.py

El fichero más importante para nuestros propósitos. Puedes saltarte, no obstante, el método `from_pretrained` de la clase `GPT` (incorpora los pesos de GPT-2 descargados de huggingface/transformers a nuestro modelo) y, especialmente, el método `configure_optimizers` (devuelve un optimizador de tipo Adam que trabaja de forma diferente según el tipo de parámetro sobre el que actúa), ya que contienen código muy específico para el sistema GPT-2. 

Estudia con detalle las clases `CausalSelfAttention` y `Block`, así como los métodos `forward`, `generate`, `__init__`, `_init_weights` y `get_default_config` de la clase `GPT`.

### Fichero generate.ipynb

Estudia este código que usa el modelo para generar texto. Es un cuaderno de Python, pero lo puedes ejecutar desde la línea de órdenes conviertiéndolo antes a un programa de Python con:
   
```bash
pip install nbconvert
jupyter nbconvert --to script generate.ipynb
python generate.py
```

Puedes cambiar la variable `model-type` para que use diferentes modelos preentrenados de GPT-2. De mayor a menor tamaño, los modelos disponibles son `gpt2-xl`, `gpt2-large`, `gpt2-medium` y `gpt2`. Si quieres poder ejecutar el código sobre CPU, cambia el valor de `device` a:

```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
```

### Fichero projects/chargpt/charpgt.py

Este código realiza el entrenamiento de un modelo de lengua a nivel de caracteres a partir del contenido del fichero `input.txt`. Puedes usar para el fichero de entrada textos como el [Quijote][quijote] o parte de la obra de [Shakespeare][shakespeare].

[quijote]: https://www.gutenberg.org/cache/epub/2000/pg2000.txt
[shakespeare]: https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

Puedes cambiar la variable `C.model.model_type` para que use modelos de diferentes tamaños (de mayor a menor, `gpt2-xl`, `gpt2-large`, `gpt2-medium`, `gpt2`, `gpt-mini`, `gpt-micro` y `gpt-nano`). Puedes ver el número de capas, el número de cabezales y el tamaño de los embeddings de cada modelo en el constructor de la clase `GPT` del fichero `mingpt/model.py`.

Lanza el programa y déjalo un tiempo entrenando con `python charpgt.py`. El modelo se va guardando en la carpeta `out`. 


## Otros programas para el transformer

La implementación en PyTorch del bloque anterior se centraba en un modelo de lengua basado en descodificador. En principio, no es necesario que estudies ningún código adicional para el transformer, pero si estás interesado en una arquitectura completa de transformer o en el codificador, puedes estudiar el código de las implementaciones que se presentan a continuación:

- Una implementación del [transformer][pytr] completo. La implementación sigue la del artículo "[Attention is all you need](https://arxiv.org/abs/1706.03762)" de 2017, que puedes consultar si necesitas más información. 
- Una implementación sencilla del [modelo BERT][pybert] con código muy parecido al del transformer completo anterior.

[pytr]: https://github.com/jaspock/me/blob/master/assets/code/transformers/transformer.py
[pybert]: https://github.com/jaspock/me/blob/master/assets/code/transformers/bert.py


## Implementaciones adicionales

El proyecto [MinT][MinT] incluye diferentes tutoriales con implementaciones desde cero de modelos tipo BERT, GPT, BART o T5. El código es ligeramente más extenso que el que hemos estudiado, pero puede servir para afianzar conocimientos en una fase avanzada. El proyecto [x-transformers] sigue un enfoque similar.

Existe cierto *pique* entre los desarrolladores por conseguir una implementación del transformer lo más compacta posible. Algunas de ellas son [minGPT][mingpt], [nanoGPT][nanogpt] y [picoGPT][picogpt]. Un aspecto destacable de estas es que son capaces de cargar los pesos de GPT-2 y realizar inferencia con ellos. Andrej Karpathy, el desarrollador de minGPT y nanoGPT tiene un [vídeo][video] muy pedagógico en el que explica el funcionamiento de su implementación.

[MinT]: https://github.com/dpressel/mint
[x-transformers]: https://github.com/lucidrains/x-transformers

[mingpt]: https://github.com/karpathy/minGPT
[nanogpt]: https://github.com/karpathy/nanoGPT
[picogpt]: https://github.com/jaymody/picoGPT
[video]: https://youtu.be/kCc8FmEb1nYç
