
<div class="content-2columns" markdown>
![](assets/imgs/logo-mask.png){: .rounded-title-img}

# Apuntes de PyTorch
</div>

Se incluyen aquí algunas notas sobre aspectos concretos de PyTorch que pueden ser útiles para entender nuestras implementaciones de modelos. Cada apartado de esta página ha sido referenciado desde otras páginas, por lo que no es necesario que la leas de principio a fin.


{%
   include-markdown "./assets/mds/texts.md"
   start="<!--nota-inicial-start-->"
   end="<!--nota-inicial-end-->"
%}

## Broadcasting en PyTorch

Observa que en la ecuación (5.12) del libro de Jurafsky y Martin el vector $\mathbf{b}$ se obtiene copiando repetidamente el valor escalar $b$. Cuando ecuaciones como esta se implementan en PyTorch, no es necesario hacer esta copia explícita gracias al mecanismo de *broadcasting* que se activa automáticamente en algunas ocasiones cuando se combinan tensores de tamaños en principio incompatibles:

``` python
import torch 
b = 10
X = torch.tensor([[1,2,3],[4,5,6]])
w = torch.tensor([-1,0,1])
yp = torch.matmul(X,w) + b
```

## Notación de Einstein

Consideremos el caso en el que tenemos un mini-batch de palabras objetivo representadas por sus embeddings $\mathbf{w}_1,\mathbf{w}_2,\ldots,\mathbf{w}_E$. Para cada palabra objetivo anterior, tenemos una palabra contextual asociada en el conjunto $\mathbf{c}_1,\mathbf{c}_2,\ldots,\mathbf{c}_E$. Para simplificar, no consideramos las muestras negativas, pero el análisis que vamos a hacer es totalmente extensible al caso en el que se incluyan. 

Sea $N$ el tamaño de los embeddings. Queremos calcular el producto escalar de cada $\mathbf{w}_i$ con cada $\mathbf{c}_i$, cálculo este que ya has visto que es fundamental en el entrenamiento y uso de los modelos de skip-grams. Para obtener estos productos escalares usando PyTorch y beneficiarnos de la eficiencia de las operaciones matriciales calculadas en GPUs, podemos empaquetar por filas los embeddings de las palabras objetivo en una matriz $A$ de tamaño $E \times N$ y los embeddings de las palabras contextuales por columnas en una matriz $B$ de tamaño $N \times E$. Si calculamos el producto $A \cdot B$ obtendremos una matriz de tamaño $E \times E$ en la que cada elemento $i,j$ es el producto escalar de $\mathbf{w}_i$ con $\mathbf{c}_j$. 

Sin embargo, nosotros solo estamos interesados en una pequeña parte de todos estos productos escalares. En concreto, aquellos que forman parte de la diagonal del resultado, que serán los de la forma  $\mathbf{w}_i$ $\mathbf{c}_i$. La multiplicación de matrices es muy ineficiente en este caso para nuestros propósitos, pero si buscamos en la documentación de PyTorch no encontraremos en principio una operación que se ajuste exactamente a nuestros intereses. 

Existe, sin embargo, en PyTorch una manera eficiente y compacta de definir operaciones matriciales basada en la notación de Einstein, de la que puedes aprender un poco leyendo hasta el apartado 2.8 aproximadamente del tutorial "[Einsum is all you need](https://rockt.github.io/2018/04/30/einsum)". En particular, podemos observar que nos interesa obtener un vector $\mathbf{d}$ tal que:

$$
\mathbf{d}_i = \mathbf{w}_i \cdot \mathbf{c}_i = \sum_{j} \mathbf{w}_{i,j} \, \mathbf{c}_{j,i}
$$

Usando la notación de Einstein con la función de PyTorch `einsum`, podemos escribir la operación matricial anterior y obtener el tensor unidimensional que queremos como sigue:

``` python
d = torch.einsum('ij,ji->i', A, B)
```


## Desestrujando tensores

Una operación frecuente en PyTorch es la de *desestrujamiento*[^1] de un tensor mediante la operación `unsqueeze`. Esta operación añade una dimensión de tamaño 1 en la posición indicada. Por ejemplo, si tenemos un tensor de forma `(2,3,4)` y aplicamos `unsqueeze(1)`, el resultado será un tensor de forma `(2,1,3,4)`. Si aplicamos `unsqueeze(0)`, el resultado será un tensor de forma `(1,2,3,4)`. Si aplicamos `unsqueeze(-1)`, el resultado será un tensor de forma `(2,3,4,1)`. Uno de los usos más típicos de `unsqueeze` es convertir un dato simple en un minibatch. Por ejemplo, imagina que tenemos un modelo de asignación de categorías léxicas (verbo, nombre, adjetivo, etc.) a palabras que recibe un minibatch de embeddings de diferentes palabras y nos devuelve para cada palabra un vector de probabilidades de asignación a cada categoría. Si queremos aplicar el modelo a una sola palabra, necesitamos convertir su embedding en un minibatch de un solo elemento y, para ello, podemos usar `unsqueeze(0)`. Si suponemos que el número de categorías es 10, tras ejecutar el modelo, el resultado será un tensor de forma `(1,10)`, que podemos convertir en un tensor de forma `(10)` con `squeeze(0)`. La operación `squeeze` es el complemento de `unsqueeze`: por defecto, elimina todas las dimensiones de tamaño 1, pero permite indicar la posición de la dimensión que queremos eliminar.

[^1]: Es difícil encontrar una traducción adecuada para los términos *squeeze* y *unsqueeze* del inglés, pero puedes asociarlos a la alteración en la forma que se produce al abrir o cerrar instrumentos (denominados *squeezebox*) como un acordeón o una concertina.


Añadir una dimensión de tamaño 1 en la posición indicada, como hace `squeeze` no afecta al número de elementos del tensor, pero sí a su forma. El bloque de datos que contiene el tensor no se modifica en memoria. El siguiente ejemplo, muestra el resultado de operaciones de desestrujamiento sobre diferentes posiciones:

``` python
import torch 
a=torch.tensor([[1,2],[3,4]])  #   [ [ 1,     2 ],     [ 3,     4 ] ]    2x2
a.squeeze(0)                   # [ [ [ 1,     2 ],     [ 3,     4 ] ] ]  1x2x2
a.squeeze(1)                   # [ [ [ 1,     2 ] ], [ [ 3,     4 ] ] ]  2x1x2
a.squeeze(2)                   # [ [ [ 1 ], [ 2 ] ], [ [ 3 ], [ 4 ] ] ]  2x2x1
a.squeeze(3)                   # exception: dimension out of range
```

Como es habitual en PyTorch, las dimensiones pueden ser negativas, lo que permite indicar la posición de la dimensión contando desde el final. En el ejemplo anterior, `a.squeeze(-1)` es equivalente a `a.squeeze(3)`. En términos de la función `view`, `t.squeeze()` es equivalente a `view(*[s for s in t.shape if s != 1])`. Por otro lado, `t.unsqueeze(i)` equivale a `view(*t.shape[:i], 1, *t.shape[i:])`.

Observar un tensor $n$-dimensional como una lista de tensores $(n-1)$-dimensionales facilita la comprensión de la representación de tensores en PyTorch. Te resultará probablemente más sencillo visualizar un tensor 5-dimensional como una lista de tensores de 4 dimensiones (y así sucesivamente) que como una matriz de cubos, por ejemplo. 


## Vectores fila y columna

La operación `squeeze` nos ayuda también a aclarar la diferencia entre la representación de vectores, vectores fila y vectores columna en PyTorch. Para empezar, considera estos dos tensores: 

``` python
a=torch.tensor([[1,2],[3,5]])
b=torch.tensor([2,3])
``` 

El tensor `a` se corresponde con una matriz de 2x2 y `b` con un vector de 2 elementos. La operación `torch.mm(a,b)` produce un error porque los tamaños son incompatibles, ya que esta operación no hace *broadcasting* y solo funciona sobre dos matrices. Podemos transformar `b` en un vector columna `[[2],[3]]` de 2x1 con ayuda de `unsqueeze` para que `torch.mm(a,b.unsqueeze(1))` funcione correctamente. También podemos transformar `b` en un vector fila `[[2,3]]` de 1x2 con ayuda de `unsqueeze` para que `torch.mm(b.unsqueeze(0),a)` funcione correctamente. Observa que el resultado de ambos productos es evidentemente distinto (los tensores resultantes, de hecho, tienen formas diferentes). Podemos usar ahora `squeeze` sobre el resultado para obtener un vector de 2 elementos. 

La operación `torch.matmul` no solo soporta *broadcasting*, sino que está preparada para operar con tensores bidimensionales y unidimensionales. El resultado es en este caso un tensor unidimensional. Las siguientes dos aserciones, por tanto, no fallan:

``` python
assert torch.equal(torch.mm(b.unsqueeze(0),a).squeeze(), torch.matmul(b,a))
assert torch.equal(torch.mm(a,b.unsqueeze(1)).squeeze(), torch.matmul(a,b))
```

## Representación en memoria de los tensores

Consideremos, para simplificar, una matriz de 4x3 inicializada como sigue:

``` python
a = torch.tensor([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
```

En memoria, los elementos de un tensor como el anterior se almacenan en posiciones consecutivas siguiendo un orden por filas, por lo que estos se encuentran dispuestos como 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12. El orden de almacenamiento de los elementos de un tensor se caracteriza mediante un concepto denominado  *stride* (que podemos traducir por *salto* o *paso*) que se puede consultar con el método `stride`:

``` python
print(a.stride())  # (3, 1)
```

La tupla `(3,1)` indica que para avanzar en la primera dimensión (las filas) de un elemento al siguiente es necesario saltar 3 posiciones en memoria y que para avanzar en la segunda dimensión (las columnas) de un elemento al siguiente es necesario saltar 1 posición en memoria. 

Hay operaciones de PyTorch (por ejemplo, la traspuesta o la función `view`) que modifican los pasos de los tensores sin mover los elementos en memoria, lo que hace que la operación sea muy eficiente al no tener que crear nuevos valores en memoria o reordenar los existentes:

``` python
t = a.t()
print(t.stride())  # (1, 3)
```

Comprueba que los pasos `(1, 3)` son los correctos si no se han modificado los datos en memoria. Muchas operaciones de PyTorch están implementadas de manera que van iterando por los datos desde la última dimensión a la primera (primero por columnas y luego por filas, por ejemplo), esperando que esto suponga comenzar por las dimensiones de paso más pequeño (columnas, en nuestro caso) e ir moviéndose hacia dimensiones con pasos menores. De esta forma, cuando el algoritmo accede al siguiente dato, este suele ser un vecino del actual y estará probablemente disponible en caché. Si los elementos estuvieran dispuestos de otra manera en memoria, el algoritmo tendría que saltar más posiciones en memoria para acceder a los datos y, por tanto, sería más lento o directamente no funcionaría. Por ello, en ocasiones algunas operaciones (por ejemplo, `t.view(-1)`) lanzan una excepción y tendremos que reordenar explícitamente los datos en memoria del tensor afectado antes de poder usar dicha operación:

``` python
print(a.is_contiguous())  # True
print(t.is_contiguous())  # False
print(a.data_ptr()==t.data_ptr())  # True
t = t.contiguous()
print(t.stride())  # (4, 1)
print(a.data_ptr()==t.data_ptr())  # False
```

La operación `contiguous` devuelve el tensor de entrada (`self`) si este ya es contiguo y devuelven una copia con los datos reorganizados en caso contrario. Para tensores contiguos de cualquier forma, el paso es siempre mayor en una dimensión dada que en la siguiente:

``` python
x= torch.ones((5, 4, 3, 2))
print(x.stride())  # (24, 6, 2, 1)
```  

## Modos de usar Matplotlib

Hay dos formas diferentes de interactuar con la librería Matplotlib. Probablemente te encontrarás con ambos estilos en el código que encuentres en la web, por lo que es importante conocerlos. La primera forma (implícita) es la más sencilla y consiste en importar la librería y llamar a sus funciones de forma directa. La segunda forma es la más completa y consiste en crear un objeto `Figure` y llamar a sus métodos usando los objetos devueltos para interactuar con la librería. En ambos casos, se trabaja internamente con una figura y uno o más marcos asociados (en inglés *axes*, pero no confundir con los ejes o *axis* de un marco), pero en el primer caso se mantiene un estado global por lo que no es necesario usar explícitamente los distintos objetos y basta con llamar a la funciones directamente.

Este es un ejemplo de código que usa la forma implícita:

``` python
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

Y este es un ejemplo de la forma explícita:

``` python
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
