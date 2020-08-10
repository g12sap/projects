# Introduction to TensorFlow for Artificial Intelligence, Machine Learning and Deep Learning

__Machine Learning__: Feed data and expected output to a machine learning algorithm and rules are predicted to solve our problem.<br>
__Neural Network__: It's an algorithm to learn patterns.


```python
import numpy as np
import pandas as pd
import tensorflow as tf
print(tf.__version__)

from tensorflow import keras
```

## Implementing a Perceptron
Below, we implement a simple neural network = __Perceptron__. It is indicated by Dense (units = 1), this indicates the shape of the story.
<br>
The complex math implemented in most machine learning is used in __TensorFlow__ as Functions.
<br/>
The two main functions are:
- Loss Functions
- Optimizers.


```python
model = keras.Sequential([keras.layers.Dense(units = 1, input_shape = [1])])
model.compile(optimizer='sgd', loss = 'mean_squared_error')
```

__Loss Function__: It is a measure of how much our prediction missed compared to the actual label.
<br/>
__Optimizer__: It takes in the input from the loss fuction and makes guesses, the logic here is that the guesses should be better and better.
<br/>
__Epochs__ : It gives the count of the training loop. 

- In this example, we have take the __loss function__ as __*mean squared error*__.
- the __optimizer__ as __*SGD*__, stands for stochastic gradient descent.


```python
xs = np.array([-1.0, 0.0,1.0,2.0,3.0,4.0], dtype = float)
ys = np.array([-3.0, -1.0, 1.0, 3.0,5.0,7.0], dtype = float)
```

Here, we have two arrays of supposedly random number.If we look at them closely, the values in $ Y $ are a result of some manipulation of values in $ X $.

Looking at it closely, we can chalk up a formula: 
$$ Y = (2x - 1) $$
- Let's take the first value:" $ -1.0 $. ". $ Y = (2 (-1.0)-1) $ , this gives us a value of $ -3.0 $ <br/>
We can observe this simple formula holds true for the rest of the values.


```python
model.fit(xs,ys, epochs = 1000)
```


```python
print(model.predict([10.0]))
```

Great! We predicted a value of $ 18.999891 $, this is close to 19. 
- Using the value $ 10 $ in our formula, $ Y = ( 2 (10.0) - 1) $, we should get a value of 19.

#### Observation

When we trained our example, we did not get the correct value, because:
1. There is very less data (6 points)and they are very linear.
2. We should use this on realistic data values. 

### Neural Networks, deal in probability ###
