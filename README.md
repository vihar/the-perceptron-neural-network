# the-perceptron-neural-network

### What is a Neural Network?

The term ‘Neural’ is derived from the human (animal) nervous system’s basic functional unit ‘neuron’ or nerve cells which are present in the brain and other parts of the human (animal) body.

### What is Artificial Neural Network?
Artificial Neural Networks are the biologically inspired simulations performed on the computer to perform certain specific tasks like clustering, classification, pattern recognition etc.

### THE PERCEPTRON 
The most basic form of an activation function is a simple binary function that has only two possible results.

### Implementation

**1.Each input gets scaled up or down**

When a signal comes in, it gets multiplied by a weight value that is assigned to this particular input. That is, if a neuron has three inputs, then it has three weights that can be adjusted individually. During the learning phase, the neural network can adjust the weights based on the error of the last test result.

**2.All signals are summed up**

In the next step, the modified input signals are summed up to a single value. In this step, an offset is also added to the sum. This offset is called bias. The neural network also adjusts the bias during the learning phase.

This is where the magic happens! At the start, all the neurons have random weights and random biases. After each learning iteration, weights and biases are gradually shifted so that the next result is a bit closer to the desired output. This way, the neural network gradually moves towards a state where the desired patterns are “learned”.

**3. Activation**

Finally, the result of the neuron’s calculation is turned into an output signal. This is done by feeding the result to an activation function (also called transfer function).


![](https://cdn.discordapp.com/attachments/391971809563508738/405209925497651213/Screen_Shot_2018-01-23_at_9.27.08_AM.png "Neural Network")

Code!

##### Make a prediction with weights, Predict Function.

```python
def predict(row, weights):
    activation = weights[0]
    for i in range(len(row) - 1):
        activation += weights[i + 1] * row[i]
    return 1.0 if activation >= 0.0 else 0.0
```


##### training data

```python
dataset = [[2.7810836, 2.550537003, 0],
           [1.465489372, 2.362125076, 0],
           [3.396561688, 4.400293529, 0],
           [1.38807019, 1.850220317, 0],
           [3.06407232, 3.005305973, 0],
           [7.627531214, 2.759262235, 1],
           [5.332441248, 2.088626775, 1],
           [6.922596716, 1.77106367, 1],
           [8.675418651, -0.242068655, 1],
           [7.673756466, 3.508563011, 1]]
```

##### Predict

```python
weights = [-0.1, 0.20653640140000007, -0.23418117710000003]

for row in dataset:
    prediction = predict(row, weights)
    #my_prediction = predict()
    print("Expected=%d, Predicted=%d" % (row[-1], prediction))
```

**Bias In Activation Function:**

```python
activation = (w1 * X1) + (w2 * X2) + bias
```

