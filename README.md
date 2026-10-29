# Self-Contained Multi-Layer Perceptron

This project demonstrates how a small neural network can be built and trained from scratch using only Python and NumPy.

The example creates a non-linear, three-class spiral dataset and trains a multilayer perceptron (MLP) to classify the points.

## What it includes

- Fully connected (dense) layers
- ReLU activation functions
- Softmax cross-entropy loss
- Forward propagation and backpropagation
- Adam optimization with bias correction
- Mini-batch training and shuffled data
- Periodic loss and accuracy reporting

## Model structure

The example network has the following layout:

1. Input layer with 2 features
2. Dense layer with 64 units
3. ReLU activation
4. Dense layer with 32 units
5. ReLU activation
6. Dense output layer with 3 units

The three output units represent the three spiral classes.

## Requirements

- Python 3.8 or newer
- NumPy

Install NumPy with:

```bash
pip install numpy
```

## Running the example

From the project directory, run:

```bash
python main.py
```

The program generates the dataset, trains the MLP for 150 epochs, and prints the training loss and classification accuracy at the beginning and at regular intervals.

## Notes

This is an educational implementation intended to illustrate the main stages of neural-network training. It does not use high-level machine-learning frameworks, external datasets, model saving, or a separate validation set.

The random seed is fixed so that the generated example and initialization are repeatable between runs.

## Current limitation

The sample dataset expression may generate `NaN` values on some runs because it raises negative random values to a fractional power. When this occurs, training produces `NaN` loss and the accuracy remains at the three-class baseline. The implementation is intentionally left unchanged here; correcting the data-generation expression would be the next step for a successful training run.

## License

No license is currently specified for this project.
