# How to run programs in this directory
1. make sure you got poetry i am using 2.1.3
2. install dependencies `poetry install`
3. get the environment `poetry env activate`
4. this will give you a command to run to activate the poetry environment
5. use the scripts below to run some programs :)

# scripts

- `mnistmlp` - will run train function in main.py for mnistMLP, this is a simple MLP to classify digits in the MNIST dataset.
- `improvedmnistmlp` - **incomplete** an improved model with more layers, neurons, and improved accuracy
- `mnistpytorch` - implement the same mlp as i did with vector math, except using additional tools from pytorch