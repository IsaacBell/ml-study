# Envelope Estimates

## Core Questions

1. How many calculations will be performed in total?
2. How many calculations can our computer(s) do in one second?

## FLOPs (Floating-Point Operation)

When we talk about training an AI, we are really talking about performing billions or trillions of these simple math operations.

## Training Time Formula

Training Time (in seconds) = (Total FLOPS required) / (FLOPS our computer can do per second)

### Deconstructing the Formula

These factors impact the total # of FLOPs required:

- *Number of Parameters:* # of internal weights updated during training. This reflects the model's size (GPT has billions of parameters).
- *FLOPS per data item:* # of calculations needed to process a single piece of data (one forward and backward pass)
- *Epochs:* # of times the model will see the entire dataset during training
- *Data Items:* total # of items in the dataset, like sentences or images
- *Model Architecture Constant:* A constant representing model-specific workload, such as the number of generation steps (e.g., for a Diffusion model) or iterations.
- *Rate of Operations:* speed of a single GPU, measured in FLOPS per second
- *Total Number of GPUs*

Ultimately, the # of FLOPs required is the total number of math problems you have to solve.

For an estimate, the # of FLOPs per data item is `6 * N_params`. That's because:

• A feedforward pass (making a prediction) takes about 2 * N_params calculations.
• A backward pass (learning from the prediction) takes about 4 * N_params calculations.

---
Calculating the FLOPs we can do per second:

It’s the speed of one computer multiplied by the number of computers you have working together.

## Inference Time Formula

`Time to process one inference = (num_params * 2 * C) / RateOfOperations`

where C is the model architecture constant (see above) and RateOfOperations is the speed of one GPU, measured in FLOPs/s.

## Deployment Time















