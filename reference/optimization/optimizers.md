# SGD Optimizer

- Single learning rate for weight training (alpha)
- Alpha doesn't change during training

> The method computes individual adaptive learning rates for different parameters from estimates of first and second moments of the gradients.

## SGD+Nesterov Momentum

Recommended by Andrej Karpathy as a decent alternative to Adam.

# Adam Optimizer

- Momentum + adaptive learning
  - Formally, Adam has the benefits of AdaGrad and RMSProp.
    - Adaptive Gradient Algo: keeps a per-param learning rate
      - improves perf with sparse gradients (good for NLP & CV)
    - Root Mean Square Progation: adapts per-param learning rates based on how quickly their gradient is changing
      - Formally, it checks the average of recent magnitudes of the gradient per weight
      - Helps with online probs and noisy probs
  - In more detail: Adam calculates an EMA of the gradient and squared gradient, with 2 params controlling decay rates (beta1 & beta2)
- Adam needs fewer hyperparams (and less hyperparam tuning) than other optimizers
  -

## Config Params

- alpha. Learning rate/step size. The proportion that weights are updated (e.g. 0.001). Larger values (0.3) give faster initial learning before the rate is updated. Smaller values (1.0E-5) slow learning down during training
- beta1. The exponential decay rate for the first moment estimates (e.g. 0.9).
- beta2. The exponential decay rate for the second-moment estimates (e.g. 0.999). Should be set close to 1.0 on problems with a sparse gradient (e.g. NLP and computer vision problems).
- epsilon. Very small number used to prevent division by zero (e.g. 10E-8).

TensorFlow and PyTorch deaults:
learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08

## Good Defaults

- Adam Paper: alpha=0.001, beta1=0.9, beta2=0.999 and epsilon=10−8
- 1e-8 for epsilon might not be a good default in general
  - when training an Inception network on ImageNet a current good choice is 1.0 or 0.1.
  -

# AdamW

Better version of Adam.

Decouples weight decay from the gradient update step. Instead of adding weight decay to the loss function, it applies weight decay directly during the parameter update.

### When to Choose AdamW

Larger models or complex, high-dimensional data.

(Why? Decoupled weight decay helps achieve better generalization)

### Code

```python
import torch.optim as optim

adam_optimizer = optim.AdamW(model.parameters, lr=1e-3)
```
