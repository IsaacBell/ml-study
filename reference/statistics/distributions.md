# Normal Distribution (a.k.a. Gaussian Distribution)

Continuous distribution, symmetrical and bell-shaped.

## Properties

- Mean, median, and mode are all equal (in a perfectly normal distribution)
- Bell-shaped curve: defined by its mean and std dev

# Binomial Distributions

Famously used to calculate coin flip probability. 

Binomial distributions model the count of successes in a series of independent trials. 

More formally, a binomial distribution is the sum of independent and identically distributed Bernoulli random variables.

## Bernoulli Distributions

- Basically a single trial of a binomial distribution. The probability of winning 1 coin toss is a Bernoulli random variable.
- All Bernoulli distributions are binomial distributions, but most binomial distributions are not Bernoulli distributions.
- If the success case has probability P and the failure case has probability 1-p, it is a Bernoulli distribution.
- If you toss a coin 5 times, you could win anything from 0 to 5 dollars. The prob of winning $5 is p^5.
  - P of getting 3 heads is (p^3) * (1 - p^2)

### Bernoulli Distributions - Data Imbalance Handling

- SMOTE: Oversample the Minority Class
- Boosting: Gradient Boosting
- Bagging: Random Forest
- Choose the Right Eval Metrics
  - F1 score
  - Recall
  - Precision
  - PR-ROC (area under ROC curve)

## Assumptions

- fixed # of independent trials
- constant probability of success
- binary outcomes (true/false, heads/tails, etc)

## Properties

- mean represents expected # successes in n trials
  - mean = np 
  - For 10 coin flips, n = 10 and p = 0.5, so mean = 5
- variance quantifies the spread around the mean
  - variance = mean * (1 - p)
  - For 10 coin flips, variance = 0.5 * 0.5 = 0.25

### Code Example

```python
from scipy.stats import bernoulli
np.random.seed(42)

# Simulate five coin flips and get the number of heads
five_coin_flips = bernoulli.rvs(p=0.5, size=5)
coin_flips_sum = sum(five_coin_flips)
```

# Poisson

Poisson distributions model discrete events in a fixed interval. 

## Assumptions

- known average rate that doesn't change over time 
- no dependence on time
- events happen independently - two events can't occur at the same time

## Properties

Mean and variance are equal. Both are represented as λ (lambda), the average # of interval events.

The more events, the more variability in the number of occurrences.

## Use Cases

- capacity planning
- rare events (ex - rare diseases, epidimiology, network modeling)
- # of emails/call received in an hour 

## Challenges

- Low event rates lead to high variability in outcomes (small λ)

### Naive Code Implementation

```python
import math

def poisson_probability(k, lam):
	"""
	Calculate the probability of observing exactly k events in a fixed interval,
	given the mean rate of events lam, using the Poisson distribution formula.
	:param k: Number of events (non-negative integer)
	:param lam: The average rate (mean) of occurrences in a fixed interval
	"""

    # math.exp: Euler's constant to the power of (num arg)
    eulers = math.exp(-lam)
	val = (lam ** k) * eulers
    val = val / math.factorial(k)
	return round(val,5)
```


