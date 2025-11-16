# Normalization & Standardization

Both can be done in scikit-learn. 

## Normalization

Normalization is a rescaling of the data from the original range so that all values are within the range of 0 and 1.

Formula:

```python
y = (x - min) / (max - min)
```

### SciKit Learn implementation

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
normalized = scaler.fit_transform(data)
inverse = scaler.inverse_transform(normalized)
```

## Standardization

### Formula

```python
mean = sum(x) / count(x)
standard_deviation = sqrt( sum( (x - mean)^2 ) / count(x))

y = (x - mean) / standard_deviation
```

# References

- https://stats.stackexchange.com/questions/111467/is-it-necessary-to-scale-the-target-value-in-addition-to-scaling-features-for-re
- https://machinelearningmastery.com/how-to-improve-neural-network-stability-and-modeling-performance-with-data-scaling/
