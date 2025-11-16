'''
Dataset represents districts.

Data fields:
    - longitude
    - latitude
    - population
    - median age
    - # households
    - median income
    - total # rooms
    - total # bedrooms

'''

from keras.datasets import california_housing as dataset # 1990 census data

(train_data, train_targets), (test_data, test_targets) = (
    dataset.load_data(version="small")
)

print(train_data.shape)
print(test_data.shape)

# feature-wise normalization
def normalize(data):
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    return (data - mean) / std

train_data = normalize(train_data)
test_data = normalize(test_data)

# todo - rescaling


