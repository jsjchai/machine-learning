import matplotlib.pyplot as plt

from housing.load import load_housing_data

housing = load_housing_data()
housing.hist(bins=50, figsize=(20, 15))
plt.show()
