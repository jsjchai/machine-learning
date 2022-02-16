from housing.download import fetch_housing_data
from housing.load import load_housing_data

# fetch_housing_data()

housing = load_housing_data()
# housing.head()
# housing.info()
# print(housing["ocean_proximity"].value_counts())
print(housing.describe())