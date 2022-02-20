from housing.download import fetch_housing_data
from housing.test import split_train_test, split_train_test_by_id
from housing.load import load_housing_data
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os.path

# fetch_housing_data()

# 加载数据
housing = load_housing_data()

# 查看图表信息
# housing.head()
# housing.info()
# print(housing["ocean_proximity"].value_counts())
# print(housing.describe())

# 创建测试集
train_set, test_set = split_train_test(housing, 0.2)
print(len(test_set))
print(len(train_set))

housing_with_id = housing.reset_index()
housing_with_id.info()
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")
print(len(test_set))
print(len(train_set))

housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "id")

print("test_set")
print(test_set.head())
print("train_set")
print(train_set.head())

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
print("sklearn test_set")
print(test_set.head())

print("-" * 100)
# 收入中位数
# housing["median_income"].hist()
# plt.show()

#
image_path = os.path.join("images", "california.png")
california_img = mpimg.imread(image_path)
ax = housing.plot(kind='scatter', x='longitude', y='latitude', alpha=0.4, s=housing['population'] / 100,
                  label='population', figsize=(10, 7),
                  c='median_house_value', cmap=plt.get_cmap('jet'), colorbar=False)
plt.imshow(california_img, extent=[-124.55, -113.80, 32.45, 42.05], alpha=0.5)
plt.ylabel("Latitude", fontsize=14)
plt.xlabel("Longitude", fontsize=14)
prices = housing["median_house_value"]
tick_values = np.linspace(prices.min(), prices.max(), 11)
cbar = plt.colorbar()
cbar.ax.set_yticklabels(["$%dk" % (round(v / 1000)) for v in tick_values], fontsize=14)
cbar.set_label('Median House Value', fontsize=16)
plt.legend(fontsize=16)
plt.show()
