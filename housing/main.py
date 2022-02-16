from housing.download import fetch_housing_data
from housing.load import load_housing_data

# fetch_housing_data()

# 加载数据
from housing.test import split_train_test, split_train_test_by_id

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
