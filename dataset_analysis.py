import os
import matplotlib.pyplot as plt
import random

train_path = 'D:/coding_practice/AgePrediction/dataset/train/'

age_count = dict()

for age in os.listdir(train_path):
    age_count[float(age)] = len(os.listdir(train_path+age))

for age in os.listdir(train_path):
    if age_count[float(age)] > 3000:
        n = age_count[float(age)]-3000
        delete_list = random.sample(os.listdir(train_path + age), n)
        # print(delete_list)
        for data in delete_list:
            os.remove(train_path + age + '/' + data)

names = list(age_count.keys())
values = list(age_count.values())

plt.figure(figsize=(15, 5))
plt.bar(range(len(age_count)), values, tick_label=names, width=0.5)
plt.xticks(rotation=70, fontsize=7)
plt.show()