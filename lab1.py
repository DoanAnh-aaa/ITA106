import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("Titanic-Dataset.csv")
print(df.head())
def bai_1():
    print("\n-10 dòng dữ liệu đầu tiên:")
    print(df.head(10))

    print("\n-Số lượng bản ghi và thuộc tính:")
    print(df.shape)
    print("-Kiểu dữ liệu của từng cột:")
    print(df.dtypes)

    print("\n---Thống kê---")
    print("-Giá trị trung bình:")
    print(df.mean(numeric_only=True))
    print("-Giá trị lớn nhất:")
    print(df.max(numeric_only=True))
    print("-Giá trị nhỏ nhất:")
    print(df.min(numeric_only=True))
    print("-Độ lệch chuẩn:")
    print(df.std(numeric_only=True))

    print("\n---Trực quan hóa dữ liệu---")
    print("-Histogram cho thuộc tính số (cột Age).")
    plt.figure(figsize=(6,4))
    plt.hist(df['Age'].dropna(), bins=20)
    plt.title('Histogram của Age')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.show()
    print("-Bar chart cho thuộc tính phân loại (cột Sex).")
    plt.figure(figsize=(5,4))
    df['Sex'].value_counts().plot(kind='bar')
    plt.title('Số lượng Nam và Nữ')
    plt.xlabel('Giới tính')
    plt.ylabel('Số lượng')
    plt.show()
bai_1()

def bai_2():
    global df
    missing_values = df.isnull().sum()
    print("\n-Số lượng dữ liệu thiếu:")
    print(missing_values)
    
    mean_age = df['Age'].mean()
    df['Age'] = df['Age'].fillna(mean_age)

    print("\n-Số dòng trùng lặp:")
    print(df.duplicated().sum())
    df = df.drop_duplicates()

    scaler = MinMaxScaler()
    numeric_cols = ['Age', 'Fare']
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # Boxplot trước xử lý
    plt.figure(figsize=(6,4))
    sns.boxplot(data=df[['Age', 'Fare']])
    plt.title('Trước khi làm sạch')
    plt.show()
    # Boxplot sau xử lý
    plt.figure(figsize=(6,4))
    sns.boxplot(data=df[['Age', 'Fare']])
    plt.title('Sau khi làm sạch')
    plt.show()
bai_2()