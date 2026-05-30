import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
import yfinance as yf
def bai_1():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    print(df.head())

    features = [
    'sepal length (cm)',
    'sepal width (cm)',
    'petal length (cm)'
    ]
    for feature in features:
        plt.figure(figsize=(6,4))
        plt.hist(df[feature], bins=20)
        plt.title(f'Histogram of {feature}')
        plt.xlabel(feature)
        plt.ylabel('Frequency')
        plt.show()
    print("""\n-Nhận xét Histogram: Sepal Length phân phối cân đối, dữ liệu tập trung trong khoảng 5-6cm;
        Sepal Width phân bố hẹp hơn và có vài giá trị bất thường; Petal Length do dữ liệu gồm 
        nhiều loại hoa khác nhau nên tách thành nhiều nhóm. """)

    plt.figure(figsize=(8,5))

    for feature in features:
        sns.kdeplot(df[feature], label=feature, fill=True)
    plt.legend()
    plt.title("Density Plot")
    plt.show()
    print("""\n-Nhận xét KDE: Sepal length có dạng gần chuẩn, Petal length có nhiều đỉnh từ đó dữ liệu
        chia nhóm rõ rệt, Sepal width phân bố hẹp.""")

    plt.figure(figsize=(8,5))
    sns.boxplot(data=df[features])
    plt.title("Boxplot")
    plt.show()
    print("""\n-Nhận xét Boxplot: Sepal width có nhiều outliers, Petal length có khoảng giá trị rộng, dữ liệu
        không hoàn toàn phân bố chuẩn.""")

bai_1()

def bai_2():
    ticker = "AAPL"
    stock = yf.download(
        ticker,
        start="2022-01-01",
        end="2024-01-01"
    )
    print(stock.head())

    plt.figure(figsize=(12,5))
    plt.plot(stock["Close"])
    plt.title("Apple Close Price")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.show()
    print("\n-Nhận xét: Giá cổ phiếu Apple có xu hướng tăng dài hạn và cũng có các giai đoạn giảm mạnh do biến động thị trường." )

    stock["MA50"] = stock["Close"].rolling(window=50).mean()
    stock["MA200"] = stock["Close"].rolling(window=200).mean()

    plt.figure(figsize=(12,5))
    plt.plot(stock["Close"], label="Close Price")
    plt.plot(stock["MA50"], label="MA50")
    plt.plot(stock["MA200"], label="MA200")
    plt.legend()
    plt.title("Moving Average Analysis")
    plt.show()
    print("\n-Nhận xét: Trong giai đoạn này Apple chủ yếu có xu hướng tăng dài hạn (MA50 nằm trên MA200).")

bai_2()
def bai_3():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    print(df.head())

    plt.figure(figsize=(10,5))
    sns.boxplot(data=df)
    plt.title("Boxplot for All Features")
    plt.show()
    
    for column in df.columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower) | (df[column] > upper)]
    print(f"\n{column}")
    print(outliers[column])

    print("""\n-Các giá trị Outlier là cá giá trị khác biệt so với các giá trị trong cùng 
    mảng dữ liệu gây ảnh hưởng đến độ chính xác phân tích.""")

bai_3()
def bai_4():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    print(df.head())

    corr_matrix = df.corr()
    print(corr_matrix)

    plt.figure(figsize=(8,6))
    sns.heatmap(
    corr_matrix,
    annot=True,
    cmap="coolwarm"
    )
    plt.title("Correlation Matrix")
    plt.show()
    
    print("\n-Petal Length và Petal Width là cặp biến có tương quan cao vì khi petal length tăng thì petal width cũng tăng.")

bai_4()
