# A Comparative Study of the Bitcoin Price Prediction Using Machine Learning and Deep Learning Methods
## Abstract
Awesome returns of Cryptocurrencies has received a
lot of attention in recent years. With more than 60% of
market cap, we are going to predict the Bitcoin prices.
We used three machine learning algorithms including linear regression, Convolutional Neural Network(CNN), and
Long Short-term Memory(LSTM) plus two time series models containing the Prophet, and Auto-Regressive Integrated
Moving Average(ARIMA) model. We found out that the Regression model performed slightly better and was able to
outperform the trivial model of just getting the yesterday’s
price as the forecast of today’s price. Prophet, LSTM and
CNN after a transformation performed more accurate. We
add a simple algorithm that can give us trading orders that
performs 5% better than long-term buy and hold strategy,
which is a significant improvement.

## 1. Introduction
Time series forecasting has been dominated by linear
methods because they are well understood and effective on
many simpler forecasting problems. However, by using
deep learning we are able to automatically learn arbitrary
complex mappings from inputs to outputs [1]. The bitcoin
price prediction that has been worked on in this project is a
regression problem, where the prediction model predicts the
next Bitcoin price based on the previous. We studied, implemented and compared various Machine Learning and deep
learning methods, such as, prophet, Convolutional Neural
Network (CNN) and long short-term memory (LSTM).

## 2. Methodology
### 2.1 Data
The dataset in the project is imported by Python using
Yahoo APIs and contains the USD bitcoin price of each day
from 2014 to 2021 and it includes six features: Open indicates the price at the beginning of the day, Close indicates
the price at the end of the day, Adjusted Close, Volume of
the sale, High price of the day and Low price of the day. The
dataset has been examined for null values and it didn’t have
any missing values. Moreover, the ‘Close’ and ‘Adj Close’
columns are identical because Cryptocurrency prices differ
from traditional stock prices. So ‘Close’ price is chosen as
the target variable.
![Figure 2. Plot of close price and volume](https://raw.githubusercontent.com/Alighaemmaghami/Bitcoin_Prediction_Comparison/refs/heads/main/figures/fig_1_bitcoin_price.png)



A figure of the ’Close’ price and the ’Volume’ is plotted
to look for any visual anomalies and as it is shown in Figure
2, There was a significant spike in Volume on the 26th of
February 2021 - around triple of its normal value. So by
checking out different data sources [2], we found that they
were aligned, and no further action was required.
The dataset is divided to 80% of training and validation
, and 20% for test data, Figure 1.

### 2.2. Checking the stationarity of data


