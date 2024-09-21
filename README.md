# A Comparative Study of the Bitcoin Price Prediction Using Machine Learning and Deep Learning Methods
## Abstract
Awesome returns of Cryptocurrencies has received a
lot of attention in recent years. With more than 60% of
market cap, we are going to predict the Bitcoin prices.
We used three machine learning algorithms including linear regression, Convolutional Neural Network(CNN), and
Long Short-term Memory(LSTM) plus one time series models namely the Prophet model. We found out that the Regression model performed slightly better and was able to
outperform other models. Prophet, LSTM and
CNN after a transformation performed more accurately. We
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
Stationary time series are defined as those whose statistical properties such as mean and variance remain unchanged over time. Our verification of time-series stationarity is necessary because a time series model cannot be
generated if it is not stationary. Using the rolling statistics
plots in conjunction with the augmented Dickey-Fuller test
results, we will be able to verify it. In Dickey-Fuller analysis, the null hypothesis is that a unit root is present in a
time series model. An alternate hypothesis is the stationarity of the model. On the other hand, if the series is integrated, then positive changes and negative changes will
happen with probabilities that are independent of the current level; in a random walk, your current position does not
influence which way you will go next.

As Figure. 6 in the Appendix shows rolling mean for
the basic data is increasing. Accordingly, the time series
does not appear to be stationary. Furthermore, the p-value
is greater than 5%, which means the null hypothesis cannot
be rejected. In the augmented Dickey-Tuller statistic, the
test statistic is negative. As it becomes more negative, the
stronger the rejection of the hypothesis that there is a unit
root. As the parameters in Figure. 6 shows -1.74 is greater
than the -2.86 critical value at the %95 confidence bound.
Therefore, we cannot reject the null hypothesis. Then to
make our time series stationary, we applied some transformations.

Time series patterns can be classified into three categories: trend, seasonality, and cycles. A trend-cycle component, a seasonal component, and a remainder component
(the rest of the time series) are usually combined, thus giving us a trend-cycle component, a seasonal component, and
a remainder component. Figure. 8 in the Appendix shows
these trends for the data.

Time series can be made stationary by estimating trends
and seasonality and then removing them from the series.
The next step is to apply the forecasting technique; the last
step is to transform the forecasted values into the original
scale by adding the estimated trend and seasonality.In addition, as the time series is constantly increasing, we apply a
log transform, and a square root transform to flatten it. After
that, We take the difference of the observation at a particular
instant with that at the previous instant.

As Figure. 7 in the Appendix shows we achieve significantly better results: The test statistic is significantly lower
than the critical value of 1%. With more than 99% confidence, we can conclude that this series is stationary. Moreover, as the p-value is inferior to the 5% threshold, the null
hypothesis is rejected, meaning that the Dickey-Fuller test
is verified. As a result, the time series is stationary. Now we
can implement our predictive models.

## 2.3. Predictive Models
This section provides brief explanation of the methods
that are used for the goal of this project.

