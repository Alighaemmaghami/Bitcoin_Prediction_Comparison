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

### 2.3.1 Linear Regression
Linear Regression models are considered to be the classical
options for forecasting the stock prices. Different scenarios
can be used using Linear Regression models. For univariate series forecasting, which uses only closing prices, use
the Simple Linear Regression (SLR) model, and the Multiple Linear Regression (MLR) model, which uses both price
and volume data [3].

We built our dataset for both models so that we had a set of
inputs (X) and outputs (Y) that were temporally dependent.
We used a one-step ahead forecast, in which the output Y is
the value from the next (future) point in time, and the inputs
X are one or more values from the past, i.e., lagged values.
The dataset for the Linear Regression and Univariate LSTM
models only contains the daily closing price series, hence
the close feature has only one lag parameter. The dataset
comprises both close and volume (USD) series in the Multiple Linear Regression and Multivariate LSTM models, thus
we utilise two separate lag settings, one for the close and
one for the volume feature [3].

We consider 5 different scenarios:
1. Forecasting with lagged prices up to 10 days
2. Forecasting with lagged prices and moving average of 3
and 9 days
3. Forecasting with lagged prices and volume values up to
10 days and moving average of 3 and 9 days
4. Forecasting with relative changes of values of lagged
prices up to 10 days (Return(t) = P(t)/P(t-1))
5. Forecasting with polynomial model with lagged prices
up to 10 days

We trained these 5 scenarios each of which one to 10
days lagged prices. Interestingly, the 4th scenario which uses the return value of each day works better in here, probably because it reduces the effect of absolute value of price
in different times. Another, interesting finding is that the
first three scenario work better when just using the last
lagged data; adding more data than the yesterday’s price
will reduce the performance of the model.
However, this does not apply to the fourth and fifth scenarios. The fifth scenario that uses a polynomial model
works better when it uses the first three lagged values; but
in general this model does not work well probably because
of over-fitting.

The fourth scenario works better. It uses the return values
instead of absolute values. This model works better when
we use six previous lagged values. We can interpret that by
using return values instead of absolute values we reduce the
effects of changing the absolute price and let the model to
predict the behaviour of the price and other external factors
better. In that situation, it can memorize the behaviour of
not just one lagged values but six values and it employs that
as a positive factor for forecasting the tomorrow’s price.

### 2.3.2 Prophet
Prophet is a timeseries modeling based on additive regressive model (GAM Formulation) to predict trends by considering yearly, weekly, and daily seasonality effects. It fits
models in Stan which is a platform for statistical modeling.
Since this modeling considers seasonality as an important
factor, we divided our data to different seasons to evaluate
and compare the model results in various seasonalities. In
terms of preprocessing the data, Prophet can handle missing
data, however, we did not have any missing values. Also,
we renamed columns “Date” and “Close” to “ds” and ”y” to
achieve an acceptable input for Prophet model. Moreover,
Prophet contains cross validation to assess forecast error by
using historical data. In our project, we chose some cutoff
points in the history indicating the validation sections for
fitting the model so that we could make a comparison between the actual values and predicted values. The output
of this procedure is a data frame with actual(y) and predicted(yhat) values.


### 2.3.3 CNN
Convolutional Neural Network models, or CNNs for short,
normally are used for two-dimensional image data. However, CNNs can be used to model univariate time series forecasting problems too. The CNN model will learn a function
that maps a sequence of past observations as input to an
output observation. In this project, we have considered a
window size of 60 days to split our data into sequences. In
the process of preparing the data we manually set the seed
to 0 to provide a better reproducibility through all the platforms and scaled it with MinMaxScaler from scikit-learn
package. Using the PyTorch package we defined our cnn
model. Since the time series data is univariate it only has
one convolutional layer with in channels equal to the window size, out channels equal to 1 and it has 128 hidden layers which is equal to the batch size. This batch size has been
chosen arbitrarily. The model is activated by a ReLU function. In order to be used in PyTorch Sequential, we have
flatten the range of dims into a tensor. To define the hidden
fully connected linear layer and we apply the PyTorch Linear function to transform it by the equation y = x * AT +
b.

### 2.3.4 Long Short-term Memory Model
The Long Short-Term Memory (LSTM) Neural Network is
another algorithm chosen for this analysis. Deep learning
recurrent neural networks (RNNs) such as LSTM are commonly used to predict time series data. Due to the LSTM’s
logic gates (input, output, and forget gates), it is capable of retaining important information and deleting unnecessary
information, making it a useful model for interpreting patterns over time. This model was built with Keras, using the
Sequential class and stacking different LSTM layers. Also,
for preventing over fitting after every LSTM layer we used
dropout layer.

Having scaled the data in the manner described in section 2.2, we reorganize the training data to make it suitable
for LSTM training. In order to do this, we defined each observation of training set as an array containing the past 90
observations that will be used to predict the next day’s price.

## 2.4. Results
### 2.4.1 Model Comparisons
The evaluation metrics that have been used in this project
are Mean absolute error(MAE) and Root-mean-square deviation(RMSE due to the fact that predicting price in time
series is a regression problem. However, we haven’t used r2ˆ
score.

### 2.4.2 Let’s make some money
In the last section, LSTM was trained most successfully, but
when will it be profitable? We decided to test this model
to see whether it would be possible to invest money in it
(please don’t!). Using $100000 as base money, we made an
algorithm that buys as much stock as possible when the next
day’s price movement is predicted to move over 0.5% upward and sells as much stock as possible when it is predicted
to move over 0.5% downward. In total, we were able to earn
$683408 from the prediction, compared to $648790 from
the buy and hold strategy, translating into a $34618 profit.
In Figure. 3, you can see the buys (green arrows)/sells (red
arrows) made by the model on the actual test set, while the
Figure 4 shows the percentage change (of the next day prediction) during the same period.

## 3. Conclusions
Generally, machine and deep learning algorithms are
not the best approach to develop trading models. However, the automate process and reinforcement learning of
the deep learning makes it a very interesting approach and it may yield productive, profitable results in future works [5].
We kept experimenting different approaches to improve the
models. The most interesting change was when we added
seasonality and changed our data scaler and as a result the
performance of all the models improved significantly since
it omitted the effect of extreme price changes.
As it can be seen in Figure. 5 our models followed the trends
very well and can capture most of the changes. Regression
model performed a little better and could outperform yesterday’s price. It is probably because the price is extremely
dependent on the previous prices and follows the previous
prices.

## References
[1] https://machinelearningmastery.com/deep-learning-for-timeseries-forecasting/

[2] https://coinmarketcap.com/historical/20210226/

[3] Uras, N., Marchesi, L., Marchesi, M., and Tonelli, R. (2020).Forecasting Bitcoin closing price series using linear regression
and neural networks models. PeerJ Computer Science, 6, e279.

[4] https://people.duke.edu/ rnau/411arim.htm

[5] https://towardsdatascience.com/using-neural-networks-topredict-stock-prices-dont-be-fooled-c43a4e26ae4e

