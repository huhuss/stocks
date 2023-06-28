Table of contents
Models
Agents
Realtime Agent
Data Explorations
Simulations
Tensorflow-js
Misc
Results
Results Agent
Results signal prediction
Results analysis
Results simulation
Contents
Models
Deep-learning models
LSTM
LSTM Bidirectional
LSTM 2-Path
GRU
GRU Bidirectional
GRU 2-Path
Vanilla
Vanilla Bidirectional
Vanilla 2-Path
LSTM Seq2seq
LSTM Bidirectional Seq2seq
LSTM Seq2seq VAE
GRU Seq2seq
GRU Bidirectional Seq2seq
GRU Seq2seq VAE
Attention-is-all-you-Need
CNN-Seq2seq
Dilated-CNN-Seq2seq
Bonus

How to use one of the model to forecast t + N, how-to-forecast.ipynb
Consensus, how to use sentiment data to forecast t + N, sentiment-consensus.ipynb
Stacking models
Deep Feed-forward Auto-Encoder Neural Network to reduce dimension + Deep Recurrent Neural Network + ARIMA + Extreme Boosting Gradient Regressor
Adaboost + Bagging + Extra Trees + Gradient Boosting + Random Forest + XGB
Agents
Turtle-trading agent
Moving-average agent
Signal rolling agent
Policy-gradient agent
Q-learning agent
Evolution-strategy agent
Double Q-learning agent
Recurrent Q-learning agent
Double Recurrent Q-learning agent
Duel Q-learning agent
Double Duel Q-learning agent
Duel Recurrent Q-learning agent
Double Duel Recurrent Q-learning agent
Actor-critic agent
Actor-critic Duel agent
Actor-critic Recurrent agent
Actor-critic Duel Recurrent agent
Curiosity Q-learning agent
Recurrent Curiosity Q-learning agent
Duel Curiosity Q-learning agent
Neuro-evolution agent
Neuro-evolution with Novelty search agent
ABCD strategy agent
Data Explorations
stock market study on TESLA stock, tesla-study.ipynb
Outliers study using K-means, SVM, and Gaussian on TESLA stock, outliers.ipynb
Overbought-Oversold study on TESLA stock, overbought-oversold.ipynb
Which stock you need to buy? which-stock.ipynb
Simulations
Simple Monte Carlo, monte-carlo-drift.ipynb
Dynamic volatility Monte Carlo, monte-carlo-dynamic-volatility.ipynb
Drift Monte Carlo, monte-carlo-drift.ipynb
Multivariate Drift Monte Carlo BTC/USDT with Bitcurate sentiment, multivariate-drift-monte-carlo.ipynb
Portfolio optimization, portfolio-optimization.ipynb, inspired from https://pythonforfinance.net/2017/01/21/investment-portfolio-optimisation-with-python/
Tensorflow-js
I code LSTM Recurrent Neural Network and Simple signal rolling agent inside Tensorflow JS, you can try it here, huseinhouse.com/stock-forecasting-js, you can download any historical CSV and upload dynamically.

Misc
fashion trending prediction with cross-validation, fashion-forecasting.ipynb
Bitcoin analysis with LSTM prediction, bitcoin-analysis-lstm.ipynb
Kijang Emas Bank Negara, kijang-emas-bank-negara.ipynb
Results
Results Agent
This agent only able to buy or sell 1 unit per transaction.

Turtle-trading agent, turtle-agent.ipynb


Moving-average agent, moving-average-agent.ipynb


Signal rolling agent, signal-rolling-agent.ipynb


Policy-gradient agent, policy-gradient-agent.ipynb


Q-learning agent, q-learning-agent.ipynb


Evolution-strategy agent, evolution-strategy-agent.ipynb


Double Q-learning agent, double-q-learning-agent.ipynb


Recurrent Q-learning agent, recurrent-q-learning-agent.ipynb


Double Recurrent Q-learning agent, double-recurrent-q-learning-agent.ipynb


Duel Q-learning agent, duel-q-learning-agent.ipynb


Double Duel Q-learning agent, double-duel-q-learning-agent.ipynb


Duel Recurrent Q-learning agent, duel-recurrent-q-learning-agent.ipynb


Double Duel Recurrent Q-learning agent, double-duel-recurrent-q-learning-agent.ipynb


Actor-critic agent, actor-critic-agent.ipynb


Actor-critic Duel agent, actor-critic-duel-agent.ipynb


Actor-critic Recurrent agent, actor-critic-recurrent-agent.ipynb


Actor-critic Duel Recurrent agent, actor-critic-duel-recurrent-agent.ipynb


Curiosity Q-learning agent, curiosity-q-learning-agent.ipynb


Recurrent Curiosity Q-learning agent, recurrent-curiosity-q-learning.ipynb


Duel Curiosity Q-learning agent, duel-curiosity-q-learning-agent.ipynb


Neuro-evolution agent, neuro-evolution.ipynb


Neuro-evolution with Novelty search agent, neuro-evolution-novelty-search.ipynb


ABCD strategy agent, abcd-strategy.ipynb


Results signal prediction
I will cut the dataset to train and test datasets,

Train dataset derived from starting timestamp until last 30 days
Test dataset derived from last 30 days until end of the dataset
So we will let the model do forecasting based on last 30 days, and we will going to repeat the experiment for 10 times. You can increase it locally if you want, and tuning parameters will help you by a lot.

LSTM, accuracy 95.693%, time taken for 1 epoch 01:09


LSTM Bidirectional, accuracy 93.8%, time taken for 1 epoch 01:40


LSTM 2-Path, accuracy 94.63%, time taken for 1 epoch 01:39


GRU, accuracy 94.63%, time taken for 1 epoch 02:10


GRU Bidirectional, accuracy 92.5673%, time taken for 1 epoch 01:40


GRU 2-Path, accuracy 93.2117%, time taken for 1 epoch 01:39


Vanilla, accuracy 91.4686%, time taken for 1 epoch 00:52


Vanilla Bidirectional, accuracy 88.9927%, time taken for 1 epoch 01:06


Vanilla 2-Path, accuracy 91.5406%, time taken for 1 epoch 01:08


LSTM Seq2seq, accuracy 94.9817%, time taken for 1 epoch 01:36


LSTM Bidirectional Seq2seq, accuracy 94.517%, time taken for 1 epoch 02:30


LSTM Seq2seq VAE, accuracy 95.4190%, time taken for 1 epoch 01:48


GRU Seq2seq, accuracy 90.8854%, time taken for 1 epoch 01:34


GRU Bidirectional Seq2seq, accuracy 67.9915%, time taken for 1 epoch 02:30


GRU Seq2seq VAE, accuracy 89.1321%, time taken for 1 epoch 01:48


Attention-is-all-you-Need, accuracy 94.2482%, time taken for 1 epoch 01:41


CNN-Seq2seq, accuracy 90.74%, time taken for 1 epoch 00:43


Dilated-CNN-Seq2seq, accuracy 95.86%, time taken for 1 epoch 00:14


Bonus

How to forecast,


Sentiment consensus,


Results analysis
Outliers study using K-means, SVM, and Gaussian on TESLA stock


Overbought-Oversold study on TESLA stock


Which stock you need to buy?


Results simulation
Simple Monte Carlo


Dynamic volatity Monte Carlo


Drift Monte Carlo


Multivariate Drift Monte Carlo BTC/USDT with Bitcurate sentiment


Portfolio optimization
