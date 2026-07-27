# Comprehensive Analysis of Forecasting Results

The bar chart above evaluates the performance of five different architectures using **Mean Absolute Percentage Error (MAPE)**. A lower MAPE value indicates a more accurate model.

The experiments are divided across three main dimensions:
1. **Algorithm Type:** Linear Regression, MLP, LSTM, BiLSTM, and Transformer.
2. **Feature Set:** `Load Only` (Autoregressive features) vs. `All Features` (Load + Weather + Time Cyclic + Holiday).
3. **Signal Processing:** `Standard` (Raw signal) vs. `VMD` (Variational Mode Decomposition).

---

## 1. The Impact of VMD (Standard vs. VMD)
**Expected Observation:** Model variations using **VMD** generally outperform their **Standard** counterparts.
* **Why this happens:** The electrical load is a highly complex, non-linear, and non-stationary signal driven by various overlapping factors (daily human routines, weekly schedules, weather anomalies). 
* **VMD's Role:** VMD acts as a mathematical filter that breaks this chaotic signal down into discrete Intrinsic Mode Functions (IMFs). By feeding IMFs into the models instead of the raw chaotic signal, the neural networks (especially LSTMs and Transformers) can easily isolate and learn the low-frequency trends (seasonality) and high-frequency patterns (noise/spikes) separately, leading to drastically reduced prediction errors.

## 2. The Impact of Exogenous Variables (Load Only vs. All Features)
**Expected Observation:** `All Features` usually yields a lower error rate than `Load Only`, though VMD sometimes narrows this gap.
* **All Features:** Weather (especially temperature for heating/cooling systems), explicit holiday flags, and cyclical time encodings provide strict "context" to the model. It helps the model understand *why* a sudden spike is happening (e.g., it's a Sunday holiday with low temperature).
* **Load Only:** Relies purely on the autoregressive nature of the data (past load predicts future load). When combined with VMD, `Load Only` can become surprisingly powerful because the IMFs themselves inherently contain the delayed effects of weather and human behavior trapped within the signal frequencies.

---

## 3. Algorithm Performance Breakdown

### A. Linear Regression (The Baseline)
* **Characteristics:** A simple, fast, and highly interpretable model that captures linear relationships.
* **Result Interpretation:** It serves as the baseline floor. If any Deep Learning model produces a higher MAPE than Linear Regression, it indicates that the neural network is heavily overfitting the training data or requires hyperparameter tuning (e.g., adjusting dropout, learning rate, or layer size).

### B. Multi-Layer Perceptron (MLP)
* **Characteristics:** A standard feed-forward neural network. It maps exact input vectors to the output without an inherent understanding of time sequences.
* **Result Interpretation:** MLP usually performs better than Linear Regression due to its ability to map non-linear weather and load relationships (using ReLU activations). However, it generally underperforms compared to LSTMs because it treats time-series sequences merely as static tabular features.

### C. Long Short-Term Memory (LSTM)
* **Characteristics:** The industry standard for time-series forecasting. It utilizes memory gates to retain context over longer sequences.
* **Result Interpretation:** We expect LSTMs to show a massive leap in accuracy over MLP and LinReg, especially on the `All Features` dataset. It perfectly handles the `(samples, time_steps, features)` 3D structure, remembering how yesterday's trend affects today's outcome.

### D. Bidirectional LSTM (BiLSTM)
* **Characteristics:** Processes the sequence both forwards and backwards simultaneously.
* **Result Interpretation:** In strict real-time forecasting, BiLSTM's advantage is sometimes negligible because the future hasn't happened yet. However, when looking at a moving window of past lags (like our 24h and 168h lags), BiLSTM can sometimes extract richer contextual feature representations than standard LSTM, yielding slightly lower MAPE.

### E. Transformer
* **Characteristics:** Utilizes a Multi-Head Attention mechanism to look at the entire input sequence simultaneously, weighing which specific past hours are most "important" to the current prediction.
* **Result Interpretation:** Transformers are incredibly powerful but are notoriously data-hungry and prone to overfitting on smaller or structurally simple datasets. 
    * If the Transformer performs **best**, it successfully utilized its attention mechanism to correlate complex weather events and IMFs to the target load better than LSTM's sequential gates.
    * If it performs **worse** than LSTM, it implies the dataset (or sequence length) might not be large enough to fully train the attention heads, or the network requires stricter regularization and hyperparameter tuning.

---

## Conclusion & Selection
To select the final model, identify the bar with the lowest MAPE. Typically, the **LSTM or BiLSTM combined with VMD and All Features** yields the most robust and accurate forecasting system for electrical load demands, balancing sequence memorization with sophisticated signal decomposition.