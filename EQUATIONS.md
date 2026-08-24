# Mathematical Equations for STLF-ENTSO-2026 Models

This document details the mathematical equations for the three neural network architectures used in the Short-Term Load Forecasting (STLF) pipeline: **LSTM**, **BiLSTM**, and **Quantile BiLSTM**.

### Notation
* $X = [x_1, x_2, \dots, x_T]$ is the input sequence with a lookback window of $T=24$.
* $H$ is the forecast horizon ($H=24$).
* $W$ and $U$ represent weight matrices, and $b$ represents bias vectors.
* $\sigma$ represents the sigmoid activation function.
* $\odot$ represents the element-wise (Hadamard) product.

---

## 1. Standard LSTM (Deterministic)
The standard LSTM updates its cell state $c_t$ and hidden state $h_t$ at each time step $t$ using the following gating mechanisms:

**Forget gate** (decides what information to discard from the cell state):
ft=σ(Wfxt+Ufht−1+bf)

**Input gate** (decides which values to update):
$$i_t = \sigma(W_i x_t + U_i h_{t-1} + b_i)$$

**Cell candidate** (creates a vector of new candidate values):
$$\tilde{c}_t = \tanh(W_c x_t + U_c h_{t-1} + b_c)$$

**Cell state update** (updates the old state $c_{t-1}$ into the new state $c_t$):
$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$

**Output gate and Hidden state** (decides what parts of the cell state to output):
$$o_t = \sigma(W_o x_t + U_o h_{t-1} + b_o)$$
$$h_t = o_t \odot \tanh(c_t)$$

**Final Prediction Layer:**
In our `return_sequences=False` architecture, only the final hidden state $h_T$ is passed to a Dense layer to produce the multi-step point forecast $\hat{Y}$ of shape $(H,)$:
$$\hat{Y} = W_{out} h_T + b_{out}$$

---

## 2. Bidirectional LSTM (BiLSTM)
A BiLSTM processes the time series in both forward and backward directions to capture context from both past and "future" (within the input window).

**Forward Pass:**
$$\overrightarrow{h}_t = \text{LSTM}_{forward}(x_t, \overrightarrow{h}_{t-1})$$

**Backward Pass:**
$$\overleftarrow{h}_t = \text{LSTM}_{backward}(x_t, \overleftarrow{h}_{t+1})$$

**Concatenation:**
For the final prediction, the model takes the last hidden state from the forward pass ($\overrightarrow{h}_T$) and the first hidden state from the backward pass ($\overleftarrow{h}_1$, since it started at $T$ and ended at $1$):
$$h_{final} = [\overrightarrow{h}_T \oplus \overleftarrow{h}_1]$$ 
*(where $\oplus$ represents vector concatenation)*

**Final Prediction Layer:**
The combined hidden state is then passed to a Dense layer to produce the multi-step point forecast $\hat{Y}$ of shape $(H,)$:
$$\hat{Y} = W_{out} h_{final} + b_{out}$$

---

## 3. Quantile BiLSTM (Probabilistic)
The architecture of the Quantile BiLSTM is identical to the BiLSTM above, but the output layer and the loss function are fundamentally changed to produce prediction intervals.

**Output Layer:**
Instead of outputting $H$ values, the Dense layer outputs $H \times k$ values (where $k$ is the number of quantiles; for P10, P50, P90, $k=3$). The output is then reshaped to $(H, k)$.
Let $Q = \{q_1, q_2, \dots, q_k\}$ be the target quantiles. The prediction for horizon step $h$ at quantile $q$ is $\hat{y}_{h,q}$:
$$\hat{Y}_{flattened} = W_q h_{final} + b_q$$

**Multi-Quantile Pinball Loss:**
Instead of Mean Squared Error (MSE), the model optimizes the Pinball Loss (Quantile Loss). For a single target value $y_h$ and a predicted quantile $\hat{y}_{h,q}$, the pinball loss $L_q$ is defined as:

$$L_q(y_h, \hat{y}_{h,q}) = \begin{cases} q (y_h - \hat{y}_{h,q}) & \text{if } y_h \ge \hat{y}_{h,q} \\ (q - 1) (y_h - \hat{y}_{h,q}) & \text{if } y_h < \hat{y}_{h,q} \end{cases}$$

This asymmetric penalty ensures that predicting the 90th percentile ($q=0.9$) penalizes under-predictions heavily and over-predictions lightly, pushing the prediction to the correct quantile boundary.

The total loss for the entire forecast window across all $k$ quantiles is the mean of all individual pinball losses:
$$\mathcal{L}_{total} = \frac{1}{H \times k} \sum_{h=1}^H \sum_{q \in Q} \max\Big(q (y_h - \hat{y}_{h,q}), \ (q - 1) (y_h - \hat{y}_{h,q})\Big)$$
