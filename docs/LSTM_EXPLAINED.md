# LSTM_EXPLAINED.md — the deep-learning stage, in plain language

A companion to `pipeline/4_LSTM_model_pipeline.ipynb`, written assuming **little prior
deep-learning knowledge**. It explains *what* an LSTM is, *why* each design choice was made,
what the **DLinear** baseline is for, and why we use **Poutyne**. Read it alongside the
notebook.

> Framing reminder: we are not trying to *win* with deep learning. Everything in this project
> says the limit is **signal in the data, not model power**. The LSTM is a **capacity-ceiling
> check** — "even a deep sequence model can't beat price / a naive baseline." That is a clean,
> honest result. See `CONFOUNDS.md` and `CONCLUSIONS.md`.

---

## 1. The problem, in one picture

Every day *t* we have a row of features (returns, on-chain metrics, …) and we want to predict
a **signal** for the near future: Buy / Hold / Sell. A tree model (XGBoost) looks at **one
row** at a time. A **sequence model** instead looks at a **window of the last N days** and asks
"given how the last 30 days *evolved*, what happens next?"

```
      features on day t-29, t-28, …, t-1, t   ->   [ sequence model ]   ->   Buy / Hold / Sell at t
      └──────────── one window (N=30 days) ────────────┘
```

The hope is that the *pattern over time* (momentum building, activity spiking) carries
information a single row misses. (Spoiler: on this data it doesn't — but proving that with a
sequence model is the point.)

---

## 2. What is an LSTM? (intuition, no math)

**RNN (Recurrent Neural Network):** a network that reads a sequence one step at a time and
carries a **memory** (a "hidden state") forward. At each day it updates its memory using the
new day's features. By the end of the window, the memory is a summary of the whole sequence.

**The problem with plain RNNs:** they forget. Over many steps the memory gets overwritten, so
they struggle with *long-term* dependencies (something 25 days ago influencing today).

**LSTM (Long Short-Term Memory):** an RNN with a smarter memory. It adds little **gates** —
think of them as valves controlled by the network itself:

- **Forget gate** — how much of the old memory to keep vs. throw away.
- **Input gate** — how much of the new day's information to write into memory.
- **Output gate** — how much of the memory to expose as this step's output.

Because the gates let the network *choose* what to remember and for how long, an LSTM can hold
onto relevant information across a long window and ignore noise. That's the whole idea: **a
memory with learnable valves.** You don't implement the gates — `nn.LSTM` does it for you.

---

## 3. Our architecture, layer by layer (and *why*)

The model in the notebook (`LSTMClassifier`) is deliberately **small** — we have only ~2,000
days of data, so a big model would just memorize noise (overfit).

```python
class LSTMClassifier(nn.Module):
    def __init__(self, n_features, hidden_size=48, num_layers=1, n_classes=3, dropout=0.3):
        self.lstm = nn.LSTM(n_features, hidden_size, num_layers,
                            batch_first=True, dropout=... )
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, n_classes))
    def forward(self, x):                 # x: (batch, seq_len, n_features)
        out, _ = self.lstm(x)             # out: (batch, seq_len, hidden_size)
        last = out[:, -1, :]              # take the LAST day's memory
        return self.head(last)            # -> (batch, 3) logits
```

**Input shape `(batch, seq_len, n_features)`** — three dimensions:
- `batch` = how many windows we process at once (for speed),
- `seq_len` = days per window (30),
- `n_features` = metrics per day (~37 for BTC).
`batch_first=True` tells PyTorch our batch dimension comes first (the intuitive order).

**`hidden_size=48`** — the size of the memory vector. Bigger = more capacity to represent
patterns, but more parameters to overfit. 48 is small on purpose (tiny data). *Choice reason:
capacity matched to data size.*

**`num_layers=1`** — one LSTM layer. Stacking layers (2–3) can capture more complex patterns
but needs more data. With ~2k rows, 1 layer is the safe default. *Choice reason: avoid
overfitting.*

**Take the last time step (`out[:, -1, :]`)** — the LSTM outputs a memory vector for *every*
day in the window, but the memory at the **final day** has "seen" the whole sequence, so it's
the natural summary to classify from. *Choice reason: it summarizes the full window.*

**`Dropout(0.3)`** — during training, randomly zero out 30% of the connections each step. This
stops the network relying too much on any one feature and is a cheap, standard **regularizer**
(anti-overfitting). *Choice reason: tiny data → strong need for regularization.*

**`Linear(hidden_size, n_classes)`** — a final layer mapping the 48-number memory to **3
numbers**, one per class (Sell / Hold / Buy). These raw numbers are called **logits**; the
loss function turns them into probabilities internally. *Choice reason: 3 classes = our
signal.* (We do **not** add a softmax here — `CrossEntropyLoss` expects raw logits.)

---

## 4. The key hyper-parameters and why these values

| Setting | Value | Why |
|--------|-------|-----|
| `SEQ_LEN` (window) | 30 | ~1 month of context; long enough for patterns, short enough to keep many samples. Try 14 / 60. |
| `hidden_size` | 48 | Small memory — matched to ~2k rows to avoid overfitting. |
| `num_layers` | 1 | One layer is plenty for this little data. |
| `dropout` | 0.3 | Standard regularization; higher = more anti-overfit. |
| `batch_size` | 64 | Windows per training step; a speed/stability trade-off. |
| `lr` (learning rate) | 1e-3 | The default Adam step size — how big each update is. |
| `epochs` | 40 (+ early stop) | Max passes over the data; early stopping ends sooner if it stops improving. |

**None of these are magic numbers.** They're sensible defaults for small tabular-sequence data.
Because our conclusion is "no signal," we deliberately **don't** run a huge hyper-parameter
search — that would risk *finding a lucky configuration* (a multiple-testing trap, see
`CONFOUNDS.md`).

---

## 5. Training choices (the non-obvious ones)

- **Loss = `CrossEntropyLoss` with class weights.** Cross-entropy is the standard loss for
  classification. Our classes are imbalanced (Hold is rare). Instead of **SMOTE oversampling**
  (which your reference notebook used — correct for tabular data but it **leaks** on a time
  series by inventing rows), we pass **inverse-frequency class weights** so the loss pays more
  attention to rare classes. *This is the single most important adaptation from your reference.*
- **Optimizer = `Adam`.** The workhorse optimizer — adapts the step size per parameter. Good
  default, rarely the thing to tune first.
- **Early stopping.** We hold out the **last 15% of each training fold** as a validation set and
  stop training when validation loss stops improving (patience 6). This prevents overfitting and
  saves time. The validation split is **time-ordered** (the *end* of the fold), never random —
  random would leak the future.
- **Per-fold scaling + walk-forward.** Exactly as in `2a`/`2b`: fit the scaler on the training
  fold only, march forward in time, never peek ahead. The windowing respects this — a window
  for day *t* uses only days ≤ *t*.

---

## 6. The DLinear baseline — and why it matters

**DLinear** ("Decomposition + Linear") is a deliberately *trivial* model from Zeng et al.
(AAAI 2023, *"Are Transformers Effective for Time Series Forecasting?"*). It's essentially **one
linear layer** — no memory, no gates, no non-linearity. That paper showed this toy model
**matched or beat** state-of-the-art Transformers on standard benchmarks — a famous reality
check that model complexity is often *not* the answer.

Our version (`LinearBaseline`) flattens the window and applies one linear layer:

```python
class LinearBaseline(nn.Module):     # DLinear-style capacity floor
    def __init__(self, n_features, seq_len, n_classes=3):
        self.fc = nn.Linear(seq_len * n_features, n_classes)
    def forward(self, x):
        return self.fc(x.flatten(1))
```

**Why include it:** it's the **capacity floor**. If the LSTM ≈ the linear model, then the
LSTM's memory and non-linearity **bought nothing** → the ceiling is the *data*, not the model.
That single comparison turns "we tried an LSTM and it failed" into "even a linear model does as
well — so complexity isn't the missing ingredient," and lets you **cite Zeng et al.** It runs on
the *same* walk-forward harness — a one-line swap of the model factory.

---

## 7. Why Poutyne (instead of a raw PyTorch loop)

Raw PyTorch makes you hand-write the training loop every time: iterate epochs, iterate batches,
zero gradients, forward pass, compute loss, backward pass, optimizer step, move tensors to the
GPU, track validation loss, implement early stopping… dozens of lines, easy to get subtly wrong.

**Poutyne is a thin Keras-style wrapper over PyTorch** that removes that boilerplate:

```python
model = Model(net, optimizer, loss_fn, batch_metrics=["accuracy"], device=DEVICE)
model.fit(X_train, y_train, validation_data=(X_val, y_val),
          epochs=40, batch_size=64, callbacks=[EarlyStopping(...)])
preds = model.predict(X_test)
```

Benefits for us specifically:
- **`fit` / `predict` like scikit-learn / Keras** — one line each; the loop is handled.
- **Callbacks** — `EarlyStopping` is one object, not 15 lines of custom logic.
- **Automatic device handling** — it moves batches to CPU/GPU for you (handy with the RTX 5070).
- **Automatic metric tracking** — train/val loss and accuracy per epoch, for free.
- **Less code = fewer bugs** — and it's what your reference notebook already uses, so the pattern
  is familiar.

Trade-off: for very custom training (multiple losses, exotic schedules) you'd drop to raw
PyTorch. We don't need that here, so Poutyne is the right tool.

---

## 8. The persistence illusion (Section 8 of the notebook) — read this twice

The most important rigor point, and where many crypto-DL papers (including your Transformer+GNN
reference reporting R²=0.9941) go wrong.

If you train a model to predict the **price level**, it learns the trivial rule *"tomorrow ≈
today."* You get a gorgeous overlay chart and **R² ≈ 0.99** — but it's just the input shifted by
one day, with **zero real predictive value**. The honest test:

1. Predict **returns**, not price levels.
2. Compare to the **naive baseline** ("tomorrow = today", i.e. predicted return = 0).
3. Report R² on *levels* (≈0.99, the illusion), R² on *returns* (≈0, the truth), **directional
   accuracy** (≈50%), and **skill vs naive** (≤0 = no skill).

The contrast between the beautiful price plot and the noise-like returns plot **is** the slide.

---

## 9. Mini-glossary

| Term | Plain meaning |
|------|---------------|
| **Epoch** | One full pass over the training data. |
| **Batch** | A small group of samples processed together (for speed). |
| **Hidden state / hidden size** | The LSTM's memory vector / its length. |
| **Gate** | A learnable "valve" that controls the LSTM's memory (forget / input / output). |
| **Logits** | Raw output numbers before they're turned into probabilities. |
| **Dropout** | Randomly disabling connections during training to prevent overfitting. |
| **Learning rate** | How big each parameter update is. |
| **Loss function** | The number the model tries to minimize (cross-entropy for classification). |
| **Class weights** | Making the loss care more about rare classes (our SMOTE replacement). |
| **Early stopping** | Halt training when validation stops improving. |
| **Overfitting** | Memorizing the training data instead of learning a general pattern. |
| **DLinear** | A one-linear-layer baseline; the capacity floor. |
| **Poutyne** | A Keras-style wrapper that removes PyTorch training boilerplate. |

---

## 10. Results (from the run, BTC 30d/1%)

- **Capacity ceiling confirmed.** LSTM per-fold F1 = 0.507 ± 0.342 (Buy) / 0.210 ± 0.330 (Sell) — std ≈ mean, unstable. MCC ~0.17 does **not** cleanly beat XGBoost (balanced F1 0.36 vs 0.41); it leans on the majority Buy class (34% opposite-direction errors). DLinear (the floor) is worst (MCC 0.06). More capacity goes from *terrible* → *weak-and-unstable*, never *good*. **The limit is the data, not the model.**
- **Persistence illusion demonstrated.** Predicting price **levels**: R² = 0.98 (the illusion). Predicting **returns**: R² = −0.74, directional accuracy 0.52 (coin flip), **negative skill vs the naive baseline** — the LSTM loses to "tomorrow = today." The gorgeous price chart is meaningless.
