"""
NN_7neurona_v2 — Loto 7/39: 7 neurona + backprop (fit).
Ulaz: empirijska distribucija. Init SEED=39. Dva CSV: loto + plus.
"""

# =============================================================================
# Loto 7/39 — 39 → 7 (hidden) → 39 (izlaz), BCE + backprop
# Trening: walk-forward — udeo do t-1 → multi-hot kola t
# Posle fita: top 7 po y_hat → next
# =============================================================================

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 39
MIN_BROJ = 1
MAX_BROJ = 39
BROJEVA_U_KOMBINACIJI = 7
N_NEURONA = 7
N_ULAZA = MAX_BROJ - MIN_BROJ + 1
MIN_HIST = SEED  # 39 kola pre prvog trening primera
LEARNING_RATE = 0.15
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
CSV_LOTO = DATA_DIR / "loto7_4658_k60_loto_2951.csv"
CSV_PLUS = DATA_DIR / "loto7_4658_k60_loto_plus_1707.csv"
NUM_COLS = ["Num1", "Num2", "Num3", "Num4", "Num5", "Num6", "Num7"]


def sigmoid(z):
    z = np.clip(z, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z))


def csv_jobs():
    return [
        (CSV_LOTO, "LOTO", "next_loto"),
        (CSV_PLUS, "LOTO PLUS", "next_loto_plus"),
    ]


def load_draws(csv_path):
    peek = pd.read_csv(csv_path, nrows=0)
    if all(c in peek.columns for c in NUM_COLS):
        df = pd.read_csv(csv_path)[NUM_COLS].astype(int)
    else:
        df = pd.read_csv(csv_path, header=None).iloc[:, :7].astype(int)
    return df.to_numpy().tolist()


def udeo_from_counts(counts):
    ukupno = float(np.sum(counts)) or 1.0
    return counts / ukupno


def multi_hot(draw):
    y = np.zeros(N_ULAZA, dtype=float)
    for n in draw:
        if MIN_BROJ <= int(n) <= MAX_BROJ:
            y[int(n) - 1] = 1.0
    return y


def napravi_trening(draws):
    """Parovi (udeo do t-1, multi-hot kola t), t >= MIN_HIST."""
    counts = np.zeros(N_ULAZA, dtype=float)
    X_list = []
    Y_list = []
    for t, draw in enumerate(draws):
        if t >= MIN_HIST and counts.sum() > 0:
            X_list.append(udeo_from_counts(counts).copy())
            Y_list.append(multi_hot(draw))
        for n in draw:
            if MIN_BROJ <= int(n) <= MAX_BROJ:
                counts[int(n) - 1] += 1.0
    if not X_list:
        raise ValueError("Premalo kola za trening")
    return np.asarray(X_list), np.asarray(Y_list), counts


def init_tezine():
    """W1 (39,7), b1 (1,7), W2 (7,39), b2 (1,39) — fiksno iz SEED."""
    W1 = np.zeros((N_ULAZA, N_NEURONA))
    b1 = np.zeros((1, N_NEURONA))
    W2 = np.zeros((N_NEURONA, N_ULAZA))
    b2 = np.zeros((1, N_ULAZA))
    for i in range(N_ULAZA):
        for j in range(N_NEURONA):
            W1[i, j] = 0.3 * np.sin((i + 1) * (j + 1) * SEED * 0.01)
            W2[j, i] = 0.3 * np.cos((i + 1) * (j + 1) * SEED * 0.01)
    for j in range(N_NEURONA):
        b1[0, j] = 0.1 * np.cos((j + 1) * SEED * 0.01)
    for i in range(N_ULAZA):
        b2[0, i] = 0.05 * np.sin((i + 1) * SEED * 0.01)
    return W1, b1, W2, b2


def bce_loss(y_true, y_pred):
    eps = 1e-8
    return -np.mean(
        y_true * np.log(y_pred + eps) + (1.0 - y_true) * np.log(1.0 - y_pred + eps)
    )


def forward(X, W1, b1, W2, b2):
    z1 = X @ W1 + b1
    a1 = sigmoid(z1)
    z2 = a1 @ W2 + b2
    y_hat = sigmoid(z2)
    return a1, y_hat


def fit(X, Y, epochs, lr=LEARNING_RATE):
    """Fit = trening (forward + BACKPROP + update) kroz epochs."""
    W1, b1, W2, b2 = init_tezine()
    n = len(X)
    last_loss = 0.0
    korak = max(1, epochs // 10)
    for epoch in range(epochs):
        # Forward
        a1, y_hat = forward(X, W1, b1, W2, b2)
        last_loss = float(bce_loss(Y, y_hat))

        # -----------------------
        #        BACKPROP
        # -----------------------
        # Output layer gradients
        dz2 = y_hat - Y                        # (n,39)
        dW2 = (a1.T @ dz2) / n                 # (7,39)
        db2 = dz2.mean(axis=0, keepdims=True)  # (1,39)
        # Hidden layer gradients
        dz1 = (dz2 @ W2.T) * a1 * (1.0 - a1)   # sigmoid derivative
        dW1 = (X.T @ dz1) / n                  # (39,7)
        db1 = dz1.mean(axis=0, keepdims=True)  # (1,7)

        # Update
        W2 -= lr * dW2
        b2 -= lr * db2
        W1 -= lr * dW1
        b1 -= lr * db1

        if epoch % korak == 0 or epoch == epochs - 1:
            print(f"epoch {epoch:4d}   loss = {last_loss:.4f}")
    return W1, b1, W2, b2, last_loss


def next_iz_yhat(y_hat_row):
    skorovi = {b: float(y_hat_row[b - 1]) for b in range(MIN_BROJ, MAX_BROJ + 1)}
    poredak = sorted(skorovi.items(), key=lambda kv: (-kv[1], kv[0]))
    nxt = tuple(sorted(b for b, _ in poredak[:BROJEVA_U_KOMBINACIJI]))
    return skorovi, nxt


def main(csv_path, label, next_key):
    draws = load_draws(csv_path)
    X, Y, counts_full = napravi_trening(draws)
    n_kola = len(draws)
    epochs = n_kola  # Loto 2951 / Plus 1707 — po CSV

    print(f"NN_7neurona_v2 — 7 neurona + backprop | {label}")
    print(f"SEED={SEED} | CSV={Path(csv_path).name} | kola={n_kola}")
    print(f"trening parova={len(X)} | epochs={epochs} | lr={LEARNING_RATE}")
    print()

    # fit (trening sa backprop)
    W1, b1, W2, b2, loss = fit(X, Y, epochs=epochs)
    print()

    x_full = udeo_from_counts(counts_full).reshape(1, -1)
    _, y_hat = forward(x_full, W1, b1, W2, b2)
    skorovi, nxt = next_iz_yhat(y_hat[0])
    top = list(nxt)

    print("broj | udeo | y_hat")
    udeo = {b: float(x_full[0, b - 1]) for b in range(MIN_BROJ, MAX_BROJ + 1)}
    for b, s in sorted(skorovi.items(), key=lambda kv: (-kv[1], kv[0]))[:15]:
        print(f"{b:4d} | {udeo[b]:.6f} | {s:.6f}")
    print("...")
    print()
    print(f"loss_final={loss:.4f}")
    print(f"{next_key}: {top}")
    return top


if __name__ == "__main__":
    for _csv, _label, _next_key in csv_jobs():
        print(f"=== {_label} ===")
        main(_csv, _label, _next_key)
        print()


"""
RUN:

=== LOTO ===
NN_7neurona_v2 — 7 neurona + backprop | LOTO
SEED=39 | CSV=loto7_4658_k60_loto_2951.csv | kola=2951
trening parova=2912 | epochs=2951 | lr=0.15

epoch    0   loss = 0.6995
epoch  295   loss = 0.4704
epoch  590   loss = 0.4704
epoch  885   loss = 0.4704
epoch 1180   loss = 0.4704
epoch 1475   loss = 0.4704
epoch 1770   loss = 0.4704
epoch 2065   loss = 0.4704
epoch 2360   loss = 0.4704
epoch 2655   loss = 0.4704
epoch 2950   loss = 0.4704

broj | udeo | y_hat
   8 | 0.028610 | 0.199532
  23 | 0.027497 | 0.193685
  22 | 0.027400 | 0.193345
  21 | 0.026577 | 0.186505
  16 | 0.026722 | 0.186480
  33 | 0.026480 | 0.186141
  38 | 0.026529 | 0.186135
  34 | 0.026529 | 0.185757
  24 | 0.026577 | 0.185453
  39 | 0.026432 | 0.185443
  35 | 0.026771 | 0.185084
  37 | 0.026141 | 0.184451
  26 | 0.026335 | 0.183727
   9 | 0.026190 | 0.183040
  13 | 0.025996 | 0.182716
...

loss_final=0.4704
next_loto: [8, x, 21, y, 23, z, 38]

=== LOTO PLUS ===
NN_7neurona_v2 — 7 neurona + backprop | LOTO PLUS
SEED=39 | CSV=loto7_4658_k60_loto_plus_1707.csv | kola=1707
trening parova=1668 | epochs=1707 | lr=0.15

epoch    0   loss = 0.7001
epoch  170   loss = 0.4703
epoch  340   loss = 0.4703
epoch  510   loss = 0.4703
epoch  680   loss = 0.4703
epoch  850   loss = 0.4703
epoch 1020   loss = 0.4703
epoch 1190   loss = 0.4703
epoch 1360   loss = 0.4703
epoch 1530   loss = 0.4703
epoch 1700   loss = 0.4703
epoch 1706   loss = 0.4703

broj | udeo | y_hat
  23 | 0.028622 | 0.199676
  11 | 0.028203 | 0.198487
   2 | 0.028120 | 0.194310
   8 | 0.027283 | 0.193047
  26 | 0.027450 | 0.192481
  34 | 0.027450 | 0.191888
   7 | 0.027032 | 0.191882
  32 | 0.027032 | 0.190059
  37 | 0.027032 | 0.187687
  18 | 0.026948 | 0.186506
  29 | 0.026697 | 0.185857
   4 | 0.026780 | 0.185832
  10 | 0.026529 | 0.184683
  25 | 0.026278 | 0.184679
  31 | 0.026027 | 0.183444
...

loss_final=0.4703
next_loto_plus: [2, x, 8, y, 23, z, 34]
"""



"""
BackTest:

Backtest NN_7neurona_v2 fit+backprop (n−500):

Loto (2451 → actual 2452)

pred: [8, 16, 21, 22, 23, 24, 35]
actual: [3, 5, 11, 15, 20, 21, 25]
HIT: False
· 1/7 (21)


Loto Plus (1207 → actual 1208)

pred: [2, 7, 9, 11, 23, 26, 37]
actual: [3, 7, 27, 29, 35, 37, 38]
HIT: False
· 2/7 (7, 37)


Backtest NN_7neurona_v2 fit+backprop (n−1000):

Loto (1951 → actual 1952)

pred: [8, 9, 21, 23, 24, 31, 38]
actual: [2, 7, 9, 13, 14, 29, 31]
HIT: False
· 2/7 (9, 31)


Loto Plus (707 → actual 708)

pred: [2, 11, 20, 23, 26, 32, 37]
actual: [12, 14, 18, 22, 31, 38, 39]
HIT: False
· 0/7 (-)


Backtest NN_7neurona_v2 fit+backprop (n−1500):

Loto (1451 → actual 1452)

pred: [8, 16, 23, 24, 32, 35, 38]
actual: [3, 14, 15, 16, 26, 27, 36]
HIT: False
· 1/7 (16)


Loto Plus (207 → actual 208)

pred: [2, 4, 8, 10, 11, 37, 39]
actual: [2, 12, 15, 16, 19, 21, 27]
HIT: False
· 1/7 (2)
"""



"""
ANALIZA — NN_7neurona_v2.py:

1. Ulaz — dva CSV-a odvojeno: Loto (2951) i Plus (1707), bez miksa.

2. Empirijska distribucija — udeo do t−1; target = multi-hot kola t.
   Nikad sirova frekvencija kao skor.

3. Mreža — 39 → 7 hidden → 39; BCE + BACKPROP (fit).
   epochs = n_kola po CSV (2951 / 1707). Init SEED=39, bez RNG.
   next = top 7 po y_hat.

4. RUN:
   next_loto:      [8, x, 21, y, 23, z, 38]
   next_loto_plus: [2, x, 8, y, 23, z, 34]
   Razlikuju se od v1 (≥1 broj). Loss ~0.47 (plafon multi-label BCE).

5. Backtest (pred n−k → actual n−k+1), k=500,1000,1500:
   Loto: 1/7, 2/7, 1/7
   Plus: 2/7, 0/7, 1/7
   Pred različiti kroz prozore (fit menja težine sa istorijom).
   Najbolji: Loto n−1000 i Plus n−500 (2/7).
"""



"""
Beleske:

v2 = v1 + backprop (fit)
39 → 7 hidden → 39 izlaz, BCE
walk-forward: udeo do t-1 → multi-hot kola t
epochs = n_kola po CSV (Loto 2951, Plus 1707) — različiti
"""
