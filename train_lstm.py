"""
Train an ensemble of LSTM classifiers on the MediaPipe landmark sequences and
evaluate it.

A single model is unstable on this small dataset: depending only on the random
seed its test accuracy swings between roughly 0.59 and 0.79. We therefore train
K models with different seeds and average their (test-time-augmented) softmax
outputs. The ensemble is far more stable and clearly more accurate than an
average single model.

Inputs : data/landmarks.npz  (X, y, split)  + labels.json
Outputs: models/ensemble/sign_lstm_<i>.keras
         models/sign_lstm.keras          (copy of member 0, for compatibility)
         models/confusion_matrix.png, models/classification_report.txt,
         models/training_history.png, models/label_map.json

Usage:
    python train_lstm.py
"""

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf

SEED = 42
ENSEMBLE_SEEDS = [1, 2, 3, 4, 5]   # one model per seed, averaged at inference
np.random.seed(SEED)
tf.random.set_seed(SEED)


def augment(X, y, rng, copies=4):
    """Create extra training samples from the small dataset.

    Applies, to every clip: a global scale change, a small random rotation in the
    image plane, Gaussian landmark jitter, a time-warp and a small temporal shift
    (signs are performed at slightly different speeds and start positions).
    Landmarks are already centred and scale-normalised, so these transforms stay
    meaningful. Only the training split is augmented.

    A left/right MIRROR is also applied (every clip is added once fully mirrored,
    and half of each augmented copy is mirrored in place). Without it the model
    is not handedness-invariant: measured on the test split, mirroring the input
    drops accuracy from ~67% to ~20%, so a live sign performed with the other
    hand than the training signer used is read as a completely different word.
    """
    X = X.astype(np.float32)
    N, T, F = X.shape
    out_X, out_y = [X], [y]
    Xm = X.copy()
    Xm[:, :, 0::3] *= -1.0                       # mirror = negate every landmark's x
    out_X.append(Xm)
    out_y.append(y)
    for _ in range(copies):
        Xa = X.copy()
        Xa *= rng.uniform(0.88, 1.12, size=(N, 1, 1)).astype(np.float32)   # scale
        pts = Xa.reshape(N, T, -1, 3)
        ang = rng.uniform(-0.20, 0.20, size=(N, T, 1)).astype(np.float32)  # rotate
        c, s = np.cos(ang), np.sin(ang)
        x, yy = pts[..., 0].copy(), pts[..., 1].copy()
        pts[..., 0] = c * x - s * yy
        pts[..., 1] = s * x + c * yy
        Xa = Xa + rng.normal(0, 0.012, Xa.shape).astype(np.float32)       # jitter
        for i in range(N):                                                # time-warp
            n = int(rng.integers(T - 6, T + 7))
            mid = Xa[i][np.linspace(0, T - 1, n).round().astype(int)]
            Xa[i] = mid[np.linspace(0, n - 1, T).round().astype(int)]
        shift = rng.integers(-2, 3, size=N)                              # temporal shift
        for i in range(N):
            if shift[i]:
                Xa[i] = np.roll(Xa[i], int(shift[i]), axis=0)
        flip = rng.random(N) < 0.5                                       # random mirror
        Xa[flip, :, 0::3] *= -1.0
        out_X.append(Xa)
        out_y.append(y)
    return np.concatenate(out_X), np.concatenate(out_y)


def build_model(input_shape, num_classes):
    """One ensemble member: a stacked LSTM with recurrent dropout + L2.

    Recurrent dropout and kernel regularisation are what stop the model from
    simply memorising the ~350 training clips.
    """
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Masking(mask_value=0.0),
        tf.keras.layers.LSTM(128, return_sequences=True, recurrent_dropout=0.3),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.LSTM(64, recurrent_dropout=0.3),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(64, activation="relu",
                              kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ])


def tta_predict(model, X, rng, copies=6):
    """Average the softmax over the input plus lightly-jittered copies.

    Test-time augmentation (TTA) makes predictions more robust: instead of one
    forward pass we average several, so a single noisy frame sequence is less
    likely to be misclassified. Uses the same (mild) transforms as training.
    """
    X = X.astype(np.float32)
    probs = model.predict(X, verbose=0)
    for _ in range(copies):
        Xa = X.copy()
        Xa *= rng.uniform(0.95, 1.05, size=(Xa.shape[0], 1, 1)).astype(np.float32)
        pts = Xa.reshape(Xa.shape[0], Xa.shape[1], -1, 3)
        ang = rng.uniform(-0.10, 0.10, size=(Xa.shape[0], Xa.shape[1], 1)).astype(np.float32)
        c, s = np.cos(ang), np.sin(ang)
        x, yy = pts[..., 0].copy(), pts[..., 1].copy()
        pts[..., 0] = c * x - s * yy
        pts[..., 1] = s * x + c * yy
        Xa = Xa + rng.normal(0, 0.01, Xa.shape).astype(np.float32)
        probs += model.predict(Xa, verbose=0)
    return probs / (copies + 1)


def main():
    data = np.load(os.path.join("data", "landmarks.npz"), allow_pickle=True)
    X, y, split = data["X"], data["y"], data["split"]

    with open("labels.json", encoding="utf-8") as f:
        labels = json.load(f)
    id_to_class = labels["id_to_class"]
    num_classes = labels["num_classes"]
    class_names = [id_to_class[str(i)]["class_name"] for i in range(num_classes)]

    tr, va, te = split == "train", split == "val", split == "test"
    Xtr, ytr = X[tr], y[tr]
    Xva, yva = X[va], y[va]
    Xte, yte = X[te], y[te]
    print(f"Train {Xtr.shape}  Val {Xva.shape}  Test {Xte.shape}  Classes {num_classes}")

    # class weights to counter the mild imbalance
    present = np.unique(ytr)
    w = compute_class_weight("balanced", classes=present, y=ytr)
    class_weight = {int(c): float(wi) for c, wi in zip(present, w)}

    os.makedirs("models/ensemble", exist_ok=True)

    ens_probs = np.zeros((Xte.shape[0], num_classes))
    singles, histories = [], []
    members = []
    for i, seed in enumerate(ENSEMBLE_SEEDS):
        tf.keras.utils.set_random_seed(seed)
        rng = np.random.default_rng(seed)
        Xa, ya = augment(Xtr, ytr, rng, copies=6)   # val/test stay untouched

        model = build_model(X.shape[1:], num_classes)
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                      loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])
        cbs = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=15,
                                                restore_best_weights=True)]
        if i == 0:
            model.summary()
        hist = model.fit(Xa, ya, validation_data=(Xva, yva), epochs=150,
                         batch_size=16, class_weight=class_weight,
                         callbacks=cbs, verbose=2)

        member_probs = tta_predict(model, Xte, np.random.default_rng(1000 + seed))
        ens_probs += member_probs
        acc = float((member_probs.argmax(1) == yte).mean())
        singles.append(acc)
        histories.append(hist)
        members.append(model)
        model.save(f"models/ensemble/sign_lstm_{i}.keras")
        print(f"[member {i} seed={seed}] test accuracy = {acc:.4f}")

    ens_probs /= len(ENSEMBLE_SEEDS)
    y_pred = ens_probs.argmax(1)
    ens_acc = float((y_pred == yte).mean())
    order = np.argsort(-ens_probs, axis=1)
    top3 = float(np.mean([yte[k] in order[k, :3] for k in range(len(yte))]))
    top5 = float(np.mean([yte[k] in order[k, :5] for k in range(len(yte))]))

    # handedness robustness: classify the SAME test clips horizontally mirrored
    Xte_m = Xte.copy()
    Xte_m[:, :, 0::3] *= -1.0
    mirror_probs = sum(tta_predict(m, Xte_m, np.random.default_rng(2000 + i))
                       for i, m in enumerate(members)) / len(members)
    mirror_acc = float((mirror_probs.argmax(1) == yte).mean())

    print(f"\nSingle-model accuracy: {np.mean(singles):.4f} "
          f"(+/- {np.std(singles):.4f}, n={len(singles)})")
    print(f"Ensemble test accuracy: {ens_acc:.4f}")
    print(f"Ensemble test accuracy (mirrored input): {mirror_acc:.4f}")
    print(f"Ensemble top-3 accuracy: {top3:.4f}")
    print(f"Ensemble top-5 accuracy: {top5:.4f}")

    report = classification_report(yte, y_pred, labels=list(range(num_classes)),
                                   target_names=class_names, zero_division=0)
    with open("models/classification_report.txt", "w", encoding="utf-8") as f:
        f.write("Ensemble of "
                f"{len(ENSEMBLE_SEEDS)} models, averaged with test-time augmentation.\n")
        f.write(f"Single-model accuracy: {np.mean(singles):.4f} "
                f"(+/- {np.std(singles):.4f})\n")
        f.write(f"Ensemble test accuracy: {ens_acc:.4f}\n")
        f.write(f"Ensemble test accuracy (mirrored input): {mirror_acc:.4f}\n")
        f.write(f"Ensemble top-3 accuracy: {top3:.4f}\n")
        f.write(f"Ensemble top-5 accuracy: {top5:.4f}\n\n{report}")
    print(report)

    cm = confusion_matrix(yte, y_pred, labels=list(range(num_classes)))
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(num_classes)); ax.set_yticks(range(num_classes))
    ax.set_xticklabels(class_names, rotation=90, fontsize=7)
    ax.set_yticklabels(class_names, fontsize=7)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Confusion matrix (test)")
    fig.colorbar(im)
    for i in range(num_classes):
        for j in range(num_classes):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=6)
    fig.tight_layout(); fig.savefig("models/confusion_matrix.png", dpi=150)

    hist = histories[0]
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1); plt.plot(hist.history["accuracy"], label="train")
    plt.plot(hist.history["val_accuracy"], label="val"); plt.title("Accuracy (member 0)")
    plt.legend()
    plt.subplot(1, 2, 2); plt.plot(hist.history["loss"], label="train")
    plt.plot(hist.history["val_loss"], label="val"); plt.title("Loss (member 0)"); plt.legend()
    plt.tight_layout(); plt.savefig("models/training_history.png", dpi=150)

    # persist the first member under the classic path + the label map for the app
    members[0].save("models/sign_lstm.keras")
    with open("models/label_map.json", "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2)
    print(f"Saved {len(members)} models -> models/ensemble/")


if __name__ == "__main__":
    main()
