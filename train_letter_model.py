"""
Train a classifier for the 26 ISL alphabet letters (A-Z) from MediaPipe
landmarks, so letters can be recognised by the live camera.

This is a SEPARATE model from the 29-sign word ensemble:
  * the word model is an LSTM over 30-frame sequences (dynamic signs),
  * the letter model is a small dense network over a single normalised
    63-value landmark frame (static hand shapes).
Keeping them apart means the existing word recognition is not touched at all.

Inputs : data/letters.npz  (X, y, split, letters)
Outputs: models/letters/letter_model.keras
         models/letters/label_map.json
         models/letters/letter_report.txt
         models/letters/letter_confusion_matrix.png

Usage:
    python train_letter_model.py
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
OUT_DIR = os.path.join("models", "letters")
np.random.seed(SEED)
tf.random.set_seed(SEED)


def augment(X, y, rng, copies=3):
    """Extra training samples: scale, in-plane rotation and landmark jitter.

    Landmarks are already centred on the wrist and scale-normalised, so these
    transforms stay meaningful (a letter signed slightly larger, slightly
    rotated, or with slightly noisy landmarks is still the same letter).
    """
    X = X.astype(np.float32)
    out_X, out_y = [X], [y]
    for _ in range(copies):
        Xa = X.copy()
        Xa *= rng.uniform(0.9, 1.1, size=(Xa.shape[0], 1)).astype(np.float32)
        pts = Xa.reshape(Xa.shape[0], -1, 3)
        ang = rng.uniform(-0.18, 0.18, size=(Xa.shape[0], 1)).astype(np.float32)
        c, s = np.cos(ang), np.sin(ang)
        x, yy = pts[..., 0].copy(), pts[..., 1].copy()
        pts[..., 0] = c * x - s * yy
        pts[..., 1] = s * x + c * yy
        Xa = pts.reshape(Xa.shape) + rng.normal(0, 0.012, Xa.shape).astype(np.float32)
        out_X.append(Xa.astype(np.float32))
        out_y.append(y)
    return np.concatenate(out_X), np.concatenate(out_y)


def build_model(num_classes):
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(63,)),
        tf.keras.layers.Dense(256, activation="relu",
                              kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(128, activation="relu",
                              kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ])


def main():
    data = np.load(os.path.join("data", "letters.npz"), allow_pickle=True)
    X, y, split = data["X"], data["y"], data["split"]
    letters = [str(c) for c in data["letters"]]
    num_classes = len(letters)

    tr, va, te = split == "train", split == "val", split == "test"
    Xtr, ytr = X[tr], y[tr]
    Xva, yva = X[va], y[va]
    Xte, yte = X[te], y[te]
    print(f"Train {Xtr.shape}  Val {Xva.shape}  Test {Xte.shape}  Classes {num_classes}")

    present = np.unique(ytr)
    w = compute_class_weight("balanced", classes=present, y=ytr)
    class_weight = {int(c): float(wi) for c, wi in zip(present, w)}

    rng = np.random.default_rng(SEED)
    Xa, ya = augment(Xtr, ytr, rng, copies=3)

    model = build_model(num_classes)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    model.summary()

    cbs = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=12,
                                            restore_best_weights=True)]
    hist = model.fit(Xa, ya, validation_data=(Xva, yva), epochs=60, batch_size=128,
                     class_weight=class_weight, callbacks=cbs, verbose=2)

    probs = model.predict(Xte, verbose=0)
    y_pred = probs.argmax(1)
    acc = float((y_pred == yte).mean())
    top3 = float(np.mean([yte[i] in np.argsort(probs[i])[::-1][:3] for i in range(len(yte))]))

    print(f"\nLetter model test accuracy : {acc:.4f}")
    print(f"Top-3 accuracy             : {top3:.4f}")

    names = [c.upper() for c in letters]
    report = classification_report(yte, y_pred, labels=list(range(num_classes)),
                                   target_names=names, zero_division=0)
    print(report)

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "letter_report.txt"), "w", encoding="utf-8") as f:
        f.write("ISL alphabet (A-Z) classifier, MediaPipe landmarks (63 features/frame)\n")
        f.write(f"Test accuracy: {acc:.4f}\nTop-3 accuracy: {top3:.4f}\n")
        f.write(f"Test samples: {len(yte)}\n\n{report}")

    cm = confusion_matrix(yte, y_pred, labels=list(range(num_classes)))
    fig, ax = plt.subplots(figsize=(11, 10))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(num_classes)); ax.set_yticks(range(num_classes))
    ax.set_xticklabels(names, fontsize=8); ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Alphabet confusion matrix (test acc {acc:.1%})")
    fig.colorbar(im)
    for i in range(num_classes):
        for j in range(num_classes):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "letter_confusion_matrix.png"), dpi=150)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1); plt.plot(hist.history["accuracy"], label="train")
    plt.plot(hist.history["val_accuracy"], label="val"); plt.title("Accuracy"); plt.legend()
    plt.subplot(1, 2, 2); plt.plot(hist.history["loss"], label="train")
    plt.plot(hist.history["val_loss"], label="val"); plt.title("Loss"); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "letter_training_history.png"), dpi=150)

    model.save(os.path.join(OUT_DIR, "letter_model.keras"))
    with open(os.path.join(OUT_DIR, "label_map.json"), "w", encoding="utf-8") as f:
        json.dump({"letters": letters, "num_classes": num_classes,
                   "test_accuracy": acc, "top3_accuracy": top3}, f, indent=2)
    print(f"Saved -> {OUT_DIR}")


if __name__ == "__main__":
    main()
