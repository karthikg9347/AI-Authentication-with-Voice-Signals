"""
AI Authentication & Classification from Voice Signals
-----------------------------------------------------
Single-file project combining:
1. Audio classification (emotion/gender/etc.)
2. Speaker authentication (voice verification)
Author: Your Name
"""

import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
from tensorflow.keras import layers, models
import joblib

# =====================================================
# =============  AUDIO FEATURE FUNCTIONS  =============
# =====================================================

def load_wav(path, sr=16000):
    y, _ = librosa.load(path, sr=sr)
    return y

def extract_mfcc(y, sr=16000, n_mfcc=40):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    delta = librosa.feature.delta(mfcc)
    delta_mean = np.mean(delta, axis=1)
    features = np.concatenate([mfcc_mean, mfcc_std, delta_mean])
    return features  # 120-D feature vector

# =====================================================
# ==============  DATA PREPARATION  ===================
# =====================================================

def prepare_classification_dataset(root_dir):
    X, y = [], []
    labels = sorted(os.listdir(root_dir))
    for label in labels:
        label_dir = os.path.join(root_dir, label)
        if not os.path.isdir(label_dir): continue
        for fn in os.listdir(label_dir):
            if not fn.endswith(".wav"): continue
            path = os.path.join(label_dir, fn)
            y_wav = load_wav(path)
            feat = extract_mfcc(y_wav)
            X.append(feat)
            y.append(label)
    X = np.array(X)
    y = np.array(y)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    return X, y_enc, le

def prepare_auth_pairs(root_dir):
    """
    Creates positive (same speaker) and negative (different speaker) pairs
    for speaker verification.
    """
    feats_by_speaker = {}
    for spk in sorted(os.listdir(root_dir)):
        spk_dir = os.path.join(root_dir, spk)
        if not os.path.isdir(spk_dir): continue
        feats = []
        for fn in os.listdir(spk_dir):
            if not fn.endswith(".wav"): continue
            path = os.path.join(spk_dir, fn)
            y = load_wav(path)
            feats.append(extract_mfcc(y))
        if feats:
            feats_by_speaker[spk] = feats

    X1, X2, y = [], [], []
    speakers = list(feats_by_speaker.keys())
    import random
    # positive pairs
    for spk in speakers:
        feats = feats_by_speaker[spk]
        for i in range(len(feats)):
            for j in range(i+1, len(feats)):
                X1.append(feats[i])
                X2.append(feats[j])
                y.append(1)
    # negative pairs
    for _ in range(len(y)):
        a, b = random.sample(speakers, 2)
        X1.append(random.choice(feats_by_speaker[a]))
        X2.append(random.choice(feats_by_speaker[b]))
        y.append(0)
    return np.array(X1), np.array(X2), np.array(y)

# =====================================================
# ===============  MODEL DEFINITIONS  =================
# =====================================================

def build_classifier(input_dim, n_classes):
    inp = layers.Input(shape=(input_dim,))
    x = layers.Dense(256, activation='relu')(inp)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    out = layers.Dense(n_classes, activation='softmax')(x)
    model = models.Model(inp, out)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def build_embedding_model(input_dim, emb_dim=128):
    inp = layers.Input(shape=(input_dim,))
    x = layers.Dense(512, activation='relu')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dense(emb_dim)(x)
    x = layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=1))(x)
    return models.Model(inp, x)

def build_siamese(embedding_model):
    inpA = layers.Input(shape=(embedding_model.input_shape[1],))
    inpB = layers.Input(shape=(embedding_model.input_shape[1],))
    embA = embedding_model(inpA)
    embB = embedding_model(inpB)
    diff = layers.Lambda(lambda t: tf.abs(t[0] - t[1]))([embA, embB])
    x = layers.Dense(64, activation='relu')(diff)
    out = layers.Dense(1, activation='sigmoid')(x)
    model = models.Model([inpA, inpB], out)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# =====================================================
# =============  TRAINING & EVALUATION  ===============
# =====================================================

def train_classifier(data_dir="data/class"):
    print("Training classification model...")
    X, y, le = prepare_classification_dataset(data_dir)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y)
    model = build_classifier(X.shape[1], len(np.unique(y)))
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=30, batch_size=32)
    os.makedirs("models", exist_ok=True)
    model.save("models/classifier.h5")
    joblib.dump(le, "models/label_encoder.joblib")
    print("Classifier saved successfully.\n")

def train_authenticator(data_dir="data/auth"):
    print("Training speaker authentication model...")
    X1, X2, y = prepare_auth_pairs(data_dir)
    X1_tr, X1_val, X2_tr, X2_val, y_tr, y_val = train_test_split(X1, X2, y, test_size=0.2, stratify=y)
    emb = build_embedding_model(X1.shape[1])
    siam = build_siamese(emb)
    siam.fit([X1_tr, X2_tr], y_tr, validation_data=([X1_val, X2_val], y_val), epochs=20, batch_size=64)
    os.makedirs("models", exist_ok=True)
    emb.save("models/embedding_model.h5")
    siam.save("models/siamese_model.h5")
    print("Authentication models saved successfully.\n")

# =====================================================
# ==================  INFERENCE  =======================
# =====================================================

def classify_file(path):
    from tensorflow.keras.models import load_model
    clf = load_model("models/classifier.h5")
    le = joblib.load("models/label_encoder.joblib")
    y = load_wav(path)
    feat = extract_mfcc(y).reshape(1, -1)
    probs = clf.predict(feat)[0]
    pred = probs.argmax()
    print(f"Prediction: {le.inverse_transform([pred])[0]}, Probabilities: {probs}")

def verify_speaker(enroll_path, test_path, threshold=0.7):
    from tensorflow.keras.models import load_model
    emb = load_model("models/embedding_model.h5")
    fa = extract_mfcc(load_wav(enroll_path)).reshape(1, -1)
    fb = extract_mfcc(load_wav(test_path)).reshape(1, -1)
    ea = emb.predict(fa)
    eb = emb.predict(fb)
    sim = cosine_similarity(ea, eb)[0, 0]
    print(f"Similarity: {sim:.3f}")
    print("Result:", "MATCH ✅" if sim >= threshold else "NO MATCH ❌")

# =====================================================
# ====================  MAIN MENU  ====================
# =====================================================

if __name__ == "__main__":
    print("\n🎙️  AI Authentication & Classification from Voice Signals")
    print("==========================================================")
    print("1. Train Classification Model")
    print("2. Train Authentication Model")
    print("3. Classify an Audio File")
    print("4. Verify a Speaker")
    print("5. Exit")

    choice = input("\nEnter your choice: ").strip()

    if choice == "1":
        train_classifier()
    elif choice == "2":
        train_authenticator()
    elif choice == "3":
        path = input("Enter path to .wav file: ")
        classify_file(path)
    elif choice == "4":
        e = input("Enter enrollment wav path: ")
        t = input("Enter test wav path: ")
        verify_speaker(e, t)
    else:
        print("Goodbye!")
