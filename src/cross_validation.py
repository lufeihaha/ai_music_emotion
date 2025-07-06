"""
5折交叉验证与学习曲线绘制
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from tensorflow.keras.utils import to_categorical
from src.deep_model import create_improved_model, prepare_data
import tensorflow as tf

def cross_validate_model(folds=5):
    X_train, X_test, y_train, y_test, class_names = prepare_data(augment=False)
    if X_train is None:
        print("数据准备失败")
        return
    X = np.concatenate([X_train, X_test], axis=0)
    y = np.concatenate([y_train, y_test], axis=0)
    num_classes = len(class_names)
    y_cat = to_categorical(y, num_classes)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    accs, f1s = [], []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"Fold {fold+1}/{folds}")
        model = create_improved_model(X.shape[1:], num_classes)
        model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        model.fit(X[train_idx], y_cat[train_idx],
                  epochs=10, batch_size=32, verbose=0)
        y_pred = np.argmax(model.predict(X[val_idx]), axis=1)
        acc = accuracy_score(y[val_idx], y_pred)
        f1 = f1_score(y[val_idx], y_pred, average='macro')
        accs.append(acc)
        f1s.append(f1)
        print(f"  Fold acc: {acc:.4f}, F1: {f1:.4f}")
    print(f"平均准确率: {np.mean(accs):.4f}, 平均F1: {np.mean(f1s):.4f}")
    return accs, f1s

def plot_learning_curve():
    X_train, X_test, y_train, y_test, class_names = prepare_data(augment=False)
    if X_train is None:
        print("数据准备失败")
        return
    num_classes = len(class_names)
    y_train_cat = to_categorical(y_train, num_classes)
    train_sizes = [0.1, 0.3, 0.5, 0.7, 1.0]
    train_scores, val_scores = [], []
    for frac in train_sizes:
        n = int(len(X_train) * frac)
        model = create_improved_model(X_train.shape[1:], num_classes)
        model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        model.fit(X_train[:n], y_train_cat[:n],
                  epochs=10, batch_size=32, verbose=0)
        train_acc = model.evaluate(X_train[:n], y_train_cat[:n], verbose=0)[1]
        val_acc = model.evaluate(X_test, to_categorical(y_test, num_classes), verbose=0)[1]
        train_scores.append(train_acc)
        val_scores.append(val_acc)
    plt.figure(figsize=(8, 5))
    plt.plot(train_sizes, train_scores, label='Train Acc')
    plt.plot(train_sizes, val_scores, label='Val Acc')
    plt.xlabel('训练集比例')
    plt.ylabel('准确率')
    plt.title('学习曲线')
    plt.legend()
    plt.tight_layout()
    plt.savefig('models/visualizations/learning_curve.png')
    plt.close()
    print("学习曲线已保存：models/visualizations/learning_curve.png")

if __name__ == "__main__":
    cross_validate_model()
    plot_learning_curve()
