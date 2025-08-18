import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, multilabel_confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# 配置参数
class Config:
    EMOTION_MODEL = "bhadresh-savani/bert-base-go-emotion"
    MAX_LENGTH = 256
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EMOTION_MODEL_PATH = "./model/emotion_model.bin"
    EMOTION_LABELS = [
        'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
        'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
        'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
        'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
        'relief', 'remorse', 'sadness', 'surprise', 'neutral'
    ]


# 自定义数据集类
class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.FloatTensor(self.labels[idx])
        }


# 数据预处理
def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path, low_memory=False)
    df = df[:7000]
    # 情感标签强制转换
    for col in Config.EMOTION_LABELS:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(float)

    # 验证数据类型
    if df[Config.EMOTION_LABELS].values.dtype.kind not in 'fiu':
        raise TypeError("Emotion labels must be numeric")

    texts = df['text'].values
    labels = df[Config.EMOTION_LABELS].values

    # 分割数据集
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )
    return train_texts, test_texts, train_labels, test_labels


# 训练函数
def train_model(model, train_loader, optimizer, device, scheduler=None, epochs=5):
    model.train()
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        total_loss = 0

        for batch in tqdm(train_loader):
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            loss = loss_fn(logits, labels)
            total_loss += loss.item()

            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()

        avg_loss = total_loss / len(train_loader)
        print(f"Training loss: {avg_loss:.4f}")

    # 保存模型
    torch.save(model.state_dict(), Config.EMOTION_MODEL_PATH)
    print(f"Model saved to {Config.EMOTION_MODEL_PATH}")


# 评估函数
def evaluate_model(model, test_loader, device):
    model.eval()
    loss_fn = nn.BCEWithLogitsLoss()
    total_loss = 0
    all_preds = []
    all_labels = []

    # 初始化准确率计算变量
    total_correct = 0
    total_samples = 0
    total_labels_count = 0

    with torch.no_grad():
        for batch in tqdm(test_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            loss = loss_fn(logits, labels)
            total_loss += loss.item()

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float().cpu().numpy()
            labels_np = labels.cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels_np)

            # 计算批次准确率
            batch_correct = np.sum(preds == labels_np)
            batch_samples = preds.shape[0]
            batch_labels = preds.shape[1]

            total_correct += batch_correct
            total_samples += batch_samples
            total_labels_count += batch_samples * batch_labels

    avg_loss = total_loss / len(test_loader)
    print(f"Validation loss: {avg_loss:.4f}")

    # 计算总准确率
    total_accuracy = total_correct / total_labels_count
    print(f"Total Accuracy: {total_accuracy:.4f}")

    # 计算样本级准确率
    sample_accuracy = np.mean([np.array_equal(p, l) for p, l in zip(all_preds, all_labels)])
    print(f"Sample-level Accuracy: {sample_accuracy:.4f}")

    return np.array(all_preds), np.array(all_labels), total_accuracy, sample_accuracy


# 可视化函数（所有中文已改为英文）
def visualize_results(true_labels, pred_labels, emotion_labels, total_accuracy, sample_accuracy):
    # ======================
    # 1. Calculate core evaluation metrics
    # ======================
    # Classification report (includes precision/recall/F1, etc.)
    report = classification_report(
        true_labels, pred_labels,
        target_names=emotion_labels,
        zero_division=0,
        output_dict=True
    )

    # Multi-label confusion matrix
    mcm = multilabel_confusion_matrix(true_labels, pred_labels)

    # ======================
    # 2. Create comprehensive visualization panel
    # ======================
    plt.figure(figsize=(20, 18))
    plt.suptitle(
        f"Multi-label Classification Model Evaluation | Total Accuracy: {total_accuracy:.2%} | Sample Accuracy: {sample_accuracy:.2%}",
        fontsize=18,
        y=0.98
    )

    # ----------------------
    # 2.1 Metrics summary panel
    # ----------------------
    ax0 = plt.subplot2grid((3, 3), (0, 0), colspan=1)
    ax0.axis('off')

    # Create metrics summary text
    report_text = (
        f"Model Performance Summary:\n\n"
        f"Total Accuracy: {total_accuracy:.2%}\n"
        f"Sample-level Accuracy: {sample_accuracy:.2%}\n"
        f"Average Precision: {report['macro avg']['precision']:.4f}\n"
        f"Average Recall: {report['macro avg']['recall']:.4f}\n"
        f"Average F1-Score: {report['macro avg']['f1-score']:.4f}"
    )

    ax0.text(0.5, 0.5, report_text,
             ha='center', va='center', fontsize=14,
             bbox=dict(boxstyle="round,pad=0.5", fc="lightyellow"))

    # ----------------------
    # 2.2 Performance heatmap per category
    # ----------------------
    ax1 = plt.subplot2grid((3, 3), (0, 1), colspan=2, rowspan=1)

    # Prepare heatmap data
    metrics_data = []
    for emotion in emotion_labels:
        metrics_data.append([
            report[emotion]['precision'],
            report[emotion]['recall'],
            report[emotion]['f1-score']
        ])

    metrics_df = pd.DataFrame(
        metrics_data,
        index=emotion_labels,
        columns=['Precision', 'Recall', 'F1-Score']
    )

    sns.heatmap(
        metrics_df,
        annot=True, fmt=".2f",
        cmap="YlGnBu",
        cbar_kws={'label': 'Score'},
        ax=ax1
    )
    ax1.set_title('Performance Metrics per Emotion')
    ax1.set_ylabel('Emotion Categories')
    ax1.set_xlabel('Evaluation Metrics')
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")

    # ----------------------
    # 2.3 Confusion matrix panel
    # ----------------------
    ax2 = plt.subplot2grid((3, 3), (1, 0), colspan=3, rowspan=1)

    # Calculate accuracy per category
    accuracy_data = []
    for i, emotion in enumerate(emotion_labels):
        tn, fp, fn, tp = mcm[i].ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        accuracy_data.append(accuracy)

    # Create confusion matrix heatmap
    sns.heatmap(
        np.array(accuracy_data).reshape(1, -1),
        annot=True, fmt=".2%",
        cmap="YlGnBu",
        xticklabels=emotion_labels,
        cbar_kws={'label': 'Accuracy'},
        ax=ax2
    )
    ax2.set_title('Accuracy per Emotion Category')
    ax2.set_ylabel('')
    ax2.set_xlabel('Emotion Categories')
    plt.setp(ax2.get_xticklabels(), rotation=90)

    # ----------------------
    # 2.4 F1-Score distribution
    # ----------------------
    ax3 = plt.subplot2grid((3, 3), (2, 0), colspan=1, rowspan=1)

    f1_scores = [report[emotion]['f1-score'] for emotion in emotion_labels]
    sns.boxplot(y=f1_scores, color="skyblue", ax=ax3)
    sns.stripplot(y=f1_scores, color="red", size=6, jitter=True, ax=ax3)
    ax3.set_title('F1-Score Distribution')
    ax3.set_ylabel('F1-Score')

    # ----------------------
    # 2.5 Support vs. Accuracy relationship
    # ----------------------
    ax4 = plt.subplot2grid((3, 3), (2, 1), colspan=1, rowspan=1)

    support = [report[emotion]['support'] for emotion in emotion_labels]
    sns.regplot(x=support, y=accuracy_data, ax=ax4)
    ax4.set_title('Sample Support vs. Accuracy')
    ax4.set_xlabel('Sample Support')
    ax4.set_ylabel('Accuracy')

    # ----------------------
    # 2.6 Emotion distribution bar chart
    # ----------------------
    ax5 = plt.subplot2grid((3, 3), (2, 2), colspan=1, rowspan=1)

    emotion_counts = true_labels.sum(axis=0)
    sns.barplot(x=emotion_labels, y=emotion_counts, ax=ax5)
    ax5.set_title('Emotion Distribution in Test Set')
    ax5.set_ylabel('Sample Count')
    plt.setp(ax5.get_xticklabels(), rotation=90)

    # ======================
    # 3. Save visualization results
    # ======================
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.savefig('emotion_model_performance.png', dpi=300)
    print("Model performance visualization saved as emotion_model_performance.png")

    # Save confusion matrix heatmap separately
    plt.figure(figsize=(15, 12))
    for i, emotion in enumerate(emotion_labels):
        plt.subplot(7, 4, i + 1)  # 28 emotions, 7 rows 4 columns
        sns.heatmap(mcm[i], annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Negative', 'Positive'],
                    yticklabels=['Negative', 'Positive'])
        plt.title(emotion)
        plt.ylabel('True')
        plt.xlabel('Predicted')

    plt.tight_layout()
    plt.savefig('emotion_confusion_matrix.png', dpi=300)
    print("Confusion matrix saved as emotion_confusion_matrix.png")


# 主函数
def main():
    # 加载数据
    try:
        train_texts, test_texts, train_labels, test_labels = load_and_prepare_data(
            'data/go_emotions_dataset.csv'
        )
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 初始化tokenizer和模型
    tokenizer = BertTokenizer.from_pretrained(Config.EMOTION_MODEL)
    model = BertForSequenceClassification.from_pretrained(
        Config.EMOTION_MODEL,
        num_labels=len(Config.EMOTION_LABELS),
        problem_type="multi_label_classification"
    )
    model = model.to(Config.DEVICE)

    # 创建数据加载器
    train_dataset = EmotionDataset(train_texts, train_labels, tokenizer, Config.MAX_LENGTH)
    test_dataset = EmotionDataset(test_texts, test_labels, tokenizer, Config.MAX_LENGTH)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # 设置优化器
    optimizer = AdamW(model.parameters(), lr=5e-5)

    # 训练模型
    train_model(model, train_loader, optimizer, Config.DEVICE, epochs=2)

    # 评估模型
    predictions, true_labels, total_acc, sample_acc = evaluate_model(
        model, test_loader, Config.DEVICE
    )

    # 可视化结果
    visualize_results(true_labels, predictions, Config.EMOTION_LABELS, total_acc, sample_acc)


if __name__ == "__main__":
    main()