# import torch
# import torch.nn as nn
# import torch.optim as optim
# from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
# from model import SentimentRNN
# from preprocess_data import train_loader, test_loader, vocab, text_pipeline
# import numpy as np

# # Thiết bị
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # Khởi tạo mô hình
# vocab_size = len(vocab)
# embed_dim = 100
# hidden_dim = 128
# output_dim = 2
# model = SentimentRNN(vocab_size, embed_dim, hidden_dim, output_dim).to(device)

# # Cấu hình
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# # Huấn luyện
# num_epochs = 10
# for epoch in range(num_epochs):
#     model.train()
#     total_loss = 0
#     correct = 0
#     total = 0
#     all_preds = []
#     all_labels = []
#     for labels, texts, lengths in train_loader:
#         labels, texts, lengths = labels.to(device), texts.to(device), lengths.to(device)
#         optimizer.zero_grad()
#         outputs = model(texts, lengths)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
#         _, predicted = torch.max(outputs, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#         all_preds.extend(predicted.cpu().numpy())
#         all_labels.extend(labels.cpu().numpy())
    
#     avg_loss = total_loss / len(train_loader)
#     accuracy = 100 * correct / total
#     precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', zero_division=0)
#     pred_pos = sum(1 for x in all_preds if x == 1)
#     pred_neg = sum(1 for x in all_preds if x == 0)
#     print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%, F1: {f1:.4f}")
#     print(f"Predictions - Positive: {pred_pos}, Negative: {pred_neg}")

# # Đánh giá
# model.eval()
# correct = 0
# total = 0
# all_preds = []
# all_labels = []
# with torch.no_grad():
#     for labels, texts, lengths in test_loader:
#         labels, texts, lengths = labels.to(device), texts.to(device), lengths.to(device)
#         outputs = model(texts, lengths)
#         _, predicted = torch.max(outputs, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#         all_preds.extend(predicted.cpu().numpy())
#         all_labels.extend(labels.cpu().numpy())

# accuracy = 100 * correct / total
# precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', zero_division=0)
# conf_matrix = confusion_matrix(all_labels, all_preds)

# print(f"Test Accuracy: {accuracy:.2f}%")
# print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")
# print("Confusion Matrix:\n", conf_matrix)

# # Dự đoán trên 5 mẫu ngẫu nhiên
# def predict_sentiment(model, text, text_pipeline, device):
#     model.eval()
#     with torch.no_grad():
#         processed_text = text_pipeline(text).unsqueeze(0).to(device)
#         lengths = torch.tensor([len(processed_text[0])]).to(device)
#         output = model(processed_text, lengths)
#         _, predicted = torch.max(output, 1)
#         return "Positive" if predicted.item() == 1 else "Negative"

# random_samples = [
#     "This movie is absolutely fantastic, a must-watch for everyone!",
#     "I was bored throughout the entire film, such a waste of time.",
#     "The acting was superb, but the plot felt a bit predictable.",
#     "Terrible direction and no chemistry between the actors.",
#     "A heartwarming story that kept me engaged till the end."
# ]

# print("\nPredictions on random samples:")
# for text in random_samples:
#     sentiment = predict_sentiment(model, text, text_pipeline, device)
#     print(f"Text: {text}\nSentiment: {sentiment}\n")

# # Lưu mô hình
# torch.save(model.state_dict(), 'imdb_sentiment_rnn.pth')




import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    precision_recall_curve,
)
import numpy as np
import os

from model import SentimentRNN
from preprocess_data import train_loader, test_loader, vocab, text_pipeline

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Cấu hình siêu tham số
lr = 0.0005
embed_dim = 100
hidden_dim = 128
num_epochs = 20

# Tạo thư mục lưu biểu đồ nếu chưa có
os.makedirs("plots", exist_ok=True)

# Khởi tạo mô hình
vocab_size = len(vocab)
output_dim = 2
model = SentimentRNN(vocab_size, embed_dim, hidden_dim, output_dim).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

all_losses = []
all_f1s = []

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []

    for labels, texts, lengths in train_loader:
        labels, texts, lengths = labels.to(device), texts.to(device), lengths.to(device)
        optimizer.zero_grad()
        outputs = model(texts, lengths)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        probs = torch.softmax(outputs, dim=1)[:, 1]
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        all_probs.extend(probs.detach().cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * correct / total
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0
    )

    all_losses.append(avg_loss)
    all_f1s.append(f1)

    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%, F1: {f1:.4f}")

# Vẽ biểu đồ Loss và F1
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), all_losses, marker='o', label='Loss')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), all_f1s, marker='o', color='green', label='F1 Score')
plt.title('F1 Score')
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.grid(True)

plt.tight_layout()
plt.savefig("plots/loss_f1_curve.png")
plt.show()

# Đánh giá trên test set
model.eval()
all_preds = []
all_labels = []
all_probs = []
with torch.no_grad():
    for labels, texts, lengths in test_loader:
        labels, texts, lengths = labels.to(device), texts.to(device), lengths.to(device)
        outputs = model(texts, lengths)
        probs = torch.softmax(outputs, dim=1)[:, 1]
        _, predicted = torch.max(outputs, 1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

# Confusion matrix
conf_matrix = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(5, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Negative", "Positive"],
            yticklabels=["Negative", "Positive"])
plt.title("Confusion Matrix on Test Set")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig("plots/confusion_matrix.png")
plt.show()

# Precision–Recall curve
precision_vals, recall_vals, _ = precision_recall_curve(all_labels, all_probs)
plt.figure(figsize=(6, 5))
plt.plot(recall_vals, precision_vals, marker='.')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision–Recall Curve')
plt.grid(True)
plt.savefig("plots/precision_recall_curve.png")
plt.show()
