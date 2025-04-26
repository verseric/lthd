import torch
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import IMDB

# Tokenizer
tokenizer = get_tokenizer('basic_english')

# Tải toàn bộ IMDB
train_iter, test_iter = IMDB(split=('train', 'test'))
train_data = [(label, text) for label, text in train_iter]
test_data = [(label, text) for label, text in test_iter]

# # Kiểm tra nhãn gốc
# print("Sample train labels:")
# for i, (label, _) in enumerate(train_data[:10]):
#     print(f"Sample {i}: Label = {label}, Type = {type(label)}")

# Xây dựng từ điển
def yield_tokens(data):
    for _, text in data:
        yield tokenizer(text)

vocab = build_vocab_from_iterator(yield_tokens(train_data), specials=['<unk>', '<pad>'], max_tokens=5000)
vocab.set_default_index(vocab['<unk>'])

# Hàm xử lý
def text_pipeline(text):
    tokens = tokenizer(text)[:100]
    return torch.tensor(vocab(tokens), dtype=torch.long)

def label_pipeline(label):
    return 1 if label == 2 else 0  # torchtext 0.16.0: 2 = Positive, 1 = Negative

# Hàm tạo batch
def collate_batch(batch):
    label_list, text_list, lengths = [], [], []
    for _label, _text in batch:
        label_list.append(label_pipeline(_label))
        processed_text = text_pipeline(_text)
        text_list.append(processed_text)
        lengths.append(len(processed_text))
    text_list = torch.nn.utils.rnn.pad_sequence(text_list, padding_value=vocab['<pad>'], batch_first=True)
    return torch.tensor(label_list, dtype=torch.long), text_list, torch.tensor(lengths)

# Tạo DataLoader
train_loader = DataLoader(train_data, batch_size=16, shuffle=True, collate_fn=collate_batch)
test_loader = DataLoader(test_data, batch_size=16, collate_fn=collate_batch)

# Kiểm tra nhãn
train_labels = []
for labels, _, _ in train_loader:
    train_labels.extend(labels.numpy())
print(f"Train labels - Positive: {sum(1 for x in train_labels if x == 1)}, Negative: {sum(1 for x in train_labels if x == 0)}")

test_labels = []
for labels, _, _ in test_loader:
    test_labels.extend(labels.numpy())
print(f"Test labels - Positive: {sum(1 for x in test_labels if x == 1)}, Negative: {sum(1 for x in test_labels if x == 0)}")