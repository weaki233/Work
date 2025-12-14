import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
import numpy as np
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# 设置随机种子
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# ==========================================
# 配置区域
# ==========================================
# 定义每一列标签的完整类别列表 (List of Lists)
# 假设我们要预测 3 个属性，请确保这里的顺序与 txt 文件中标签列的顺序一致
ALL_CLASS_DEFINITIONS = [
    # 第 1 个属性列的所有可能类别 (例如：新闻分类)
    ["体育", "财经", "科技", "娱乐", "政治", "教育"],

    # 第 2 个属性列的所有可能类别 (例如：情感分析)
    ["正面", "负面", "中性"],

]

# 自动计算任务数量
LABEL_COLUMN_COUNT = len(ALL_CLASS_DEFINITIONS)
# 假设 txt 文件没有表头(header=None)，第一列是文本，后面 N 列是标签
# 如果有表头，请将 header=None 改为 header=0
DATA_SEP = '\t'  # txt 分隔符，根据实际情况修改，如 ',' 或 ' '

# ==========================================

class MultiTaskDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_len=128, is_train=True, class_definitions=None):
        self.tokenizer = tokenizer
        self.max_len = max_len
        # 使用用户传入的类别定义
        self.class_definitions = class_definitions
        self.num_tasks = len(class_definitions)

        print(f"加载文件: {file_path}")

        # 1. 加载数据
        try:
            # header=None 表示无表头
            df = pd.read_csv(file_path, sep=DATA_SEP, header=None, dtype=str)
            df = df.dropna()

            # 检查列数是否匹配
            if df.shape[1] < (1 + self.num_tasks):
                print(f"警告: 数据列数 ({df.shape[1]}) 小于预期 (文本列 + {self.num_tasks}个标签列)")

            self.data = df
        except Exception as e:
            print(f"加载数据失败: {e}")
            self.data = pd.DataFrame()
            self.texts = []
            return

        self.texts = self.data.iloc[:, 0].tolist()

        # 获取标签原始文本
        self.labels_raw = [self.data.iloc[:, i + 1].tolist() for i in range(self.num_tasks)]

        # 2. 构建固定映射 (Label Mapping)
        # 无论是不是训练集，都必须使用 class_definitions 生成映射，保证 ID 一致
        self.label_mappings = []
        for i in range(self.num_tasks):
            # 获取用户定义的完整类别列表
            defined_labels = self.class_definitions[i]

            # 构建映射字典
            label2id = {label: idx for idx, label in enumerate(defined_labels)}
            id2label = {idx: label for label, idx in label2id.items()}

            self.label_mappings.append({
                'label2id': label2id,
                'id2label': id2label,
                'num_classes': len(defined_labels)
            })

            # 3. 数据校验 (可选，但在实际工程中很重要)
            # 检查当前数据中是否有“未定义”的标签
            current_data_labels = set(self.labels_raw[i])
            defined_set = set(defined_labels)
            unknown_labels = current_data_labels - defined_set
            if unknown_labels:
                print(f"错误: 任务 {i + 1} 的数据中发现了未定义的标签: {unknown_labels}")
                print(f"请在 ALL_CLASS_DEFINITIONS 中补充这些类别，或清洗数据。")
                # 这里可以选择 raise ValueError 终止程序

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = str(self.texts[index])

        label_ids = []
        for i in range(self.num_tasks):
            label_str = self.labels_raw[i][index]
            mapping = self.label_mappings[i]['label2id']

            # 如果数据中的标签不在我们定义的列表中，这里会报错或需要处理
            if label_str not in mapping:
                # 遇到未定义标签，暂时给一个默认值 0 (或者你可以选择报错)
                print(f"警告: 样本 {index} 含有未定义标签 {label_str}")
                label_id = 0
            else:
                label_id = mapping[label_str]

            label_ids.append(label_id)

        # 滑动窗口逻辑 (保持不变)
        window_size = 128
        stride = 64
        max_windows = 3

        input_ids_list = []
        attention_mask_list = []

        for i in range(0, len(text), stride):
            if len(input_ids_list) >= max_windows:
                break
            window_text = text[i:i + window_size]
            encoding = self.tokenizer.encode_plus(
                window_text, add_special_tokens=True, max_length=self.max_len,
                padding='max_length', truncation=True, return_attention_mask=True, return_tensors='pt'
            )
            input_ids_list.append(encoding['input_ids'])
            attention_mask_list.append(encoding['attention_mask'])

        if len(input_ids_list) == 0:
            encoding = self.tokenizer.encode_plus(
                text, add_special_tokens=True, max_length=self.max_len,
                padding='max_length', truncation=True, return_attention_mask=True, return_tensors='pt'
            )
            input_ids_list.append(encoding['input_ids'])
            attention_mask_list.append(encoding['attention_mask'])

        input_ids = torch.cat(input_ids_list, dim=0)
        attention_mask = torch.cat(attention_mask_list, dim=0)

        return {
            'input_ids': input_ids.flatten(),
            'attention_mask': attention_mask.flatten(),
            # 返回一个张量包含所有任务的标签
            'labels': torch.tensor(label_ids, dtype=torch.long)
        }


# 多任务分类模型
class MultiTaskBERT(nn.Module):
    def __init__(self, model_path, label_mappings):
        super(MultiTaskBERT, self).__init__()
        self.bert = BertModel.from_pretrained(model_path)
        self.dropout = nn.Dropout(0.3)
        self.label_mappings = label_mappings

        # 创建多个分类头 (ModuleList)
        # 每个任务对应一个独立的 Linear 层
        self.classifiers = nn.ModuleList()
        for mapping in label_mappings:
            n_classes = mapping['num_classes']
            self.classifiers.append(nn.Linear(self.bert.config.hidden_size, n_classes))

    def forward(self, input_ids, attention_mask):
        # 处理滑动窗口维度 (Batch * Windows, Seq)
        batch_size = input_ids.size(0)  # 注意：这里实际上是 batch_size，但如果 dataloader 这里的 shape 是 [B, num_win*seq]，需要处理
        # 这里的 input_ids 已经是 flatten 过的，我们需要根据实际 batch 拆分
        # 但为简化，我们在 Dataset 里 flatten 了，这里直接进 BERT
        # 修正：Dataset 输出 input_ids 是 [num_windows * seq_len]，DataLoader 会增加 batch 维 -> [B, num_windows * seq_len]

        # 获取正确的 seq_length (假设 max_len)
        # 简单处理：视为 [B * num_windows, seq_len]
        total_len = input_ids.size(1)
        # 这里的 reshape 逻辑需根据 max_len 动态调整，或假设 Dataset 输出固定长度
        # 简单起见，利用 attention_mask 的形状推断

        # 此时 input_ids: [Batch, Total_Seq_Len] -> 需要 Reshape 为 [Batch * Windows, Single_Seq_Len]
        # 假设 Dataset 中的 max_len 是单个窗口长度
        # 这是一个 Trick: 为了处理不定长窗口，这里简单假设 Flatten 后的维度被正确 Reshape
        # 实际代码中，建议 Dataset 返回 [Batch, Windows, Seq]

        # 为了兼容之前的逻辑，我们假设 input_ids 已经被正确 view
        # 实际上 Dataset 返回的是 flatten 的，所以我们需要知道每个样本有多少个 windows
        # 原代码逻辑：input_ids.view(-1, seq_length) 依赖于 batch_size 和 seq_length

        # 重新推导维度
        real_batch_size = input_ids.size(0)
        seq_len_per_window = 128  # 需与 Dataset max_len 一致，建议传入参数

        input_ids = input_ids.view(-1, seq_len_per_window)
        attention_mask = attention_mask.view(-1, seq_len_per_window)

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # [Batch * Windows, Hidden]

        # 聚合窗口：还原为 [Batch, Windows, Hidden] -> Mean -> [Batch, Hidden]
        pooled_output = pooled_output.view(real_batch_size, -1, pooled_output.size(-1))
        pooled_output = torch.mean(pooled_output, dim=1)

        pooled_output = self.dropout(pooled_output)

        # 多个分类头并行输出
        logits_list = []
        for classifier in self.classifiers:
            logits_list.append(classifier(pooled_output))

        return logits_list


def evaluate_model(model, data_loader, device):
    model.eval()
    num_tasks = len(model.classifiers)

    # 记录每个任务的正确数
    correct_counts = [0] * num_tasks
    total = 0

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluation"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)  # [Batch, Num_Tasks]

            # 获取所有任务的 logits
            logits_list = model(input_ids, attention_mask)

            for task_idx, logits in enumerate(logits_list):
                _, preds = torch.max(logits, dim=1)
                task_labels = labels[:, task_idx]
                correct_counts[task_idx] += torch.sum(preds == task_labels).item()

            total += labels.size(0)

    accuracies = [count / total for count in correct_counts] if total > 0 else [0] * num_tasks
    return accuracies


def train_model(model, train_loader, val_loader, optimizer, scheduler, device, num_epochs=5):
    best_avg_acc = 0
    criterion = nn.CrossEntropyLoss()
    num_tasks = len(model.classifiers)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')

        # Training
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc="Training"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)  # [Batch, Num_Tasks]

            optimizer.zero_grad()

            logits_list = model(input_ids, attention_mask)

            loss = 0
            # 计算所有任务的 Loss 之和
            for task_idx, logits in enumerate(logits_list):
                task_loss = criterion(logits, labels[:, task_idx])
                loss += task_loss

            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Train Loss: {avg_loss:.4f}")

        # Validation
        val_accuracies = evaluate_model(model, val_loader, device)
        avg_val_acc = sum(val_accuracies) / len(val_accuracies)

        acc_str = ", ".join([f"Task{i + 1}: {acc:.4f}" for i, acc in enumerate(val_accuracies)])
        print(f"Val Accuracies: [{acc_str}] | Avg: {avg_val_acc:.4f}")

        if avg_val_acc > best_avg_acc:
            best_avg_acc = avg_val_acc
            torch.save(model.state_dict(), 'best_model_multitask.pth')
            print("保存最佳模型")

    return model


def main():
    # 参数设置
    BERT_MODEL_PATH = './bert-base-multilingual-cased'  # 请确保此路径正确
    MAX_LEN = 128  # 与 Dataset 中的 window_size 对应
    BATCH_SIZE = 8
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 10

    # 文件路径 (假设都在当前目录或指定目录)
    # 输入文件格式：txt/csv, 无表头
    # Col 0: Text, Col 1: Label1, Col 2: Label2, Col 3: Label3...
    train_path = "train.txt"
    dev_path = "dev.txt"
    test_path = "test.txt"

    # 检查配置
    if not ALL_CLASS_DEFINITIONS:
        print("错误: 请在代码顶部配置 ALL_CLASS_DEFINITIONS")
        return
    if not os.path.exists(BERT_MODEL_PATH):
        print(f"错误: 模型路径 {BERT_MODEL_PATH} 不存在")
        return

    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH)

    # 1. 加载训练集并构建映射
    print("正在加载训练数据...")
    if not os.path.exists(train_path):
        print(f"训练文件 {train_path} 不存在，请检查路径")
        return

    train_dataset = MultiTaskDataset(train_path, tokenizer, max_len=MAX_LEN, is_train=True, class_definitions=ALL_CLASS_DEFINITIONS)
    if len(train_dataset) == 0: return

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 2. 加载验证集 (使用训练集的映射)
    if os.path.exists(dev_path):
        val_dataset = MultiTaskDataset(dev_path, tokenizer, max_len=MAX_LEN, is_train=False,
                                       class_definitions=ALL_CLASS_DEFINITIONS)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    else:
        print("未找到验证集，使用训练集作为验证（仅供调试）")
        val_loader = train_loader

    # 3. 初始化模型
    model = MultiTaskBERT(BERT_MODEL_PATH, train_dataset.label_mappings)
    model = model.to(device)

    # 4. 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * NUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # 5. 训练
    model = train_model(model, train_loader, val_loader, optimizer, scheduler, device, NUM_EPOCHS)

    # 6. 测试 (可选)
    if os.path.exists(test_path):
        print("\n开始测试...")
        test_dataset = MultiTaskDataset(test_path, tokenizer, max_len=MAX_LEN, is_train=False, class_definitions=ALL_CLASS_DEFINITIONS)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # 加载最佳权重
        model.load_state_dict(torch.load('best_model_multitask.pth'))

        test_accs = evaluate_model(model, test_loader, device)
        acc_str = ", ".join([f"Task{i + 1}: {acc:.4f}" for i, acc in enumerate(test_accs)])
        print(f"测试集最终结果: [{acc_str}]")


if __name__ == '__main__':
    main()