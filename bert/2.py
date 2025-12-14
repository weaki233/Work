import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score
import numpy as np
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# 设置随机种子确保结果可复现
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


# 数据加载和预处理
class MultiLevelTextDataset(Dataset):
    def __init__(self, ctg_file, dcp_file, tokenizer, max_len=128, is_train=True, train_dataset=None):
        self.tokenizer = tokenizer
        self.max_len = max_len

        print(f"加载第一级文件: {ctg_file}")
        print(f"加载第二级文件: {dcp_file}")

        # 加载第一级分类数据
        try:
            if ctg_file.endswith('.tsv'):
                ctg_df = pd.read_csv(ctg_file, sep='\t')
            else:
                ctg_df = pd.read_csv(ctg_file)
            print(f"第一级数据形状: {ctg_df.shape}")
            print(f"第一级数据列名: {ctg_df.columns.tolist()}")
        except Exception as e:
            print(f"加载第一级数据失败: {e}")
            ctg_df = pd.DataFrame()

        # 加载第二级分类数据
        try:
            if dcp_file.endswith('.tsv'):
                dcp_df = pd.read_csv(dcp_file, sep='\t')
            else:
                dcp_df = pd.read_csv(dcp_file)
            print(f"第二级数据形状: {dcp_df.shape}")
            print(f"第二级数据列名: {dcp_df.columns.tolist()}")
        except Exception as e:
            print(f"加载第二级数据失败: {e}")
            dcp_df = pd.DataFrame()

        if ctg_df.empty or dcp_df.empty:
            print("数据加载失败，创建空数据集")
            self.data = pd.DataFrame()
            return

        # 根据数据形状确定文本列和标签列
        if ctg_df.shape[1] >= 3:
            ctg_text_col = ctg_df.columns[1]  # 第二列为文本
            ctg_label_col = ctg_df.columns[2]  # 第三列为标签
            ctg_df = ctg_df.rename(columns={ctg_text_col: 'text', ctg_label_col: 'label_ctg'})
        else:
            print("第一级数据列数不足3列")
            self.data = pd.DataFrame()
            return

        if dcp_df.shape[1] >= 3:
            dcp_text_col = dcp_df.columns[1]  # 第二列为文本
            dcp_label_col = dcp_df.columns[2]  # 第三列为标签
            dcp_df = dcp_df.rename(columns={dcp_text_col: 'text', dcp_label_col: 'label_dcp'})
        else:
            print("第二级数据列数不足3列")
            self.data = pd.DataFrame()
            return

        print(f"处理后的第一级数据列名: {ctg_df.columns.tolist()}")
        print(f"处理后的第二级数据列名: {dcp_df.columns.tolist()}")

        # 根据索引合并数据
        ctg_df = ctg_df.reset_index().rename(columns={'index': 'id'})
        dcp_df = dcp_df.reset_index().rename(columns={'index': 'id'})

        # 合并数据
        self.data = pd.merge(ctg_df[['id', 'text', 'label_ctg']],
                             dcp_df[['id', 'text', 'label_dcp']],
                             on='id',
                             suffixes=('_ctg', '_dcp'))

        print(f"合并后数据形状: {self.data.shape}")
        print(f"合并后数据列名: {self.data.columns.tolist()}")

        # 创建标签映射
        if is_train:
            self.ctg_label2id = {label: idx for idx, label in enumerate(sorted(self.data['label_ctg'].unique()))}
            self.ctg_id2label = {idx: label for label, idx in self.ctg_label2id.items()}

            # 为每个第一级类别创建第二级标签映射
            self.dcp_label_mappings = {}
            for ctg_label in self.ctg_label2id.keys():
                dcp_labels = self.data[self.data['label_ctg'] == ctg_label]['label_dcp'].unique()
                self.dcp_label_mappings[ctg_label] = {
                    'label2id': {label: idx for idx, label in enumerate(sorted(dcp_labels))},
                    'id2label': {idx: label for idx, label in enumerate(sorted(dcp_labels))},
                    'num_classes': len(dcp_labels)
                }
        else:
            # 验证/测试集：从训练集复制标签映射
            if train_dataset is not None:
                self.ctg_label2id = train_dataset.ctg_label2id
                self.ctg_id2label = train_dataset.ctg_id2label
                self.dcp_label_mappings = train_dataset.dcp_label_mappings
            else:
                # 如果没有提供训练集引用，尝试从数据推断（简化版）
                self.ctg_label2id = {label: idx for idx, label in enumerate(sorted(self.data['label_ctg'].unique()))}
                self.ctg_id2label = {idx: label for label, idx in self.ctg_label2id.items()}
                self.dcp_label_mappings = {}
                for ctg_label in self.ctg_label2id.keys():
                    dcp_labels = self.data[self.data['label_ctg'] == ctg_label]['label_dcp'].unique()
                    self.dcp_label_mappings[ctg_label] = {
                        'label2id': {label: idx for idx, label in enumerate(sorted(dcp_labels))},
                        'id2label': {idx: label for idx, label in enumerate(sorted(dcp_labels))},
                        'num_classes': len(dcp_labels)
                    }

        self.texts = self.data['text_ctg'].tolist()
        self.ctg_labels = self.data['label_ctg'].tolist()
        self.dcp_labels = self.data['label_dcp'].tolist()

        print(f"样本数量: {len(self.texts)}")
        if len(self.texts) > 0:
            print(f"第一个样本文本: {self.texts[0][:50]}...")
            print(f"第一级标签: {self.ctg_labels[0]}")
            print(f"第二级标签: {self.dcp_labels[0]}")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = str(self.texts[index])
        ctg_label = self.ctg_labels[index]
        dcp_label = self.dcp_labels[index]

        # 滑动窗口参数
        window_size = 128  # 每个窗口的大小
        stride = 64  # 滑动步长
        max_windows = 3  # 最多取多少个窗口

        # 初始化输入列表
        input_ids_list = []
        attention_mask_list = []

        # 使用滑动窗口处理长文本
        for i in range(0, len(text), stride):
            if len(input_ids_list) >= max_windows:
                break

            # 获取当前窗口的文本
            window_text = text[i:i + window_size]

            # 编码当前窗口的文本
            encoding = self.tokenizer.encode_plus(
                window_text,
                add_special_tokens=True,
                max_length=self.max_len,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt',
            )

            input_ids_list.append(encoding['input_ids'])
            attention_mask_list.append(encoding['attention_mask'])

        # 如果没有获取到任何窗口，使用全文本
        if len(input_ids_list) == 0:
            encoding = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=self.max_len,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt',
            )
            input_ids_list.append(encoding['input_ids'])
            attention_mask_list.append(encoding['attention_mask'])

        # 将多个窗口的输入堆叠起来
        input_ids = torch.cat(input_ids_list, dim=0)
        attention_mask = torch.cat(attention_mask_list, dim=0)

        return {
            'input_ids': input_ids.flatten(),
            'attention_mask': attention_mask.flatten(),
            'ctg_label': torch.tensor(self.ctg_label2id[ctg_label], dtype=torch.long),
            'dcp_label': torch.tensor(self.dcp_label_mappings[ctg_label]['label2id'][dcp_label], dtype=torch.long),
            'ctg_class': ctg_label,
            'num_windows': len(input_ids_list)  # 添加窗口数量信息
        }


# 多级分类模型
class MultiLevelBERT(nn.Module):
    def __init__(self, n_ctg_classes, dcp_label_mappings, model_path):
        super(MultiLevelBERT, self).__init__()
        # 使用本地模型路径
        self.bert = BertModel.from_pretrained(model_path)
        self.dropout = nn.Dropout(0.3)

        # 第一级分类器
        self.ctg_classifier = nn.Linear(self.bert.config.hidden_size, n_ctg_classes)

        # 存储第二级分类器信息
        self.dcp_label_mappings = dcp_label_mappings
        self.dcp_classifiers = nn.ModuleList()

        # 为每个第一级类别创建第二级分类器
        self.dcp_classifier_dict = {}
        for ctg_label, mapping in dcp_label_mappings.items():
            n_dcp_classes = mapping['num_classes']
            classifier = nn.Linear(self.bert.config.hidden_size, n_dcp_classes)
            self.dcp_classifier_dict[ctg_label] = classifier
            self.dcp_classifiers.append(classifier)

    def forward(self, input_ids, attention_mask, ctg_class=None):
        # 获取batch大小和窗口数量
        batch_size = len(ctg_class)
        seq_length = input_ids.size(1) // batch_size

        # 重塑输入形状 (batch_size * num_windows, seq_len)
        input_ids = input_ids.view(-1, seq_length)
        attention_mask = attention_mask.view(-1, seq_length)

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # 获取每个窗口的pooled_output
        pooled_output = outputs.pooler_output

        # 计算每个样本的平均表示
        pooled_output = pooled_output.view(batch_size, -1, pooled_output.size(-1))
        pooled_output = torch.mean(pooled_output, dim=1)  # 对窗口取平均

        pooled_output = self.dropout(pooled_output)

        # 第一级分类
        ctg_logits = self.ctg_classifier(pooled_output)

        # 第二级分类 - 分别处理每个样本
        batch_size = pooled_output.size(0)
        dcp_logits_list = []

        for i in range(batch_size):
            class_name = ctg_class[i]
            classifier = self.dcp_classifier_dict[class_name]
            sample_output = pooled_output[i].unsqueeze(0)
            dcp_logit = classifier(sample_output)
            dcp_logits_list.append(dcp_logit)

        # 创建一个占位符张量来存储所有第二级logits
        max_dcp_classes = max([mapping['num_classes'] for mapping in self.dcp_label_mappings.values()])
        dcp_logits = torch.full((batch_size, max_dcp_classes), -100.0, device=device)

        for i, logit in enumerate(dcp_logits_list):
            num_classes = logit.size(1)
            dcp_logits[i, :num_classes] = logit

        return ctg_logits, dcp_logits


# 测试评估函数
def evaluate_model(model, data_loader, device):
    model.eval()
    ctg_correct = 0
    dcp_correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Testing"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            ctg_labels = batch['ctg_label'].to(device)
            dcp_labels = batch['dcp_label'].to(device)
            ctg_class = batch['ctg_class']

            ctg_logits, dcp_logits = model(input_ids, attention_mask, ctg_class)

            # 计算第二级分类的mask
            dcp_mask = (dcp_logits != -100.0).any(dim=1)
            valid_dcp_logits = dcp_logits[dcp_mask]
            valid_dcp_labels = dcp_labels[dcp_mask]

            # 计算准确率
            _, ctg_preds = torch.max(ctg_logits, dim=1)
            ctg_correct += torch.sum(ctg_preds == ctg_labels)

            if len(valid_dcp_logits) > 0:
                _, dcp_preds = torch.max(valid_dcp_logits, dim=1)
                dcp_correct += torch.sum(dcp_preds == valid_dcp_labels)

            total += ctg_labels.size(0)

    ctg_acc = ctg_correct.double() / total
    dcp_acc = dcp_correct.double() / total if total > 0 else 0

    return ctg_acc.item(), dcp_acc.item()


# 训练函数
def train_model(model, train_loader, val_loader, optimizer, scheduler, device, num_epochs=10):
    best_accuracy = 0
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        # 训练阶段
        model.train()
        train_loss = 0.0
        ctg_correct = 0
        dcp_correct = 0
        total = 0

        for batch in tqdm(train_loader, desc="Training"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            ctg_labels = batch['ctg_label'].to(device)
            dcp_labels = batch['dcp_label'].to(device)
            ctg_class = batch['ctg_class']

            optimizer.zero_grad()

            ctg_logits, dcp_logits = model(input_ids, attention_mask, ctg_class)

            # 计算第二级分类的mask，只计算有效类别的损失
            dcp_mask = (dcp_logits != -100.0).any(dim=1)
            valid_dcp_logits = dcp_logits[dcp_mask]
            valid_dcp_labels = dcp_labels[dcp_mask]

            loss_ctg = criterion(ctg_logits, ctg_labels)
            loss_dcp = criterion(valid_dcp_logits, valid_dcp_labels) if len(valid_dcp_logits) > 0 else 0
            loss = loss_ctg + loss_dcp

            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

            # 计算准确率
            _, ctg_preds = torch.max(ctg_logits, dim=1)
            ctg_correct += torch.sum(ctg_preds == ctg_labels)

            # 计算第二级准确率（只计算有效预测）
            if len(valid_dcp_logits) > 0:
                _, dcp_preds = torch.max(valid_dcp_logits, dim=1)
                dcp_correct += torch.sum(dcp_preds == valid_dcp_labels)

            total += ctg_labels.size(0)

        train_loss = train_loss / len(train_loader)
        train_ctg_acc = ctg_correct.double() / total
        train_dcp_acc = dcp_correct.double() / total if total > 0 else 0

        print(f'Train loss: {train_loss:.4f}, CTG Acc: {train_ctg_acc:.4f}, DCP Acc: {train_dcp_acc:.4f}')

        # 验证阶段
        model.eval()
        val_loss = 0.0
        ctg_correct = 0
        dcp_correct = 0
        total = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                ctg_labels = batch['ctg_label'].to(device)
                dcp_labels = batch['dcp_label'].to(device)
                ctg_class = batch['ctg_class']

                ctg_logits, dcp_logits = model(input_ids, attention_mask, ctg_class)

                # 计算第二级分类的mask
                dcp_mask = (dcp_logits != -100.0).any(dim=1)
                valid_dcp_logits = dcp_logits[dcp_mask]
                valid_dcp_labels = dcp_labels[dcp_mask]

                loss_ctg = criterion(ctg_logits, ctg_labels)
                loss_dcp = criterion(valid_dcp_logits, valid_dcp_labels) if len(valid_dcp_logits) > 0 else 0
                loss = loss_ctg + loss_dcp

                val_loss += loss.item()

                _, ctg_preds = torch.max(ctg_logits, dim=1)
                ctg_correct += torch.sum(ctg_preds == ctg_labels)

                if len(valid_dcp_logits) > 0:
                    _, dcp_preds = torch.max(valid_dcp_logits, dim=1)
                    dcp_correct += torch.sum(dcp_preds == valid_dcp_labels)

                total += ctg_labels.size(0)

        val_loss = val_loss / len(val_loader)
        val_ctg_acc = ctg_correct.double() / total
        val_dcp_acc = dcp_correct.double() / total if total > 0 else 0

        print(f'Val loss: {val_loss:.4f}, CTG Acc: {val_ctg_acc:.4f}, DCP Acc: {val_dcp_acc:.4f}')

        # 保存最佳模型（删除了输出语句）
        if val_dcp_acc > best_accuracy:
            best_accuracy = val_dcp_acc
            torch.save(model.state_dict(), 'best_model.pth')

    print(f'训练完成，最佳准确率: {best_accuracy:.4f}')
    return model


# 主函数
def main():
    # 参数设置
    BERT_MODEL_PATH = './bert-base-multilingual-cased'
    MAX_LEN = 256
    BATCH_SIZE = 8
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 5

    # 检查模型路径
    if not os.path.exists(BERT_MODEL_PATH):
        print(f"错误: 模型路径 {BERT_MODEL_PATH} 不存在")
        return

    print("正在加载本地BERT模型...")
    try:
        tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH)
        print("分词器加载成功")
    except Exception as e:
        print(f"加载分词器失败: {e}")
        return

    # 加载数据
    base_path = "cls"
    ctg_train_path = os.path.join(base_path, "cls_ctg", "train.tsv")
    dcp_train_path = os.path.join(base_path, "cls_dcp", "train.tsv")
    ctg_dev_path = os.path.join(base_path, "cls_ctg", "dev.tsv")
    dcp_dev_path = os.path.join(base_path, "cls_dcp", "dev.tsv")
    ctg_test_path = os.path.join(base_path, "cls_ctg", "test.csv")
    dcp_test_path = os.path.join(base_path, "cls_dcp", "test.csv")

    # 检查文件是否存在
    for path in [ctg_train_path, dcp_train_path]:
        if not os.path.exists(path):
            print(f"错误: 文件 {path} 不存在")
            print("请检查文件路径")
            return

    print("正在加载训练数据...")
    try:
        train_dataset = MultiLevelTextDataset(ctg_train_path, dcp_train_path, tokenizer, MAX_LEN)

        if len(train_dataset) == 0:
            print("训练数据为空，无法继续")
            return

        print(f"训练数据大小: {len(train_dataset)}")
        print(f"第一级类别数量: {len(train_dataset.ctg_label2id)}")
        print(f"第一级类别: {list(train_dataset.ctg_label2id.keys())}")

        # 显示每个第一级类别对应的第二级类别数量
        for ctg_label, mapping in train_dataset.dcp_label_mappings.items():
            print(f"第一级类别 '{ctg_label}' 有 {mapping['num_classes']} 个第二级类别")

        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

        # 加载验证数据
        val_dataset = MultiLevelTextDataset(ctg_dev_path, dcp_dev_path, tokenizer, MAX_LEN,
                                            is_train=False, train_dataset=train_dataset)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # 初始化模型
        n_ctg_classes = len(train_dataset.ctg_label2id)
        model = MultiLevelBERT(n_ctg_classes, train_dataset.dcp_label_mappings, BERT_MODEL_PATH)
        model = model.to(device)

        # 设置优化器
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
        total_steps = len(train_loader) * NUM_EPOCHS
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )

        print("开始训练模型...")
        model = train_model(model, train_loader, val_loader, optimizer, scheduler, device, NUM_EPOCHS)

        print("训练完成！")

        # 测试集评估
        if os.path.exists(ctg_test_path) and os.path.exists(dcp_test_path):
            print("\n" + "=" * 50)
            print("开始测试集评估...")
            print("=" * 50)

            # 加载测试数据
            test_dataset = MultiLevelTextDataset(ctg_test_path, dcp_test_path, tokenizer, MAX_LEN,
                                                 is_train=False, train_dataset=train_dataset)
            test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

            print(f"测试数据大小: {len(test_dataset)}")

            # 加载最佳模型
            if os.path.exists('best_model.pth'):
                model.load_state_dict(torch.load('best_model.pth'))
                print("已加载最佳模型进行测试")

            # 在测试集上评估
            test_ctg_acc, test_dcp_acc = evaluate_model(model, test_loader, device)

            print("\n" + "=" * 50)
            print("测试集最终结果:")
            print("=" * 50)
            print(f"第一级分类准确率 (CTG Acc): {test_ctg_acc:.4f}")
            print(f"第二级分类准确率 (DCP Acc): {test_dcp_acc:.4f}")
            print("=" * 50)

        else:
            print(f"\n测试集文件不存在，跳过测试评估")
            print(f"第一级测试文件路径: {ctg_test_path}")
            print(f"第二级测试文件路径: {dcp_test_path}")

    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()