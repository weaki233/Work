# 导入所有需要的库
import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score
import numpy as np
from tqdm import tqdm  # 用于显示进度条，方便观察训练过程
import warnings

# 忽略一些不影响程序运行的警告信息
warnings.filterwarnings('ignore')

# --- 1. 全局设置 ---

# 设置随机种子，以确保每次运行代码时，随机过程（如参数初始化、数据打乱）的结果都是一样的，便于复现实验结果
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
# 如果使用CUDA，也为GPU设置随机种子
torch.cuda.manual_seed_all(SEED)

# 自动选择设备：如果检测到有可用的NVIDIA GPU，则使用'cuda'，否则使用'cpu'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


# --- 2. 数据集定义 (MultiLevelTextDataset) ---
# 这是整个代码的核心部分之一，负责加载、合并和预处理两级分类的数据。
class MultiLevelTextDataset(Dataset):
    """
    自定义的PyTorch数据集类，用于处理两级（层级）文本分类任务。
    它会同时加载一级分类和二级分类的数据，并根据原始行号（索引）将它们合并。
    """

    def __init__(self, ctg_file, dcp_file, tokenizer, max_len=128, is_train=True, train_dataset=None):
        """
        初始化数据集。
        :param ctg_file: str, 第一级分类数据的文件路径 (e.g., train.tsv for categories)
        :param dcp_file: str, 第二级分类数据的文件路径 (e.g., train.tsv for descriptions)
        :param tokenizer: BertTokenizer, 用于文本编码的分词器
        :param max_len: int, 文本编码后的最大长度，超过则截断，不足则填充
        :param is_train: bool, 标记当前是训练集还是验证/测试集。训练集需要创建标签映射，而验证/测试集需要复用训练集的映射。
        :param train_dataset: MultiLevelTextDataset, 当 is_train=False 时，必须传入训练集对象，以便共享标签映射。
        """
        self.tokenizer = tokenizer
        self.max_len = max_len

        print(f"加载第一级文件: {ctg_file}")
        print(f"加载第二级文件: {dcp_file}")

        # --- 数据加载 ---
        # 使用 try-except 结构来增加代码的鲁棒性，防止因文件不存在或格式错误导致程序崩溃
        try:
            # 根据文件扩展名判断分隔符，.tsv 使用制表符，其他（如.csv）使用逗号
            if ctg_file.endswith('.tsv'):
                ctg_df = pd.read_csv(ctg_file, sep='\t')
            else:
                ctg_df = pd.read_csv(ctg_file)
            print(f"第一级数据形状: {ctg_df.shape}")
            print(f"第一级数据列名: {ctg_df.columns.tolist()}")
        except Exception as e:
            print(f"加载第一级数据失败: {e}")
            ctg_df = pd.DataFrame()  # 如果加载失败，创建一个空的DataFrame以避免后续代码出错

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

        # 如果任何一个文件加载失败，则无法继续，创建一个空数据集并返回
        if ctg_df.empty or dcp_df.empty:
            print("数据加载失败，创建空数据集")
            self.data = pd.DataFrame()
            return

        # --- 数据预处理与合并 ---
        # 假设数据格式是固定的：第一列是ID(可选)，第二列是文本，第三列是标签
        # 自动识别文本列和标签列，并重命名为统一的'text'和'label_ctg'/'label_dcp'，增强代码的通用性
        if ctg_df.shape[1] >= 3:
            ctg_text_col = ctg_df.columns[1]  # 第二列为文本
            ctg_label_col = ctg_df.columns[2]  # 第三列为标签
            ctg_df = ctg_df.rename(columns={ctg_text_col: 'text', ctg_label_col: 'label_ctg'})
        else:
            print("第一级数据列数不足3列，无法处理")
            self.data = pd.DataFrame()
            return

        if dcp_df.shape[1] >= 3:
            dcp_text_col = dcp_df.columns[1]  # 第二列为文本
            dcp_label_col = dcp_df.columns[2]  # 第三列为标签
            dcp_df = dcp_df.rename(columns={dcp_text_col: 'text', dcp_label_col: 'label_dcp'})
        else:
            print("第二级数据列数不足3列，无法处理")
            self.data = pd.DataFrame()
            return

        print(f"处理后的第一级数据列名: {ctg_df.columns.tolist()}")
        print(f"处理后的第二级数据列名: {dcp_df.columns.tolist()}")

        # 核心步骤：合并数据。
        # 假设 ctg_file 和 dcp_file 中的数据行是一一对应的。
        # .reset_index() 会将原始的行号变成一个名为 'index' 的新列，我们用它作为唯一ID来合并。
        ctg_df = ctg_df.reset_index().rename(columns={'index': 'id'})
        dcp_df = dcp_df.reset_index().rename(columns={'index': 'id'})

        # 使用 pd.merge 根据 'id' 列合并两个DataFrame。
        # 假设两个文件中的文本内容可能不同，通过 suffixes 参数区分它们。
        self.data = pd.merge(ctg_df[['id', 'text', 'label_ctg']],
                             dcp_df[['id', 'text', 'label_dcp']],
                             on='id',
                             suffixes=('_ctg', '_dcp'))

        print(f"合并后数据形状: {self.data.shape}")
        print(f"合并后数据列名: {self.data.columns.tolist()}")

        # --- 创建标签映射 ---
        # 标签映射：将字符串类型的标签（如'体育'）转换为模型可以处理的数字索引（如 0）。
        if is_train:
            # 仅在处理训练集时创建新的映射
            # 1. 第一级标签映射
            self.ctg_label2id = {label: idx for idx, label in enumerate(sorted(self.data['label_ctg'].unique()))}
            self.ctg_id2label = {idx: label for label, idx in self.ctg_label2id.items()}

            # 2. 第二级标签映射（这是分层任务的关键）
            # 我们需要为 *每一个* 第一级类别，单独创建一个第二级标签的映射。
            # 例如，'体育'下的二级标签和'科技'下的二级标签是完全不同的集合。
            self.dcp_label_mappings = {}
            for ctg_label in self.ctg_label2id.keys():
                # 筛选出当前一级类别下的所有数据
                subset_df = self.data[self.data['label_ctg'] == ctg_label]
                # 获取该子集下所有唯一的二级标签
                dcp_labels = subset_df['label_dcp'].unique()
                # 为这个一级类别创建它专属的二级标签映射
                self.dcp_label_mappings[ctg_label] = {
                    'label2id': {label: idx for idx, label in enumerate(sorted(dcp_labels))},
                    'id2label': {idx: label for idx, label in enumerate(sorted(dcp_labels))},
                    'num_classes': len(dcp_labels)  # 记录该一级类别下有多少个二级类别
                }
        else:
            # 如果是验证集或测试集，必须复用训练集的标签映射，以保证标签的一致性。
            if train_dataset is not None:
                self.ctg_label2id = train_dataset.ctg_label2id
                self.ctg_id2label = train_dataset.ctg_id2label
                self.dcp_label_mappings = train_dataset.dcp_label_mappings
            else:
                # 备用逻辑：如果没有提供训练集，则根据当前数据自己创建映射（不推荐，可能导致不一致）
                print("警告: 未提供训练集映射，验证/测试集将根据自身数据创建映射。")
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

        # 将数据列转换为列表，方便索引
        self.texts = self.data['text_ctg'].tolist()  # 使用第一级分类的文本作为输入
        self.ctg_labels = self.data['label_ctg'].tolist()
        self.dcp_labels = self.data['label_dcp'].tolist()

        # 打印一些样本信息以供检查
        print(f"样本数量: {len(self.texts)}")
        if len(self.texts) > 0:
            print(f"第一个样本文本: {self.texts[0][:50]}...")
            print(f"第一级标签: {self.ctg_labels[0]}")
            print(f"第二级标签: {self.dcp_labels[0]}")

    def __len__(self):
        # 返回数据集的总样本数
        return len(self.texts)

    def __getitem__(self, index):
        # 根据索引获取单个样本
        text = str(self.texts[index])
        ctg_label_str = self.ctg_labels[index]
        dcp_label_str = self.dcp_labels[index]

        # --- 文本编码 ---
        # 使用分词器将文本转换为BERT模型可以接受的格式
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,  # 添加 [CLS] 和 [SEP] 特殊标记
            max_length=self.max_len,  # 设置最大长度
            padding='max_length',  # 填充到最大长度
            truncation=True,  # 超过最大长度则截断
            return_attention_mask=True,  # 返回 attention mask，告诉模型哪些是真实token，哪些是padding
            return_tensors='pt',  # 返回 PyTorch 张量
        )

        # 将字符串标签转换为数字ID
        ctg_label_id = self.ctg_label2id[ctg_label_str]
        # 获取二级标签ID时，需要先指定一级标签，再从对应的映射中查找
        dcp_label_id = self.dcp_label_mappings[ctg_label_str]['label2id'][dcp_label_str]

        # 返回一个字典，包含了模型训练所需的所有信息
        return {
            'input_ids': encoding['input_ids'].flatten(),  # 输入的token ID
            'attention_mask': encoding['attention_mask'].flatten(),  # attention mask
            'ctg_label': torch.tensor(ctg_label_id, dtype=torch.long),  # 第一级标签的数字ID
            'dcp_label': torch.tensor(dcp_label_id, dtype=torch.long),  # 第二级标签的数字ID
            'ctg_class': ctg_label_str  # **非常重要**: 同时返回第一级标签的字符串，模型在forward时需要用它来选择正确的二级分类器
        }


# --- 3. 模型定义 (MultiLevelBERT) ---
# 这是模型的架构设计，也是分层任务的另一个核心。
class MultiLevelBERT(nn.Module):
    """
    用于多级分类的BERT模型。
    包含一个共享的BERT主干网络。
    一个用于第一级分类的分类头。
    多个用于第二级分类的分类头，每个头对应一个第一级类别。
    """

    def __init__(self, n_ctg_classes, dcp_label_mappings, model_path):
        """
        :param n_ctg_classes: int, 第一级分类的总类别数
        :param dcp_label_mappings: dict, 从数据集中获取的、包含所有二级分类信息的字典
        :param model_path: str, 预训练BERT模型的本地路径
        """
        super(MultiLevelBERT, self).__init__()
        # 加载预训练的BERT模型作为特征提取器
        self.bert = BertModel.from_pretrained(model_path)
        # Dropout层用于防止过拟合
        self.dropout = nn.Dropout(0.3)

        # 定义第一级分类器：一个简单的线性层，将BERT的输出[CLS]向量映射到第一级类别的数量
        self.ctg_classifier = nn.Linear(self.bert.config.hidden_size, n_ctg_classes)

        # --- 动态创建第二级分类器 ---
        self.dcp_label_mappings = dcp_label_mappings
        # 使用 nn.ModuleList 来妥善管理多个线性层，确保它们能被PyTorch正确识别和训练
        self.dcp_classifiers = nn.ModuleList()
        # 同时使用一个字典，方便在forward过程中通过一级类别的名字快速查找对应的分类器
        self.dcp_classifier_dict = {}

        for ctg_label, mapping in dcp_label_mappings.items():
            # 获取当前一级类别下的二级类别数量
            n_dcp_classes = mapping['num_classes']
            # 创建一个专属的线性分类器
            classifier = nn.Linear(self.bert.config.hidden_size, n_dcp_classes)
            # 添加到字典和ModuleList中
            self.dcp_classifier_dict[ctg_label] = classifier
            self.dcp_classifiers.append(classifier)

    def forward(self, input_ids, attention_mask, ctg_class=None):
        """
        模型的前向传播逻辑。
        :param ctg_class: list of str, **关键参数**，一个批次中每个样本的真实一级类别字符串。
                          在训练和验证时，我们使用真实标签来选择分类器；在推理时，我们会先预测一级类别，再用预测结果来选择。
        """
        # 1. 通过BERT模型获取文本的特征表示
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # pooled_output 对应 [CLS] token 的输出向量，通常用于整个句子的分类任务
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)

        # 2. 第一级分类
        # 将特征向量输入第一级分类器，得到logits
        ctg_logits = self.ctg_classifier(pooled_output)

        # 3. 第二级分类 (这是最复杂的部分)
        # 因为一个批次(batch)中的样本可能属于不同的一级类别，所以它们需要使用不同的二级分类器。
        # 我们不能像一级分类那样用一次矩阵乘法完成，必须逐个样本处理。
        batch_size = pooled_output.size(0)
        dcp_logits_list = []

        for i in range(batch_size):
            # 获取当前样本的真实一级类别名称
            class_name = ctg_class[i]
            # 从字典中找到对应的二级分类器
            classifier = self.dcp_classifier_dict[class_name]
            # 获取当前样本的特征向量
            sample_output = pooled_output[i].unsqueeze(0)  # 保持批次维度
            # 进行二级分类，得到logits
            dcp_logit = classifier(sample_output)
            dcp_logits_list.append(dcp_logit)

        # 4. 组合二级分类结果
        # 问题：dcp_logits_list 中的每个logit张量维度可能不同（因为不同一级类别下的二级类别数不同）。
        # 解决方法：创建一个足够大的“画布”张量，将所有logit“粘贴”上去，空余位置用一个特殊值（如-100）填充。
        # CrossEntropyLoss 会自动忽略 label 为-100的损失计算。

        # 找到所有二级分类器中最大的类别数
        max_dcp_classes = max([mapping['num_classes'] for mapping in self.dcp_label_mappings.values()])
        # 创建一个填充了-100的张量，形状为 (batch_size, max_dcp_classes)
        dcp_logits = torch.full((batch_size, max_dcp_classes), -100.0, device=device)

        # 遍历每个样本的logit，将其复制到“画布”的对应行
        for i, logit in enumerate(dcp_logits_list):
            num_classes = logit.size(1)  # 当前logit的实际类别数
            dcp_logits[i, :num_classes] = logit

        return ctg_logits, dcp_logits


# --- 4. 评估函数 (evaluate_model) ---
def evaluate_model(model, data_loader, device):
    """
    在测试集或验证集上评估模型性能。
    """
    model.eval()  # 将模型设置为评估模式，关闭Dropout等
    ctg_correct = 0
    dcp_correct = 0
    total = 0

    with torch.no_grad():  # 在此代码块中，不计算梯度，节省计算资源
        for batch in tqdm(data_loader, desc="Testing"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            ctg_labels = batch['ctg_label'].to(device)
            dcp_labels = batch['dcp_label'].to(device)
            ctg_class = batch['ctg_class']

            # 获取模型输出
            ctg_logits, dcp_logits = model(input_ids, attention_mask, ctg_class)

            # --- 处理二级分类的输出 ---
            # dcp_logits 中含有填充值-100，计算准确率前需要过滤掉
            # 创建一个掩码(mask)，标记哪些行是有效的（即没有被完全填充）
            dcp_mask = (dcp_logits != -100.0).any(dim=1)
            valid_dcp_logits = dcp_logits[dcp_mask]
            valid_dcp_labels = dcp_labels[dcp_mask]

            # --- 计算准确率 ---
            # 第一级准确率
            _, ctg_preds = torch.max(ctg_logits, dim=1)
            ctg_correct += torch.sum(ctg_preds == ctg_labels)

            # 第二级准确率
            if len(valid_dcp_logits) > 0:
                _, dcp_preds = torch.max(valid_dcp_logits, dim=1)
                dcp_correct += torch.sum(dcp_preds == valid_dcp_labels)

            total += ctg_labels.size(0)

    # 计算最终的平均准确率
    ctg_acc = ctg_correct.double() / total
    dcp_acc = dcp_correct.double() / total if total > 0 else 0

    return ctg_acc.item(), dcp_acc.item()


# --- 5. 训练函数 (train_model) ---
def train_model(model, train_loader, val_loader, optimizer, scheduler, device, num_epochs=10):
    """
    模型训练的主循环。
    """
    best_accuracy = 0
    # 定义损失函数，CrossEntropyLoss适用于多分类任务
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        # --- 训练阶段 ---
        model.train()  # 将模型设置为训练模式
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

            optimizer.zero_grad()  # 每个batch开始前清空梯度

            # 前向传播
            ctg_logits, dcp_logits = model(input_ids, attention_mask, ctg_class)

            # --- 计算损失 ---
            # 同样需要过滤掉二级分类中的填充值
            dcp_mask = (dcp_logits != -100.0).any(dim=1)
            valid_dcp_logits = dcp_logits[dcp_mask]
            valid_dcp_labels = dcp_labels[dcp_mask]

            # 分别计算两级的损失
            loss_ctg = criterion(ctg_logits, ctg_labels)
            loss_dcp = criterion(valid_dcp_logits, valid_dcp_labels) if len(valid_dcp_logits) > 0 else 0
            # 总损失是两级损失之和，也可以加权
            loss = loss_ctg + loss_dcp

            # 反向传播和优化
            loss.backward()
            optimizer.step()
            scheduler.step()  # 更新学习率

            train_loss += loss.item()

            # --- 计算训练集准确率 ---
            _, ctg_preds = torch.max(ctg_logits, dim=1)
            ctg_correct += torch.sum(ctg_preds == ctg_labels)

            if len(valid_dcp_logits) > 0:
                _, dcp_preds = torch.max(valid_dcp_logits, dim=1)
                dcp_correct += torch.sum(dcp_preds == valid_dcp_labels)

            total += ctg_labels.size(0)

        # 计算并打印当前epoch的训练集平均损失和准确率
        train_loss = train_loss / len(train_loader)
        train_ctg_acc = ctg_correct.double() / total
        train_dcp_acc = dcp_correct.double() / total if total > 0 else 0
        print(f'Train loss: {train_loss:.4f}, CTG Acc: {train_ctg_acc:.4f}, DCP Acc: {train_dcp_acc:.4f}')

        # --- 验证阶段 ---
        model.eval()
        val_loss = 0.0
        ctg_correct = 0
        dcp_correct = 0
        total = 0

        with torch.no_grad():  # 不计算梯度
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                ctg_labels = batch['ctg_label'].to(device)
                dcp_labels = batch['dcp_label'].to(device)
                ctg_class = batch['ctg_class']

                ctg_logits, dcp_logits = model(input_ids, attention_mask, ctg_class)

                # 同样的方式过滤和计算损失
                dcp_mask = (dcp_logits != -100.0).any(dim=1)
                valid_dcp_logits = dcp_logits[dcp_mask]
                valid_dcp_labels = dcp_labels[dcp_mask]
                loss_ctg = criterion(ctg_logits, ctg_labels)
                loss_dcp = criterion(valid_dcp_logits, valid_dcp_labels) if len(valid_dcp_logits) > 0 else 0
                loss = loss_ctg + loss_dcp
                val_loss += loss.item()

                # 计算验证集准确率
                _, ctg_preds = torch.max(ctg_logits, dim=1)
                ctg_correct += torch.sum(ctg_preds == ctg_labels)
                if len(valid_dcp_logits) > 0:
                    _, dcp_preds = torch.max(valid_dcp_logits, dim=1)
                    dcp_correct += torch.sum(dcp_preds == valid_dcp_labels)
                total += ctg_labels.size(0)

        # 计算并打印当前epoch的验证集平均损失和准确率
        val_loss = val_loss / len(val_loader)
        val_ctg_acc = ctg_correct.double() / total
        val_dcp_acc = dcp_correct.double() / total if total > 0 else 0
        print(f'Val loss: {val_loss:.4f}, CTG Acc: {val_ctg_acc:.4f}, DCP Acc: {val_dcp_acc:.4f}')

        # 保存表现最好的模型
        # 以第二级分类的验证集准确率为标准
        if val_dcp_acc > best_accuracy:
            best_accuracy = val_dcp_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"找到更优模型，已保存到 best_model.pth (DCP Acc: {best_accuracy:.4f})")

    print(f'训练完成，最佳准确率: {best_accuracy:.4f}')
    return model


# --- 6. 主函数 (main) ---
# 程序的入口，负责组织整个流程
def main():
    # --- 超参数设置 ---
    BERT_MODEL_PATH = './bert-base-multilingual-cased'  # 本地预训练模型路径
    MAX_LEN = 256  # 文本最大长度
    BATCH_SIZE = 8  # 批次大小
    LEARNING_RATE = 2e-5  # 学习率
    NUM_EPOCHS = 5  # 训练轮次
    CONTINUE_TRAIN_MODEL_PATH = "best_model.pth"
    # --- 环境检查 ---
    # 检查本地模型路径是否存在
    if not os.path.exists(BERT_MODEL_PATH):
        print(f"错误: 模型路径 {BERT_MODEL_PATH} 不存在")
        return

    print("正在加载本地BERT模型的分词器...")
    try:
        tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH)
        print("分词器加载成功")
    except Exception as e:
        print(f"加载分词器失败: {e}")
        return

    # --- 数据路径定义 ---
    base_path = "cls"
    ctg_train_path = os.path.join(base_path, "cls_ctg", "train.tsv")
    dcp_train_path = os.path.join(base_path, "cls_dcp", "train.tsv")
    ctg_dev_path = os.path.join(base_path, "cls_ctg", "dev.tsv")
    dcp_dev_path = os.path.join(base_path, "cls_dcp", "dev.tsv")
    ctg_test_path = os.path.join(base_path, "cls_ctg", "test.csv")
    dcp_test_path = os.path.join(base_path, "cls_dcp", "test.csv")

    # 检查训练文件是否存在，保证程序能正常启动
    for path in [ctg_train_path, dcp_train_path]:
        if not os.path.exists(path):
            print(f"错误: 训练文件 {path} 不存在，请检查路径")
            return

    # --- 执行流程 ---
    try:
        # 1. 加载训练和验证数据
        print("正在加载训练数据...")
        train_dataset = MultiLevelTextDataset(ctg_train_path, dcp_train_path, tokenizer, MAX_LEN)

        if len(train_dataset) == 0:
            print("训练数据为空，程序终止")
            return

        print("\n--- 数据集信息概览 ---")
        print(f"训练数据大小: {len(train_dataset)}")
        print(f"第一级类别数量: {len(train_dataset.ctg_label2id)}")
        print(f"第一级类别: {list(train_dataset.ctg_label2id.keys())}")
        for ctg_label, mapping in train_dataset.dcp_label_mappings.items():
            print(f"  - 第一级类别 '{ctg_label}' 下有 {mapping['num_classes']} 个第二级类别")
        print("---------------------\n")

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

        print("正在加载验证数据...")
        val_dataset = MultiLevelTextDataset(ctg_dev_path, dcp_dev_path, tokenizer, MAX_LEN,
                                            is_train=False, train_dataset=train_dataset)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # 2. 初始化模型
        n_ctg_classes = len(train_dataset.ctg_label2id)
        model = MultiLevelBERT(n_ctg_classes, train_dataset.dcp_label_mappings, BERT_MODEL_PATH)
        model = model.to(device)  # 将模型移动到GPU或CPU
        if os.path.exists(CONTINUE_TRAIN_MODEL_PATH):
            print("\n" + "=" * 50)
            print(f"检测到预训练权重: {CONTINUE_TRAIN_MODEL_PATH}")
            print("正在加载权重以继续训练...")

            # 使用 map_location=device 确保权重能正确加载到当前设备 (CPU或GPU)
            # 使用 strict=False 来“宽容地”加载，忽略不匹配的层
            try:
                model.load_state_dict(
                    torch.load(CONTINUE_TRAIN_MODEL_PATH, map_location=device),
                    strict=False
                )
                print("权重加载成功！(strict=False 模式)")
            except Exception as e:
                print(f"加载权重失败: {e}。将从头开始训练。")

            print("=" * 50 + "\n")
        else:
            print(f"\n未找到预训练权重 {CONTINUE_TRAIN_MODEL_PATH}。")
            print("将从 BERT 基础模型开始训练。\n")
        # 3. 设置优化器和学习率调度器
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
        total_steps = len(train_loader) * NUM_EPOCHS
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,  # 预热步数
            num_training_steps=total_steps
        )

        # 4. 开始训练
        print("开始训练模型...")
        model = train_model(model, train_loader, val_loader, optimizer, scheduler, device, NUM_EPOCHS)
        print("训练完成！")

        # 5. 在测试集上评估最终模型
        if os.path.exists(ctg_test_path) and os.path.exists(dcp_test_path):
            print("\n" + "=" * 50)
            print("开始在测试集上进行最终评估...")
            print("=" * 50)

            test_dataset = MultiLevelTextDataset(ctg_test_path, dcp_test_path, tokenizer, MAX_LEN,
                                                 is_train=False, train_dataset=train_dataset)
            test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
            print(f"测试数据大小: {len(test_dataset)}")

            # 加载训练过程中保存的最佳模型
            if os.path.exists('best_model.pth'):
                model.load_state_dict(torch.load('best_model.pth'))
                print("已加载验证集上表现最佳的模型进行测试")

            test_ctg_acc, test_dcp_acc = evaluate_model(model, test_loader, device)

            print("\n" + "=" * 50)
            print("测试集最终结果:")
            print("=" * 50)
            print(f"第一级分类准确率 (CTG Acc): {test_ctg_acc:.4f}")
            print(f"第二级分类准确率 (DCP Acc): {test_dcp_acc:.4f}")
            print("=" * 50)

        else:
            print(f"\n测试集文件不存在，跳过测试评估。")
            print(f"期望的第一级测试文件路径: {ctg_test_path}")
            print(f"期望的第二级测试文件路径: {dcp_test_path}")

    except Exception as e:
        # 捕获所有可能的异常，并打印详细的错误信息，方便调试
        print(f"程序运行中发生错误: {e}")
        import traceback
        traceback.print_exc()


# 当该脚本被直接运行时，执行main()函数
if __name__ == '__main__':
    main()