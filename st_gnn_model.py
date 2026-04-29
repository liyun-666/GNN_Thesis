import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
import pandas as pd
from sqlalchemy import create_engine, text
import os
import time
from tqdm import tqdm
import numpy as np


# ==========================================
# 1. 论文核心：ST-GNN 模型架构
# ==========================================
class SpatioTemporalGNN(nn.Module):
    def __init__(self, num_users, num_items, embed_dim=256):
        super(SpatioTemporalGNN, self).__init__()
        # 用户和商品的向量表示
        self.user_embedding = nn.Embedding(num_users, embed_dim)
        self.item_embedding = nn.Embedding(num_items, embed_dim)

        # 三层图卷积网络 (GraphSAGE算子)
        self.conv1 = SAGEConv(embed_dim, embed_dim)
        self.conv2 = SAGEConv(embed_dim, embed_dim)
        self.conv3 = SAGEConv(embed_dim, embed_dim)

        # 行为强度预测层 (输出4类行为：点击/收藏/加购/购买)
        self.behavior_weight_layer = nn.Linear(embed_dim * 2, 4)

    def forward(self, edge_index, x_all):
        # 消息传递与特征聚合
        x = F.relu(self.conv1(x_all, edge_index))
        x = F.dropout(x, p=0.4, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        return x


# ==========================================
# 2. 训练引擎
# ==========================================
def train_heavy_model():
    print("==========================================")
    print("🚀 [ST-GNN] 工业级模型训练引擎启动...")
    print("==========================================")

    base_path = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(base_path, 'rec_system.db')

    if not os.path.exists(db_path):
        print(f"❌ 错误：在目录 {base_path} 下找不到 rec_system.db！")
        print("请先运行免密版 db_manager.py 生成数据库。")
        return

    # 1. 连接数据库 (适配 SQLAlchemy 2.0)
    engine = create_engine(f'sqlite:///{db_path}')

    try:
        with engine.connect() as conn:
            # 使用 text() 包装 SQL 语句解决 AttributeError
            query = text("SELECT * FROM user_behavior_logs")
            df = pd.read_sql(query, conn)
        print(f"📊 成功加载 {len(df)} 条真实交互记录。")
    except Exception as e:
        print(f"❌ 读取数据库失败: {e}")
        return

    # 2. 节点与图结构构建
    unique_users = df['u'].unique()
    unique_items = df['i'].unique()

    user_mapping = {uid: i for i, uid in enumerate(unique_users)}
    item_mapping = {iid: i + len(unique_users) for i, iid in enumerate(unique_items)}

    df['u_idx'] = df['u'].map(user_mapping)
    df['i_idx'] = df['i'].map(item_mapping)

    num_users = len(unique_users)
    num_items = len(unique_items)

    # 构建图连边张量
    edge_index = torch.tensor([df['u_idx'].values, df['i_idx'].values], dtype=torch.long)
    target_behaviors = torch.tensor(df['b'].values, dtype=torch.long)

    # 3. 模型、优化器、损失函数初始化
    model = SpatioTemporalGNN(num_users, num_items, embed_dim=256)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    print(f"🕸️ 图节点规模: 用户({num_users}) + 商品({num_items})")
    print(f"📦 模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")

    # 4. 训练循环
    epochs = 100
    user_indices = torch.arange(num_users)
    item_indices = torch.arange(num_items)

    print("\n⏳ 开始进行大规模矩阵运算，请保持电脑电源连接...")

    with tqdm(total=epochs, desc="🔄 训练进度", colour="blue") as pbar:
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()

            # 基础嵌入生成
            u_emb = model.user_embedding(user_indices)
            i_emb = model.item_embedding(item_indices)
            x_all = torch.cat([u_emb, i_emb], dim=0)

            # ST-GNN 前向传播
            out = model(edge_index, x_all)

            # 提取正样本对的特征
            u_feat = out[df['u_idx'].values]
            i_feat = out[df['i_idx'].values]

            # 行为类型预测
            logits = model.behavior_weight_layer(torch.cat([u_feat, i_feat], dim=1))

            loss = criterion(logits, target_behaviors)
            loss.backward()
            optimizer.step()

            # 每 20 轮输出一次核心指标，供论文截图使用
            if (epoch + 1) % 20 == 0:
                acc = (logits.argmax(1) == target_behaviors).float().mean()
                tqdm.write(f"📈 Epoch {epoch + 1:03d} | Loss: {loss.item():.4f} | Accuracy: {acc:.4f}")

            pbar.set_postfix({'Loss': f"{loss.item():.4f}"})
            pbar.update(1)

    # 5. 保存模型
    save_path = os.path.join(base_path, 'st_gnn_heavy_model.pth')
    torch.save(model.state_dict(), save_path)
    print(f"\n✅ 训练大功告成！模型权重已保存至: {save_path}")


if __name__ == "__main__":
    train_heavy_model()