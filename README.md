# ST-GNN 多行为序列推荐系统

本项目实现了一个基于时空图神经网络（ST-GNN）的多行为序列商品推荐系统，覆盖数据清洗、模型训练、推荐推理、在线交互更新、桌面端演示、移动端 PWA 演示和论文实验可视化等完整流程。

## 项目亮点

- 多行为建模：支持点击、收藏、加购、购买等行为序列建模。
- 时空图推荐：融合用户-商品图、商品转移图、行为语义和时间序列特征。
- 在线交互更新：用户提交新行为后，可实时刷新推荐结果。
- 双工作区桌面端：内置 Demo 数据演示，也支持加载自定义 SQLite 数据库训练。
- Inspector 检验工具：支持批量评分、专项商品诊断和论文结果导出。
- 多端演示：包含 Streamlit 演示、桌面应用、FastAPI 后端和移动端 PWA。

## 目录结构

```text
.
├── app.py                    # Streamlit 推荐系统演示
├── desktop_app_v2.py          # 桌面端主程序
├── api_server.py              # FastAPI 后端服务
├── recommender_engine.py      # ST-GNN 推荐核心逻辑
├── st_gnn_model.py            # 模型结构
├── train_stgnn.py             # 训练脚本
├── experiment_suite.py        # 实验、消融和敏感性分析
├── qa_tool.py                 # Inspector 检验工具
├── data_quality.py            # 数据质量检查与清洗
├── mobile_app_pwa/            # 移动端 Web/PWA 前端
├── artifacts/                 # 小型模型产物和实验结果
├── sample_data/               # 示例数据
└── docs/*.md                  # 项目、论文和部署说明文档
```

## 环境要求

- Python 3.10+
- Windows 10/11（桌面端脚本按 Windows 环境编写）
- 推荐使用虚拟环境安装依赖

安装依赖：

```bash
pip install -r requirements.txt
```

## 快速运行

### 1. Streamlit 演示

```bash
streamlit run app.py
```

### 2. 桌面端演示

Windows 下可双击：

```text
start_desktop_app.bat
```

或直接运行：

```bash
python desktop_app_v2.py
```

### 3. API + 移动端 PWA

启动后端：

```bash
uvicorn api_server:app --host 0.0.0.0 --port 8000 --app-dir .
```

访问移动端页面：

```text
http://127.0.0.1:8000/web/index.html
```

## 数据格式

推荐系统默认使用 SQLite 或 CSV 行为数据，核心字段如下：

| 字段 | 含义 |
| --- | --- |
| `u` | 用户 ID |
| `i` | 商品 ID |
| `b` | 行为类型，`0=click`, `1=favorite`, `2=cart`, `3=buy` |
| `t` | Unix 时间戳 |

自定义数据库需要包含上述字段，桌面端会进行格式校验。

## 模型与实验

训练模型：

```bash
python train_stgnn.py
```

运行实验：

```bash
python experiment_suite.py
```

生成论文图表：

```bash
python generate_thesis_figures.py
```

## GitHub 上传说明

仓库默认排除了以下内容，避免超过 GitHub 限制或泄露本地数据：

- 构建产物：`build/`, `dist/`, `installer_out/`, `*.exe`
- 超大原始数据：`UserBehavior.csv`
- 本地账号和临时数据库：`app_accounts.db`, `custom_*.db`
- Python 缓存、IDE 配置和运行日志

如需共享大文件，建议使用 GitHub Releases、网盘或 Git LFS。

## 相关文档

- `PROJECT_FUNCTION_SPEC.md`：项目功能与实现细节
- `APP_BUILD_GUIDE.md`：应用启动与打包说明
- `DEPLOY_APP_GUIDE.md`：部署说明
- `THESIS_ARCH_OUTLINE.md`：论文架构说明
- `FIGURE_INDEX.md`：图表索引
