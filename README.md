# MockMate — 智能面试伙伴

基于 RAG（检索增强生成）技术的智能面试模拟系统。上传你的知识库文档，与 AI 进行真实感面试对话，支持混合向量检索、多轮对话和完整的检索过程追踪。

---

## 功能特性

- **知识库驱动对话** — 上传 PDF、Word、Excel 文档，AI 基于文档内容回答问题
- **混合向量检索** — 密集向量（BGE-M3）+ BM25 稀疏向量，兼顾语义和关键词匹配
- **三级分块策略** — 细粒度检索 + 粗粒度上下文，自动合并相关分块
- **智能查询重写** — Step-back、HyDE、复杂查询多种策略，提升检索召回率
- **RAG 追踪可视化** — 前端实时展示检索过程、评分和来源文档
- **流式响应** — 实时流式输出，对话体验流畅
- **多用户支持** — JWT 认证 + 基于角色的权限控制（用户/管理员）
- **异步文档处理** — 后台任务处理大文件上传和向量化，不阻塞主流程

---

## 技术栈

| 层级 | 技术 |
|------|------|
| 后端框架 | FastAPI + Uvicorn |
| AI 框架 | LangChain + LangGraph |
| 向量模型 | BAAI/bge-m3（本地部署） |
| 向量数据库 | Milvus 2.5 |
| 关系数据库 | PostgreSQL 15 |
| 缓存 | Redis 7 |
| 容器化 | Docker Compose |

---

## 系统架构

```
用户请求
   │
   ▼
FastAPI 后端
   │
   ├── 认证层（JWT）
   │
   ├── LangChain Agent
   │      │
   │      └── LangGraph RAG 管道
   │             ├── 混合检索（Milvus）
   │             ├── 相关性评分
   │             ├── 查询重写（Step-back / HyDE）
   │             ├── Auto-merging
   │             └── 可选 Reranking
   │
   ├── 文档处理
   │      ├── 三级分块（L1/L2/L3）
   │      ├── 向量化（BGE-M3 密集 + BM25 稀疏）
   │      └── 写入 Milvus + PostgreSQL
   │
   └── 会话管理（Redis 缓存 + PostgreSQL 持久化）
```

---

## 快速开始

### 前置依赖

- Python 3.11+
- Docker & Docker Compose
- [uv](https://github.com/astral-sh/uv)（Python 包管理器）

### 1. 克隆项目

```bash
git clone https://github.com/NingXiao-114/MockMate.git
cd MyRagBot
```

### 2. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env`，填入必要配置：

```env
# LLM（兼容 OpenAI 格式的 API，如火山引擎 Ark）
ARK_API_KEY=your_api_key
MODEL=your_model_name
BASE_URL=https://ark.cn-beijing.volces.com/api/v3

# 评分模型（可与主模型相同）
GRADE_API_KEY=your_api_key
GRADE_MODEL=your_model_name
GRADE_BASE_URL=https://ark.cn-beijing.volces.com/api/v3

# 向量化（本地模型，首次运行自动下载）
EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_DEVICE=cpu          # 有 GPU 改为 cuda

# 认证
JWT_SECRET_KEY=change-this-to-a-random-secret
ADMIN_INVITE_CODE=your_invite_code
```

### 3. 启动基础服务

```bash
# 启动 PostgreSQL + Redis
docker-compose up -d

# 启动 Milvus
docker-compose -f milvus-docker-compose.yml up -d
```

### 4. 安装依赖并启动后端

```bash
cd backend
uv sync
uv run python app.py
```

后端默认运行在 `http://localhost:8000`

### 5. 访问前端

直接用浏览器打开 `frontend/index.html`，或通过任意静态文件服务器托管。

---

## 使用流程

1. **注册账号** — 普通用户直接注册；管理员注册需要邀请码
2. **上传文档**（管理员）— 支持 PDF、Word、Excel，上传后自动向量化入库
3. **开始对话** — 创建新会话，向 AI 提问，AI 基于知识库回答
4. **查看 RAG 追踪** — 点击消息旁的追踪按钮，查看检索来源和评分详情

---

## API 文档

启动后端后访问 `http://localhost:8000/docs` 查看完整的 Swagger 文档。

主要端点：

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/auth/register` | 用户注册 |
| POST | `/auth/login` | 用户登录 |
| POST | `/chat/stream` | 流式对话 |
| GET | `/sessions` | 获取会话列表 |
| POST | `/documents/upload/async` | 异步上传文档 |
| DELETE | `/documents/{filename}` | 删除文档 |

---

## 项目结构

```
MyRagBot/
├── backend/
│   ├── app.py                # 应用入口
│   ├── api.py                # 路由定义
│   ├── agent.py              # LangChain Agent
│   ├── rag_pipeline.py       # LangGraph RAG 工作流
│   ├── embedding.py          # 向量化服务
│   ├── milvus_client.py      # Milvus 客户端
│   ├── document_loader.py    # 文档加载与分块
│   ├── models.py             # 数据库模型
│   └── auth.py               # 认证逻辑
├── frontend/
│   ├── index.html
│   ├── script.js             # Vue 3 应用
│   └── style.css
├── docker-compose.yml
├── milvus-docker-compose.yml
└── .env.example
```

---

## 配置说明

### GPU 加速

将 `EMBEDDING_DEVICE=cpu` 改为 `EMBEDDING_DEVICE=cuda`，向量化速度可提升数倍。

---

## License

MIT
