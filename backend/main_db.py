import models  # noqa: F401 — registers all tables onto Base before init_db
from database import init_db, DATABASE_URL, engine
from sqlalchemy import text

print(f"--- 调试信息 ---")
print(f"当前使用的 URL: {DATABASE_URL}")

with engine.connect() as conn:
    db_name = conn.execute(text("SELECT current_database();")).scalar()
    print(f"数据库服务器报告当前数据库名: {db_name}")

print(f"--- 开始创建表 ---")
init_db()
print(f"--- 创建尝试完成 ---")
