import pandas as pd
from sqlalchemy import create_engine
import os


def init_db():
    print("🔗 正在连接免密数据库 (SQLite)...")
    base_path = os.path.dirname(os.path.abspath(__file__))
    csv_file = os.path.join(base_path, 'final_real_data.csv')

    # 建立一个名为 rec_system.db 的本地文件数据库
    db_path = os.path.join(base_path, 'rec_system.db')
    engine = create_engine(f'sqlite:///{db_path}')

    try:
        df = pd.read_csv(csv_file)
        df.to_sql(name='user_behavior_logs', con=engine, if_exists='replace', index=False)
        print(f"🎉 大功告成！数据已存入本地数据库文件: {db_path}")
    except Exception as e:
        print(f"❌ 还是失败了: {e}")


if __name__ == "__main__":
    init_db()