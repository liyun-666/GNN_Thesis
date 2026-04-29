import pandas as pd
import os

# 🚀 自动获取当前脚本所在的文件夹，不管你文件夹叫什么名字都能找对！
base_path = os.path.dirname(os.path.abspath(__file__))
input_file = os.path.join(base_path, 'UserBehavior.csv')
output_file = os.path.join(base_path, 'final_real_data.csv')

print(f"当前脚本运行目录: {base_path}")
print(f"正在寻找目标文件: {input_file}")

if not os.path.exists(input_file):
    print(f"❌ 依旧没找到！请确认 UserBehavior.csv 确实在这个文件夹里: {base_path}")
    # 打印当前文件夹下所有的文件，帮你看看文件到底叫啥
    print(f"当前文件夹内的文件列表: {os.listdir(base_path)}")
else:
    print(f"🎯 找到了！开始提取 10 万条精华数据...")
    try:
        # 读取前 100,000 条
        df = pd.read_csv(input_file, header=None, names=['u', 'i', 'c', 'b', 't'], nrows=100000)

        # 映射行为
        mapping = {'pv': 0, 'fav': 1, 'cart': 2, 'buy': 3}
        df['b'] = df['b'].map(mapping)

        # 保存精华版
        df[['u', 'i', 'b', 't']].to_csv(output_file, index=False)
        print(f"✅ 成功！‘精华版’数据已保存至: {output_file}")
    except Exception as e:
        print(f"❌ 提取过程中出错: {e}")