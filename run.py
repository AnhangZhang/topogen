import matplotlib
# 强制使用非交互式后端（必须在导入 pyplot 或 generate 之前设置）
matplotlib.use('Agg')

import os
import sys
import random
import networkx as nx

# 尝试导入项目中的主函数
try:
    from generate import simple_diagram
except ImportError:
    print("错误：找不到 generate.py。请确保 run.py 在项目根目录下。")
    sys.exit(1)

def create_fake_data():
    """
    创建一个简单的 5 节点电力系统拓扑，并赋予初始随机坐标
    """
    G = nx.Graph()

    # --- 1. 添加节点 (必须带 x, y 坐标) ---
    # 我们生成 5 个节点
    nodes = [1, 2, 3, 4, 5]
    
    for i in nodes:
        # 核心修复：给每个节点赋予一个随机的初始坐标
        # 同时也加上 type 和 name，防止后面画图用到这些属性报错
        G.add_node(i, 
                   x=random.uniform(0, 10),  # 初始 X 坐标
                   y=random.uniform(0, 10),  # 初始 Y 坐标
                   type='substation' if i == 1 else 'load', # 假装第一个是变电站
                   name=f"Bus-{i}"
        )

    # --- 2. 添加边 (连接关系) ---
    edges = [
        (1, 2),
        (2, 3),
        (3, 1), # 这是一个环
        (3, 4),
        (4, 5)
    ]
    G.add_edges_from(edges)
    
    return G

def main():
    # 1. 准备输出目录
    output_dir = 'output_demo'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 2. 造数据
    print("正在构造带坐标的测试数据...")
    fake_graph = create_fake_data()
    
    case_name = "Demo_Case_5_Bus"

    print(f"图数据构造完成: {fake_graph}")
    print(f"节点1的属性示例: {fake_graph.nodes[1]}") # 打印出来确认一下有没有 x, y
    print("-" * 30)

    # 3. 调用 generate.py 中的核心流程
    print("开始运行布局优化 (simple_diagram)...")
    try:
        simple_diagram(
            graph=fake_graph, 
            graph_name=case_name,
            max_layout_retry=3, 
            save_diagram=True,
            save_dir=output_dir
        )
        print("-" * 30)
        print(f"运行成功！结果已保存到: {os.path.abspath(output_dir)}")
        print("请查看目录下的 raw.pdf, cross.pdf, layout.pdf")
        
    except Exception as e:
        print("\n运行依然报错！")
        print("详细错误信息:", e)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()