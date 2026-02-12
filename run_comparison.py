import matplotlib
# 强制使用非交互式后端
matplotlib.use('Agg')

import os
import sys
import random
import networkx as nx
import time
from utils.plotting import plot_graph

try:
    from generate import simple_diagram
except ImportError:
    print("错误：找不到 generate.py。请确保脚本在项目根目录下。")
    sys.exit(1)

def create_baseline_case():
    """
    Baseline: 简单的5节点系统 (用于对比的基准)
    """
    G = nx.Graph()
    nodes = ['1', '2', '3', '4', '5']
    
    for i in nodes:
        G.add_node(i, 
                   x=random.uniform(0, 10),
                   y=random.uniform(0, 10),
                   type='substation' if i == '1' else 'load',
                   name=f"Bus-{i}"
        )
    
    # 简单的链式结构
    edges = [('1', '2'), ('2', '3'), ('3', '4'), ('4', '5')]
    G.add_edges_from(edges)
    
    return G

def create_case1():
    """
    Case 1: 中等规模系统 (10节点，有环状结构)
    """
    G = nx.Graph()
    nodes = [str(i) for i in range(1, 11)]  # '1'-'10'
    
    for i in nodes:
        G.add_node(i, 
                   x=random.uniform(0, 15),
                   y=random.uniform(0, 15),
                   type='substation' if int(i) <= 2 else 'load',
                   name=f"Bus-{i}"
        )
    
    # 创建环状连接和一些交叉连接
    edges = [
        ('1', '2'), ('2', '3'), ('3', '4'), ('4', '5'), ('5', '1'),  # 环1
        ('5', '6'), ('6', '7'), ('7', '8'), ('8', '9'), ('9', '5'),  # 环2
        ('3', '8'), ('4', '9'), ('2', '10'), ('7', '10')             # 交叉连接
    ]
    G.add_edges_from(edges)
    
    return G

def create_case2():
    """
    Case 2: 中等复杂系统 (20节点，多环结构)
    """
    G = nx.Graph()
    
    # 节点数据: ID, 名称, 坐标, 电压等级
    nodes_data = {
        '1': ("Center", 506, 500, 500),
        '2': ("Inner1", 697, 526, 220),
        '3': ("Inner2", 421, 666, 220),
        '4': ("Inner3", 354, 534, 220),
        '5': ("Inner4", 196, 475, 220),
        '6': ("Inner5", 446, 450, 220),
        '7': ("Outer1a", 550, 700, 220),
        '8': ("Outer1b", 460, 620, 220),
        '9': ("Outer2a", 495, 861, 220),
        '10': ("Outer2b", 365, 855, 220),
        '11': ("Outer3a", 381, 395, 220),
        '12': ("Outer3b", 236, 431, 220),
        '13': ("A1", 592, 267, 220),
        '14': ("A2", 622, 355, 220),
        '15': ("A3", 568, 212, 220),
        '16': ("A4", 447, 330, 220),
        '17': ("B1", 660, 475, 220),
        '18': ("B2", 775, 483, 220),
        '19': ("B3", 796, 539, 220),
        '20': ("B4", 639, 540, 220)
    }
    
    # 添加节点（y坐标取负，使图片下方对应更大的原始y值）
    for node_id, (name, x, y, voltage) in nodes_data.items():
        G.add_node(node_id,
                   x=x,  # 不缩放坐标，保持原始值
                   y=y,  # y取负，翻转坐标系
                   type='substation' if voltage == 500 else 'load',
                   name=name,
                   voltage=voltage)
        print(f"添加节点 {node_id}: {name} ({x}, {y}), 电压等级: {voltage}")
    
    # 连接关系
    edges = [
        ('1', '2'), ('1', '3'), ('1', '4'), ('1', '5'), ('1', '6'),
        ('2', '7'), ('7', '8'), ('1', '8'), ('8', '9'), ('9', '10'), ('10', '3'),
        ('4', '11'), ('11', '12'), ('12', '5'),
        ('1', '16'), ('16', '15'), ('15', '13'), ('13', '14'), ('14', '1'),
        ('17', '18'), ('18', '19'), ('19', '20'), ('20', '1'), ('1', '17')
    ]
    
    G.add_edges_from(edges)
    
    return G

def create_case3():
    """
    Case 3: 真实深圳电力系统 (16节点，实际拓扑数据)
    """
    G = nx.Graph()
    
    # 真实节点数据: ID, 名称, 坐标, 电压等级
    nodes_data = {
        '1': ("现代", 700, 832, 500),
        '2': ("白石洲", 553, 987, 220),
        '3': ("深圳湾", 605, 1012, 220),
        '4': ("翡翠", 577, 909, 220),
        '5': ("祥和", 658, 937, 220),
        '6': ("梅林B", 773, 887, 220),
        '7': ("庙西", 758, 1001, 220),
        '8': ("皇岗", 817, 1048, 220),
        '9': ("滨河", 818, 985, 220),
        '10': ("福慧", 878, 976, 220),
        '11': ("梨园", 948, 847, 220),
        '12': ("湖贝", 1049, 928, 220),
        '13': ("清水河", 984, 841, 220),
        '14': ("上步", 949, 896, 220),
        '15': ("中航", 929, 935, 220),
        '16': ("龙塘B", 753, 740, 220)
    }
    
    # 添加节点（y坐标取负，使图片下方对应更大的原始y值）
    for node_id, (name, x, y, voltage) in nodes_data.items():
        print(f"添加节点 {node_id}: {name} ({x}, {-y}), 电压等级: {voltage}")
        G.add_node(node_id,
                   x=x,  # 不缩放坐标，保持原始值
                   y=-y,  # y取负，翻转坐标系
                   type='substation' if voltage == 500 else 'load',
                   name=name,
                   voltage=voltage)
    
    # 真实连接关系
    edges = [
        ('1', '2'), ('2', '3'), ('3', '4'), ('4', '5'), ('5', '6'),
        ('6', '1'), ('4', '1'), ('7', '1'), ('7', '8'), ('8', '9'),
        ('9', '10'), ('10', '1'), ('1', '11'), ('11', '12'), ('12', '13'),
        ('13', '14'), ('14', '1'), ('14', '15'), ('16', '1')
    ]
    
    G.add_edges_from(edges)
    
    return G

def run_comparison_experiment():
    """
    运行对比实验，生成baseline和三个case的结果
    """
    # 设置随机种子以确保可重复性
    random.seed(42)
    
    cases = {
        'baseline': (create_baseline_case, 'Baseline_5_Nodes_Simple'),
        'case1': (create_case1, 'Case1_10_Nodes_Medium'),
        'case2': (create_case2, 'Case2_20_Nodes_MultiRing'),
        'case3': (create_case3, 'Case3_Shenzhen_Real_16_Nodes')
    }
    
    results = {}
    
    for case_id, (create_func, case_name) in cases.items():
        print("\n" + "="*60)
        print(f"运行 {case_name}")
        print("="*60)
        
        # 创建输出目录
        output_dir = f'output_comparison/{case_id}'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 生成图数据
        graph = create_func()
        print(f"图数据构造完成: 节点数={graph.number_of_nodes()}, 边数={graph.number_of_edges()}")
        

        # 运行优化
        try:
            start_time = time.time()
            simple_diagram(
                graph=graph,
                graph_name=case_name,
                max_layout_retry=10,
                save_diagram=True,
                save_dir=output_dir,
                seed=42
            )
            total_time = time.time() - start_time
            
            results[case_id] = {
                'name': case_name,
                'nodes': graph.number_of_nodes(),
                'edges': graph.number_of_edges(),
                'time': total_time,
                'status': 'success'
            }
            
            print(f"✓ {case_name} 完成！总时间: {total_time:.2f}s")
            print(f"  结果保存在: {os.path.abspath(output_dir)}")
            
        except Exception as e:
            print(f"✗ {case_name} 运行失败！")
            print(f"  错误信息: {e}")
            import traceback
            traceback.print_exc()
            
            results[case_id] = {
                'name': case_name,
                'nodes': graph.number_of_nodes(),
                'edges': graph.number_of_edges(),
                'time': 0,
                'status': 'failed',
                'error': str(e)
            }
    
    # 打印总结报告
    print("\n" + "="*60)
    print("对比实验总结")
    print("="*60)
    print(f"{'Case':<15} {'节点数':<8} {'边数':<8} {'时间(s)':<10} {'状态':<10}")
    print("-"*60)
    
    for case_id in ['case3']:
        r = results[case_id]
        print(f"{case_id:<15} {r['nodes']:<8} {r['edges']:<8} {r['time']:<10.2f} {r['status']:<10}")
    
    print("="*60)
    print(f"\n所有结果保存在: {os.path.abspath('output_comparison')}")
    print("每个case包含: raw.pdf (原始), cross.pdf (交叉优化), layout.pdf (最终布局)")
    
    return results

def main():
    print("开始运行论文对比实验...")
    print("将生成: Baseline + 3个对比Case")
    print()
    
    results = run_comparison_experiment()
    
    print("\n实验完成！")
    print("你现在可以使用这些图表进行论文对比分析。")

if __name__ == "__main__":
    main()
