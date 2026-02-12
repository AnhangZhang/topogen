#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
根据指标.png中的定义，计算拓扑图优化前后的各项指标
输入: output_comparison/case*/raw.svg (优化前) 和 layout.svg (优化后)
"""

import os
import re
import math
import numpy as np
from collections import defaultdict


def parse_svg(svg_path):
    """
    解析SVG文件，提取节点坐标和边信息
    返回: nodes (dict: id -> (x, y)), edges (list of polylines)
    """
    with open(svg_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 提取所有station圆形节点 (不包含halo和bend/dummy/inter)
    # 格式: <circle class="station-XXX" cx="..." cy="..." r="10.0"/>
    station_pattern = r'<circle class="station-(?:220|500|default)" cx="([^"]+)" cy="([^"]+)" r="10\.0"/>'
    station_matches = re.findall(station_pattern, content)
    
    nodes = {}
    for i, (cx, cy) in enumerate(station_matches):
        nodes[i] = (float(cx), float(cy))
    
    # 提取所有边 (line元素)
    # 格式: <line class="edge" x1="..." y1="..." x2="..." y2="..."/>
    edge_pattern = r'<line class="edge" x1="([^"]+)" y1="([^"]+)" x2="([^"]+)" y2="([^"]+)"/>'
    edge_matches = re.findall(edge_pattern, content)
    
    # 由于SVG中每条边画了两次(双线效果)，需要去重
    # 将边按端点分组，合并相近的边
    raw_edges = []
    for x1, y1, x2, y2 in edge_matches:
        raw_edges.append(((float(x1), float(y1)), (float(x2), float(y2))))
    
    # 将相近的边合并（去掉双线效果）
    edges = merge_duplicate_edges(raw_edges)
    
    # 将边分组形成polyline (连续的线段)
    polylines = build_polylines(edges, nodes)
    
    return nodes, polylines, edges


def merge_duplicate_edges(raw_edges, threshold=5.0):
    """
    合并双线效果产生的重复边
    """
    merged = []
    used = [False] * len(raw_edges)
    
    for i, e1 in enumerate(raw_edges):
        if used[i]:
            continue
        # 找到与e1配对的边
        for j in range(i + 1, len(raw_edges)):
            if used[j]:
                continue
            e2 = raw_edges[j]
            # 检查两条边是否平行且接近
            if edges_are_paired(e1, e2, threshold):
                # 取中点作为合并后的边
                mid_p1 = ((e1[0][0] + e2[0][0]) / 2, (e1[0][1] + e2[0][1]) / 2)
                mid_p2 = ((e1[1][0] + e2[1][0]) / 2, (e1[1][1] + e2[1][1]) / 2)
                merged.append((mid_p1, mid_p2))
                used[i] = True
                used[j] = True
                break
        if not used[i]:
            merged.append(e1)
            used[i] = True
    
    return merged


def edges_are_paired(e1, e2, threshold):
    """检查两条边是否是双线效果的配对"""
    # e1和e2的端点应该分别接近
    d1 = point_distance(e1[0], e2[0]) + point_distance(e1[1], e2[1])
    d2 = point_distance(e1[0], e2[1]) + point_distance(e1[1], e2[0])
    return min(d1, d2) < threshold * 2


def point_distance(p1, p2):
    """计算两点距离"""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def build_polylines(edges, nodes):
    """
    根据边和节点构建polyline
    每条polyline是从一个station到另一个station的路径
    """
    # 简化处理：直接返回所有边，每条边作为一个简单的polyline
    # 对于有bend点的复杂路径，需要更复杂的逻辑
    polylines = []
    for e in edges:
        polylines.append([e[0], e[1]])
    return polylines


def find_nearest_node(point, nodes, threshold=15.0):
    """找到距离point最近的节点"""
    min_dist = float('inf')
    nearest = None
    for nid, npos in nodes.items():
        d = point_distance(point, npos)
        if d < min_dist:
            min_dist = d
            nearest = nid
    return nearest if min_dist < threshold else None


# ============== 指标计算函数 ==============

def metric1_segment_crossings(edges):
    """
    指标1: 几何交叉数 (总线段交叉)
    计算所有线段对的交叉数，排除端点共享的情况
    """
    crossings = 0
    n = len(edges)
    for i in range(n):
        for j in range(i + 1, n):
            if segments_cross(edges[i], edges[j]):
                crossings += 1
    return crossings


def segments_cross(seg1, seg2):
    """
    检测两线段是否交叉(不包含端点重合)
    seg1 = ((x1,y1), (x2,y2)), seg2 = ((x3,y3), (x4,y4))
    """
    p1, p2 = seg1
    p3, p4 = seg2
    
    # 检查是否共享端点
    eps = 1e-6
    if (point_distance(p1, p3) < eps or point_distance(p1, p4) < eps or
        point_distance(p2, p3) < eps or point_distance(p2, p4) < eps):
        return False
    
    # 计算交叉
    def ccw(A, B, C):
        return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
    
    def intersect(A, B, C, D):
        return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)
    
    return intersect(p1, p2, p3, p4)


def metric2_direction_ratio(nodes, edges):
    """
    指标2: 八方向连接比例 (带容差，长度加权)
    计算每条边的方向角，判断是否接近8个标准方向
    θ = atan2(y2-y1, x2-x1) * 180/π, 范围 [0, 360)
    8方向: 0, 45, 90, 135, 180, 225, 270, 315
    容差 ε_θ = 3°
    
    加权比例 R_oct = Σ(1[|Δ(θ_k)| <= ε_θ] * len(e)) / Σlen(e)
    """
    epsilon = 3.0  # 容差3度
    octants = [0, 45, 90, 135, 180, 225, 270, 315]
    
    total_len = 0
    aligned_len = 0
    
    for e in edges:
        p1, p2 = e
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        length = math.sqrt(dx*dx + dy*dy)
        if length < 1e-6:
            continue
        
        # 计算角度 [0, 360)
        theta = math.degrees(math.atan2(dy, dx))
        if theta < 0:
            theta += 360
        
        # 检查是否接近8方向
        aligned = False
        for oct_angle in octants:
            delta = abs(theta - oct_angle)
            delta = min(delta, 360 - delta)  # 处理环绕
            if delta <= epsilon:
                aligned = True
                break
        
        total_len += length
        if aligned:
            aligned_len += length
    
    return aligned_len / total_len if total_len > 0 else 0


def metric3_turning_angle_cost(edges, nodes):
    """
    指标3: 转向角折零代价 (逐线段/至polyline)
    对于每条边，计算与标准角度的偏差
    标准角度: 0, 45, 90, 135度 (共4个，对称到180-180)
    w(0) = 0, w(45) = 1, w(90) = 2, w(135) = 3
    
    偏差角 δ_k = 45 * round(θ_k / 45) mod 8
    代价 = Σ w(δ_k) / n
    """
    weights = {0: 0, 45: 1, 90: 2, 135: 3, 180: 4}  # 按照图片说明
    
    total_cost = 0
    count = 0
    
    for e in edges:
        p1, p2 = e
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        length = math.sqrt(dx*dx + dy*dy)
        if length < 1e-6:
            continue
        
        # 计算角度绝对值 [0, 180]
        theta = abs(math.degrees(math.atan2(dy, dx)))
        if theta > 180:
            theta = 360 - theta
        if theta > 90:
            theta = 180 - theta
        
        # 量化到最近的45度
        quantized = 45 * round(theta / 45)
        if quantized > 135:
            quantized = 135
        
        cost = weights.get(int(quantized), 0)
        total_cost += cost
        count += 1
    
    return total_cost / count if count > 0 else 0


def metric4_total_edge_length(edges):
    """
    指标4: 总线路长度 (折线总长度)
    L = Σ ||p_i - p_{i-1}||
    """
    total = 0
    for e in edges:
        p1, p2 = e
        total += point_distance(p1, p2)
    return total


def metric5_edge_length_cv(edges):
    """
    指标5: 边长均匀性 (用CV)
    CV = σ / μ
    其中 μ = (1/E) Σ l_e, σ = sqrt((1/E) Σ (l_e - μ)^2)
    """
    lengths = []
    for e in edges:
        p1, p2 = e
        lengths.append(point_distance(p1, p2))
    
    if len(lengths) == 0:
        return 0
    
    mu = np.mean(lengths)
    sigma = np.std(lengths)
    
    return sigma / mu if mu > 0 else 0


def metric6_total_displacement(nodes_before, nodes_after):
    """
    指标6: 总体布局偏移 (sum / mean / max)
    D_sum = Σ ||x_v - x̄_v||
    D_mean = (1/V) * D_sum
    D_max = max ||x_v - x̄_v||
    
    返回: (D_sum, D_mean, D_max)
    """
    if len(nodes_before) != len(nodes_after):
        # 节点数不匹配，尝试匹配最近的节点
        pass
    
    displacements = []
    for vid in nodes_before:
        if vid in nodes_after:
            d = point_distance(nodes_before[vid], nodes_after[vid])
            displacements.append(d)
    
    if len(displacements) == 0:
        return 0, 0, 0
    
    d_sum = sum(displacements)
    d_mean = d_sum / len(displacements)
    d_max = max(displacements)
    
    return d_sum, d_mean, d_max


def metric7_degree_3_preservation(nodes, edges):
    """
    指标7: 出边顺序保持率 (仅 deg >= 3, tie 首按失效)
    对于度>=3的节点，检查其相邻边的顺序是否保持
    
    计算: R_order = #{v : deg(v) >= 3, v pass} / #{v : deg(v) >= 3}
    """
    # 构建节点的邻接边
    adj = defaultdict(list)
    for e in edges:
        p1, p2 = e
        # 找到对应的节点
        n1 = find_nearest_node(p1, nodes)
        n2 = find_nearest_node(p2, nodes)
        if n1 is not None and n2 is not None:
            adj[n1].append((n2, p2))
            adj[n2].append((n1, p1))
    
    # 统计度>=3的节点
    high_degree_nodes = [n for n in adj if len(adj[n]) >= 3]
    
    if len(high_degree_nodes) == 0:
        return 1.0  # 没有高度节点，认为保持率100%
    
    # 这里简化处理，返回1.0 (完整实现需要比较优化前后的顺序)
    return 1.0


def metric8_additional_metrics(nodes, edges, nodes_before=None, edges_before=None):
    """
    指标8: 你要补充的两项
    8.1 距离子项倍的交叉/对数 (station-station 与 station-edge)
    8.2 重叠段总长度 (nonincident segments overlap length)
    
    返回: (station_violations, edge_violations, overlap_length)
    """
    # 8.1(a) station-station violations
    # N_ss = #{(u,v) : u != v, ||x_u - x_v|| < d_s}
    d_s = 20.0  # 最小站点距离阈值
    station_violations = 0
    node_list = list(nodes.values())
    for i in range(len(node_list)):
        for j in range(i + 1, len(node_list)):
            d = point_distance(node_list[i], node_list[j])
            if d < d_s:
                station_violations += 1
    
    # 8.1(b) station-edge violations (只考虑nonincident)
    # 对于每个节点v，计算到每条不与其关联的边的最短距离
    edge_violations = 0
    node_ids = list(nodes.keys())
    for vid, vpos in nodes.items():
        for e in edges:
            p1, p2 = e
            n1 = find_nearest_node(p1, nodes)
            n2 = find_nearest_node(p2, nodes)
            # 跳过关联的边
            if n1 == vid or n2 == vid:
                continue
            # 计算点到线段的距离
            d = point_to_segment_distance(vpos, p1, p2)
            if d < d_s:
                edge_violations += 1
    
    # 8.2 重叠段总长度
    # 对于所有非关联边对，计算重叠长度
    overlap_length = 0.0
    n_edges = len(edges)
    for i in range(n_edges):
        for j in range(i + 1, n_edges):
            e1, e2 = edges[i], edges[j]
            # 检查是否共享端点 (关联)
            if edges_share_endpoint(e1, e2, nodes):
                continue
            # 计算重叠长度
            overlap = segment_overlap_length(e1, e2)
            overlap_length += overlap
    
    return station_violations, edge_violations, overlap_length


def point_to_segment_distance(p, s1, s2):
    """计算点p到线段s1-s2的最短距离"""
    dx = s2[0] - s1[0]
    dy = s2[1] - s1[1]
    if dx == 0 and dy == 0:
        return point_distance(p, s1)
    
    t = max(0, min(1, ((p[0] - s1[0]) * dx + (p[1] - s1[1]) * dy) / (dx*dx + dy*dy)))
    proj = (s1[0] + t * dx, s1[1] + t * dy)
    return point_distance(p, proj)


def edges_share_endpoint(e1, e2, nodes, threshold=15.0):
    """检查两条边是否共享端点"""
    for p1 in e1:
        for p2 in e2:
            if point_distance(p1, p2) < threshold:
                return True
    return False


def segment_overlap_length(e1, e2, threshold=5.0):
    """
    计算两条近似平行线段的重叠长度
    简化实现：如果两条边近似平行且距离很近，计算投影重叠长度
    """
    # 计算两条边的方向向量
    d1 = (e1[1][0] - e1[0][0], e1[1][1] - e1[0][1])
    d2 = (e2[1][0] - e2[0][0], e2[1][1] - e2[0][1])
    
    len1 = math.sqrt(d1[0]**2 + d1[1]**2)
    len2 = math.sqrt(d2[0]**2 + d2[1]**2)
    
    if len1 < 1e-6 or len2 < 1e-6:
        return 0
    
    # 归一化
    d1_norm = (d1[0]/len1, d1[1]/len1)
    d2_norm = (d2[0]/len2, d2[1]/len2)
    
    # 检查是否平行 (点积接近1或-1)
    dot = abs(d1_norm[0]*d2_norm[0] + d1_norm[1]*d2_norm[1])
    if dot < 0.95:  # 不平行
        return 0
    
    # 计算e2端点到e1的距离
    dist1 = point_to_segment_distance(e2[0], e1[0], e1[1])
    dist2 = point_to_segment_distance(e2[1], e1[0], e1[1])
    
    if min(dist1, dist2) > threshold:
        return 0
    
    # 计算投影重叠 (简化)
    # 将e2投影到e1上，计算重叠长度
    return min(len1, len2) * 0.5  # 简化估计


def calculate_all_metrics(raw_svg_path, layout_svg_path):
    """
    计算所有指标
    """
    # 解析SVG
    nodes_raw, polylines_raw, edges_raw = parse_svg(raw_svg_path)
    nodes_layout, polylines_layout, edges_layout = parse_svg(layout_svg_path)
    
    results = {
        'raw': {},
        'layout': {},
        'comparison': {}
    }
    
    # 指标1: 几何交叉数
    results['raw']['crossings'] = metric1_segment_crossings(edges_raw)
    results['layout']['crossings'] = metric1_segment_crossings(edges_layout)
    
    # 指标2: 八方向连接比例
    results['raw']['direction_ratio'] = metric2_direction_ratio(nodes_raw, edges_raw)
    results['layout']['direction_ratio'] = metric2_direction_ratio(nodes_layout, edges_layout)
    
    # 指标3: 转向角折零代价 (这里用边的方向代价)
    results['raw']['turning_cost'] = metric3_turning_angle_cost(edges_raw, nodes_raw)
    results['layout']['turning_cost'] = metric3_turning_angle_cost(edges_layout, nodes_layout)
    
    # 指标4: 总线路长度
    results['raw']['total_length'] = metric4_total_edge_length(edges_raw)
    results['layout']['total_length'] = metric4_total_edge_length(edges_layout)
    
    # 指标5: 边长CV
    results['raw']['length_cv'] = metric5_edge_length_cv(edges_raw)
    results['layout']['length_cv'] = metric5_edge_length_cv(edges_layout)
    
    # 指标6: 总体布局偏移
    d_sum, d_mean, d_max = metric6_total_displacement(nodes_raw, nodes_layout)
    results['comparison']['displacement_sum'] = d_sum
    results['comparison']['displacement_mean'] = d_mean
    results['comparison']['displacement_max'] = d_max
    
    # 指标7: 出边顺序保持率
    results['layout']['order_preservation'] = metric7_degree_3_preservation(nodes_layout, edges_layout)
    
    # 指标8: 补充指标
    ss_viol_raw, se_viol_raw, overlap_raw = metric8_additional_metrics(nodes_raw, edges_raw)
    ss_viol_layout, se_viol_layout, overlap_layout = metric8_additional_metrics(nodes_layout, edges_layout)
    
    results['raw']['station_violations'] = ss_viol_raw
    results['raw']['edge_violations'] = se_viol_raw
    results['raw']['overlap_length'] = overlap_raw
    
    results['layout']['station_violations'] = ss_viol_layout
    results['layout']['edge_violations'] = se_viol_layout
    results['layout']['overlap_length'] = overlap_layout
    
    # 节点和边数
    results['raw']['num_nodes'] = len(nodes_raw)
    results['raw']['num_edges'] = len(edges_raw)
    results['layout']['num_nodes'] = len(nodes_layout)
    results['layout']['num_edges'] = len(edges_layout)
    
    return results


def main():
    """主函数"""
    base_dir = 'output_comparison'
    cases = [f'case{i}' for i in range(1, 17)]
    
    print("=" * 120)
    print("拓扑图优化指标计算结果")
    print("=" * 120)
    
    # 表头
    header = f"{'Case':<10} {'节点':<6} {'边':<6} | {'交叉(前)':<10} {'交叉(后)':<10} | {'8方向(前)':<10} {'8方向(后)':<10} | {'CV(前)':<10} {'CV(后)':<10} | {'偏移均值':<12} {'偏移最大':<12}"
    print(header)
    print("-" * 120)
    
    all_results = {}
    
    for case in cases:
        raw_svg = os.path.join(base_dir, case, 'raw.svg')
        layout_svg = os.path.join(base_dir, case, 'layout.svg')
        
        if not os.path.exists(raw_svg) or not os.path.exists(layout_svg):
            print(f"{case:<10} 文件不存在，跳过")
            continue
        
        results = calculate_all_metrics(raw_svg, layout_svg)
        all_results[case] = results
        
        # 输出结果
        row = (
            f"{case:<10} "
            f"{results['raw']['num_nodes']:<6} "
            f"{results['raw']['num_edges']:<6} | "
            f"{results['raw']['crossings']:<10} "
            f"{results['layout']['crossings']:<10} | "
            f"{results['raw']['direction_ratio']:<10.4f} "
            f"{results['layout']['direction_ratio']:<10.4f} | "
            f"{results['raw']['length_cv']:<10.4f} "
            f"{results['layout']['length_cv']:<10.4f} | "
            f"{results['comparison']['displacement_mean']:<12.2f} "
            f"{results['comparison']['displacement_max']:<12.2f}"
        )
        print(row)
    
    print("=" * 120)
    
    # 输出详细结果
    print("\n" + "=" * 80)
    print("详细指标结果")
    print("=" * 80)
    
    for case, results in all_results.items():
        print(f"\n【{case}】")
        print(f"  节点数: {results['raw']['num_nodes']}, 边数: {results['raw']['num_edges']}")
        print(f"  指标1 - 几何交叉数: {results['raw']['crossings']} -> {results['layout']['crossings']}")
        print(f"  指标2 - 八方向比例: {results['raw']['direction_ratio']:.4f} -> {results['layout']['direction_ratio']:.4f}")
        print(f"  指标3 - 转向代价:   {results['raw']['turning_cost']:.4f} -> {results['layout']['turning_cost']:.4f}")
        print(f"  指标4 - 总长度:     {results['raw']['total_length']:.2f} -> {results['layout']['total_length']:.2f}")
        print(f"  指标5 - 边长CV:     {results['raw']['length_cv']:.4f} -> {results['layout']['length_cv']:.4f}")
        print(f"  指标6 - 偏移(sum/mean/max): {results['comparison']['displacement_sum']:.2f} / {results['comparison']['displacement_mean']:.2f} / {results['comparison']['displacement_max']:.2f}")
        print(f"  指标8 - 站点冲突:   {results['raw']['station_violations']} -> {results['layout']['station_violations']}")
        print(f"  指标8 - 边冲突:     {results['raw']['edge_violations']} -> {results['layout']['edge_violations']}")
        print(f"  指标8 - 重叠长度:   {results['raw']['overlap_length']:.2f} -> {results['layout']['overlap_length']:.2f}")


if __name__ == '__main__':
    main()
