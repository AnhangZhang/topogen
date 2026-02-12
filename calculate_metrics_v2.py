#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
严格按照指标.png中的定义计算8个指标
输入: output_comparison/case*/raw.svg (优化前) 和 layout.svg (优化后)
"""

import os
import re
import math
import numpy as np
from collections import defaultdict


def parse_svg(svg_path):
    """
    解析SVG文件，提取节点坐标和边(线段)信息
    返回: nodes (dict: id -> (x, y)), segments (所有线段列表)
    """
    with open(svg_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 提取所有station圆形节点 (class="station-XXX" r="10.0")
    station_pattern = r'<circle class="station-(?:220|500|default)" cx="([^"]+)" cy="([^"]+)" r="10\.0"/>'
    station_matches = re.findall(station_pattern, content)
    
    nodes = {}
    for i, (cx, cy) in enumerate(station_matches):
        nodes[i] = (float(cx), float(cy))
    
    # 提取所有边线段
    edge_pattern = r'<line class="edge" x1="([^"]+)" y1="([^"]+)" x2="([^"]+)" y2="([^"]+)"/>'
    edge_matches = re.findall(edge_pattern, content)
    
    # 由于SVG中每条边画了两次(双线效果)，需要去重合并
    raw_segments = []
    for x1, y1, x2, y2 in edge_matches:
        raw_segments.append(((float(x1), float(y1)), (float(x2), float(y2))))
    
    # 合并双线效果
    segments = merge_duplicate_segments(raw_segments)
    
    return nodes, segments


def merge_duplicate_segments(raw_segments, threshold=5.0):
    """合并双线效果产生的重复线段"""
    merged = []
    used = [False] * len(raw_segments)
    
    for i, s1 in enumerate(raw_segments):
        if used[i]:
            continue
        for j in range(i + 1, len(raw_segments)):
            if used[j]:
                continue
            s2 = raw_segments[j]
            if segments_are_paired(s1, s2, threshold):
                # 取中点作为合并后的线段
                mid_p1 = ((s1[0][0] + s2[0][0]) / 2, (s1[0][1] + s2[0][1]) / 2)
                mid_p2 = ((s1[1][0] + s2[1][0]) / 2, (s1[1][1] + s2[1][1]) / 2)
                merged.append((mid_p1, mid_p2))
                used[i] = True
                used[j] = True
                break
        if not used[i]:
            merged.append(s1)
            used[i] = True
    
    return merged


def segments_are_paired(s1, s2, threshold):
    """检查两条线段是否是双线效果的配对"""
    d1 = point_dist(s1[0], s2[0]) + point_dist(s1[1], s2[1])
    d2 = point_dist(s1[0], s2[1]) + point_dist(s1[1], s2[0])
    return min(d1, d2) < threshold * 2


def point_dist(p1, p2):
    """计算两点欧氏距离"""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def find_nearest_node(point, nodes, threshold=15.0):
    """找到距离point最近的节点"""
    min_dist = float('inf')
    nearest = None
    for nid, npos in nodes.items():
        d = point_dist(point, npos)
        if d < min_dist:
            min_dist = d
            nearest = nid
    return nearest if min_dist < threshold else None


def segments_share_endpoint(s1, s2, threshold=10.0):
    """检查两条线段是否共享端点"""
    for p1 in s1:
        for p2 in s2:
            if point_dist(p1, p2) < threshold:
                return True
    return False


# ============== 指标1: 几何交叉数 (C_cross) ==============
def metric1_crossings(segments):
    """
    指标1: 几何交叉数 (总线段交叉)
    count of segment intersections excluding incident and excluding intersections "at interchange"
    
    收集所有线段集合 S = {s_k}
    对任意两条线段 e, f：
    - e ≠ f 不相邻（不共享端点）：若共享端点直接跳过
    - 计算线段交叉
    
    输出: C_cross
    """
    C_cross = 0
    n = len(segments)
    
    for i in range(n):
        for j in range(i + 1, n):
            s1, s2 = segments[i], segments[j]
            # 排除共享端点的线段对
            if segments_share_endpoint(s1, s2):
                continue
            # 检测交叉
            if segments_intersect(s1, s2):
                C_cross += 1
    
    return C_cross


def segments_intersect(seg1, seg2):
    """检测两线段是否相交（不包含端点接触）"""
    p1, p2 = seg1
    p3, p4 = seg2
    
    def ccw(A, B, C):
        return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
    
    def intersect(A, B, C, D):
        return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)
    
    return intersect(p1, p2, p3, p4)


# ============== 指标2: 八方向连接比例 (R_oct) ==============
def metric2_octilinear_ratio(segments, epsilon_theta=3.0):
    """
    指标2: 八方向连接比例 (带容差，长度加权)
    
    对每条线段 e = ((x_a,y_a), (x_b,y_b))：
    1. 线段方向角（度）: θ = atan2(y_b - y_a, x_b - x_a) · 180/π，范围 θ ∈ [0, 360)
    2. 最近八方向: k = round(θ/45) mod 8，最近方向角 θ_k = 45k
    3. 偏离度: Δ(θ_k) = min{|θ - θ_k|, 360 - |θ - θ_k|}
    4. 依照容差: 若 Δ(θ) > ε_θ 则该线段视为偏离
    
    定义长度加权比例:
    R_oct = Σ(1[|Δ(θ_k)| <= ε_θ] * len(e)) / Σlen(e)
    
    输出: R_oct
    """
    total_length = 0.0
    aligned_length = 0.0
    
    for seg in segments:
        p1, p2 = seg
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        length = math.sqrt(dx*dx + dy*dy)
        
        if length < 1e-6:
            continue
        
        # 计算方向角 θ ∈ [0, 360)
        theta = math.degrees(math.atan2(dy, dx))
        if theta < 0:
            theta += 360
        
        # 最近八方向 k = round(θ/45) mod 8
        k = round(theta / 45) % 8
        theta_k = 45 * k
        
        # 偏离度 Δ(θ_k)
        delta = abs(theta - theta_k)
        delta = min(delta, 360 - delta)  # 处理环绕
        
        total_length += length
        if delta <= epsilon_theta:
            aligned_length += length
    
    R_oct = aligned_length / total_length if total_length > 0 else 0
    return R_oct


# ============== 指标3: 转向角折零代价 (B) ==============
def metric3_turning_cost(segments, nodes):
    """
    指标3: 转向角折零代价 (逐线段/至polyline)
    
    对每条边对应折线 P_i = (p_1, ..., p_k)，对每个中间点 i = 1...k-1:
    - 设 a = p_i - p_{i-1}, b = p_{i+1} - p_i
    - 转角角 (0~180°): δ_i = arccos((a·b)/(|a||b|)) · 180/π
    - 将 δ_i 量化到 {0, 45, 90, 135}:
      δ̂_i = 45 · round(δ_i/45)
    - 权重代价: w(0) = 0, w(45) = 1, w(90) = 2, w(135) = 3
    
    归纳的单代价: B_i = Σ w(δ̂_i)
    全图总代价: B = Σ B_i
    
    输出: B (也报告平均: B/|E|)
    """
    # 首先需要将线段组装成折线（polyline）
    # 按照共享端点将线段连接成边的路径
    polylines = build_polylines_from_segments(segments, nodes)
    
    weights = {0: 0, 45: 1, 90: 2, 135: 3, 180: 4}
    
    B_total = 0
    
    for polyline in polylines:
        if len(polyline) < 3:
            continue  # 至少需要3个点才有转角
        
        B_i = 0
        for i in range(1, len(polyline) - 1):
            p_prev = polyline[i - 1]
            p_curr = polyline[i]
            p_next = polyline[i + 1]
            
            # 向量 a = p_curr - p_prev, b = p_next - p_curr
            a = (p_curr[0] - p_prev[0], p_curr[1] - p_prev[1])
            b = (p_next[0] - p_curr[0], p_next[1] - p_curr[1])
            
            len_a = math.sqrt(a[0]**2 + a[1]**2)
            len_b = math.sqrt(b[0]**2 + b[1]**2)
            
            if len_a < 1e-6 or len_b < 1e-6:
                continue
            
            # 计算转角 δ_i = arccos((a·b)/(|a||b|)) · 180/π
            dot = a[0]*b[0] + a[1]*b[1]
            cos_angle = max(-1, min(1, dot / (len_a * len_b)))
            delta_i = math.degrees(math.acos(cos_angle))
            
            # 量化到 {0, 45, 90, 135, 180}
            delta_hat = 45 * round(delta_i / 45)
            delta_hat = min(delta_hat, 180)
            
            # 权重代价
            w = weights.get(int(delta_hat), 4)
            B_i += w
        
        B_total += B_i
    
    return B_total


def build_polylines_from_segments(segments, nodes):
    """
    将线段组装成折线（从station到station的路径）
    简化处理：每条线段作为独立的2点折线
    """
    polylines = []
    for seg in segments:
        polylines.append([seg[0], seg[1]])
    return polylines


# ============== 指标4: 总线路长度 (L) ==============
def metric4_total_length(segments):
    """
    指标4: 总线路长度 (折线总长度)
    
    对每条边: l_e = Σ ||p_i - p_{i-1}||
    输出: L = Σ l_e
    """
    L = 0.0
    for seg in segments:
        p1, p2 = seg
        L += point_dist(p1, p2)
    return L


# ============== 指标5: 边长均匀性 (CV) ==============
def metric5_length_cv(segments):
    """
    指标5: 边长均匀性 (用CV)
    
    用上面的 l_e:
    μ = (1/E) Σ l_e
    σ = sqrt((1/E) Σ (l_e - μ)^2)
    CV = σ/μ
    
    输出: CV
    """
    lengths = []
    for seg in segments:
        p1, p2 = seg
        lengths.append(point_dist(p1, p2))
    
    if len(lengths) == 0:
        return 0.0
    
    mu = np.mean(lengths)
    sigma = np.std(lengths)
    
    CV = sigma / mu if mu > 0 else 0.0
    return CV


# ============== 指标6: 总体布局偏移 (D_sum, D_mean, D_max) ==============
def metric6_displacement(nodes_raw, nodes_layout):
    """
    指标6: 总体布局偏移 (sum/mean/max)
    
    D_sum = Σ ||x_v - x̄_v||
    D_mean = (1/V) D_sum
    D_max = max ||x_v - x̄_v||
    
    输出三项: D_sum, D_mean, D_max
    """
    displacements = []
    
    # 按索引匹配节点
    for vid in nodes_raw:
        if vid in nodes_layout:
            d = point_dist(nodes_raw[vid], nodes_layout[vid])
            displacements.append(d)
    
    if len(displacements) == 0:
        return 0.0, 0.0, 0.0
    
    D_sum = sum(displacements)
    D_mean = D_sum / len(displacements)
    D_max = max(displacements)
    
    return D_sum, D_mean, D_max


# ============== 指标7: 出边顺序保持率 (R_order) ==============
def metric7_order_preservation(nodes_raw, segments_raw, nodes_layout, segments_layout):
    """
    指标7: 出边顺序保持率 (仅 deg>=3; tie 首按失效)
    
    对每个顶点 v（只限 deg(v) >= 3）:
    1. 输入进入该点的初始顺序 Q^raw (按CCW或CW固定)
    2. 输出中，取等条 incident 边的方向
    3. 若存在 tie（接近角度），则该点 fail
    4. 否则，若 Q^raw 与 Q^layout 循环一致，则 pass，否则 fail
    
    保持率: R_order = #{v : deg(v) >= 3, v pass} / #{v : deg(v) >= 3}
    """
    def get_incident_angles(nodes, segments, node_id, threshold=15.0):
        """获取节点的所有关联边的角度"""
        node_pos = nodes[node_id]
        angles = []
        
        for seg in segments:
            p1, p2 = seg
            # 检查是否与该节点关联
            if point_dist(p1, node_pos) < threshold:
                # 边从node_pos出发到p2
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                angle = math.degrees(math.atan2(dy, dx))
                if angle < 0:
                    angle += 360
                angles.append(angle)
            elif point_dist(p2, node_pos) < threshold:
                # 边从node_pos出发到p1
                dx = p1[0] - p2[0]
                dy = p1[1] - p2[1]
                angle = math.degrees(math.atan2(dy, dx))
                if angle < 0:
                    angle += 360
                angles.append(angle)
        
        return sorted(angles)
    
    def check_cyclic_order(order1, order2, tie_epsilon=5.0):
        """检查两个角度序列是否循环一致"""
        if len(order1) != len(order2):
            return False
        if len(order1) == 0:
            return True
        
        n = len(order1)
        # 检查是否存在tie（相邻角度差太小）
        for i in range(n):
            diff = abs(order1[i] - order1[(i+1) % n])
            diff = min(diff, 360 - diff)
            if diff < tie_epsilon:
                return False  # tie导致fail
        
        # 检查循环一致性
        # 尝试所有起点
        for start in range(n):
            match = True
            for i in range(n):
                diff = abs(order1[i] - order2[(start + i) % n])
                diff = min(diff, 360 - diff)
                if diff > 30:  # 允许一定容差
                    match = False
                    break
            if match:
                return True
        return False
    
    # 计算每个节点的度
    degrees_raw = defaultdict(int)
    for seg in segments_raw:
        for nid, npos in nodes_raw.items():
            if point_dist(seg[0], npos) < 15 or point_dist(seg[1], npos) < 15:
                degrees_raw[nid] += 1
    
    # 统计度>=3的节点
    high_degree_nodes = [nid for nid, deg in degrees_raw.items() if deg >= 3]
    
    if len(high_degree_nodes) == 0:
        return 1.0  # 没有高度节点
    
    pass_count = 0
    for nid in high_degree_nodes:
        if nid not in nodes_layout:
            continue
        
        angles_raw = get_incident_angles(nodes_raw, segments_raw, nid)
        angles_layout = get_incident_angles(nodes_layout, segments_layout, nid)
        
        if check_cyclic_order(angles_raw, angles_layout):
            pass_count += 1
    
    R_order = pass_count / len(high_degree_nodes) if len(high_degree_nodes) > 0 else 1.0
    return R_order


# ============== 指标8: 补充指标 ==============
def metric8_additional(nodes, segments):
    """
    指标8: 你要补充的两项
    
    8.1 距离子项的冲突数 (station-station 与 station-edge)
    
    (a) station-station violations
        N_ss = #{(u,v) ⊂ V : u ≠ v, ||x_u - x_v|| < d_s}
        也报告最小值 min_{u,v} ||x_u - x_v||
    
    (b) station-edge violations (只考虑 nonincident)
        对每个站点 v，对每条不与其关联的边 e，计算到线段最小距离
        N_se = #{(v,e) : v not incident to e, d(v,e) < d_s}
    
    8.2 重叠段总长度 (nonincident segments overlap length)
        对所有非关联边对 {e, f}:
        - 排除共享端点的
        - 计算重叠部分长度 len(e ∩ f)
        L_overlap = Σ len(e ∩ f)
    
    输出: N_ss, N_se, L_overlap
    """
    d_s = 20.0  # 最小距离阈值
    
    # 8.1(a) station-station violations
    node_list = list(nodes.items())
    N_ss = 0
    min_ss_dist = float('inf')
    
    for i in range(len(node_list)):
        for j in range(i + 1, len(node_list)):
            nid1, pos1 = node_list[i]
            nid2, pos2 = node_list[j]
            d = point_dist(pos1, pos2)
            if d < d_s:
                N_ss += 1
            min_ss_dist = min(min_ss_dist, d)
    
    # 8.1(b) station-edge violations (nonincident)
    N_se = 0
    for nid, npos in nodes.items():
        for seg in segments:
            p1, p2 = seg
            # 检查是否关联
            if point_dist(p1, npos) < 15 or point_dist(p2, npos) < 15:
                continue  # 跳过关联的边
            # 计算点到线段距离
            d = point_to_segment_dist(npos, p1, p2)
            if d < d_s:
                N_se += 1
    
    # 8.2 重叠段总长度
    L_overlap = 0.0
    n_seg = len(segments)
    for i in range(n_seg):
        for j in range(i + 1, n_seg):
            s1, s2 = segments[i], segments[j]
            # 排除共享端点的
            if segments_share_endpoint(s1, s2):
                continue
            # 计算重叠长度
            overlap = compute_segment_overlap(s1, s2)
            L_overlap += overlap
    
    return N_ss, N_se, L_overlap


def point_to_segment_dist(p, s1, s2):
    """计算点p到线段s1-s2的最短距离"""
    dx = s2[0] - s1[0]
    dy = s2[1] - s1[1]
    len_sq = dx*dx + dy*dy
    
    if len_sq < 1e-10:
        return point_dist(p, s1)
    
    t = max(0, min(1, ((p[0] - s1[0]) * dx + (p[1] - s1[1]) * dy) / len_sq))
    proj = (s1[0] + t * dx, s1[1] + t * dy)
    return point_dist(p, proj)


def compute_segment_overlap(s1, s2, threshold=5.0):
    """
    计算两条近似平行线段的重叠长度
    """
    d1 = (s1[1][0] - s1[0][0], s1[1][1] - s1[0][1])
    d2 = (s2[1][0] - s2[0][0], s2[1][1] - s2[0][1])
    
    len1 = math.sqrt(d1[0]**2 + d1[1]**2)
    len2 = math.sqrt(d2[0]**2 + d2[1]**2)
    
    if len1 < 1e-6 or len2 < 1e-6:
        return 0.0
    
    # 归一化方向向量
    d1_norm = (d1[0]/len1, d1[1]/len1)
    d2_norm = (d2[0]/len2, d2[1]/len2)
    
    # 检查是否平行 (点积绝对值接近1)
    dot = abs(d1_norm[0]*d2_norm[0] + d1_norm[1]*d2_norm[1])
    if dot < 0.95:  # 不平行
        return 0.0
    
    # 计算两条线段端点之间的距离
    dist1 = point_to_segment_dist(s2[0], s1[0], s1[1])
    dist2 = point_to_segment_dist(s2[1], s1[0], s1[1])
    
    if min(dist1, dist2) > threshold:
        return 0.0
    
    # 计算投影重叠
    # 将s2投影到s1的方向上
    def project_point(p, origin, direction):
        return (p[0] - origin[0]) * direction[0] + (p[1] - origin[1]) * direction[1]
    
    # s1的投影范围 [0, len1]
    # s2端点在s1方向上的投影
    proj_s2_0 = project_point(s2[0], s1[0], d1_norm)
    proj_s2_1 = project_point(s2[1], s1[0], d1_norm)
    
    # 计算重叠区间
    s2_min = min(proj_s2_0, proj_s2_1)
    s2_max = max(proj_s2_0, proj_s2_1)
    
    overlap_start = max(0, s2_min)
    overlap_end = min(len1, s2_max)
    
    overlap_len = max(0, overlap_end - overlap_start)
    return overlap_len


# ============== 主计算函数 ==============
def calculate_all_metrics(raw_svg_path, layout_svg_path):
    """计算所有8个指标"""
    
    # 解析SVG
    nodes_raw, segments_raw = parse_svg(raw_svg_path)
    nodes_layout, segments_layout = parse_svg(layout_svg_path)
    
    results = {}
    
    # 基本信息
    results['num_nodes'] = len(nodes_raw)
    results['num_edges'] = len(segments_raw)
    
    # 指标1: 几何交叉数 C_cross
    results['M1_crossings_raw'] = metric1_crossings(segments_raw)
    results['M1_crossings_layout'] = metric1_crossings(segments_layout)
    
    # 指标2: 八方向连接比例 R_oct
    results['M2_octilinear_raw'] = metric2_octilinear_ratio(segments_raw)
    results['M2_octilinear_layout'] = metric2_octilinear_ratio(segments_layout)
    
    # 指标3: 转向角折零代价 B
    results['M3_turning_cost_raw'] = metric3_turning_cost(segments_raw, nodes_raw)
    results['M3_turning_cost_layout'] = metric3_turning_cost(segments_layout, nodes_layout)
    
    # 指标4: 总线路长度 L
    results['M4_total_length_raw'] = metric4_total_length(segments_raw)
    results['M4_total_length_layout'] = metric4_total_length(segments_layout)
    
    # 指标5: 边长均匀性 CV
    results['M5_length_cv_raw'] = metric5_length_cv(segments_raw)
    results['M5_length_cv_layout'] = metric5_length_cv(segments_layout)
    
    # 指标6: 总体布局偏移
    D_sum, D_mean, D_max = metric6_displacement(nodes_raw, nodes_layout)
    results['M6_displacement_sum'] = D_sum
    results['M6_displacement_mean'] = D_mean
    results['M6_displacement_max'] = D_max
    
    # 指标7: 出边顺序保持率 R_order
    results['M7_order_preservation'] = metric7_order_preservation(
        nodes_raw, segments_raw, nodes_layout, segments_layout)
    
    # 指标8: 补充指标
    N_ss_raw, N_se_raw, L_overlap_raw = metric8_additional(nodes_raw, segments_raw)
    N_ss_layout, N_se_layout, L_overlap_layout = metric8_additional(nodes_layout, segments_layout)
    
    results['M8_station_violations_raw'] = N_ss_raw
    results['M8_station_violations_layout'] = N_ss_layout
    results['M8_edge_violations_raw'] = N_se_raw
    results['M8_edge_violations_layout'] = N_se_layout
    results['M8_overlap_length_raw'] = L_overlap_raw
    results['M8_overlap_length_layout'] = L_overlap_layout
    
    return results


def main():
    """主函数"""
    base_dir = 'output_comparison'
    cases = [f'case{i}' for i in range(1, 17)]
    
    print("=" * 140)
    print("拓扑图优化指标计算结果 (严格按照指标定义)")
    print("=" * 140)
    
    all_results = {}
    
    for case in cases:
        raw_svg = os.path.join(base_dir, case, 'raw.svg')
        layout_svg = os.path.join(base_dir, case, 'layout.svg')
        
        if not os.path.exists(raw_svg) or not os.path.exists(layout_svg):
            print(f"{case}: 文件不存在，跳过")
            continue
        
        results = calculate_all_metrics(raw_svg, layout_svg)
        all_results[case] = results
    
    # 输出表格
    print("\n" + "=" * 140)
    print("指标汇总表")
    print("=" * 140)
    
    # 表头
    print(f"{'Case':<8} | {'V':<4} {'E':<4} | "
          f"{'M1:交叉(前→后)':<16} | "
          f"{'M2:八方向(前→后)':<20} | "
          f"{'M3:转角代价(前→后)':<20} | "
          f"{'M4:总长度(前→后)':<22}")
    print("-" * 140)
    
    for case, r in all_results.items():
        print(f"{case:<8} | {r['num_nodes']:<4} {r['num_edges']:<4} | "
              f"{r['M1_crossings_raw']:>5} → {r['M1_crossings_layout']:<5} | "
              f"{r['M2_octilinear_raw']:>7.4f} → {r['M2_octilinear_layout']:<7.4f} | "
              f"{r['M3_turning_cost_raw']:>7} → {r['M3_turning_cost_layout']:<7} | "
              f"{r['M4_total_length_raw']:>9.1f} → {r['M4_total_length_layout']:<9.1f}")
    
    print()
    print(f"{'Case':<8} | "
          f"{'M5:CV(前→后)':<20} | "
          f"{'M6:偏移(sum/mean/max)':<32} | "
          f"{'M7:顺序保持率':<14} | "
          f"{'M8:N_ss(前→后)':<14} | "
          f"{'M8:N_se(前→后)':<14}")
    print("-" * 140)
    
    for case, r in all_results.items():
        print(f"{case:<8} | "
              f"{r['M5_length_cv_raw']:>7.4f} → {r['M5_length_cv_layout']:<7.4f} | "
              f"{r['M6_displacement_sum']:>8.1f} / {r['M6_displacement_mean']:>6.1f} / {r['M6_displacement_max']:<6.1f} | "
              f"{r['M7_order_preservation']:>12.4f} | "
              f"{r['M8_station_violations_raw']:>5} → {r['M8_station_violations_layout']:<5} | "
              f"{r['M8_edge_violations_raw']:>5} → {r['M8_edge_violations_layout']:<5}")
    
    print("=" * 140)
    
    # 详细结果
    print("\n" + "=" * 100)
    print("各Case详细指标")
    print("=" * 100)
    
    for case, r in all_results.items():
        print(f"\n【{case}】 节点数: {r['num_nodes']}, 边数: {r['num_edges']}")
        print(f"  指标1 (C_cross)  几何交叉数:       {r['M1_crossings_raw']:>5} → {r['M1_crossings_layout']:<5}")
        print(f"  指标2 (R_oct)    八方向比例:       {r['M2_octilinear_raw']:>7.4f} → {r['M2_octilinear_layout']:<7.4f}")
        print(f"  指标3 (B)        转向角代价:       {r['M3_turning_cost_raw']:>7} → {r['M3_turning_cost_layout']:<7}")
        print(f"  指标4 (L)        总线路长度:       {r['M4_total_length_raw']:>9.2f} → {r['M4_total_length_layout']:<9.2f}")
        print(f"  指标5 (CV)       边长均匀性CV:     {r['M5_length_cv_raw']:>7.4f} → {r['M5_length_cv_layout']:<7.4f}")
        print(f"  指标6 (D)        布局偏移 sum:     {r['M6_displacement_sum']:.2f}")
        print(f"                           mean:    {r['M6_displacement_mean']:.2f}")
        print(f"                           max:     {r['M6_displacement_max']:.2f}")
        print(f"  指标7 (R_order)  顺序保持率:       {r['M7_order_preservation']:.4f}")
        print(f"  指标8 (N_ss)     站点冲突数:       {r['M8_station_violations_raw']:>5} → {r['M8_station_violations_layout']:<5}")
        print(f"  指标8 (N_se)     边冲突数:         {r['M8_edge_violations_raw']:>5} → {r['M8_edge_violations_layout']:<5}")
        print(f"  指标8 (L_overlap)重叠长度:         {r['M8_overlap_length_raw']:>7.2f} → {r['M8_overlap_length_layout']:<7.2f}")


if __name__ == '__main__':
    main()
