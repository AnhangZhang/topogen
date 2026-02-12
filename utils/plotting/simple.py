import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.font_manager as fm
import os
import math
from skgeom import Point2
from skgeom.draw import draw
from matplotlib.patches import Polygon as PolygonPatch
from utils.crossing import find_crossings
from .save import save_plot

# 设置中文字体支持
rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False
# 强制重新加载字体
fm._load_fontmanager(try_read_cache=False)

# SVG样式常量
SVG_STATION_RADIUS = 10.0
SVG_FONT_SIZE = 8
SVG_PADDING = 50


def save_svg(graph, save_path, hide_dummy=False):
    """
    生成SVG格式的拓扑图
    :param graph: networkx.Graph
    :param save_path: str, SVG保存路径
    :param hide_dummy: boolean, 是否隐藏虚拟节点
    """
    # 确保目录存在
    dir_path = os.path.dirname(save_path)
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
    
    # 获取节点坐标范围
    coords = [(graph.nodes[n]['x'], graph.nodes[n]['y']) for n in graph.nodes]
    if not coords:
        return
    
    xs, ys = zip(*coords)
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    
    # 计算坐标范围
    data_width = max_x - min_x
    data_height = max_y - min_y
    
    # 如果坐标范围太小（布局优化后的归一化坐标），需要缩放到合理的可视化尺寸
    # 目标：确保图像至少有600像素的最小维度
    min_target_size = 600
    if data_width > 0 and data_height > 0:
        current_max = max(data_width, data_height)
        if current_max < min_target_size:
            scale = min_target_size / current_max
        else:
            scale = 1.0
    elif data_width > 0:
        scale = min_target_size / data_width if data_width < min_target_size else 1.0
    elif data_height > 0:
        scale = min_target_size / data_height if data_height < min_target_size else 1.0
    else:
        scale = 1.0
    
    padding = SVG_PADDING
    scaled_width = data_width * scale + 2 * padding
    scaled_height = data_height * scale + 2 * padding
    
    with open(save_path, 'w', encoding='utf-8') as f:
        # SVG 头部
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write(f'<svg width="{scaled_width}" height="{scaled_height}" ')
        f.write(f'viewBox="0 0 {scaled_width} {scaled_height}" ')
        f.write('xmlns="http://www.w3.org/2000/svg">\n')
        
        # 样式定义（使用固定值，与C++版本一致）
        station_radius = SVG_STATION_RADIUS  # 10.0
        font_size = SVG_FONT_SIZE  # 8
        stroke_width = 2
        f.write('<defs>\n')
        f.write('  <style>\n')
        f.write(f'    .station-220 {{ fill: #2563eb; stroke: #1e40af; stroke-width: {stroke_width}; }}\n')
        f.write(f'    .station-500 {{ fill: #ef4444; stroke: #1e40af; stroke-width: {stroke_width}; }}\n')
        f.write(f'    .station-default {{ fill: #2563eb; stroke: #1e40af; stroke-width: {stroke_width}; }}\n')
        f.write(f'    .bend {{ fill: #111827; stroke: #111827; stroke-width: 1; }}\n')
        f.write(f'    .dummy {{ fill: #8b4513; stroke: #8b4513; stroke-width: 1; }}\n')
        f.write(f'    .inter {{ fill: #6b7280; stroke: #6b7280; stroke-width: 1; }}\n')
        f.write(f'    .station-label {{ font-family: Arial, sans-serif; font-size: {font_size}px; '
                f'fill: white; text-anchor: middle; dominant-baseline: central; }}\n')
        f.write(f'    .edge {{ stroke: #6b7280; stroke-width: {stroke_width}; fill: none; }}\n')
        f.write('    .background { fill: #f8fafc; }\n')
        f.write('  </style>\n')
        f.write('</defs>\n')
        
        # 背景
        f.write(f'<rect class="background" width="{scaled_width}" height="{scaled_height}"/>\n')
        
        # 绘制边（双线效果）
        f.write('<!-- Edges -->\n')
        for u, v in graph.edges:
            x1 = (graph.nodes[u]['x'] - min_x) * scale + padding
            y1 = (graph.nodes[u]['y'] - min_y) * scale + padding
            x2 = (graph.nodes[v]['x'] - min_x) * scale + padding
            y2 = (graph.nodes[v]['y'] - min_y) * scale + padding
            
            # 计算双线偏移
            dx = x2 - x1
            dy = y2 - y1
            length = math.sqrt(dx * dx + dy * dy)
            if length > 1e-9:
                ux, uy = dx / length, dy / length
                nx_, ny = -uy, ux
                offset = 2.0  # 双线偏移量
                
                # 双线
                f.write(f'<line class="edge" x1="{x1 + nx_ * offset}" y1="{y1 + ny * offset}" '
                        f'x2="{x2 + nx_ * offset}" y2="{y2 + ny * offset}"/>\n')
                f.write(f'<line class="edge" x1="{x1 - nx_ * offset}" y1="{y1 - ny * offset}" '
                        f'x2="{x2 - nx_ * offset}" y2="{y2 - ny * offset}"/>\n')
            else:
                f.write(f'<line class="edge" x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}"/>\n')
        
        # 绘制节点遮罩（halo，用于遮挡穿过节点的边）
        f.write('<!-- Node Halos -->\n')
        halo_size = 2.0
        for node in graph.nodes:
            x = (graph.nodes[node]['x'] - min_x) * scale + padding
            y = (graph.nodes[node]['y'] - min_y) * scale + padding
            is_dummy = is_dummy_node(node)
            
            if is_dummy:
                r = station_radius * 0.35 + halo_size
            else:
                r = station_radius + halo_size
            f.write(f'<circle class="background" cx="{x}" cy="{y}" r="{r}"/>\n')
        
        # 绘制节点
        f.write('<!-- Stations -->\n')
        for node in graph.nodes:
            x = (graph.nodes[node]['x'] - min_x) * scale + padding
            y = (graph.nodes[node]['y'] - min_y) * scale + padding
            is_dummy = is_dummy_node(node)
            
            if is_dummy:
                if hide_dummy:
                    # 小灰点作为拐点
                    r = station_radius * 0.35
                    f.write(f'<circle class="bend" cx="{x}" cy="{y}" r="{r}"/>\n')
                else:
                    # 显示虚拟节点
                    r = station_radius * 0.5
                    node_str = str(node).lower()
                    if node_str.startswith('d'):
                        f.write(f'<circle class="dummy" cx="{x}" cy="{y}" r="{r}"/>\n')
                    else:
                        f.write(f'<circle class="inter" cx="{x}" cy="{y}" r="{r}"/>\n')
                    f.write(f'<text class="station-label" x="{x}" y="{y}">{node}</text>\n')
            else:
                # 真实节点
                voltage = graph.nodes[node].get('voltage', 220)
                station_class = 'station-500' if voltage == 500 else 'station-220'
                f.write(f'<circle class="{station_class}" cx="{x}" cy="{y}" r="{station_radius}"/>\n')
                
                # 节点标签
                # label = graph.nodes[node].get('name', str(node))
                # f.write(f'<text class="station-label" x="{x}" y="{y}">{label}</text>\n')
        
        f.write('</svg>\n')
    
    print(f'SVG saved: {save_path}')


def is_dummy_node(node):
    """检查节点是否为虚拟节点（D*或I*开头）"""
    node_str = str(node).lower()
    return node_str.startswith('d') or node_str.startswith('i')


def create_display_graph(graph, hide_dummy=True):
    """
    创建用于显示的图，可选择隐藏虚拟节点
    :param graph: networkx.Graph
    :param hide_dummy: boolean, 是否隐藏虚拟节点
    :return: networkx.Graph, 用于显示的图
    """
    if not hide_dummy:
        return graph
    
    # 创建不包含虚拟节点的新图
    display_graph = nx.Graph()
    
    # 只添加非虚拟节点
    for node, attrs in graph.nodes(data=True):
        if not is_dummy_node(node):
            display_graph.add_node(node, **attrs)
    
    # 重新连接边：如果边的两端都是真实节点，直接连接
    # 如果边经过虚拟节点，需要找到真实节点之间的路径
    real_nodes = set(display_graph.nodes())
    
    # 对于每对真实节点，检查它们在原图中是否连通（可能经过虚拟节点）
    for u in real_nodes:
        for v in real_nodes:
            if u >= v:
                continue
            # 检查u和v在原图中是否通过路径连接（路径上可能有虚拟节点）
            if graph.has_edge(u, v):
                display_graph.add_edge(u, v)
            else:
                # 检查是否通过虚拟节点连接
                try:
                    path = nx.shortest_path(graph, u, v)
                    # 如果路径上除了u和v之外都是虚拟节点，则连接u和v
                    intermediate_nodes = path[1:-1]
                    if all(is_dummy_node(n) for n in intermediate_nodes) and len(intermediate_nodes) > 0:
                        display_graph.add_edge(u, v)
                except nx.NetworkXNoPath:
                    pass
    
    return display_graph


def plot_graph(graph, figsize=10, node_size=80, highlight_node=None, draw_crossing=True,
               draw_area=False, highlight_area=None, draw_cell=False, cell=None, area=None, draw_label=True,
               save_figure=True, save_path='graph.pdf', hide_dummy=False, save_svg_file=True):
    """
    Draw a networkx.Graph and optionally save it to a file
    :param graph: networkx.Graph
    :param figsize: figure size in inches
    :param node_size: node size
    :param highlight_node: list of str, names of the nodes to be highlighted
    :param draw_crossing: boolean, whether to highlight crossings
    :param draw_area: boolean, whether to draw nodes in different areas with different colors
    :param highlight_area: list of int, areas not in the list are drawn as gray
    :param draw_cell: boolean, whether to draw the partition cell
    :param cell: numpy.array of shape (n, 2), coordinates of cell nodes
    :param area: int, index of area
    :param draw_label: boolean, whether to draw labels
    :param save_figure: boolean, whether to save the drawing to an .pdf file
    :param save_path: str, the path to save the figure
    :param hide_dummy: boolean, 是否隐藏虚拟节点标签(D*, I*)，但保留拐点
    :return: None
    """
    if not highlight_node:
        highlight_node = []
    pos, labels, sizes, colors = dict(), dict(), [], []
    n_dummy = 0
    for node in graph.nodes:
        pos[node] = (graph.nodes[node]['x'], graph.nodes[node]['y'])
        
        # 判断是否为虚拟节点
        is_dummy = is_dummy_node(node)
        
        if is_dummy:
            n_dummy += 1
            if hide_dummy:
                # 隐藏虚拟节点：不显示标签，节点大小设为很小的点
                labels[node] = ''  # 空标签
                sizes.append(node_size * 0.15)  # 很小的点作为拐点
                colors.append('tab:gray')  # 灰色小点
            else:
                # 显示虚拟节点
                labels[node] = node
                sizes.append(0.8 * node_size)
                if str(node).lower().startswith('d'):
                    colors.append('tab:brown')
                else:
                    colors.append('tab:gray')
        else:
            # 真实节点：使用name属性作为标签
            labels[node] = graph.nodes[node].get('name', node)
            sizes.append(node_size)
            if node in highlight_node:
                colors.append('tab:red')
            else:
                if not draw_area:
                    colors.append('tab:blue')
                else:
                    if highlight_area is None:
                        colors.append(graph.nodes[node]['area'])
                    else:
                        if graph.nodes[node]['area'] == highlight_area:
                            colors.append('tab:blue')
                        else:
                            colors.append('tab:gray')

    if cell is None:
        x_min, y_min = np.min(np.array(list(pos.values())), axis=0)
        x_max, y_max = np.max(np.array(list(pos.values())), axis=0)
    else:
        x_min, y_min = np.min(cell, axis=0)
        x_max, y_max = np.max(cell, axis=0)
    x_delta, y_delta = (x_max - x_min) / 15, (y_max - y_min) / 15
    x_min, x_max = x_min - x_delta, x_max + x_delta
    y_min, y_max = y_min - y_delta, y_max + y_delta
    figsize = (figsize, figsize * (y_max - y_min) / (x_max - x_min) + 0.8)
    plt.figure(1, figsize=figsize, dpi=120)
    plt.clf()  # 清除之前的内容，防止图像叠加！
    nx.draw(graph, pos, node_size=sizes, with_labels=draw_label, labels=labels, font_size=node_size / 16,
            font_color='white', node_color=colors, font_family='WenQuanYi Micro Hei')
    annotation = []
    if area is not None:
        annotation.append('Area = %s' % str(area))
    if draw_crossing:
        crossings = find_crossings(graph)
        # 如果隐藏虚拟节点，不计算虚拟节点相关的交叉
        annotation.append('Crossings = %d' % len(crossings))
    if draw_cell and cell is not None:
        ax = plt.gca()
        ax.add_patch(PolygonPatch(cell, alpha=0.2))
    anno = ', '.join(annotation)
    plt.text(0.05, 0.95, s=anno, ha='left', va='top', transform=plt.gca().transAxes, color='tab:orange')
    if save_figure:
        figure = plt.gcf()
        save_plot(figure, save_path)
        # 同时生成SVG文件
        if save_svg_file:
            svg_path = save_path.rsplit('.', 1)[0] + '.svg'
            save_svg(graph, svg_path, hide_dummy=hide_dummy)
    ax = plt.gca()
    ax.set_aspect('equal')
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()
    pass


def plot_comparison_graph(raw_graph, crossing_reduced_graph, optimized_graph, force_directed_graph,
                          save_figure=True, save_path='graph.pdf'):
    """
    Plot a 2x2 diagram for comparison
    :param raw_graph: upper left graph
    :param crossing_reduced_graph: upper right graph
    :param optimized_graph: lower left graph
    :param force_directed_graph: lower right graph
    :param save_figure: boolean, whether to save the figure as a file
    :param save_path: str, path to save the figure
    :return: None
    """
    fig, axs = plt.subplots(2, 2, figsize=(7.5, 7.5), layout='constrained', linewidth=4,
                            gridspec_kw={'hspace': 0, 'wspace': 0})

    def _plot_graph(graph, ax):
        plt.sca(ax)
        ax.margins(0)
        pos, sizes, colors = dict(), [], []
        dummy_pos, inter_pos = [], []
        for node in graph.nodes:
            pos[node] = (graph.nodes[node]['x'], graph.nodes[node]['y'])
            colors.append('tab:blue')
            if str(node).lower().startswith('d'):
                sizes.append(0)
                dummy_pos.append(pos[node])
            elif str(node).lower().startswith('i'):
                sizes.append(0)
                inter_pos.append(pos[node])
            else:
                sizes.append(10)
        nx.draw(graph, pos, node_size=sizes, node_color=colors, with_labels=False, width=0.5)
        crossings = find_crossings(graph)
        for _, (x, y) in crossings:
            draw(Point2(x, y), color='tab:orange', s=4, zorder=2)
        for x, y in dummy_pos:
            draw(Point2(x, y), color='tab:brown', s=4, zorder=2)
        plt.axis('on')
        plt.setp(ax.spines.values(), linewidth=1)
        pass

    _plot_graph(raw_graph, axs[0][0])
    _plot_graph(crossing_reduced_graph, axs[0][1])
    _plot_graph(optimized_graph, axs[1][0])
    _plot_graph(force_directed_graph, axs[1][1])

    if save_figure:
        save_plot(fig, save_path)
    plt.show()
    pass
