"""
生成图像与原始图像对比可视化
展示8个循环（从小到大）的Generated、Target和Absolute Difference
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.io import loadmat

# 设置所有绘图参数
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['axes.unicode_minus'] = False


def convert_to_uint8(gaf_cell):
    """
    将GAF_cell转换为uint8 (0-255)格式
    """
    converted_gaf = np.empty(gaf_cell.shape, dtype=object)
    
    for i in range(gaf_cell.shape[0]):
        if gaf_cell[i, 0] is not None and len(gaf_cell[i, 0]) > 0:
            img = gaf_cell[i, 0].astype(np.float64)
            
            # 归一化到[0, 1]
            min_val = img.min()
            max_val = img.max()
            
            if max_val > min_val:
                img = (img - min_val) / (max_val - min_val)
            else:
                img = np.zeros_like(img)
            
            # 转换到uint8 [0, 255]
            converted_gaf[i, 0] = np.round(img * 255).astype(np.uint8)
        else:
            converted_gaf[i, 0] = np.zeros((128, 128), dtype=np.uint8)
    
    return converted_gaf


def load_gaf_data(generated_file, original_file):
    """加载生成的和原始的GAF数据"""
    gen_data = loadmat(generated_file)
    orig_data = loadmat(original_file)
    
    gen_gaf = gen_data['GAF_cell']
    orig_gaf = orig_data['GAF_cell']
    
    # 将生成的GAF转换为uint8格式
    print("正在将生成的GAF转换为uint8 (0-255)...")
    gen_gaf = convert_to_uint8(gen_gaf)
    print("✅ 已转换为uint8 (0-255)")
    
    return gen_gaf, orig_gaf


def visualize_comparison(gen_gaf, orig_gaf, cycle_indices, save_path):
    """
    可视化生成图像与原始图像的对比
    """
    # Mondrian渐变色
    mondrian_colors = ['#4573b4', '#acd2e5', '#fff4ae', '#fcb777']
    cmap_mondrian = LinearSegmentedColormap.from_list("mondrian", mondrian_colors, N=256)
    
    n_cycles = len(cycle_indices)
    
    # 手动布局参数
    img_width = 0.28       # 增大宽度
    img_height = 0.12      # 减小高度
    col_gap = 0.005         # 列间距
    row_gap = 0.01        # 行间距
    left_margin = 0.06     # 左边距（留给标签）
    top_margin = 0.96      # 顶部边距
    bottom_margin = 0.02   # 底部边距
    cbar_height = 0.008    # colorbar高度
    cbar_gap = 0.005       # colorbar与图的间距
    label_x = 0.02         # 标签x位置
    label_fontsize = 20    # 标签字体大小
    cbar_fontsize = 18    # colorbar字体大小
    title_fontsize = 20    # 列标题字体大小
    
    # 创建figure
    fig = plt.figure(figsize=(10, 2.8*n_cycles))
    
    # 手动创建每个子图
    axes = []
    
    for row in range(n_cycles):
        row_axes = []
        for col in range(3):
            left = left_margin + col * (img_width + col_gap)
            bottom = top_margin - (row + 1) * img_height - row * row_gap
            ax = fig.add_axes([left, bottom, img_width, img_height])
            row_axes.append(ax)
        axes.append(row_axes)
    
    # 添加行标签
    for row_idx, cycle_idx in enumerate(cycle_indices):
        label_y = top_margin - (row_idx + 0.5) * img_height - row_idx * row_gap
        fig.text(label_x, label_y, f'Cycle {cycle_idx + 1}', 
                fontsize=label_fontsize, fontweight='normal',
                ha='left', va='center', rotation=90)
    
    # 添加列标题（只在第一行上方）
    column_titles = ['Generated', 'Target', 'Absolute Difference']
    for col_idx, col_title in enumerate(column_titles):
        pos = axes[0][col_idx].get_position()
        fig.text(pos.x0 + pos.width/2, top_margin + 0.005, col_title,
                fontsize=title_fontsize, fontweight='bold',
                ha='center', va='bottom')
    
    # 绘制图像
    for idx, cycle_idx in enumerate(cycle_indices):
        gen = gen_gaf[cycle_idx, 0].astype(np.float64)
        orig = orig_gaf[cycle_idx, 0].astype(np.float64)
        error = np.abs(gen - orig)
        
        # 计算当前循环的误差最大值
        max_error = error.max()
        
        # Generated（第0列）
        axes[idx][0].imshow(gen, cmap=cmap_mondrian, vmin=0, vmax=255)
        axes[idx][0].set_xticks([])
        axes[idx][0].set_yticks([])
        for spine in axes[idx][0].spines.values():
            spine.set_visible(False)
        
        # Target（第1列）
        axes[idx][1].imshow(orig, cmap=cmap_mondrian, vmin=0, vmax=255)
        axes[idx][1].set_xticks([])
        axes[idx][1].set_yticks([])
        for spine in axes[idx][1].spines.values():
            spine.set_visible(False)
        
        # Error Map（第2列）- 使用当前循环的最大值
        im = axes[idx][2].imshow(error, cmap=cmap_mondrian, vmin=0, vmax=max_error)
        axes[idx][2].set_xticks([])
        axes[idx][2].set_yticks([])
        for spine in axes[idx][2].spines.values():
            spine.set_visible(False)
        
        # 只为第三列添加colorbar在右侧（按当前循环的误差范围）
        pos = axes[idx][2].get_position()
        cbar_width = 0.015  # colorbar宽度
        cbar_left = pos.x0 + pos.width + 0.01  # colorbar左边距
        cax = fig.add_axes([cbar_left, pos.y0, cbar_width, pos.height])
        cbar = plt.colorbar(im, cax=cax, orientation='vertical')
        cbar.ax.tick_params(labelsize=cbar_fontsize)
    
    # 保存图片
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.savefig(save_path, dpi=600, bbox_inches='tight', facecolor='white')
    plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    print(f"✅ Saved comparison visualization: {save_path}")
    plt.close(fig)


def main():
    """主函数"""
    # 配置路径
    generated_dir = "./generated_data"  # 生成数据的目录
    original_dir = "./data_for_generated"  # 原始数据的目录
    output_dir = "./comparison_visualizations"  # 输出目录
    
    # 配置要对比的文件和循环
    battery_file = "025Cbattery6_02_test_sliding.mat"  # 修改为你的文件名
    generated_file = os.path.join(generated_dir, f"generated_{battery_file}")
    original_file = os.path.join(original_dir, battery_file)
    
    # 选择8个循环索引（从小到大）
    # 例如：选择第1, 50, 100, 150, 200, 250, 300, 350个循环
    # # 注意：索引从0开始，所以实际循环编号要+1
    cycle_indices = [3, 23, 42, 63, 85, 102, 116, 124]

    # cycle_indices = [132, 156, 178, 189, 201, 212, 224, 239]
    
    # 加载数据
    print(f"Loading generated data from: {generated_file}")
    print(f"Loading original data from: {original_file}")
    gen_gaf, orig_gaf = load_gaf_data(generated_file, original_file)
    
    print(f"Generated GAF shape: {gen_gaf.shape}")
    print(f"Original GAF shape: {orig_gaf.shape}")
    
    # 自动选择均匀分布的8个循环（如果不想手动指定）
    total_cycles = min(gen_gaf.shape[0], orig_gaf.shape[0])
    # cycle_indices = np.linspace(0, total_cycles - 1, 8, dtype=int).tolist()
    print(f"Selected cycle indices: {[i+1 for i in cycle_indices]}")  # 打印实际循环编号
    
    # 生成可视化
    save_path = os.path.join(output_dir, f"{battery_file.replace('.mat', '')}_comparison.png")
    visualize_comparison(gen_gaf, orig_gaf, cycle_indices, save_path)
    
    print("Visualization completed!")


if __name__ == "__main__":
    main()