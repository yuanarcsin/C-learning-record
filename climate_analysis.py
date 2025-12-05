# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# 解决控制台中文输出问题（Windows）
if sys.platform == 'win32':
    os.system('chcp 65001 > nul')
sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf8', buffering=1)

print("=== 气候数据分析开始 ===")

# 1. 数据读取与预处理
print("\n1. 读取Excel数据...")

excel_file = r"D:\yuan\Documents\excel\climate_py_work\data.xlsx"

try:
    df = pd.read_excel(excel_file)
    print("数据读取成功！")
except Exception as e:
    print(f"读取文件失败: {e}")
    print("请检查文件路径是否正确")
    exit()

print(f"\n数据形状: {df.shape}")
print(f"数据列名: {list(df.columns)}")

print("\n数据完整性检查：")
print("缺失值统计：")
print(df.isnull().sum())

print("\n数据描述统计：")
print(df.describe())

# 检查异常值
print(f"\n气温范围: {df['平均气温'].min():.1f}℃ 到 {df['平均气温'].max():.1f}℃")
if df['平均气温'].min() < -50 or df['平均气温'].max() > 50:
    print("警告：发现可能的异常气温值！")
else:
    print("气温数据在合理范围内")

# 添加季节列
def get_season(month):
    if month in [3, 4, 5]:
        return '春季'
    elif month in [6, 7, 8]:
        return '夏季'
    elif month in [9, 10, 11]:
        return '秋季'
    else:
        return '冬季'

df['季节'] = df['月份'].apply(get_season)

print("\n添加季节列后的前5行数据：")
print(df.head())

# 2. 数据统计分析
print("\n2. 数据统计分析...")

# 年平均气温
annual_avg = df.groupby('年份')['平均气温'].mean().sort_index()
print("\n年平均气温（按年份升序）：")
for year, temp in annual_avg.items():
    print(f"{year}年: {temp:.2f}℃")

# 季节平均气温
seasonal_avg = df.groupby('季节')['平均气温'].mean()
season_order = ['春季', '夏季', '秋季', '冬季']
print("\n季节平均气温：")
for season in season_order:
    print(f"{season}: {seasonal_avg[season]:.2f}℃")

# 3. 数据可视化
print("\n3. 生成图表...")

# 先设置图表样式，再设置中文字体（避免样式覆盖字体配置）
plt.style.use('seaborn-v0_8')
# 解决中文显示+负号显示问题
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']  # 优先中文字体
plt.rcParams['axes.unicode_minus'] = False

fig = plt.figure(figsize=(14, 10))

# 3.1 年平均气温折线图
plt.subplot(2, 1, 1)
years = annual_avg.index
temps = annual_avg.values

# 主折线（年平均气温实际值）
line = plt.plot(years, temps, marker='o', linewidth=2.5, markersize=8,
                markerfacecolor='red', markeredgecolor='darkred',
                color='#2E86AB', label='年平均气温')

# 趋势线（气温变化趋势）
z = np.polyfit(years, temps, 1)
p = np.poly1d(z)
trend_line = plt.plot(years, p(years), "r--", linewidth=2, alpha=0.8,
                      label=f'趋势线 (斜率: {z[0]:.3f})')

# 美化+添加线条作用标注
plt.title('2010-2020年年平均气温变化趋势', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('年份', fontsize=12)
plt.ylabel('气温（℃）', fontsize=12)
plt.grid(True, alpha=0.3, linestyle='--')
plt.legend(fontsize=11)

# 标注线条作用（图表左上角添加说明框）
plt.text(0.02, 0.95, '线条说明：\n• 蓝色实线：各年份实际年平均气温\n• 红色虚线：气温变化趋势（斜率代表年变化率）',
         transform=plt.gca().transAxes, fontsize=10,
         bbox=dict(facecolor='white', alpha=0.9, edgecolor='lightgray', boxstyle='round,pad=0.5'))

plt.xticks(years, rotation=45)
plt.ylim(11.5, 13.6)

# 添加数值标签
for i, (year, temp) in enumerate(zip(years, temps)):
    plt.annotate(f'{temp:.1f}℃', (year, temp),
                 textcoords="offset points", xytext=(0,10),
                 ha='center', fontsize=9, fontweight='bold')

# 3.2 季节箱线图
plt.subplot(2, 1, 2)
season_data = [df[df['季节'] == season]['平均气温'] for season in season_order]

box_plot = plt.boxplot(season_data, labels=season_order, patch_artist=True,
                       widths=0.6, showmeans=True, meanline=True,
                       meanprops=dict(linestyle='--', linewidth=2, color='red'))

# 颜色设置
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
for patch, color in zip(box_plot['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

for element in ['whiskers', 'caps', 'medians']:
    for line in box_plot[element]:
        line.set_color('black')
        line.set_linewidth(1.5)

for flier in box_plot['fliers']:
    flier.set(marker='o', color='red', alpha=0.5)

# 标题与标签
plt.title('四季气温分布箱线图', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('季节', fontsize=12)
plt.ylabel('气温（℃）', fontsize=12)
plt.grid(True, alpha=0.3, linestyle='--')

# 添加中位数和均值标签
for i, (data, color) in enumerate(zip(season_data, colors)):
    median = np.median(data)
    mean = np.mean(data)
    plt.text(i+1, median, f'中位: {median:.1f}℃', ha='center', va='bottom', fontsize=9, fontweight='bold')
    plt.text(i+1, mean, f'平均: {mean:.1f}℃', ha='center', va='top', fontsize=9, fontweight='bold')

plt.tight_layout(pad=3.0)

# 保存高清图
plt.savefig('气温分析结果_最终版.png', dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.show()

# 4. 结果分析
print("\n4. 结果分析：")

trend_slope = z[0]
if trend_slope > 0.01:
    trend_direction = "明显上升"
elif trend_slope > 0:
    trend_direction = "轻微上升"
elif trend_slope < -0.01:
    trend_direction = "明显下降"
elif trend_slope < 0:
    trend_direction = "轻微下降"
else:
    trend_direction = "基本平稳"

print(f"气温变化趋势：2010-2020年间，该地区年平均气温呈现{trend_direction}趋势")
print(f"趋势斜率: {trend_slope:.4f} (每年变化量)")

season_std = df.groupby('季节')['平均气温'].std()
most_variable_season = season_std.idxmax()
least_variable_season = season_std.idxmin()

print(f"\n季节特征分析：")
print(f"- {most_variable_season}的气温波动最大 (标准差: {season_std[most_variable_season]:.2f}℃)")
print(f"- {least_variable_season}的气温波动最小 (标准差: {season_std[least_variable_season]:.2f}℃)")

print("\n各季节气温波动情况：")
for season in season_order:
    data = df[df['季节'] == season]['平均气温']
    print(f"- {season}: 平均{data.mean():.1f}℃, 波动{data.std():.2f}℃, 范围{data.min():.1f}~{data.max():.1f}℃")

total_change = annual_avg.iloc[-1] - annual_avg.iloc[0]
print(f"\n总体变化：2010年({annual_avg.iloc[0]:.1f}℃)到2020年({annual_avg.iloc[-1]:.1f}℃)，气温变化{total_change:+.1f}℃")

print("\n=== 分析完成！图表已保存为 '气温分析结果_最终版.png' ===")

# 输出总结表格
summary_table = pd.DataFrame({
    '指标': ['年均气温变化趋势', '趋势斜率', '最大波动季节', '最小波动季节', '总变化'],
    '结果': [
        trend_direction,
        f"{trend_slope:.4f}",
        most_variable_season,
        least_variable_season,
        f"{total_change:+.1f}℃"
    ]
})

print("\n=== 总结报告 ===")
print(summary_table.to_string(index=False))