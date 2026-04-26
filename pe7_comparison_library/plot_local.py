#!/usr/bin/env python3
"""
Local plotting script for scenario comparison
Usage: python plot_local.py
"""
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 설정
DATA_DIR = Path('./data')  # 이 파일과 같은 디렉토리에 data 폴더 있어야 함
plt.rcParams['font.size'] = 8

def load_data():
    """데이터 로드"""
    # Targetable results 로드
    with open(DATA_DIR / 'targetable_results.json', 'r') as f:
        targetable_results = json.load(f)
    
    # Variant efficiency 로드
    efficiency_df = pd.read_csv(DATA_DIR / 'variant_efficiency.csv')
    
    # Comparison df 로드
    comparison_df = pd.read_csv(DATA_DIR / 'comparison_df.csv')
    
    return targetable_results, efficiency_df, comparison_df

def plot_targetable_comparison(targetable_results):
    """Targetable edit 비율 비교 barplot"""
    scenario_labels = {
        'DP_base': 'DP-base\n(NGG)',
        'DP_DLD1_PE2max': 'DP-PE2max\n(NGG)',
        'DP7_NGG': 'DP7\n(NGG)',
        'DP7_NGG_NRCH': 'DP7\n(NGG+NRCH)',
        'DP7_NGG_NRTH': 'DP7\n(NGG+NRTH)',
        'DP7_Full': 'DP7\n(All PAMs)'
    }
    
    scenario_colors = {
        'DP_base': '#D3D3D3',          # Light gray
        'DP_DLD1_PE2max': '#A9A9A9',  # Dark gray
        'DP7_NGG': '#E64B35',          # Red (DN)
        'DP7_NGG_NRCH': '#4DBBD5',    # Blue (DC)
        'DP7_NGG_NRTH': '#00A087',    # Green (DT)
        'DP7_Full': '#3C5488'          # Dark blue
    }
    
    scenarios = ['DP_base', 'DP_DLD1_PE2max', 'DP7_NGG', 'DP7_NGG_NRCH', 'DP7_NGG_NRTH', 'DP7_Full']
    
    # Barplot 생성
    fig, ax = plt.subplots(figsize=(3.5, 3))
    
    # X축 위치
    x = np.arange(len(scenarios))
    width = 0.35
    
    # 데이터 준비
    ratios_10 = [targetable_results[s]['ratio_10'] for s in scenarios]
    ratios_50 = [targetable_results[s]['ratio_50'] for s in scenarios]
    
    # Bar 그리기 (10% 먼저, 배경)
    bars1 = ax.bar(x - width/2, ratios_10, width, label='> 10%', color='#F0F0F0', edgecolor='gray', linewidth=1.5)
    
    # Bar 그리기 (50%, 위 레이어)
    bars2 = ax.bar(x + width/2, ratios_50, width, label='> 50%',
                  color=[scenario_colors[s] for s in scenarios],
                  edgecolor='black', linewidth=1)
    
    # 10% 수치 표시
    for bar, ratio in zip(bars1, ratios_10):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
               f'{ratio:.1f}%',
               ha='center', va='bottom', fontsize=7, fontweight='bold', color='gray')
    
    # 50% 수치 표시
    for bar, ratio in zip(bars2, ratios_50):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
               f'{ratio:.1f}%',
               ha='center', va='bottom', fontsize=7, fontweight='bold', color='black')
    
    # X축 설정
    ax.set_xticks(x)
    ax.set_xticklabels([scenario_labels[s] for s in scenarios], fontsize=7, rotation=30, ha='right')
    
    # Y축 설정
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_yticklabels(['0', '20', '40', '60', '80', '100'], fontsize=8)
    ax.set_ylabel('Targetable Edits (%)', fontsize=8, fontweight='bold')
    ax.set_xlabel('', fontsize=8, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.set_axisbelow(True)
    sns.despine(ax=ax, top=True, right=True)
    
    # 범례
    ax.legend(fontsize=6, loc='upper left', framealpha=0.95)
    
    plt.title('Targetable Edits Comparison (Mean Efficiency > 10% and > 50%)', fontsize=10, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(DATA_DIR / 'targetable_edits_barplot.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {DATA_DIR / 'targetable_edits_barplot.png'}")
    plt.show()

def plot_efficiency_violin(comparison_df):
    """Efficiency violin plot"""
    scenarios = ['DP_base', 'DP_DLD1_PE2max', 'DP7_NGG', 'DP7_NGG_NRCH', 'DP7_NGG_NRTH', 'DP7_Full']
    
    scenario_labels = {
        'DP_base': 'DP-base\n(NGG)',
        'DP_DLD1_PE2max': 'DP-PE2max\n(NGG)',
        'DP7_NGG': 'DP7\n(NGG)',
        'DP7_NGG_NRCH': 'DP7\n(NGG+NRCH)',
        'DP7_NGG_NRTH': 'DP7\n(NGG+NRTH)',
        'DP7_Full': 'DP7\n(All PAMs)'
    }
    
    # Prepare data for violin plot
    plot_data = []
    for scenario in scenarios:
        scenario_df = comparison_df[comparison_df['scenario'] == scenario]
        for idx, row in scenario_df.iterrows():
            # DP-base/PE2max는 _eval_score (DP7_NGG) 사용, 나머지는 자체 예측치
            if scenario in ['DP_base', 'DP_DLD1_PE2max']:
                pred_score = row['_eval_score']
            else:
                pred_score = row[scenario]
            plot_data.append({'Scenario': scenario, 'Prediction': pred_score})
    
    plot_df = pd.DataFrame(plot_data)
    
    fig, ax = plt.subplots(figsize=(3.5, 3))
    
    scenario_colors = {
        'DP_base': '#D3D3D3',
        'DP_DLD1_PE2max': '#A9A9A9',
        'DP7_NGG': '#E64B35',
        'DP7_NGG_NRCH': '#4DBBD5',
        'DP7_NGG_NRTH': '#00A087',
        'DP7_Full': '#3C5488'
    }
    palette = [scenario_colors[s] for s in scenarios]
    
    sns.violinplot(data=plot_df, x='Scenario', y='Prediction', ax=ax,
                  palette=palette, order=scenarios)
    
    ax.set_xticklabels([scenario_labels.get(s, s) for s in scenarios], fontsize=7, rotation=30, ha='right')
    ax.set_ylabel('Predicted PE Efficiency (%)', fontsize=8, fontweight='bold')
    ax.set_xlabel('', fontsize=8, fontweight='bold')
    ax.set_ylim(0, 100)
    sns.despine(ax=ax, top=True, right=True)
    
    plt.tight_layout()
    plt.savefig(DATA_DIR / 'scenario_comparison_violin.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {DATA_DIR / 'scenario_comparison_violin.png'}")
    plt.show()

if __name__ == '__main__':
    print("Loading data...")
    targetable_results, efficiency_df, comparison_df = load_data()
    
    print(f"Targetable results scenarios: {list(targetable_results.keys())}")
    print(f"Comparison df shape: {comparison_df.shape}")
    
    print("\nGenerating plots...")
    plot_targetable_comparison(targetable_results)
    plot_efficiency_violin(comparison_df)
    
    print("\n✓ Done!")
