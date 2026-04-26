"""
DeepPrime vs DeepPrime7 scenario comparison

Analysis:
- Compute model predictions for each scenario
- Select top 10 pegRNAs per variant (pegRNA must be available in all scenarios)
- Compare prediction score distributions (violin/box plot)

Usage:
    python eval/clinvar/scenario_comparison.py

Environment variables:
    DEEPPRIME_ROOT: Path to the DeepPrime repository root (default: ../DeepPrime relative to this repo)
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import warnings
import json
from multiprocessing import Pool, cpu_count, set_start_method
import torch

# Set multiprocessing start method to avoid CUDA issues
try:
    set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set

warnings.filterwarnings('ignore')

# --- Path configuration (portable, no hardcoded absolute paths) ---
SCRIPT_DIR  = Path(__file__).resolve().parent          # eval/clinvar/
REPO_ROOT   = SCRIPT_DIR.parent.parent                 # DeepPrime7 repo root

# DeepPrime (original) repo: override via DEEPPRIME_ROOT env var if not a sibling directory
_dp_default = REPO_ROOT.parent / 'DeepPrime'
DEEPPRIME_ROOT = Path(os.environ.get('DEEPPRIME_ROOT', _dp_default))

if not DEEPPRIME_ROOT.exists():
    raise EnvironmentError(
        f"DeepPrime repository not found at '{DEEPPRIME_ROOT}'.\n"
        "Set the DEEPPRIME_ROOT environment variable to its location:\n"
        "  export DEEPPRIME_ROOT=/path/to/DeepPrime"
    )

sys.path.insert(0, str(SCRIPT_DIR.parent))  # add eval/ so run_deepprime_original is importable
sys.path.insert(0, str(DEEPPRIME_ROOT))

import run_deepprime_original as rdo

# --- Model paths ---
DP_MODELS = {
    'DP_base':        DEEPPRIME_ROOT / 'models' / 'ontarget' / 'final',
    'DP_DLD1_PE2max': DEEPPRIME_ROOT / 'models' / 'ontarget_variants' / 'DP_variant_DLD1_PE2max_Opti_221114',
}

DP7_MODELS = {
    'DP7':        REPO_ROOT / 'models' / 'deepprime7.pth',
    'DP7_scaler': REPO_ROOT / 'models' / 'deepprime7_scaler.pth',
}


def _tensorize_chunk(args):
    """Helper function for multiprocessing - tensorize a chunk of data (CPU only)"""
    df_chunk, _ = args
    # Process on CPU to avoid CUDA fork issues
    device = torch.device('cpu')
    try:
        g_tensor, x_tensor = rdo._tensorize_inputs(df_chunk, device)
        return (g_tensor, x_tensor)  # Keep on CPU for now
    except Exception as e:
        print(f"Error in chunk processing: {e}")
        raise


class ScenarioComparison:
    """Compare 6 pegRNA selection scenarios."""

    def __init__(self, features_dir=None, device='cuda:0'):
        if features_dir is None:
            features_dir = SCRIPT_DIR / 'deepprime_features'
        self.features_dir = Path(features_dir)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.results_dir = SCRIPT_DIR / 'scenario_results'
        self.results_dir.mkdir(parents=True, exist_ok=True)

        print(f"Using device: {self.device}")

        # Load all pegRNA features
        self.load_all_features()

    def load_all_features(self):
        """Load and filter all chunk files."""
        print("Loading pegRNA features from chunks...")

        chunks = []
        # File pattern: deepprime_pegrna_features_000001-000100.csv
        chunk_files = sorted(self.features_dir.glob('deepprime_pegrna_features_*.csv'))

        for chunk_file in tqdm(chunk_files):
            df = pd.read_csv(chunk_file)
            chunks.append(df)

        self.df = pd.concat(chunks, ignore_index=True)
        print(f"Loaded {len(self.df)} pegRNAs from {len(chunk_files)} chunks")
        print(f"Unique variants: {self.df['variant_id'].nunique()}")

    def _save_checkpoint(self, stage_name):
        """Save intermediate result checkpoint."""
        checkpoint_file = self.results_dir / f'checkpoint_{stage_name}.pkl'
        import pickle
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(self.df_pred, f)
        print(f"✓ Checkpoint saved: {checkpoint_file}")

    def _load_checkpoint(self, stage_name):
        """Load from checkpoint."""
        checkpoint_file = self.results_dir / f'checkpoint_{stage_name}.pkl'
        if checkpoint_file.exists():
            import pickle
            with open(checkpoint_file, 'rb') as f:
                self.df_pred = pickle.load(f)
            print(f"✓ Loaded checkpoint: {checkpoint_file}")
            return True
        return False

    def prepare_for_prediction(self):
        """Prepare data for prediction."""
        print("Preparing data for prediction...")

        self.df_pred = self.df.copy()

        # Generate required columns for DeepPrime prediction
        # Use Target and Spacer sequences to create WT74_On and Edited74_On
        self.df_pred['WT74_On'] = self.df_pred['Target'].str.slice(0, 74).fillna('')
        self.df_pred['Edited74_On'] = self.df_pred['Target'].str.slice(0, 74).fillna('')

        # Rename columns to match DeepPrime schema
        column_map = {
            "PBS_len": "PBSlen",
            "RTT_len": "RTlen",
            "RT-PBS_len": "RT-PBSlen",
            "Tm1_PBS": "Tm1",
            "Tm2_RTT_cTarget_sameLength": "Tm2",
            "Tm3_RTT_cTarget_replaced": "Tm2new",
            "Tm4_cDNA_PAM-oppositeTarget": "Tm3",
            "Tm5_RTT_cDNA": "Tm4",
            "deltaTm_Tm4-Tm2": "TmD",
            "GC_count_PBS": "nGCcnt1",
            "GC_count_RTT": "nGCcnt2",
            "GC_count_RT-PBS": "nGCcnt3",
            "GC_contents_PBS": "fGCcont1",
            "GC_contents_RTT": "fGCcont2",
            "GC_contents_RT-PBS": "fGCcont3",
            "MFE_RT-PBS-polyT": "MFE3",
            "MFE_Spacer": "MFE4",
        }

        # Only rename columns that exist
        rename_cols = {k: v for k, v in column_map.items() if k in self.df_pred.columns}
        self.df_pred = self.df_pred.rename(columns=rename_cols)

        print(f"Data prepared: {len(self.df_pred)} pegRNAs")
        print(f"Renamed {len(rename_cols)} columns for DeepPrime schema")
        print(f"Generated WT74_On and Edited74_On columns\n")

    def get_dp_predictions(self):
        """Run DeepPrime ensemble predictions."""
        print("=" * 80)
        print("Computing DeepPrime predictions...")
        print("=" * 80)

        # Load models
        base_models = sorted((DP_MODELS['DP_base']).glob('model_*.pt'))
        dld1_models = sorted((DP_MODELS['DP_DLD1_PE2max']).glob('final_model_*.pt'))

        print(f"Loaded {len(base_models)} base models")
        print(f"Loaded {len(dld1_models)} DLD1 PE2max models\n")

        # Tensorize inputs with multiprocessing optimization
        print("Tensorizing inputs (with multiprocessing)...")
        g_tensor, x_tensor = self._tensorize_inputs_parallel(self.df_pred)

        # Get predictions
        print("Getting DP_base predictions...")
        pred_dp_base = rdo.ensemble_predict(base_models, g_tensor, x_tensor,
                                            self.device, batch_size=1024)

        print("Getting DP_DLD1_PE2max predictions...")
        pred_dp_dld1 = rdo.ensemble_predict(dld1_models, g_tensor, x_tensor,
                                            self.device, batch_size=1024)

        self.df_pred['DP_base'] = pred_dp_base
        self.df_pred['DP_DLD1_PE2max'] = pred_dp_dld1

        print("DeepPrime predictions complete\n")

    def _tensorize_inputs_parallel(self, df, chunk_size=50000):
        """Tensorize inputs using multiprocessing for speed"""
        print(f"Processing {len(df)} samples with {cpu_count()} processes (CPU)...")

        # Split dataframe into chunks
        n_chunks = max(1, len(df) // chunk_size)
        chunks = [df.iloc[i::n_chunks].reset_index(drop=True) for i in range(n_chunks)]

        print(f"Processing {len(chunks)} chunks...\n")

        # Use multiprocessing pool to process chunks on CPU
        with Pool(processes=min(cpu_count() - 1, len(chunks))) as pool:
            args_list = [(chunk, 'cpu') for chunk in chunks]
            results = list(tqdm(pool.imap_unordered(_tensorize_chunk, args_list),
                               total=len(chunks), desc="Tensorizing (CPU)"))

        # Concatenate results
        g_tensors = [r[0] for r in results]
        x_tensors = [r[1] for r in results]

        g_tensor = torch.cat(g_tensors, dim=0)
        x_tensor = torch.cat(x_tensors, dim=0)

        # Move to GPU
        print(f"\nMoving tensors to {self.device}...")
        g_tensor = g_tensor.to(self.device)
        x_tensor = x_tensor.to(self.device)

        print(f"✓ Tensorized: g_tensor {g_tensor.shape}, x_tensor {x_tensor.shape}\n")

        return g_tensor, x_tensor

    def get_dp7_predictions(self):
        """Run DeepPrime7 predictions."""
        print("=" * 80)
        print("Computing DeepPrime7 predictions...")
        print("=" * 80)

        # Load DP7 model
        dp7_input = rdo._dp_dataset_to_dp7(self.df_pred)
        _, dp7_pred_map = rdo.deepprime7_predict(
            dp7_input,
            self.device,
            DP7_MODELS['DP7'],
            DP7_MODELS['DP7_scaler'],
            batch_size=1024
        )

        # Save raw per-endpoint predictions
        self.df_pred['DP7_DN'] = dp7_pred_map.get('DN_M_EF')
        self.df_pred['DP7_DC'] = dp7_pred_map.get('DC_M_EF')
        self.df_pred['DP7_DT'] = dp7_pred_map.get('DT_M_EF')
        # AN endpoint may be available but not used for these scenarios
        if 'AN_M_EF' in dp7_pred_map:
            self.df_pred['DP7_AN'] = dp7_pred_map.get('AN_M_EF')

        # Precompute scenario-specific DP7 combined scores (use max across endpoints)
        # DP7_NGG: DN only
        self.df_pred['DP7_NGG'] = self.df_pred['DP7_DN']
        # DP7_NGG_NRCH: best of DN or DC
        self.df_pred['DP7_NGG_NRCH'] = self.df_pred[['DP7_DN', 'DP7_DC']].max(axis=1)
        # DP7_NGG_NRTH: best of DN or DT
        self.df_pred['DP7_NGG_NRTH'] = self.df_pred[['DP7_DN', 'DP7_DT']].max(axis=1)
        # DP7_Full: best of DN, DC, or DT
        self.df_pred['DP7_Full'] = self.df_pred[['DP7_DN', 'DP7_DC', 'DP7_DT']].max(axis=1)

        print("DeepPrime7 predictions complete\n")

    def select_top10_per_variant(self):
        """Select top 10 pegRNAs per variant for each scenario.

        Special logic for DP_base and DP_DLD1_PE2max: pegRNAs are ranked by their
        own model scores, but prediction efficiency is evaluated using DP7_NGG scores
        for a fair comparison across all scenarios.
        """
        print("=" * 80)
        print("Selecting top 10 pegRNAs per variant")
        print("=" * 80)

        scenarios = ['DP_base', 'DP_DLD1_PE2max', 'DP7_NGG',
                    'DP7_NGG_NRCH', 'DP7_NGG_NRTH', 'DP7_Full']

        results = []
        variant_info = {}

        for variant_id in tqdm(self.df_pred['variant_id'].unique()):
            var_df = self.df_pred[self.df_pred['variant_id'] == variant_id].copy()

            # Count available pegRNAs per scenario
            available = {}
            for scenario in scenarios:
                if scenario == 'DP_base':
                    # NGG pegRNAs only
                    available[scenario] = (var_df['PAM_NGG'] == 1).sum()
                elif scenario == 'DP_DLD1_PE2max':
                    # NGG pegRNAs only
                    available[scenario] = (var_df['PAM_NGG'] == 1).sum()
                elif scenario == 'DP7_NGG':
                    # NGG only
                    available[scenario] = (var_df['PAM_NGG'] == 1).sum()
                elif scenario == 'DP7_NGG_NRCH':
                    # NGG + NRCH
                    available[scenario] = ((var_df['PAM_NGG'] == 1) | (var_df['PAM_NRCH'] == 1)).sum()
                elif scenario == 'DP7_NGG_NRTH':
                    # NGG + NRTH
                    available[scenario] = ((var_df['PAM_NGG'] == 1) | (var_df['PAM_NRTH'] == 1)).sum()
                else:  # DP7_Full: all PAMs
                    available[scenario] = len(var_df)

            # Include variant if at least one scenario has at least 1 pegRNA
            if any(count >= 1 for count in available.values()):
                variant_info[variant_id] = available

                # Select top 10 per scenario
                for scenario in scenarios:
                    if scenario == 'DP_base':
                        # Ranked by DP_base; evaluated by DP7_NGG for cross-scenario fairness
                        ngg_df = var_df[var_df['PAM_NGG'] == 1].nlargest(10, 'DP_base')
                        top10 = ngg_df.copy()
                        top10['_eval_score'] = ngg_df['DP7_NGG'].values
                        top10['scenario'] = scenario

                    elif scenario == 'DP_DLD1_PE2max':
                        # Ranked by DP_DLD1_PE2max; evaluated by DP7_NGG for cross-scenario fairness
                        ngg_df = var_df[var_df['PAM_NGG'] == 1].nlargest(10, 'DP_DLD1_PE2max')
                        top10 = ngg_df.copy()
                        top10['_eval_score'] = ngg_df['DP7_NGG'].values
                        top10['scenario'] = scenario

                    elif scenario == 'DP7_NGG':
                        top10 = var_df[var_df['PAM_NGG'] == 1].nlargest(10, scenario).copy()
                        top10['_eval_score'] = top10[scenario]
                        top10['scenario'] = scenario

                    elif scenario == 'DP7_NGG_NRCH':
                        valid_df = var_df[(var_df['PAM_NGG'] == 1) | (var_df['PAM_NRCH'] == 1)]
                        top10 = valid_df.nlargest(10, scenario).copy()
                        top10['_eval_score'] = top10[scenario]
                        top10['scenario'] = scenario

                    elif scenario == 'DP7_NGG_NRTH':
                        valid_df = var_df[(var_df['PAM_NGG'] == 1) | (var_df['PAM_NRTH'] == 1)]
                        top10 = valid_df.nlargest(10, scenario).copy()
                        top10['_eval_score'] = top10[scenario]
                        top10['scenario'] = scenario

                    else:  # DP7_Full
                        top10 = var_df.nlargest(10, scenario).copy()
                        top10['_eval_score'] = top10[scenario]
                        top10['scenario'] = scenario

                    results.append(top10)

        self.comparison_df = pd.concat(results, ignore_index=True)

        print(f"\n✓ Selected top 10 pegRNAs per variant")
        print(f"  Total pegRNA records: {len(self.comparison_df)}")
        print(f"  Variants included: {len(variant_info)}")
        print(f"  Average pegRNA per variant per scenario: {len(self.comparison_df) / (len(variant_info) * 6):.1f}")
        print()

    def generate_plots(self):
        """Generate comparison violin plots.

        DP_base and DP_DLD1_PE2max are plotted using DP7_NGG scores (_eval_score)
        for a fair comparison with DP7 scenarios.
        """
        print("=" * 80)
        print("Generating comparison plots")
        print("=" * 80)

        scenarios = ['DP_base', 'DP_DLD1_PE2max', 'DP7_NGG',
                    'DP7_NGG_NRCH', 'DP7_NGG_NRTH', 'DP7_Full']

        # Prepare data for violin plot
        plot_data = []
        for scenario in scenarios:
            scenario_df = self.comparison_df[self.comparison_df['scenario'] == scenario]
            for idx, row in scenario_df.iterrows():
                # DP_base/PE2max use _eval_score (DP7_NGG); others use their own predictions
                if scenario in ['DP_base', 'DP_DLD1_PE2max']:
                    pred_score = row['_eval_score']
                else:
                    pred_score = row[scenario]
                plot_data.append({'Scenario': scenario, 'Prediction': pred_score})

        plot_df = pd.DataFrame(plot_data)

        # Create boxplot
        fig, ax = plt.subplots(figsize=(4, 3))

        scenario_labels = {
            'DP_base': 'DP-base\n(NGG)',
            'DP_DLD1_PE2max': 'DP-FT PE2max\n(NGG)',
            'DP7_NGG': 'DP7\n(NGG)',
            'DP7_NGG_NRCH': 'DP7\n(NGG+NRCH)',
            'DP7_NGG_NRTH': 'DP7\n(NGG+NRTH)',
            'DP7_Full': 'DP7\n(NGG+NRCH+NRTH)'
        }

        # Color palette: DP Base/DLD1 = light gray, DP7 NGG/DC/DT endpoints, DP7 Full = dark gray
        color_palette = {
            'DP_base': '#D3D3D3',          # Light gray
            'DP_DLD1_PE2max': '#A9A9A9',   # Dark gray
            'DP7_NGG': '#E64B35',          # Red (DN)
            'DP7_NGG_NRCH': '#4DBBD5',    # Blue (DC)
            'DP7_NGG_NRTH': '#00A087',    # Green (DT)
            'DP7_Full': '#3C5488'          # Dark gray
        }
        palette = [color_palette[s] for s in scenarios]

        sns.violinplot(data=plot_df, x='Scenario', y='Prediction', ax=ax,
                      palette=palette, order=scenarios)

        ax.set_xticklabels([scenario_labels.get(s, s) for s in scenarios], fontsize=7,
                          rotation=30, ha='right')
        ax.set_yticks(ax.get_yticks())
        ax.set_yticklabels([f'{int(y)}' for y in ax.get_yticks()], fontsize=8)
        ax.set_ylabel('Predicted PE Efficiency (%)', fontsize=8, fontweight='bold')
        ax.set_xlabel('', fontsize=8, fontweight='bold')
        ax.set_ylim(0, 100)
        sns.despine(ax=ax, top=True, right=True)

        plt.tight_layout()
        plot_file = self.results_dir / 'scenario_comparison_boxplot.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {plot_file}\n")

        return plot_df

    def generate_statistics_report(self):
        """Generate statistics report.

        DP_base and DP_DLD1_PE2max use DP7_NGG scores (_eval_score) for comparison.
        """
        print("=" * 80)
        print("Generating Statistics Report")
        print("=" * 80)

        scenarios = ['DP_base', 'DP_DLD1_PE2max', 'DP7_NGG',
                    'DP7_NGG_NRCH', 'DP7_NGG_NRTH', 'DP7_Full']

        report = {}
        for scenario in scenarios:
            scenario_df = self.comparison_df[self.comparison_df['scenario'] == scenario]
            # DP_base/PE2max use _eval_score (DP7_NGG); others use their own predictions
            if scenario in ['DP_base', 'DP_DLD1_PE2max']:
                scenario_preds = scenario_df['_eval_score'].values
            else:
                scenario_preds = scenario_df[scenario].values

            report[scenario] = {
                'mean': float(scenario_preds.mean()),
                'median': float(np.median(scenario_preds)),
                'std': float(scenario_preds.std()),
                'min': float(scenario_preds.min()),
                'max': float(scenario_preds.max()),
                'q25': float(np.percentile(scenario_preds, 25)),
                'q75': float(np.percentile(scenario_preds, 75)),
                'count': int(len(scenario_preds))
            }

        # Print report
        print("\nPrediction Statistics:")
        print("-" * 80)
        for scenario in scenarios:
            stats = report[scenario]
            print(f"\n{scenario}:")
            print(f"  Count: {stats['count']}")
            print(f"  Mean: {stats['mean']:.4f}")
            print(f"  Median: {stats['median']:.4f}")
            print(f"  Std: {stats['std']:.4f}")
            print(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
            print(f"  IQR: [{stats['q25']:.4f}, {stats['q75']:.4f}]")

        # Save to JSON
        report_file = self.results_dir / 'scenario_statistics.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n✓ Saved: {report_file}\n")

        return report

    def analyze_targetable_edits(self):
        """Analyze targetable edits: variants with top-10 pegRNA mean efficiency > 10% and > 50%."""
        print("=" * 80)
        print("Analyzing Targetable Edits (Mean Efficiency > 10% and > 50%)")
        print("=" * 80)

        scenarios = ['DP_base', 'DP_DLD1_PE2max', 'DP7_NGG', 'DP7_NGG_NRCH', 'DP7_NGG_NRTH', 'DP7_Full']

        # Compute mean efficiency of top 10 pegRNAs per variant per scenario
        variant_efficiency = {}
        for scenario in scenarios:
            scenario_df = self.comparison_df[self.comparison_df['scenario'] == scenario]
            var_means = scenario_df.groupby('variant_id')['_eval_score'].mean()
            variant_efficiency[scenario] = var_means

        # Compute targetable edit ratios (mean > 10% and > 50%)
        targetable_results = {}
        for scenario in scenarios:
            var_means = variant_efficiency[scenario]
            num_targetable_10 = (var_means > 10).sum()
            num_targetable_50 = (var_means > 50).sum()
            total_variants = 1000  # total number of variants in the ClinVar set
            ratio_10 = (num_targetable_10 / total_variants * 100)
            ratio_50 = (num_targetable_50 / total_variants * 100)

            targetable_results[scenario] = {
                'num_targetable_10': int(num_targetable_10),
                'num_targetable_50': int(num_targetable_50),
                'total_variants': int(total_variants),
                'ratio_10': float(ratio_10),
                'ratio_50': float(ratio_50)
            }

            print(f"\n{scenario}:")
            print(f"  Variants with mean efficiency > 10%: {num_targetable_10}/{total_variants} ({ratio_10:.2f}%)")
            print(f"  Variants with mean efficiency > 50%: {num_targetable_50}/{total_variants} ({ratio_50:.2f}%)")

        # Generate barplot
        self._plot_targetable_comparison(targetable_results, scenarios)

        # Save detailed results
        detail_file = self.results_dir / 'targetable_edits_detail.csv'
        detail_data = []
        for scenario in scenarios:
            var_means = variant_efficiency[scenario]
            for var_id, mean_eff in var_means.items():
                detail_data.append({
                    'variant_id': var_id,
                    'scenario': scenario,
                    'mean_efficiency': mean_eff,
                    'is_targetable_10': mean_eff > 10,
                    'is_targetable_50': mean_eff > 50
                })

        detail_df = pd.DataFrame(detail_data)
        detail_df.to_csv(detail_file, index=False)
        print(f"\n✓ Saved detailed results: {detail_file}")

        # Save JSON summary
        summary_file = self.results_dir / 'targetable_edits_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(targetable_results, f, indent=2)
        print(f"✓ Saved summary: {summary_file}\n")

        return targetable_results, variant_efficiency

    def _plot_targetable_comparison(self, targetable_results, scenarios):
        """Targetable edit ratio barplot (10% threshold: light gray; 50% threshold: scenario color)."""
        scenario_labels = {
            'DP_base': 'DP-base\n(NGG)',
            'DP_DLD1_PE2max': 'DP-FT PE2max\n(NGG)',
            'DP7_NGG': 'DP7\n(NGG)',
            'DP7_NGG_NRCH': 'DP7\n(NGG+NRCH)',
            'DP7_NGG_NRTH': 'DP7\n(NGG+NRTH)',
            'DP7_Full': 'DP7\n(NGG+NRCH+NRTH)'
        }

        scenario_colors = {
            'DP_base': '#D3D3D3',          # Light gray
            'DP_DLD1_PE2max': '#A9A9A9',  # Dark gray
            'DP7_NGG': '#E64B35',          # Red (DN)
            'DP7_NGG_NRCH': '#4DBBD5',    # Blue (DC)
            'DP7_NGG_NRTH': '#00A087',    # Green (DT)
            'DP7_Full': '#3C5488'          # Dark blue
        }

        # Generate barplot
        fig, ax = plt.subplots(figsize=(4, 3))

        x = np.arange(len(scenarios))
        width = 0.35

        ratios_10 = [targetable_results[s]['ratio_10'] for s in scenarios]
        ratios_50 = [targetable_results[s]['ratio_50'] for s in scenarios]

        # Draw > 10% bars (background, light gray)
        bars1 = ax.bar(x - width/2, ratios_10, width, label='> 10%', color='#F0F0F0', edgecolor='gray', linewidth=0.)

        # Draw > 50% bars (foreground, scenario color)
        bars2 = ax.bar(x + width/2, ratios_50, width, label='> 50%',
                      color=[scenario_colors[s] for s in scenarios],
                      edgecolor='black', linewidth=0.)

        # Annotate > 10% values
        for bar, ratio in zip(bars1, ratios_10):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{ratio:.1f}%',
                   ha='center', va='bottom', fontsize=7, fontweight='bold', color='gray')

        # Annotate > 50% values
        for bar, ratio in zip(bars2, ratios_50):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{ratio:.1f}%',
                   ha='center', va='bottom', fontsize=7, fontweight='bold', color='black')

        # X axis
        ax.set_xticks(x)
        ax.set_xticklabels([scenario_labels[s] for s in scenarios], fontsize=7, rotation=30, ha='right')

        # Y axis
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.set_yticklabels(['0', '20', '40', '60', '80', '100'], fontsize=8)
        ax.set_ylabel('Targetable Edits (%)', fontsize=8, fontweight='bold')
        ax.set_xlabel('', fontsize=8, fontweight='bold')
        ax.set_ylim(0, 100)
        ax.set_axisbelow(True)
        sns.despine(ax=ax, top=True, right=True)

        # Legend
        ax.legend(fontsize=6, loc='upper left', framealpha=0.95)

        plot_file = self.results_dir / 'targetable_edits_barplot.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved barplot: {plot_file}")
        plt.close()

    def _save_data_for_local_plotting(self, targetable_results, variant_efficiency):
        """Save intermediate data needed to regenerate plots locally."""
        print("\n" + "=" * 80)
        print("Saving data for local plotting")
        print("=" * 80)

        # 1. Targetable results (JSON)
        results_file = self.results_dir / 'targetable_results.json'
        with open(results_file, 'w') as f:
            json.dump(targetable_results, f, indent=2)
        print(f"✓ Saved targetable_results: {results_file}")

        # 2. Variant efficiency (CSV)
        efficiency_data = []
        for scenario in ['DP_base', 'DP_DLD1_PE2max', 'DP7_NGG', 'DP7_NGG_NRCH', 'DP7_NGG_NRTH', 'DP7_Full']:
            for var_id, efficiency in variant_efficiency[scenario].items():
                efficiency_data.append({
                    'variant_id': var_id,
                    'scenario': scenario,
                    'mean_efficiency': efficiency
                })

        efficiency_df = pd.DataFrame(efficiency_data)
        efficiency_file = self.results_dir / 'variant_efficiency.csv'
        efficiency_df.to_csv(efficiency_file, index=False)
        print(f"✓ Saved variant_efficiency: {efficiency_file}")

        # 3. Top-10 comparison data (CSV)
        comparison_file = self.results_dir / 'comparison_df.csv'
        self.comparison_df.to_csv(comparison_file, index=False)
        print(f"✓ Saved comparison_df: {comparison_file}")
        print()

    def run_full_analysis(self):
        """Run the full analysis pipeline."""
        print("\n" + "=" * 80)
        print("FULL SCENARIO COMPARISON ANALYSIS")
        print("=" * 80 + "\n")

        # Try loading from checkpoint (predictions already computed)
        if self._load_checkpoint('predictions'):
            print("Skipping prediction steps - using cached results\n")
        else:
            # Run full prediction pipeline
            self.prepare_for_prediction()
            self._save_checkpoint('prepared')

            self.get_dp_predictions()
            self._save_checkpoint('dp_predictions')

            self.get_dp7_predictions()
            self._save_checkpoint('predictions')
            print("\n" + "=" * 80 + "\n")

        self.select_top10_per_variant()
        plot_df = self.generate_plots()
        report = self.generate_statistics_report()

        # Targetable edit analysis
        targetable_results, variant_efficiency = self.analyze_targetable_edits()

        # Save comparison table
        output_file = self.results_dir / 'scenario_comparison_top10.csv'
        self.comparison_df.to_csv(output_file, index=False)
        print(f"✓ Saved comparison table: {output_file}\n")

        # Save data for local plotting
        self._save_data_for_local_plotting(targetable_results, variant_efficiency)

        print("=" * 80)
        print("✓ Analysis Complete!")
        print("=" * 80)
        print(f"\nResults saved in: {self.results_dir}")


def main():
    print("\n" + "=" * 80)
    print("DeepPrime vs DeepPrime7 Scenario Comparison")
    print("=" * 80 + "\n")

    # Check device
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")

    # Run analysis
    comparison = ScenarioComparison(device=device)
    comparison.run_full_analysis()


if __name__ == '__main__':
    main()
