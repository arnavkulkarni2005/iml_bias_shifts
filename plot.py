import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. YOUR EXACT TERMINAL DATA
# ==========================================
# Format: [Input, Middle, Output]
DATA = {
    "Baseline": {
        "bias_gap":      [0.0000, 0.0010, -0.0003], 
        "real_probe":    [0.43,   0.93,   0.93],   
        "control_probe": [0.43,   0.53,   0.50]    
    },
    "Neutral": {
        "bias_gap":      [0.0000, 0.0005, 0.0001], 
        "real_probe":    [0.43,   0.93,   1.00],    
        "control_probe": [0.43,   0.37,   0.43]    
    },
    "Progressive": {
        "bias_gap":      [0.0000, 0.0005, -0.0003], 
        "real_probe":    [0.43,   0.97,   1.00],    
        "control_probe": [0.43,   0.40,   0.63]    
    }
}

LAYERS = ["Input\n(L0)", "Middle\n(L16)", "Output\n(Final)"]

# ==========================================
# 2. PLOTTING CONFIGURATION
# ==========================================
plt.rcParams.update({
    'font.size': 10, 'font.family': 'sans-serif', 'figure.dpi': 300
})

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=False)
plt.subplots_adjust(wspace=0.35) 

colors = {'real': '#2b7bba', 'control': '#bfbfbf', 'bias': '#d62728'}

for i, (persona, metrics) in enumerate(DATA.items()):
    ax1 = axes[i]
    x = np.arange(len(LAYERS))
    width = 0.35

    # --- BARS (Probes) ---
    ax1.bar(x - width/2, metrics['real_probe'], width, label='Gender Probe (Real)', color=colors['real'], alpha=0.9, zorder=3)
    ax1.bar(x + width/2, metrics['control_probe'], width, label='Control Probe (Random)', color=colors['control'], hatch='//', alpha=0.6, zorder=3)
    
    ax1.set_ylim(0.0, 1.1)
    ax1.set_ylabel('Probe Accuracy', fontweight='bold', color='#333333')
    ax1.set_title(f"Condition: {persona}", fontweight='bold', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(LAYERS)
    ax1.grid(axis='y', linestyle='--', alpha=0.3)
    ax1.axhline(0.5, color='black', linestyle=':', linewidth=1) # Chance line

    # --- LINE (Bias Gap) ---
    ax2 = ax1.twinx()
    ax2.plot(x, metrics['bias_gap'], color=colors['bias'], marker='o', linewidth=2.5, markersize=8, label='Cosine Bias Gap')
    
    ax2.set_ylim(-0.005, 0.005) # Zoom in to show it is ZERO
    ax2.set_ylabel('Cosine Bias Gap ($\Delta$)', color=colors['bias'], fontweight='bold')
    ax2.tick_params(axis='y', labelcolor=colors['bias'])
    ax2.axhline(0, color=colors['bias'], linestyle='--', linewidth=0.8, alpha=0.5)

    if i == 0:
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper center', bbox_to_anchor=(1.7, -0.15), ncol=3, frameon=False)

plt.suptitle("The Fairness Facade: Gender Information Emerges in L16 and Persists at Output Despite Zero Bias Gap", fontsize=14, fontweight='bold', y=0.98)
plt.savefig('final_results_plot.png', bbox_inches='tight')
print("Plot saved.")
plt.show()