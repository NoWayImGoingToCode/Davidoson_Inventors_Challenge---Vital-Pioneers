import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

csv_data = """time_ms,phase,V_in,R_mem,G_mem
0,LTP,4.500,39600.00,0.00002525
22,LTP,4.500,39288.00,0.00002545
45,LTP,4.500,38980.00,0.00002565
67,LTP,4.500,38668.00,0.00002586
90,LTP,4.500,38360.00,0.00002607
112,LTP,4.500,38048.00,0.00002628
136,LTP,4.500,37744.00,0.00002649
158,LTP,4.500,37432.00,0.00002672
181,LTP,4.500,37124.00,0.00002694
203,LTP,4.500,36812.00,0.00002717
226,LTP,4.500,36504.00,0.00002739
248,LTP,4.500,36192.00,0.00002763
272,LTP,4.500,35888.00,0.00002786
294,LTP,4.500,35576.00,0.00002811
317,LTP,4.500,35268.00,0.00002835
339,LTP,4.500,34956.00,0.00002861
362,LTP,4.500,34648.00,0.00002886
385,LTP,4.500,34340.00,0.00002912
408,LTP,4.500,34032.00,0.00002938
431,LTP,4.500,33724.00,0.00002965
453,LTP,4.500,33412.00,0.00002993
476,LTP,4.500,33104.00,0.00003021
498,LTP,4.500,32792.00,0.00003050
521,LTP,4.500,32484.00,0.00003078
544,LTP,4.500,32176.00,0.00003108
567,LTP,4.500,31868.00,0.00003138
589,LTP,4.500,31556.00,0.00003169
612,LTP,4.500,31248.00,0.00003200
634,LTP,4.500,30936.00,0.00003232
657,LTP,4.500,30628.00,0.00003265
680,LTP,4.500,30320.00,0.00003298
703,LTP,4.500,30012.00,0.00003332
726,LTP,4.500,29704.00,0.00003367
748,LTP,4.500,29392.00,0.00003402
771,LTP,4.500,29084.00,0.00003438
794,LTP,4.500,28776.00,0.00003475
817,LTP,4.500,28468.00,0.00003513
839,LTP,4.500,28156.00,0.00003552
862,LTP,4.500,27848.00,0.00003591
884,LTP,4.500,27536.00,0.00003632
907,Low-frequency,4.500,27228.00,0.00003673
1110,Low-frequency,4.500,27228.00,0.00003673 
1312,Low-frequency,4.500,27228.00,0.00003673 
1516,Low-frequency,4.500,27228.00,0.00003673 
1719,Low-frequency,4.500,27228.00,0.00003673 
1922,Low-frequency,4.500,27228.00,0.00003673 
2124,Low-frequency,4.500,27228.00,0.00003673 
2327,Low-frequency,4.500,27228.00,0.00003673 
2530,Low-frequency,4.500,27228.00,0.00003673 
2733,Low-frequency,4.500,27228.00,0.00003673 
2935,Low-frequency,4.500,27228.00,0.00003673 
3138,Low-frequency,4.500,27228.00,0.00003673 
3341,Low-frequency,4.500,27228.00,0.00003673 
3544,Low-frequency,4.500,27228.00,0.00003673 
3746,Low-frequency,4.500,27228.00,0.00003673 
3949,Low-frequency,4.500,27228.00,0.00003673 
4152,Low-frequency,4.500,27228.00,0.00003673 
4355,Low-frequency,4.500,27228.00,0.00003673 
4557,Low-frequency,4.500,27228.00,0.00003673 
4557,Low-frequency,4.500,27228.00,0.00003673 
4760,Low-frequency,4.500,27228.00,0.00003673
4963,LTD,0.000,27268.00,0.00003667
5265,LTD,0.000,27308.00,0.00003662 
5568,LTD,0.000,27348.00,0.00003657 
5871,LTD,0.000,27388.00,0.00003651 
6173,LTD,0.000,27428.00,0.00003646 
6476,LTD,0.000,27468.00,0.00003641 
6778,LTD,0.000,27508.00,0.00003635 
7081,LTD,0.000,27548.00,0.00003630 
7384,LTD,0.000,27588.00,0.00003625 
7687,LTD,0.000,27628.00,0.00003620 
7989,LTD,0.000,27668.00,0.00003614 
8292,LTD,0.000,27708.00,0.00003609 
8594,LTD,0.000,27748.00,0.00003604 
8897,LTD,0.000,27788.00,0.00003599 
9199,LTD,0.000,27828.00,0.00003594 
9502,LTD,0.000,27868.00,0.00003588 
9805,LTD,0.000,27908.00,0.00003583 
10107,LTD,0.000,27948.00,0.00003578 
10411,LTD,0.000,27988.00,0.00003573 
10713,LTD,0.000,28028.00,0.00003568
11016,LTP,4.500,28028.00,0.00003568
11927,Low-frequency,4.500,15672.00,0.00006381
14969,Low-frequency,4.500,15672.00,0.00006381
15984,LTD,0.000,15712.00,0.00006365
21735,LTD,0.000,16472.00,0.00006071
22038,LTP,4.500,16472.00,0.00006071
22220,LTP,4.500,14000.00,0.00007143
22880,LTP,4.500,5040.00,0.00019841
22903,LTP,4.500,4732.00,0.00021133
22926,LTP,4.500,4424.00,0.00022604
22948,Low-frequency,4.500,4112.00,0.00024319
27004,LTD,0.000,4152.00,0.00024085
30030,LTD,0.000,4552.00,0.00021968
30333,LTD,0.000,4592.00,0.00021777
30636,LTD,0.000,4632.00,0.00021589
30939,LTD,0.000,4672.00,0.00021404
31241,LTD,0.000,4712.00,0.00021222
31544,LTD,0.000,4752.00,0.00021044
31847,LTD,0.000,4792.00,0.00020868
32149,LTD,0.000,4832.00,0.00020695
32452,LTD,0.000,4872.00,0.00020525
32754,LTD,0.000,4912.00,0.00020358
33057,LTP,4.500,4912.00,0.00020358
33080,LTP,4.500,4604.00,0.00021720
33102,LTP,4.500,4292.00,0.00023299
33125,LTP,4.500,3984.00,0.00025100
33148,LTP,4.500,3676.00,0.00027203
33171,LTP,4.500,3368.00,0.00029691
33193,LTP,4.500,3056.00,0.00032723
33216,LTP,4.500,2748.00,0.00036390
33239,LTP,4.500,2440.00,0.00040984
33262,LTP,4.500,2132.00,0.00046904
33285,LTP,4.500,2000.00,0.00050000
33307,LTP,4.500,2000.00,0.00050000
33330,LTP,4.500,2000.00,0.00050000
33352,LTP,4.500,2000.00,0.00050000
33376,LTP,4.500,2000.00,0.00050000
33398,LTP,4.500,2000.00,0.00050000
33421,LTP,4.500,2000.00,0.00050000
33443,LTP,4.500,2000.00,0.00050000
33466,LTP,4.500,2000.00,0.00050000
33489,LTP,4.500,2000.00,0.00050000
33512,LTP,4.500,2000.00,0.00050000
33534,LTP,4.500,2000.00,0.00050000
"""

df = pd.read_csv(StringIO(csv_data))


G_exp = df['G_mem'].values  # Conductance in Siemens 
t_exp = df['time_ms'].values / 1000.0  # Convert milliseconds to seconds
phases = df['phase'].values  # Phase labels: 'LTP', 'LTD', or 'Low-frequency'

# Also extract other useful columns
V_in = df['V_in'].values  # Input voltage (0V for LTD, 4.5V for LTP/Low-freq)
R_mem = df['R_mem'].values  # Resistance in Ohms

# Time differences between measurements
df['time_diff_ms'] = df['time_ms'].diff().fillna(0)
df['time_diff_s'] = df['time_diff_ms'] / 1000.0



fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# G(t) vs t with timing markers
ax1 = axes[0, 0]
ax1.plot(df['time_ms'], df['G_mem'] * 1e6, 'b-', linewidth=2, label='Conductance')
ax1.set_xlabel('Time (ms)')
ax1.set_ylabel('Conductance (µS)', color='b')
ax1.tick_params(axis='y', labelcolor='b')
ax1.grid(True, alpha=0.3)

# Add second y-axis for time differences
ax1b = ax1.twinx()
time_diffs = df['time_diff_ms'].values
time_diffs[0] = 0  # First point has no difference
ax1b.plot(df['time_ms'], time_diffs, 'r--', linewidth=1, alpha=0.7, label='Time Δ (ms)')
ax1b.set_ylabel('Time between measurements (ms)', color='r')
ax1b.tick_params(axis='y', labelcolor='r')
ax1b.set_ylim([0, max(time_diffs) * 1.1])

# Highlight very short intervals
short_idx = np.where(time_diffs < 10)[0]
if len(short_idx) > 0:
    ax1.scatter(df['time_ms'].iloc[short_idx], df['G_mem'].iloc[short_idx] * 1e6,
               color='red', s=50, zorder=5, label='Δt < 10ms')
ax1.set_title('Conductance with Timing Information')
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1b.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

# Histogram of time intervals
ax2 = axes[0, 1]
bins = np.arange(0, df['time_diff_ms'].max() + 50, 50)
ax2.hist(df['time_diff_ms'], bins=bins, edgecolor='black', alpha=0.7)
ax2.axvline(x=10, color='red', linestyle='--', label='Δt = 10ms')
ax2.set_xlabel('Time between measurements (ms)')
ax2.set_ylabel('Frequency')
ax2.set_title('Distribution of Measurement Intervals')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Conductance change rate vs time interval
ax3 = axes[1, 0]
df['delta_G'] = df['G_mem'].diff()
df['rate_G'] = df['delta_G'] / (df['time_diff_s'] + 1e-9)  
valid_mask = (df['time_diff_ms'] > 1) & (np.abs(df['rate_G']) < 1e-3)
if valid_mask.any():
    scatter = ax3.scatter(df.loc[valid_mask, 'time_diff_ms'], 
                         df.loc[valid_mask, 'rate_G'] * 1e6,  # Convert to µS/s
                         c=df.loc[valid_mask, 'G_mem'] * 1e6,  # Color by conductance
                         cmap='viridis', alpha=0.7, s=50, edgecolors='k')
    
    ax3.set_xlabel('Time interval (ms)')
    ax3.set_ylabel('Conductance change rate (µS/s)')
    ax3.set_title('Change Rate vs Measurement Interval')
    ax3.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax3)
    cbar.set_label('Current Conductance (µS)')

# Phase-specific timing
ax4 = axes[1, 1]
colors = {'LTP': 'green', 'LTD': 'orange', 'Low-frequency': 'gray'}
phase_data = []
for phase, color in colors.items():
    phase_times = df[df['phase'] == phase]['time_diff_ms'].values
    if len(phase_times) > 0:
        phase_times = phase_times[1:] if len(phase_times) > 1 else phase_times
        phase_data.append(phase_times)
        jitter = np.random.normal(0, 0.1, len(phase_times))
        y_pos = list(colors.keys()).index(phase) + 1
        ax4.scatter(phase_times + jitter, [y_pos] * len(phase_times), 
                   color=color, alpha=0.6, s=30, label=phase)

# Create boxplot
bp = ax4.boxplot(phase_data, positions=range(1, len(phase_data) + 1), 
                 patch_artist=True, widths=0.6)
for patch, color in zip(bp['boxes'], colors.values()):
    patch.set_facecolor(color)
    patch.set_alpha(0.5)

ax4.set_xlabel('Time interval (ms)')
ax4.set_ylabel('Phase')
ax4.set_yticks(range(1, len(colors) + 1))
ax4.set_yticklabels(list(colors.keys()))
ax4.set_title('Phase-Specific Timing Intervals')
ax4.legend()
ax4.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.show()



# Look at early measurements (first 10%)
early_idx = int(len(df) * 0.1)
early_data = df.iloc[:early_idx]


# Model with retention
class RetentionAwareSynapse:
    def __init__(self, G_init, eta_ltp, eta_ltd, G_min, G_max,
                 tau_phys=0.05, tau_act=0.02, V_th=3.5):
        self.G = G_init 
        self.G_target = G_init # Plasticity target
        self.eta_ltp = eta_ltp #learning rate for LTP
        self.eta_ltd = eta_ltd #learning rate for LTD
        self.G_min = G_min
        self.G_max = G_max
        self.tau_phys = tau_phys # Ion migration time (s)
        self.tau_act = tau_act # Activation delay (s)
        self.V_th = V_th

        self.x = 0.0                        

        self.G_history = []
        self.t_history = []

    def update_target(self, phase, V, dt):
        act_inf = 1.0 if V > self.V_th else 0.0
        self.x += (act_inf - self.x) * (1 - np.exp(-dt / self.tau_act))

        if self.x < 0.1:
            return  
        w = (self.G_target - self.G_min) / (self.G_max - self.G_min)

        if phase == 'LTP':
            eta_eff = self.eta_ltp * (1 - w)
            self.G_target += eta_eff * (self.G_max - self.G_target)

        elif phase == 'LTD':
            eta_eff = self.eta_ltd * w
            self.G_target -= eta_eff * (self.G_target - self.G_min)

        self.G_target = np.clip(self.G_target, self.G_min, self.G_max)

    def evolve(self, dt):
        alpha = 1 - np.exp(-dt / self.tau_phys)
        self.G += alpha * (self.G_target - self.G)

    def simulate(self, pulse_times, pulse_types, V_in):
        self.G_history = [self.G]
        self.t_history = [pulse_times[0]]

        for i in range(1, len(pulse_times)):
            dt = pulse_times[i] - pulse_times[i - 1]

            # Update plasticity 
            self.update_target(pulse_types[i], V_in[i], dt)

            self.evolve(dt)
            self.G_history.append(self.G)
            self.t_history.append(pulse_times[i])

        return np.array(self.t_history), np.array(self.G_history)

eta_ltp = 5.9558e-02
eta_ltd = 2.8956e-02
G_min = df['G_mem'].min()
G_max = df['G_mem'].max()

# Model with no retention
class SimpleSynapse:
    
    def __init__(self, G_init, eta_ltp, eta_ltd, G_min, G_max):
        self.eta_ltp = eta_ltp
        self.eta_ltd = eta_ltd
        self.G_min = G_min
        self.G_max = G_max
        self.G = G_init
        self.target_G = G_init
        self.x = 0.0

        
    def ltp(self):
        if self.G < self.G_max:
            delta_G = self.eta_ltp * (self.G_max - self.G)
            self.G += delta_G
            if self.G > self.G_max:
                self.G = self.G_max
    
    def ltd(self):
        if self.G > self.G_min:
            delta_G = self.eta_ltd * (self.G - self.G_min)
            self.G -= delta_G
            if self.G < self.G_min:
                self.G = self.G_min
    
    def low_freq(self):
        pass
    
    def simulate(self, pulse_times, pulse_types):
        G_history = [self.G]
        for i in range(1, len(pulse_times)):
            phase = pulse_types[i]
            if phase == 'LTP':
                self.ltp()
            elif phase == 'LTD':
                self.ltd()
            elif phase == 'Low-frequency':
                self.low_freq()
            G_history.append(self.G)
        return np.array(pulse_times), np.array(G_history)



# Simple model (no retention)
simple_synapse = SimpleSynapse(G_exp[0], eta_ltp, eta_ltd, G_min, G_max)
t_simple, G_simple = simple_synapse.simulate(t_exp, phases)

# Model with retention 
retention_factors = [0.999, 0.99, 0.95]
G_retention_models = []
labels = []

for rf in retention_factors:
    synapse = RetentionAwareSynapse(
    G_init=G_exp[0],
    eta_ltp=eta_ltp,
    eta_ltd=eta_ltd,
    G_min=G_min,
    G_max=G_max,
    tau_phys=0.05,
    tau_act=0.02,
    V_th=3.5
)

t_ret, G_ret = synapse.simulate(t_exp, phases, V_in)

G_retention_models.append(G_ret)
labels.append(f'Retention={rf}')


fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Early phase 
ax1 = axes[0, 0]
early_idx = np.where(t_exp * 1000 < 1000)[0]
if len(early_idx) > 0:
    ax1.plot(t_exp[early_idx] * 1000, G_exp[early_idx] * 1e6, 
             'ko-', linewidth=2, markersize=5, label='Experiment', alpha=0.8)
    ax1.plot(t_simple[early_idx] * 1000, G_simple[early_idx] * 1e6,
             'r--', linewidth=2, label='Simple Model', alpha=0.8)
    
    for i, G_ret in enumerate(G_retention_models):
        ax1.plot(t_exp[early_idx] * 1000, G_ret[early_idx] * 1e6,
                 linewidth=1.5, label=labels[i], alpha=0.7)
    
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Conductance (µS)')
    ax1.set_title('EARLY PHASE (0-1000 ms) - Steep Start Analysis')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

# Full Timescale comparison
ax2 = axes[0, 1]
ax2.plot(t_exp * 1000, G_exp * 1e6, 'ko-', linewidth=2, markersize=3, 
         label='Experiment', alpha=0.8)
ax2.plot(t_simple * 1000, G_simple * 1e6, 'r--', linewidth=2, 
         label='Simple Model', alpha=0.8)

for i, G_ret in enumerate(G_retention_models):
    ax2.plot(t_exp * 1000, G_ret * 1e6, linewidth=1.5, 
             label=labels[i], alpha=0.7)

ax2.set_xlabel('Time (ms)')
ax2.set_ylabel('Conductance (µS)')
ax2.set_title('FULL TIMESCALE COMPARISON')
ax2.legend(loc='upper left')
ax2.grid(True, alpha=0.3)

# Error comparison
ax3 = axes[1, 0]
errors_simple = np.abs(G_simple - G_exp) * 1e6
ax3.plot(t_exp * 1000, errors_simple, 'r-', linewidth=2, 
         label='Simple Model Error', alpha=0.8)

for i, G_ret in enumerate(G_retention_models):
    errors_ret = np.abs(G_ret - G_exp) * 1e6
    ax3.plot(t_exp * 1000, errors_ret, linewidth=1.5, 
             label=f'{labels[i]} Error', alpha=0.7)

ax3.set_xlabel('Time (ms)')
ax3.set_ylabel('Absolute Error (µS)')
ax3.set_title('MODEL ERRORS')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Early phase error zoom
ax4 = axes[1, 1]
if len(early_idx) > 0:
    ax4.plot(t_exp[early_idx] * 1000, errors_simple[early_idx], 
             'r-', linewidth=2, label='Simple Model Error', alpha=0.8)
    
    for i, G_ret in enumerate(G_retention_models):
        errors_ret = np.abs(G_ret[early_idx] - G_exp[early_idx]) * 1e6
        ax4.plot(t_exp[early_idx] * 1000, errors_ret,
                 linewidth=1.5, label=f'{labels[i]} Error', alpha=0.7)
    
    ax4.set_xlabel('Time (ms)')
    ax4.set_ylabel('Absolute Error (µS)')
    ax4.set_title('EARLY PHASE ERRORS (Zoom)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
