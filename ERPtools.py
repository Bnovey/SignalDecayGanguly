import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.interpolate import interp1d

# Define standard bands
BANDS = {
    'Delta': (128, 256),
    'Theta': (256, 384),
    'Beta': (512, 640),
    'LG': (640, 768),
    'HG': (768, None)  # HG goes to end
}

def process_band_states(all_data, band_name, start_idx, end_idx):
    """
    Extract ERP data for a specific frequency band.
    Returns:
        D_dict: dict of stacked data per target
        time_to_target: (2,8) success counts
        state_lengths: tuple of lengths (state1_len, state2_len, state3_len, state4_len)
    """
    erp_data = {target: [] for target in range(1, 8)}
    time_to_target = np.zeros((2, 8))
    fs = None

    trial_count = 0
    state_lengths = None
    for subj_id, subj_files in all_data.items():
        for fname, mat_file in subj_files.items():
            trial_count += 1
            print(f"  [{band_name}] Processing trial {trial_count}")

            features_list = mat_file['SmoothedNeuralFeatures']            
            features = np.array(features_list).squeeze().T  # (896, trials)

            if end_idx is None:
                features = features[start_idx:, :]
            else:
                features = features[start_idx:end_idx, :]

            # Get metadata
            fs = float(np.array(mat_file['Params']['UpdateRate']).flat[0])
            kinax = np.array(mat_file['TaskState']).flatten()
            target_id = int(np.array(mat_file['TargetID']).flat[0])
            # click_counter = int(np.array(mat_file['Params']['ClickCounter']).flat[0])

            # Find states
            state1 = np.where(kinax == 1)[0]
            state2 = np.where(kinax == 2)[0]
            state3 = np.where(kinax == 3)[0]
            state4 = np.where(kinax == 4)[0]

            # Interpolate state3
            tmp_data = features[:, state3]
            tb = np.arange(1, tmp_data.shape[1] + 1) / fs
            t = np.arange(1, 11) / fs
            tb_scaled = tb * (t[-1] / tb[-1])

            tmp_data_interp = np.zeros((features.shape[0], len(t)))
            for ch in range(features.shape[0]):
                f = interp1d(tb_scaled, tmp_data[ch, :], kind='cubic', 
                            fill_value='extrapolate')
                tmp_data_interp[ch, :] = f(t)

            # Concatenate states
            data = np.hstack([
                features[:, state1],
                features[:, state2],
                tmp_data_interp,
                features[:, state4]
            ])

            padded = 0
            if len(state1) < 8:
                data = np.hstack([data[:, [0]], data])
                padded = 1
            if state_lengths is None:
                state_lengths = (len(state1) + padded, len(state2), tmp_data_interp.shape[1], len(state4))

            # Track success
            # trial_dur = (len(state3) - click_counter) / fs
            # time_to_target[1, target_id] += 1
            # if trial_dur <= 3:
            #     time_to_target[0, target_id] += 1

            erp_data[target_id].append(data)

    D_dict = {}
    for target in range(1, 8):
        if erp_data[target]:
            D_dict[target] = np.stack(erp_data[target], axis=2)
        else:
            D_dict[target] = np.array([])

    return D_dict, time_to_target, state_lengths


def plot_erps(results, band_name, output_dir=None, ch_map=None):
    """
    Plot ERPs for a specific band across all 7 targets in an 8x16 channel grid.
    Uses bootstrap CI and permutation null distribution for significance testing.
    """
    D_dict = results[band_name]['D']
    state_lengths = results[band_name].get('state_lengths')

    if ch_map is None:
        ch_map = np.arange(1, 129).reshape(8, 16)

    for target_id in range(1, 8):
        if D_dict[target_id].size == 0:
            print(f"  Target {target_id}: No data. Skipping.")
            continue

        data = D_dict[target_id]  
        num_channels, total_time, num_trials = data.shape
        print(f"  Target {target_id}: {num_trials} trials, {total_time} timepoints")

        state_boundaries = []
        if state_lengths:   
            cumsum = 0
            for s_len in state_lengths[:-1]:
                cumsum += s_len
                state_boundaries.append(cumsum)

        fig = plt.figure(figsize=(20, 10))
        fig.suptitle(f"{band_name} Band - Target {target_id} ({num_trials} trials)", fontsize=16)
        fig.patch.set_facecolor('white')
        gs = GridSpec(8, 16, figure=fig, hspace=0.5, wspace=0.5)
        axes = {}
        for row in range(8):
            for col in range(16):
                axes[(row, col)] = fig.add_subplot(gs[row, col])

        for ch_idx in range(1, min(num_channels + 1, 129)):
            pos = np.where(ch_map == ch_idx)
            if len(pos[0]) == 0:
                continue
            row, col = pos[0][0], pos[1][0]
            ax = axes[(row, col)]

            erps = data[ch_idx - 1, :, :]
            chdata = erps.T

            baseline = chdata[:, :8]
            m_base = np.mean(baseline)
            s_base = np.std(baseline)
            chdata = (chdata - m_base) / (s_base + 1e-10)

            m = np.mean(chdata, axis=0)
            mb_samples = []
            for _ in range(1000):
                boot_idx = np.random.choice(chdata.shape[0], chdata.shape[0], replace=True)
                mb_samples.append(np.mean(chdata[boot_idx, :], axis=0))
            mb_samples = np.array(mb_samples)
            ci_low = np.percentile(mb_samples, 2.5, axis=0)
            ci_high = np.percentile(mb_samples, 97.5, axis=0)

            null_samples = []
            for _ in range(1000):
                tmp = chdata.copy()
                tmp = tmp.flatten()
                np.random.shuffle(tmp)
                tmp = tmp.reshape(chdata.shape)
                baseline_null = tmp[:, :8]
                m_null = np.mean(baseline_null)
                s_null = np.std(baseline_null)
                tmp_norm = (tmp - m_null) / (s_null + 1e-10)
                null_samples.append(np.mean(tmp_norm, axis=0))
            null_samples = np.array(null_samples)
            null_low = np.percentile(null_samples, 2.5, axis=0)
            null_high = np.percentile(null_samples, 97.5, axis=0)

            tt = np.arange(m.shape[0])
            ax.fill_between(tt, ci_low, ci_high, color=[0.3, 0.3, 0.7], alpha=0.2)
            ax.plot(tt, m, color='b', linewidth=1.5)
            ax.fill_between(tt, null_low, null_high, color=[0.7, 0.3, 0.3], alpha=0.2)

            for boundary in state_boundaries:
                ax.axvline(x=boundary, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)

            pvals = []
            for idx in range(total_time):
                p = np.sum(np.abs(m[idx]) >= np.abs(null_samples[:, idx])) / float(null_samples.shape[0])
                pvals.append(p)
            pvals = np.array(pvals)

            sig_count = np.sum(pvals >= 0.95)
            is_significant = sig_count >= 7

            ax.set_ylabel(str(ch_idx), fontsize=8)
            ax.set_ylim([-2, 4])
            ax.axhline(0, color='k', linewidth=1.5)
            ax.set_xticks([])
            ax.set_yticks([])

            if is_significant:
                for spine in ax.spines.values():
                    spine.set_edgecolor('green')
                    spine.set_linewidth(2)

        if output_dir:
            fname = f"{band_name}_Target{target_id}.png"
            fpath = os.path.join(output_dir, fname)
            plt.savefig(fpath, dpi=150, bbox_inches='tight')
            print(f"  Saved: {fpath}")

        plt.show()

def plot_erps_side_by_side(results_long, results_short, band_name, output_dir=None, ch_map=None):
    """
    Plot ERPs for a specific band across all 7 targets in an 8x32 channel grid,
    showing long and short data side by side for each channel.
    Uses bootstrap CI and permutation null distribution for significance testing.
    """
    D_long = results_long[band_name]['D']
    D_short = results_short[band_name]['D']
    state_lengths_long = results_long[band_name].get('state_lengths')
    state_lengths_short = results_short[band_name].get('state_lengths')

    def get_boundaries(state_lengths):
        boundaries = []
        if state_lengths:
            cumsum = 0
            for s_len in state_lengths[:-1]:
                cumsum += s_len
                boundaries.append(cumsum)
        return boundaries

    boundaries_long = get_boundaries(state_lengths_long)
    boundaries_short = get_boundaries(state_lengths_short)

    if ch_map is None:
        ch_map = np.arange(1, 129).reshape(8, 16)

    for target_id in range(1, 8):
        if D_long[target_id].size == 0 and D_short[target_id].size == 0:
            print(f"  Target {target_id}: No data in either set. Skipping.")
            continue

        fig = plt.figure(figsize=(40, 10))
        fig.suptitle(f"{band_name} Band - Target {target_id} (Long vs Short)", fontsize=16)
        fig.patch.set_facecolor('white')
        gs = GridSpec(8, 32, figure=fig, hspace=0.5, wspace=0.5)
        axes = {}
        for row in range(8):
            for col in range(32):
                axes[(row, col)] = fig.add_subplot(gs[row, col])

        max_ch = min(129, max(
            D_long[target_id].shape[0] if D_long[target_id].size > 0 else 0,
            D_short[target_id].shape[0] if D_short[target_id].size > 0 else 0
        ) + 1)

        for ch_idx in range(1, max_ch):
            pos = np.where(ch_map == ch_idx)
            if len(pos[0]) == 0:
                continue
            row, col = pos[0][0], pos[1][0]

            # LONG handling
            ax_long = axes[(row, col)]
            is_significant_long = False
            if D_long[target_id].size > 0:
                erps = D_long[target_id][ch_idx - 1, :, :]
                chdata = erps.T
                baseline = chdata[:, :8]
                m_base = np.mean(baseline)
                s_base = np.std(baseline)
                chdata = (chdata - m_base) / (s_base + 1e-10)
                m = np.mean(chdata, axis=0)
                
                # CI Bootstrapping (Long)
                mb_samples = []
                for _ in range(100): # Reduced from 1000 for speed in side-by-side if you want
                    boot_idx = np.random.choice(chdata.shape[0], chdata.shape[0], replace=True)
                    mb_samples.append(np.mean(chdata[boot_idx, :], axis=0))
                mb_samples = np.array(mb_samples)
                ci_low = np.percentile(mb_samples, 2.5, axis=0)
                ci_high = np.percentile(mb_samples, 97.5, axis=0)
                
                # Null Samples (Long)
                null_samples = []
                for _ in range(100): # Reduced from 1000 for speed
                    tmp = chdata.copy().flatten()
                    np.random.shuffle(tmp)
                    tmp = tmp.reshape(chdata.shape)
                    baseline_null = tmp[:, :8]
                    m_null = np.mean(baseline_null)
                    s_null = np.std(baseline_null)
                    tmp_norm = (tmp - m_null) / (s_null + 1e-10)
                    null_samples.append(np.mean(tmp_norm, axis=0))
                null_samples = np.array(null_samples)
                null_low = np.percentile(null_samples, 2.5, axis=0)
                null_high = np.percentile(null_samples, 97.5, axis=0)
                tt = np.arange(m.shape[0])
                
                ax_long.fill_between(tt, ci_low, ci_high, color=[0.3, 0.3, 0.7], alpha=0.2)
                ax_long.plot(tt, m, color='b', linewidth=1.5)
                ax_long.fill_between(tt, null_low, null_high, color=[0.7, 0.3, 0.3], alpha=0.2)

                for boundary in boundaries_long:
                    ax_long.axvline(x=boundary, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)

                ax_long.set_title(f"Ch {ch_idx} L", fontsize=8)
                ax_long.set_xticks([])
                ax_long.set_yticks([])
                ax_long.set_ylim([-2, 4])
                ax_long.axhline(0, color='k', linewidth=1.5)

                pvals = [np.sum(np.abs(m[idx]) >= np.abs(null_samples[:, idx])) / float(null_samples.shape[0]) for idx in range(m.shape[0])]
                is_significant_long = np.sum(np.array(pvals) >= 0.95) >= 7

            # SHORT handling
            ax_short = axes[(row, col + 16)]
            is_significant_short = False
            if D_short[target_id].size > 0:
                erps = D_short[target_id][ch_idx - 1, :, :]
                chdata = erps.T
                baseline = chdata[:, :8]
                m_base = np.mean(baseline)
                s_base = np.std(baseline)
                chdata = (chdata - m_base) / (s_base + 1e-10)
                m = np.mean(chdata, axis=0)
                
                # CI Bootstrapping (Short)
                mb_samples = []
                for _ in range(100):
                    boot_idx = np.random.choice(chdata.shape[0], chdata.shape[0], replace=True)
                    mb_samples.append(np.mean(chdata[boot_idx, :], axis=0))
                mb_samples = np.array(mb_samples)
                ci_low = np.percentile(mb_samples, 2.5, axis=0)
                ci_high = np.percentile(mb_samples, 97.5, axis=0)
                
                # Null Samples (Short)
                null_samples = []
                for _ in range(100):
                    tmp = chdata.copy().flatten()
                    np.random.shuffle(tmp)
                    tmp = tmp.reshape(chdata.shape)
                    baseline_null = tmp[:, :8]
                    m_null = np.mean(baseline_null)
                    s_null = np.std(baseline_null)
                    tmp_norm = (tmp - m_null) / (s_null + 1e-10)
                    null_samples.append(np.mean(tmp_norm, axis=0))
                null_samples = np.array(null_samples)
                null_low = np.percentile(null_samples, 2.5, axis=0)
                null_high = np.percentile(null_samples, 97.5, axis=0)
                tt = np.arange(m.shape[0])
                
                ax_short.fill_between(tt, ci_low, ci_high, color=[0.7, 0.3, 0.3], alpha=0.2)
                ax_short.plot(tt, m, color='r', linewidth=1.5)
                ax_short.fill_between(tt, null_low, null_high, color=[0.3, 0.3, 0.7], alpha=0.1)

                for boundary in boundaries_short:
                    ax_short.axvline(x=boundary, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)

                ax_short.set_title(f"Ch {ch_idx} S", fontsize=8)
                ax_short.set_xticks([])
                ax_short.set_yticks([])
                ax_short.set_ylim([-2, 4])
                ax_short.axhline(0, color='k', linewidth=1.5)

                pvals = [np.sum(np.abs(m[idx]) >= np.abs(null_samples[:, idx])) / float(null_samples.shape[0]) for idx in range(m.shape[0])]
                is_significant_short = np.sum(np.array(pvals) >= 0.95) >= 7

            if is_significant_long:
                for spine in ax_long.spines.values():
                    spine.set_edgecolor('green')
                    spine.set_linewidth(2)
            if is_significant_short:
                for spine in ax_short.spines.values():
                    spine.set_edgecolor('green')
                    spine.set_linewidth(2)

        if output_dir:
            fname = f"{band_name}_Target{target_id}_side_by_side.png"
            fpath = os.path.join(output_dir, fname)
            plt.savefig(fpath, dpi=150, bbox_inches='tight')

        plt.show()

def plot_sig_per_channel(results_long, results_short, band_name, ch_map=None):
    """
    For each target, plot long and short sig counts per channel overlapped.
    """
    D_long = results_long[band_name]['D']
    D_short = results_short[band_name]['D']

    if ch_map is None:
        ch_map = np.arange(1, 129).reshape(8, 16)

    def compute_sig_counts(D_dict, target_id):
        if D_dict[target_id].size == 0:
            return []
        data = D_dict[target_id]
        num_channels, total_time, num_trials = data.shape
        sig_counts = []
        for ch_idx in range(1, min(num_channels + 1, 129)):
            erps = data[ch_idx - 1, :, :]
            chdata = erps.T

            baseline = chdata[:, :8]
            m_base = np.mean(baseline)
            s_base = np.std(baseline)
            chdata = (chdata - m_base) / (s_base + 1e-10)

            m = np.mean(chdata, axis=0)

            null_samples = []
            for _ in range(100): # 100 iterations for performance in overview mode
                tmp = chdata.copy().flatten()
                np.random.shuffle(tmp)
                tmp = tmp.reshape(chdata.shape)
                baseline_null = tmp[:, :8]
                m_null = np.mean(baseline_null)
                s_null = np.std(baseline_null)
                tmp_norm = (tmp - m_null) / (s_null + 1e-10)
                null_samples.append(np.mean(tmp_norm, axis=0))
            null_samples = np.array(null_samples)

            pvals = [np.sum(np.abs(m[idx]) >= np.abs(null_samples[:, idx])) / float(null_samples.shape[0]) for idx in range(total_time)]
            sig_counts.append(np.sum(np.array(pvals) >= 0.95))
        return sig_counts

    for target_id in range(1, 8):
        sig_long = compute_sig_counts(D_long, target_id)
        sig_short = compute_sig_counts(D_short, target_id)

        if not sig_long and not sig_short:
            continue

        channels = np.arange(1, 129)
        width = 0.4

        plt.figure(figsize=(16, 4))
        if sig_long:
            plt.bar(channels - width/2, sig_long, width=width, color='steelblue', alpha=0.7, label='Long')
        if sig_short:
            plt.bar(channels + width/2, sig_short, width=width, color='coral', alpha=0.7, label='Short')
        plt.axhline(7, color='r', linestyle='--', linewidth=1, label='threshold (7)')
        plt.xlabel('Channel')
        plt.ylabel('# significant bins')
        plt.title(f'{band_name} - Target {target_id}')
        plt.xlim(0, 129)
        plt.legend()
        plt.tight_layout()
        plt.show()