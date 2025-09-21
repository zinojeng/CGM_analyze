import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def _get_profile(profile_config):
    if profile_config is not None:
        return profile_config
    from profile_config import PATIENT_PROFILES, DEFAULT_PROFILE_KEY  # 延後載入以避免循環
    return PATIENT_PROFILES[DEFAULT_PROFILE_KEY]


def _percentage_in_range(values, minimum=None, maximum=None, include_lower=True, include_upper=False):
    mask = pd.Series(True, index=values.index)
    if minimum is not None:
        mask &= values >= minimum if include_lower else values > minimum
    if maximum is not None:
        mask &= values <= maximum if include_upper else values < maximum
    if mask.empty:
        return 0.0
    return mask.mean()


def calculate_metrics(cgm_df, profile_config=None):
    profile = _get_profile(profile_config)
    glucose_values = pd.to_numeric(cgm_df['Sensor Glucose (mg/dL)'], errors='coerce').dropna()

    if glucose_values.empty:
        return {}

    metrics = {}
    mean_glucose = glucose_values.mean()
    metrics['Mean Glucose (mg/dL)'] = mean_glucose

    for range_config in profile['ranges']:
        proportion = _percentage_in_range(
            glucose_values,
            minimum=range_config.get('min'),
            maximum=range_config.get('max'),
            include_lower=range_config.get('include_lower', True),
            include_upper=range_config.get('include_upper', False)
        )
        metrics[range_config['metric_label']] = proportion

    cv = glucose_values.std() / mean_glucose if mean_glucose else np.nan
    metrics['CV'] = cv

    metrics['GMI'] = 3.31 + (0.02392 * mean_glucose)

    gri = np.log(glucose_values / 100) ** 2
    metrics['GRI'] = gri.mean() * 100

    metrics['MAGE'] = calculate_mage(glucose_values)

    return metrics


def calculate_mage(glucose_values, threshold=1):
    differences = np.abs(np.diff(glucose_values))
    significant_changes = differences[differences > threshold * glucose_values.std()]
    if significant_changes.size == 0:
        return 0.0
    return np.mean(significant_changes)


def create_agp(df, profile_config=None):
    profile = _get_profile(profile_config)
    agp_df = df.copy()
    agp_df['Time'] = agp_df['Timestamp'].dt.strftime('%H:%M')
    grouped = agp_df.groupby('Time')['Sensor Glucose (mg/dL)']
    percentiles = grouped.quantile([0.05, 0.25, 0.5, 0.75, 0.95]).unstack()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.fill_between(percentiles.index, percentiles[0.05], percentiles[0.95], alpha=0.2, color='blue')
    ax.fill_between(percentiles.index, percentiles[0.25], percentiles[0.75], alpha=0.2, color='blue')
    ax.plot(percentiles.index, percentiles[0.5], color='blue', linewidth=2)
    ax.set_ylim(40, 400)
    ax.set_xlabel('Time of Day')
    ax.set_ylabel('Glucose (mg/dL)')
    ax.set_title('Ambulatory Glucose Profile')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xticks(ax.get_xticks()[::2])

    target_lower, target_upper = profile['target_range']
    ax.axhline(y=target_lower, color='green', linestyle='--', alpha=0.7)
    ax.axhline(y=target_upper, color='green', linestyle='--', alpha=0.7)

    return fig


def create_daily_clusters(df, profile_config=None):
    profile = _get_profile(profile_config)
    cluster_df = df.copy()
    cluster_df['Date'] = cluster_df['Timestamp'].dt.date
    ranges = list(reversed(profile['ranges']))

    def _daily_summary(day_df):
        values = pd.to_numeric(day_df['Sensor Glucose (mg/dL)'], errors='coerce')
        return {
            range_config['daily_label']: _percentage_in_range(
                values,
                minimum=range_config.get('min'),
                maximum=range_config.get('max'),
                include_lower=range_config.get('include_lower', True),
                include_upper=range_config.get('include_upper', False)
            ) * 100
            for range_config in ranges
        }

    if cluster_df.empty:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_title('Daily Glucose Cluster Chart', fontsize=16)
        ax.set_ylabel('Percentage (%)', fontsize=12)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylim(0, 100)
        ax.grid(True, linestyle='--', alpha=0.7)
        return fig

    daily_stats = cluster_df.groupby('Date').apply(lambda day: pd.Series(_daily_summary(day))).fillna(0)

    fig, ax = plt.subplots(figsize=(12, 6))
    bottoms = np.zeros(len(daily_stats))

    for range_config in ranges:
        label = range_config['daily_label']
        ax.bar(
            daily_stats.index,
            daily_stats[label],
            bottom=bottoms,
            label=label,
            color=range_config.get('color', '#999999')
        )
        bottoms += daily_stats[label]

    ax.set_title('Daily Glucose Cluster Chart', fontsize=16)
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylim(0, 100)
    ax.grid(True, linestyle='--', alpha=0.7)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5), title='Glucose Range')

    plt.tight_layout()
    return fig


def analyze_hypoglycemia(cgm_df):
    low_glucose = cgm_df[cgm_df['Sensor Glucose (mg/dL)'] < 70]
    very_low_glucose = cgm_df[cgm_df['Sensor Glucose (mg/dL)'] < 54]

    low_events = low_glucose.groupby(low_glucose['Timestamp'].dt.date).size()
    very_low_events = very_low_glucose.groupby(very_low_glucose['Timestamp'].dt.date).size()

    low_times = low_glucose.groupby(low_glucose['Timestamp'].dt.hour).size()
    very_low_times = very_low_glucose.groupby(very_low_glucose['Timestamp'].dt.hour).size()

    return {
        'low_events': low_events,
        'very_low_events': very_low_events,
        'low_times': low_times,
        'very_low_times': very_low_times
    }


def analyze_hyperglycemia(cgm_df):
    high_glucose = cgm_df[cgm_df['Sensor Glucose (mg/dL)'] > 180]
    very_high_glucose = cgm_df[cgm_df['Sensor Glucose (mg/dL)'] > 250]

    high_events = high_glucose.groupby(high_glucose['Timestamp'].dt.date).size()
    very_high_events = very_high_glucose.groupby(very_high_glucose['Timestamp'].dt.date).size()

    high_times = high_glucose.groupby(high_glucose['Timestamp'].dt.hour).size()
    very_high_times = very_high_glucose.groupby(very_high_glucose['Timestamp'].dt.hour).size()

    return {
        'high_events': high_events,
        'very_high_events': very_high_events,
        'high_times': high_times,
        'very_high_times': very_high_times
    }
