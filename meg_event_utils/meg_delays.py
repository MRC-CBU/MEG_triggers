import mne
import pandas as pd
import json
import os
import numpy as np


# =============================================================================
# Function to load delay CSV files
# =============================================================================
def load_delay_df(delay_csv_file):
    """Load delay CSV and parse the Date column as datetime, sorted ascending."""
    df = pd.read_csv(delay_csv_file)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df = df.sort_values('Date')
    return df

# =============================================================================
# Function to get auditory and visual delays
# =============================================================================
def get_delays_for_fif(raw_file, delay_auditory_df, delay_visual_df):
    """
        Retrieves the auditory and visual delays in samples for a given raw fif file,
        based on the measurement date and provided delay dataframes.
        Args:
            raw_file (str): Path to the raw fif file.
            delay_auditory_df (pd.DataFrame): DataFrame containing auditory delay information,
                with a 'Date' column (datetime objects) and a 'Mean Delay Rounded (ms)' column.
            delay_visual_df (pd.DataFrame): DataFrame containing visual delay information,
                with a 'Date' column (datetime objects) and a 'Mean Delay Rounded (ms)' column.
        Returns:
            tuple: A tuple containing:
                - meas_date (datetime.datetime): The measurement date of the raw file.
                - auditory_delay_samples (int): Auditory delay in samples.
                - visual_delay_samples (int): Visual delay in samples.
                - auditory_visual_delay_samples (int): Average of auditory and visual delays in samples.
        Raises:
            ValueError: If no matching delay is found in either delay_auditory_df or
                delay_visual_df for the measurement date.
        """
    
    raw = mne.io.read_raw_fif(raw_file, preload=False, verbose=False)
    meas_date = raw.info['meas_date'].replace(tzinfo=None)  # remove timezone if present

    matched_auditory_delay = delay_auditory_df[delay_auditory_df['Date'] <= meas_date]
    if matched_auditory_delay.empty:
        raise ValueError(f"No auditory delay found for measurement date: {meas_date}")
      
    matched_visual_delay = delay_visual_df[delay_visual_df['Date'] <= meas_date]
    if matched_visual_delay.empty:
        raise ValueError(f"No visual delay found for measurement date: {meas_date}")
    
    auditory_delay_ms = matched_auditory_delay.iloc[-1]['Mean Delay Rounded (ms)']
    visual_delay_ms = matched_visual_delay.iloc[-1]['Mean Delay Rounded (ms)']
    
    sfreq = raw.info['sfreq']
    auditory_delay_samples = int(round(auditory_delay_ms * sfreq / 1000))
    visual_delay_samples = int(round(visual_delay_ms * sfreq / 1000))
    
    auditory_visual_delay_samples = int(round((auditory_delay_samples + visual_delay_samples) / 2))
    
    return meas_date, auditory_delay_samples, visual_delay_samples, auditory_visual_delay_samples

# =============================================================================
# Function to apply delays to events
# =============================================================================
def apply_delays_to_events(events, event_info, task, raw_file_path, auditory_delay_df, visual_delay_df):
    delay_applied = np.zeros(len(events), dtype=int)
    if task in event_info and 'delays' in event_info[task]:
        meas_date, delay_audio, delay_vis, delay_audio_vis = get_delays_for_fif(
            raw_file_path, auditory_delay_df, visual_delay_df
        )
        delay_values = {"audio": delay_audio, "visual": delay_vis, "audio_visual": delay_audio_vis}
        task_delays = {
            event_code: delay_values[delay_type]
            for event_code, delay_type in event_info.get(task, {}).get('delays', {}).items()
        }
        for i, event in enumerate(events):
            event_code = event[2]
            if event_code in task_delays:
                delay_samples = task_delays[event_code]
                event[0] += delay_samples
                delay_applied[i] = delay_samples
    
    # Sort events by onset (first column) and keep delays in sync
    sort_idx = np.argsort(events[:, 0])
    events = events[sort_idx]
    delay_applied = delay_applied[sort_idx]
    
    return events, delay_applied

# =============================================================================
# Function to add delay information to events.tsv and events.json files
# =============================================================================

def add_delay_to_events_tsv_and_json(
    events_tsv,
    delay_applied,
    description="Delay in samples that was applied to the event onset, to correct for known auditory or visual stimulus presentation lag.",
    units="samples"
):
    """
    Add a 'delay_applied' column to the given BIDS events.tsv file, and document it in the events.json file.

    Parameters
    ----------
    events_tsv : str
        Path to the events.tsv file.
    delay_applied : array-like
        Values of delay for each event (must match the number of rows in events.tsv *excluding* BAD_ACQ_SKIP).
    description : str
        Description for the 'delay_applied' field in events.json.
    units : str
        Units for the 'delay_applied' field.
    """
    if not (os.path.exists(events_tsv) and delay_applied is not None):
        print(f"[INFO] File not found or no delays: {events_tsv}")
        return

    df_events = pd.read_csv(events_tsv, sep='\t')
    # We want to add 'n/a' for BAD_ACQ_SKIP, and use delay_applied for all other rows
    out_col = []
    delay_idx = 0

    for _, row in df_events.iterrows():
        if 'trial_type' in row and row['trial_type'] == 'BAD_ACQ_SKIP':
            out_col.append(0)
        else:
            if delay_idx < len(delay_applied):
                out_col.append(delay_applied[delay_idx])
                delay_idx += 1
            else:
                # Something is mismatched, fail early
                raise ValueError("Not enough entries in delay_applied for the number of non-BAD_ACQ_SKIP events!")
    if delay_idx != len(delay_applied):
        raise ValueError("delay_applied vector is longer than the number of non-BAD_ACQ_SKIP events!")
    df_events['delay_applied'] = out_col
    df_events.to_csv(events_tsv, sep='\t', index=False)

    events_json = events_tsv.replace('_events.tsv', '_events.json')
    if os.path.exists(events_json):
        with open(events_json, 'r') as f:
            events_dict = json.load(f)
    else:
        events_dict = {}
    events_dict['delay_applied'] = {
        "Description": description,
        "Units": units
    }
    with open(events_json, 'w') as f:
        json.dump(events_dict, f, indent=4)

