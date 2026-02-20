import numpy as np
import mne
from collections import namedtuple
from .trigger_misc import decode_sti_value_full_info

#===============================================================================
def decompose_sti101_in_individual_channels(raw_file=None, data=None, times=None, raw=None, verbose=True):
  """
  Decompose the STI101 channel into individual channels and create a time-series matrix.

  Parameters:
  -----------
  raw_file : str, optional
    Path to the raw MEG file in FIF format.
  data : numpy.ndarray, optional
    Preloaded data array containing the STI101 channel time series.
  times : numpy.ndarray, optional
    Preloaded array containing the time points corresponding to the data samples.
  raw : mne.io.Raw, optional
    Raw MEG data object.
  verbose : bool, optional
    If True, prints the unique values and individual channels. If False, suppresses printing. Default is True.

  Returns:
  --------
  time_series_array : numpy.ndarray
    Array containing the time series for each individual channel.
  channel_names : list of str
    List of individual channel names.
  raw : mne.io.Raw
    Raw MEG object containing metadata like sampling frequency.

  Example:
  --------
  >>> time_series_array, channel_names = decompose_sti101_in_individual_channels('sample_raw.fif')
  Unique values found: [1, 2, 3, 4, 5, 4096]
  Individual channels: ['STI001', 'STI002', 'STI003', 'STI013', 'STI016']
  """
  # Load raw STI101 data
  if raw is None:
    if raw_file is not None:
      raw = mne.io.read_raw_fif(raw_file, preload=False, verbose="WARNING")
      data, times = raw.get_data(picks="STI101", return_times=True)
    elif data is None or times is None:
      raise ValueError("Either raw_file or both data and times must be provided.")
  else:
    data, times = raw.get_data(picks="STI101", return_times=True)

  # Extract unique STI values (ignoring 0)
  unique_values = np.unique(data[data != 0]).astype(int)
  if unique_values.size == 0 and verbose:
    print("No events detected in STI101.")
    return np.empty((0, 0), dtype=int), [], raw

  # Decode values into individual STI channels
  decoded = {val: decode_sti_value_full_info(val) for val in unique_values}
  individual_channels = sorted({ch for val, (channels, _) in decoded.items() for ch in channels})
  
  if verbose:
    print(f"Unique values found: {unique_values.tolist()}; Individual channels: {individual_channels}")

  # Precompute channel indices for fast access
  channel_indices = {ch: idx for idx, ch in enumerate(individual_channels)}

  # Initialize the time-series matrix
  num_samples, num_channels = len(times), len(individual_channels)
  time_series_array = np.zeros((num_samples, num_channels), dtype=int)

  # Fill in the matrix efficiently
  for t in range(num_samples):
    current_value = int(data[0, t])
    if current_value != 0:
      channels, bits = decode_sti_value_full_info(current_value)
      for ch, bit in zip(channels, bits):
        time_series_array[t, channel_indices[ch]] = bit

  return time_series_array, individual_channels, raw

#===============================================================================
def clean_sti101_timeseries(data101, sti_channels, min_samples=2, max_button_samples=20000, verbose=True, steps=["remove_long_press", "remove_sti003", "remove_short_events", "remove_isolated_events"]):
    """
    This function applies a series of cleaning steps to the STI101 time series data,
    including removing events during prolonged button presses, eliminating short events,
    and addressing specific artifacts related to STI003 channel activity.

    Args:
      data101 (np.ndarray): A 2D NumPy array representing the STI101 time series data
        (time x channels).
      sti_channels (list): A list of strings representing the names of the STI channels.
        This list is used to map channel names to their indices in the data array.
      min_samples (int, optional): The minimum number of consecutive samples required for
        an event to be considered valid. Events shorter than this duration are removed.
        Defaults to 2.
      max_button_samples (int, optional): The maximum duration (in samples) that the STI016
        button can be pressed continuously. If the button is pressed for longer than this,
        all events during that period are removed. Defaults to 20000.
      verbose (bool, optional): If True, prints detailed information about the cleaning steps
        and the number of events removed. Defaults to True.
    steps (list, optional): A list of strings specifying the cleaning steps to apply.
        By default, all steps are applied in the following order:
        - remove_long_press: Remove events during prolonged button presses.
        - remove_sti003: Remove STI003 events that start during STI013, STI015 or STI016 activity.
        - remove_short_events: Remove short events that are shorter than min_samples.
        - remove_isolated_events: Remove isolated events that are not preceded or followed by
          other events within min_samples. 
          Defaults to ["remove_long_press", "remove_sti003", "remove_short_events", "remove_isolated_events"].

    Returns:
      tuple: A tuple containing two elements:
        - filtered_time_series (np.ndarray): The cleaned STI101 time series data.
        - removed_events (list): A list of tuples, where each tuple contains the
          (start_time, channel_index) of a removed event.

    """
    # Create a copy of the original time series to modify

    filtered_time_series = data101.copy()
    removed_events = []  # List to store removed event locations

    # Create a binary version for event detection (preserves original values)
    binary_time_series = (data101 != 0).astype(int)

    # Identify channel indices
    channel_indices = {name: i for i, name in enumerate(sti_channels)}

    channel_indices = {name: i for i, name in enumerate(sti_channels)}

    # The problematic channels
    sti003_idx = channel_indices.get("STI003")
    sti013_idx = channel_indices.get("STI013")
    sti015_idx = channel_indices.get("STI015")
    sti016_idx = channel_indices.get("STI016")
    
    all_channels = [i for name, i in channel_indices.items() if name.startswith("STI")]

    # Identify STI001-STI008 channel indices
    sti001_to_sti008_indices = [i for name, i in channel_indices.items() 
                                if name.startswith("STI") and name[3:].isdigit() 
                                and 1 <= int(name[3:]) <= 8]
    # Step 1: Remove **all events** if **STI016** is on continuously for more than **X seconds**
    if "remove_long_press" in steps and sti016_idx is not None:
      
        sti016_changes = np.diff(np.concatenate(([0], binary_time_series[:, sti016_idx], [0])))  # Detect button press on/off. The values in this array will be either 1 (button press start), -1 (button press end), or 0 (no change).
        
        sti016_starts = np.where(sti016_changes == 1)[0]  # Indices where button press starts
        sti016_ends = np.where(sti016_changes == -1)[0]   # Indices where button press ends

        for start, end in zip(sti016_starts, sti016_ends):
            button_duration = end - start  # Duration of the button press
            
            if button_duration > max_button_samples:  # Check if sti016 press is too long
                for j in all_channels:  # Remove all events occurring during this time
                    filtered_time_series[start:end, j] = 0  
                    removed_events.append((start, j))  
                
                if verbose:
                    print(f"Removed ALL events from {start} to {end} because STI016 was pressed for over {max_button_samples} samples.")
        
        #  Recalculate binary_time_series 
        binary_time_series = (filtered_time_series != 0).astype(int) 
                        
    # Step 2: Remove **STI003** if it started while **STI015 or STI016** was active,
    # but only at timepoints where no other STI001-STI008 channels are active
    if "remove_sti003" in steps and sti003_idx is not None and (sti013_idx is not None or sti015_idx is not None or sti016_idx is not None):
        #  Recalculate binary_time_series 
        binary_time_series = (filtered_time_series != 0).astype(int) 
        
        sti003_changes = np.diff(np.concatenate(([0], binary_time_series[:, sti003_idx], [0])))  # Detect sti003 onsets
        sti003_starts = np.where(sti003_changes == 1)[0]  
        sti003_ends = np.where(sti003_changes == -1)[0]   

        for start, end in zip(sti003_starts, sti003_ends):
            # Check if sti013 or sti015 OR sti016 was active at the onset
            if (sti013_idx is not None and binary_time_series[start, sti013_idx] == 1) or \
              (sti015_idx is not None and binary_time_series[start, sti015_idx] == 1) or \
                (sti016_idx is not None and binary_time_series[start, sti016_idx] == 1):
                
                # Instead of removing the entire duration, check each timepoint
                removed_count = 0
                for t in range(start, end):
                    # Check if any other STI001-STI008 channels are active at this timepoint
                    other_sti_active = False
                    for idx in sti001_to_sti008_indices:
                        if idx != sti003_idx and binary_time_series[t, idx] == 1:
                            other_sti_active = True
                            break
                    
                    # Only remove STI003 at this timepoint if no other STI001-STI008 channels are active
                    if not other_sti_active:
                        filtered_time_series[t, sti003_idx] = 0
                        removed_events.append((t, sti003_idx))
                        removed_count += 1
                
                if verbose and removed_count > 0:
                    print(f"STI003 came ON while STI013/015/016 was ON; Removed STI003 {removed_count} timepoints between {start}-{end} where no other STI001-STI008 channels were active.")
                    
                # Recalculate binary_time_series 
                binary_time_series = (filtered_time_series != 0).astype(int)  

    # Step 3: Remove short events
    if "remove_short_events" in steps:
        for i in range(filtered_time_series.shape[1]):  
            event_changes = np.diff(np.concatenate(([0], binary_time_series[:, i], [0])))  # Use binary version for event detection
            event_starts = np.where(event_changes == 1)[0]  # Indices where events start
            event_ends = np.where(event_changes == -1)[0]   # Indices where events end

            for start, end in zip(event_starts, event_ends):
                duration = end - start  # Calculate event duration in timepoints
                
                if duration < min_samples:
                    filtered_time_series[start:end, i] = 0  
                    removed_events.append((start, i)) 
                    # Recalculate binary_time_series 
                    binary_time_series = (filtered_time_series != 0).astype(int)  
                    if verbose:
                        print(f"Removed short {sti_channels[i]} event at timepoints {start} to {end}.")

    # Step 4: Remove single-channel non-button events if they are immediately followed by a summed non-button event
    if "remove_isolated_events" in steps:
        for i in sti001_to_sti008_indices:
            for t in range(len(filtered_time_series) - 1):  # Iterate over timepoints
                if (
                    filtered_time_series[t, i] != 0 and  # Current timepoint has an event
                    np.sum(filtered_time_series[t, sti001_to_sti008_indices]) == filtered_time_series[t, i] and  # It's the only active event
                    np.sum(filtered_time_series[t + 1, sti001_to_sti008_indices]) > filtered_time_series[t, i]  # Next timepoint has a summed event
                ):
                    filtered_time_series[t, i] = 0  # Remove the isolated event
                    removed_events.append((t, i))
                    
                    # Recalculate binary_time_series 
                    binary_time_series = (filtered_time_series != 0).astype(int)
                    
                    if verbose:
                        print(f"Removed isolated {sti_channels[i]} at timepoint {t} because a summed event followed.")
    
    return filtered_time_series, removed_events

#===============================================================================
def sum_stim_channels_and_find_events(cleaned_time_series, individual_channels, raw, stim_range=(1, 9), min_duration=0.002, shortest_event=1, consecutive=True, verbose=True):
  """
  Sum stimulus channels and detect events from cleaned time series data.

  This function takes the cleaned time series for individual STI channels, 
  sums the stimulus channels (STI001-STI008), and uses MNE to find events.
  
  Parameters:
  -----------
  cleaned_time_series : numpy.ndarray
    Array containing the cleaned time series for each individual channel.
  individual_channels : list of str
    List of individual channel names corresponding to the columns in cleaned_time_series.
  raw : mne.io.Raw
    Raw MEG object containing metadata like sampling frequency.
  stim_range : tuple, optional
    Range of STI channel numbers to consider as stimulus channels (default: (1, 9)).
  min_duration : float, optional
    Minimum duration (in seconds) for an event to be considered valid (default: 0.002).
  shortest_event : int, optional
    Minimum number of samples for an event to be considered valid (default: 1).
  consecutive : bool, optional
    If True, only considers consecutive events as valid (default: True).
  verbose : bool, optional
    If True, prints verbose output (default: True).
  

  Returns:
  --------
  events : numpy.ndarray
    Array containing the detected events.
  event_counts : dict
    Dictionary containing the count of each event.
  """
  
  # Define stimulus channels (STI001 - STI008)
  stimulus_channels = [f'STI{str(i).zfill(3)}' for i in range(*stim_range)]
  if verbose:
    print(f"Stimulus channels considered for summing: {stimulus_channels}")

  # Get indices of stimulus channels that exist in `individual_channels`
  stimulus_indices = np.array([i for i, ch in enumerate(individual_channels) if ch in stimulus_channels])

  # Sum the values across stim channels
  stimulus_codes = np.sum(cleaned_time_series[:, stimulus_indices], axis=1, keepdims=True)  # Faster reshape

  # Get indices and names of non-stimulus channels
  button_indices = np.setdiff1d(np.arange(len(individual_channels)), stimulus_indices)  # Set operation is faster
  button_channel_names = [f'new{individual_channels[i]}' for i in button_indices]

  # Extract button channels' codes
  button_codes = cleaned_time_series[:, button_indices]

  # Merge stimulus and button codes
  event_codes = np.hstack([stimulus_codes, button_codes]) 

  # Create dummy info and raw object for MNE
  ch_names = ['summedSTI'] + button_channel_names
  info = mne.create_info(ch_names, raw.info['sfreq'], ['stim'] + ['misc'] * len(button_channel_names)) 

  # Create MNE Raw object
  raw_temp = mne.io.RawArray(event_codes.T, info, first_samp=raw.first_samp, verbose="WARNING")

  # Detect events
  events = mne.find_events(
      raw_temp, 
      stim_channel=ch_names,
      min_duration=min_duration, # not neccessary if channels cleaned properly, but doesn't harm to have it
      shortest_event=shortest_event,
      consecutive=consecutive,
      uint_cast=True,
      verbose='WARNING'
  )
  
  # Count events
  event_counts = mne.count_events(events, ids=None)
  
  return events, event_counts

#===============================================================================
def translate_channels_to_events(time_series_array, channel_names):
    """
    Translate individual STI channels to events and create a new time-series matrix.

    Parameters:
    -----------
    time_series_array : numpy.ndarray
        Array containing the time series for each individual channel.
    channel_names : list of str
        List of individual channel names.

    Returns:
    --------
    event_time_series : numpy.ndarray
        Array containing the time series for each individual event.
    event_names : list of str
        List of individual event names.
    """
    
    bitwise_channel_map = {
        'STI001': 1, 'STI002': 2, 'STI003': 4, 'STI004': 8, 'STI005': 16,
        'STI006': 32, 'STI007': 64, 'STI008': 128
    }

    label_channel_map = {
        'STI009': 'btnL_1', 'STI010': 'btnL_2', 'STI011': 'btnL_3', 'STI012': 'btnL_4',
        'STI013': 'btnR_1', 'STI014': 'btnR_2', 'STI015': 'btnR_3', 'STI016': 'btnR_4'
    }

    # Get indices for the first 8 channels (STI001 to STI008)
    first_8_indices = [i for i, ch in enumerate(channel_names) if ch in bitwise_channel_map]
    event_series = np.zeros(time_series_array.shape[0], dtype=int)  # Initialize event series

    # Iterate over each channel in STI001-STI008 and apply the numeric mapping
    for i in first_8_indices:
        event_series += (time_series_array[:, i] != 0) * bitwise_channel_map[channel_names[i]]

    # Identify timepoints where multiple first 8 channels are active
    multi_channel_mask = np.sum(time_series_array[:, first_8_indices] != 0, axis=1) > 1

    # Extract unique event values that occur when multiple channels are active
    unique_events = np.unique(event_series[multi_channel_mask])
    unique_events = unique_events[unique_events > 0]  # Exclude zero

    # Initialize new event columns (one per unique event)
    event_columns = np.zeros((time_series_array.shape[0], len(unique_events)), dtype=int)

    # Assign events to respective columns
    for idx, event_value in enumerate(unique_events):
        event_columns[:, idx] = (event_series == event_value) * event_value

    # Remove values from individual channels only where a new event was created
    filtered_time_series = time_series_array.copy()
    filtered_time_series[np.ix_(multi_channel_mask, first_8_indices)] = 0  # Fix shape mismatch

    # Concatenate the new event series with the filtered time-series data
    processed_time_series = np.hstack((filtered_time_series, event_columns))

    # Rename existing STI channels to "Event_X" for numeric mappings
    renamed_channels = [
        f"Event_{bitwise_channel_map[ch]}" if ch in bitwise_channel_map else label_channel_map.get(ch, ch)
        for ch in channel_names
    ]

    # Add new event column names
    processed_event_names = renamed_channels + [f"Event_{event}" for event in unique_events]

    # Remove columns that are entirely zero
    nonzero_mask = np.any(processed_time_series != 0, axis=0)
    event_time_series = processed_time_series[:, nonzero_mask]
    event_names = [name for name, keep in zip(processed_event_names, nonzero_mask) if keep]
    
    
    # Get sorted indices based on processed_event_names
    sorted_indices = np.argsort(event_names)
    event_time_series = event_time_series[:, sorted_indices]
    event_names = [event_names[i] for i in sorted_indices]
    
    return event_time_series, event_names

#===============================================================================
def create_event_onsets(time_series_array, series_names, raw, verbose=True):
    info = mne.create_info(series_names, raw.info['sfreq'], 'stim')

    raw_temp = mne.io.RawArray(time_series_array, info, first_samp=raw.first_samp, verbose="WARNING")
    
    events = mne.find_events(
      raw_temp, 
      stim_channel=series_names, 
      consecutive='increasing', 
      #min_duration=0.002,
      uint_cast=True, 
      verbose='WARNING')  

    event_counts = mne.count_events(events, ids=None)

    if verbose:
      print("\nEvent Counts")
      for event, count in event_counts.items():
        print(f"    Event {event}: {count} occurrences")  

    return events, event_counts
  
#===============================================================================
def find_corrected_events(raw_file=None, raw=None, cleaning_steps=["remove_long_press", "remove_sti003", "remove_short_events", "remove_isolated_events"], min_duration=0, shortest_event=1, consecutive=True, verbose=True):
  """
  Find corrected events from raw MEG data file.

  This function decomposes the STI101 channel into individual channels, cleans the time series data,
  sums the stimulus channels, and detects events using MNE.

  Parameters:
  -----------
  raw_file : str, optional
    Path to the raw MEG file in FIF format.
  raw : mne.io.Raw, optional
    Raw MEG data object.
  verbose : bool, optional
    If True, prints the unique values and individual channels. If False, suppresses printing. Default is True.

  Returns:
  --------
  EventResults : namedtuple
    A namedtuple containing the following fields:
    - events : numpy.ndarray
        Array containing the detected events.
    - event_counts : dict
        Dictionary containing the count of each event.
    - time_series_array : numpy.ndarray
        Array containing the time series for each individual channel.
    - individual_channels : list of str
        List of individual channel names.
    - raw : mne.io.Raw
        Raw MEG object containing metadata like sampling frequency.
    - cleaned_time_series : numpy.ndarray
        Array containing the cleaned time series for each individual channel.
    - removed_events : list
        List of tuples, where each tuple contains the (start_time, channel_index) of a removed event.
  Example:
  --------
  >>> event_results = find_corrected_events('sample_raw.fif')
  >>> print(event_results.events, event_results.cleaned_time_series)
  """
  
  EventResults = namedtuple("EventResults", ["events", "event_counts", "time_series_array", "individual_channels", "raw", "cleaned_time_series", "removed_events"])

  
  # Decompose STI101 into individual channels
  time_series_array, individual_channels, raw = decompose_sti101_in_individual_channels(raw_file=raw_file, raw=raw, verbose=verbose)

  # Clean STI101 time series data
  cleaned_time_series, removed_events = clean_sti101_timeseries(time_series_array, individual_channels, steps=cleaning_steps, verbose=verbose)

  # Sum stimulus channels and find events
  events, event_counts = sum_stim_channels_and_find_events(cleaned_time_series, individual_channels, raw, min_duration=min_duration, shortest_event=shortest_event, consecutive=consecutive)

  return EventResults(events, event_counts, time_series_array, individual_channels, raw, cleaned_time_series, removed_events)