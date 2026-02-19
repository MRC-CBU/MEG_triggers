import numpy as np
import mne

#===============================================================================
def get_original_events(raw_file, consecutive='increasing', verbose=True):
  """
  Extract and count events from the raw MEG file.

  Parameters:
  -----------
  raw_file : str
    Path to the raw MEG file in FIF format.
  consecutive : bool|'increasing'
      The type of consecutive event detection to use based on MNE parameter for mne.find_events.
  verbose : bool, optional
        If True, prints event details. If False, suppresses printing. Default is True.

  Returns:
  --------
  events : numpy.ndarray
    Array containing the events detected in the raw MEG file.

  Example:
  --------
  >>> get_original_events(raw_file)
  Channel: STI001
    Event 5: 84 occurrences
  Channel: STI002
    Event 5: 80 occurrences
  Channel: STI003
    Event 5: 14 occurrences
  Channel: STI101
    Event 1: 38 occurrences
    Event 2: 40 occurrences
    Event 3: 40 occurrences
    Event 4: 4 occurrences
    Event 5: 4 occurrences
    Event 32768: 129 occurrences
    Event 32772: 6 occurrences
  """
  
  raw = mne.io.read_raw_fif(raw_file, preload=False, verbose=False)
  
  # Get the channels of interest
  event_channels = [channel for channel in raw.ch_names if channel in ['STI101'] + [f'STI{str(i).zfill(3)}' for i in range(1, 17)]]
  
  # for each channel, extract events
  for channel in event_channels: 
    
    # Extract events
    events = mne.find_events(raw, stim_channel=channel, min_duration=0.002, 
                            shortest_event=1, uint_cast=True, verbose=False, consecutive=consecutive)
    
    # if no events are found, skip to the next channel
    if len(events) == 0:
        continue
    
    if verbose:
      print(f"Channel: {channel}")      
      # Count events
      event_counts = mne.count_events(events, ids=None)
      
      # Display event counts for the subject
      for event, count in event_counts.items():
          print(f"    Event {event}: {count} occurrences")
        
  return events

#===============================================================================
def get_used_sti_channels(raw_file):
  """
  Get the list of used STI channels from the raw MEG file.

  Parameters:
  -----------
  raw_file : str
    Path to the raw MEG file in FIF format.

  Returns:
  --------
  used_sti_channels : list of str
    List of STI channel names that are used (i.e., contain non-zero values).

  Example:
  --------
  >>> get_used_sti_channels(raw_file)
  ['STI001', 'STI002', 'STI003', 'STI101', 'STI201', 'STI301']
  """
  raw = mne.io.read_raw_fif(raw_file, preload=False, verbose=False)
  sti_channels = [channel for channel in raw.ch_names if channel.startswith('STI')]
  
  used_sti_channels = []
  for channel in sti_channels:
    data = raw.copy().pick([channel]).get_data()
    if np.any(data != 0):
      used_sti_channels.append(channel)
      
  return used_sti_channels

#===============================================================================
def print_unique_channel_values(raw_file):
  """
  Get unique values for each STI channel in the raw MEG file.

  Parameters:
  -----------
  raw_file : str
    Path to the raw MEG file in FIF format.

  Returns:
  --------
  None
    Prints the unique values for each active STI channel.

  Example:
  --------
  >>> print_unique_channel_values(raw_file)
  Channel STI001 unique values: [0 5]
  Channel STI002 unique values: [0 5]
  Channel STI003 unique values: [0 5]
  Channel STI101 unique values: [-32768 -32764      0      1      2      3      4      5]
  Channel STI201 unique values: [   0  256  768 1792 3840 7936]
  Channel STI301 unique values: [-9 -8 -7 -6 -5 -4 -3 -2 -1  0  1  2  3  4  5  6  7  8  9]
  """
  # Load MEG raw data
  raw = mne.io.read_raw_fif(raw_file, preload=False, verbose=False)

  # Get all available STI channels
  available_sti_channels = [ch for ch in raw.ch_names if ch.startswith('STI')]

  # Dictionary to store unique values for each channel
  sti_values = {}

  # Iterate over each STI channel
  for sti_channel in available_sti_channels:
      # Extract data for this channel
      sti_data = raw.get_data(picks=[sti_channel])[0].astype(int)
      
      # Get unique values in this channel
      unique_values = np.unique(sti_data)
      
      # Store values if any of them are nonzero
      if np.any(unique_values):
          sti_values[sti_channel] = unique_values

  # Print unique values for each channel
  for channel, values in sti_values.items():
      print(f"Channel {channel} unique values: {values}") 
      
#===============================================================================
def get_sti101_values(raw_file=None, raw=None):
  """
  Get the unique values for STI101 channel in the raw MEG file.

  Parameters:
  -----------
  raw_file : str, optional
    Path to the raw MEG file in FIF format.
  raw : mne.io.Raw, optional
    Preloaded raw MEG data object.

  Returns:
  --------
  unique_values : numpy.ndarray
    Unique values for STI101 channel in the raw MEG file.

  Example:
  --------
  >>> get_sti101_values(raw_file='path/to/file.fif')
  array([-32768, -32764,      0,      1,      2,      3,      4,      5])
  >>> get_sti101_values(raw=raw)
  array([-32768, -32764,      0,      1,      2,      3,      4,      5])
  """
  if raw is None:
    if raw_file is None:
      raise ValueError("Either raw_file or raw must be provided.")
    raw = mne.io.read_raw_fif(raw_file, preload=False, verbose=False)

  # Extract data for STI101 channel
  sti101_data = raw.get_data(picks=['STI101'])[0].astype(int)
  
  # Get unique values in STI101 channel
  unique_values = np.unique(sti101_data)
  
  return unique_values

#===============================================================================
def decode_sti_value_full_info(value):
  """
  Decode a signed 16-bit integer STI value to individual channel bits and labels.

  Parameters:
  -----------
  value : int
    Signed 16-bit integer representing the STI value.

  Returns:
  --------
  channels : list of str
    List of channel labels corresponding to the active bits in the STI value.
  bits : list of int
    List of power-of-2 values representing the active bits in the STI value.

  Example:
  --------
  >>> decode_sti_value_full_info(32769)
  (['STI001', 'STI016'], [1, 32768])
  """
  
  if value < 0:
      value = (1 << 16) + value  # Convert negative values to unsigned

  indices = [i for i in range(16) if value & (1 << i)]  # Get active bit indices
  bits = [1 << i for i in indices]  # Convert indices to power-of-2 values
  channels = [f"STI{str(i+1).zfill(3)}" for i in indices]  # Convert to STI labels

  return channels, bits

#===============================================================================
def decode_sti_value(value):
  """
  Decode a signed 16-bit integer STI value to a combination of summed small bits and individual large bits.

  Parameters:
  -----------
  value : int
    Signed 16-bit integer representing the STI value.

  Returns:
  --------
  decoded_sti_value : int or list of int
    If the value contains only small bits (< 256), returns their sum.
    If the value contains only large bits (>= 256), returns a list of those bits.
    If the value contains both, returns a list with the sum of small bits followed by the large bits.

  Example:
  --------
  >>> decode_sti_value(32772)
  [4, 32768]
  >>> decode_sti_value(32773)
  [5, 32768]
  >>> decode_sti_value(3)
  3
  """
  if value < 0:
      value = (1 << 16) + value  # Convert negative values to unsigned

  indices = [i for i in range(16) if value & (1 << i)]  # Get active bit indices
  bits = [1 << i for i in indices]  # Convert indices to power-of-2 values

  # Separate bits into two groups: those < 256 and those >= 256
  small_bits = [b for b in bits if b < 256]  # Values to be summed
  large_bits = [b for b in bits if b >= 256]  # Keep as separate elements

  # Sum small values and combine with larger values
  if small_bits:
      summed_value = sum(small_bits)  # Sum bits < 256
      return [summed_value] + large_bits if large_bits else summed_value
  decoded_sti_value = large_bits if len(large_bits) > 1 else large_bits[0]  # Single large values remain as is
  
  return decoded_sti_value

#===============================================================================
def print_decoded_sti101_channel(raw_file, consecutive='increasing'):
    """
    Decode STI101 channel from a raw MEG file.

    Parameters:
    -----------
    raw_file : str
      Path to the raw MEG file in FIF format.
    consecutive : bool|'increasing'
      The type of consecutive event detection to use based on MNE parameter for mne.find_events.

    Returns:
    --------
    None
      Prints the decoded STI101 events and their counts.

    Example:
    --------
    >>> print_decoded_sti101_channel('sample_raw.fif')
    STI101 events:  [1, 2, 3, 4, 5, 32768, 32772]
    Decoded events: [1, 2, 3, 4, 5, 32768, [4, 32768]]
        Event 1: 38 occurrences
        Event 2: 40 occurrences
        Event 3: 40 occurrences
        Event 4: 4 occurrences
        Event 5: 4 occurrences
        Event 32768: 129 occurrences
        Event 32772: 6 occurrences
    """
    raw = mne.io.read_raw_fif(raw_file, preload=False, verbose=False)
    
    events = mne.find_events(raw, stim_channel='STI101', min_duration=0.002, shortest_event=1, uint_cast=True, verbose=False, consecutive=consecutive)
    
    if len(events) == 0:
        print("No events found.")
        return
    
    event_counts = mne.count_events(events, ids=None)
    sti101_values = list(event_counts.keys())

    # Decode STI101 values
    decoded_values = [decode_sti_value(val) for val in sti101_values]

    print(f"STI101 events:  {sti101_values}")
    print(f"Decoded events: {decoded_values}")
          # Display event counts for the subject
    for event, count in event_counts.items():
        print(f"    Event {event}: {count} occurrences")
        
#===============================================================================
def split_sti101(raw_file=None, data=None, times=None):
  """
  Decompose the STI101 channel into individual time series for each unique value.

  Parameters:
  -----------
  raw_file : str, optional
    Path to the raw MEG file in FIF format.
  data : numpy.ndarray, optional
    Preloaded data array containing the STI101 channel time series.
  times : numpy.ndarray, optional
    Preloaded array containing the time points corresponding to the data samples.

  Returns:
  --------
  data101 : numpy.ndarray
    Array containing the time series for each unique value in the STI101 channel.
  times : numpy.ndarray
    Array containing the time points corresponding to the data samples.
  sti_channels : list of str
    List of channel names corresponding to the columns in the data101 array.

  Example:
  --------
  >>> data101, times, sti_channels = split_sti101('sample_raw.fif')
  Unique values found: [1, 2, 3, 4, 5, 32768, 32772]
  Decoded channels: ['STI001', 'STI002', 'STI003', 'STI004', 'STI005', 'STI016', 'STI004 STI016']
  """
  if raw_file is not None:
    raw = mne.io.read_raw_fif(raw_file, preload=False, verbose=False)
    data, times = raw.get_data(picks='STI101', return_times=True)
  elif data is None or times is None:
    raise ValueError("Either raw_file or both data and times must be provided.")

  # Extract unique values that are greater than 0
  unique_values = np.unique(data)
  unique_values = unique_values[unique_values != 0].astype(int).tolist()
  print(f"Unique values found: {unique_values}")

  # Decode values to the original channels
  decoded = {val: decode_sti_value_full_info(val) for val in unique_values}
  sti_channels = [' '.join(channels) for val, (channels, bits) in decoded.items()]

  print(f"Decoded channels: {sti_channels}")

  # Create a time-series per each unique value
  data101 = np.zeros((len(times), len(unique_values)))
  for i, val in enumerate(unique_values):
    temp = data.copy()
    temp[temp != val] = 0  # Set all other values to 0
    temp[temp == val] = 1.5  # Set the selected value to 5
    data101[:, i] = temp  # Assign to the correct column

  return data101, times, sti_channels

#===============================================================================
def get_channel_timecourses(raw_file):
  """
  Extract and decode STI channel time courses from raw MEG data.
  Combines the original STI channels (STI001 to STI016) with additional channels decoded from STI101.
  The function returns the combined time courses, time points, and channel names.
  
  # Uses a function: decode_sti_value_full_info

  Parameters:
  -----------
  raw_file : str
    Path to the raw MEG data file.

  Returns:
  --------
  data : numpy.ndarray (n_samples, n_channels)
    Combined time courses of the original STI channels and additional ones decoded from STI101.
  times : numpy.ndarray (n_samples,)
    Time points corresponding to the data samples.
  channel_lables : list
    List of channel names corresponding to the columns in the data array.
  """
  # Load raw data
  raw = mne.io.read_raw_fif(raw_file, preload=False, verbose=False)
  
  print("\nGetting active channel timecourses...")

  # -----------------------------------------------------------------------------------------------
  # Get the original STI001 to STI016 channels
  # -----------------------------------------------------------------------------------------------
  original_channels = [channel for channel in raw.ch_names if channel in [f'STI{str(i).zfill(3)}' for i in range(1, 17)]]
  print(f"Original channels: {original_channels}")
  data_original = raw.get_data(picks=original_channels)

  # Identify which channels have at least one nonzero value
  active_channels = np.any(data_original, axis=1)

  # Get the corresponding channel names
  kept_original_channels = [name for name, keep in zip(original_channels, active_channels) if keep]
  removed_channels = [name for name, keep in zip(original_channels, active_channels) if not keep]

  # Filter out the inactive channels
  data_original = data_original[active_channels,:]
  
  # replace all non-zero values with 1
  data_original[data_original != 0] = 1.5 # default is 5

  print(f"Kept active Channels: {kept_original_channels}")
  print(f"Removed Channels: {removed_channels}")

  # -----------------------------------------------------------------------------------------------
  # Get the STI101 channel and decode each unique value to the original channels
  # -----------------------------------------------------------------------------------------------
  raw_sti101 = raw.copy().pick('STI101')
  data_101, times = raw_sti101.get_data(return_times=True) 

  # Extract unique values that are greater than 0
  unique_values = np.unique(data_101)
  unique_values = unique_values[unique_values != 0].astype(int).tolist()
  print(f"Unique values in STI101: {unique_values}")

  # Decode STI101 values to the original channels
  decoded = {val: decode_sti_value_full_info(val) for val in unique_values}
  sti101_channels = [' '.join(channels) for val, (channels, bits) in decoded.items()]

  print(f"Decoded STI101 values: {sti101_channels}")

  # Keep only the channels that are not in the original set
  mask = np.array([name not in kept_original_channels for name in sti101_channels])
  filtered_unique_values = np.array(unique_values)[mask].tolist()
  kept_sti101_channels = np.array(sti101_channels)[mask].tolist()
  
  print(f"Kept STI101 Channels: {kept_sti101_channels}")

  # # Create a time-series per each unique value in the filtered STI101
  data101 = np.zeros((len(times), len(filtered_unique_values)))
  for i, val in enumerate(filtered_unique_values):
      temp = data_101.copy()
      temp[temp != val] = 0  # Set all other values to 0
      temp[temp == val] = 1.5  # Set the selected value to 5
      data101[:, i] = temp  # Assign to the correct column

  # -----------------------------------------------------------------------------------------------

  # merge data and data_original
  data = np.concatenate((data101, data_original.T), axis=1)

  # and merge sti101_channels with original_channels
  channel_lables = kept_sti101_channels + kept_original_channels
  
  print(f"Final channel labels: {channel_lables}")
  
  return data, times, channel_lables

#===============================================================================
def create_event_onsets_version1(time_series_array, channel_names, raw, verbose=True):
    """
    Create corrected events from the decomposed STI101 channel time-series.

    This function should be used together with the decompose_sti101_in_individual_channels function.

    Parameters:
    -----------
    time_series_array : numpy.ndarray
      Array containing the time series for each individual channel.
    channel_names : list of str
      List of individual channel names.
    verbose : bool, optional
        If True, prints event details. If False, suppresses printing. Default is True.

    Returns:
    --------
    events : numpy.ndarray
      Array containing the corrected events.
    event_counts : dict
      Dictionary containing the count of each event.

    Example:
    --------
    >>> time_series_array, channel_names = decompose_sti101_in_individual_channels('sample_raw.fif')
    >>> events, event_counts = create_event_onsets(time_series_array, channel_names)
    """
    num_samples = time_series_array.shape[0]
    stim_channels = [f'STI{str(i).zfill(3)}' for i in range(1, 9)]
    response_channels = [f'STI{str(i).zfill(3)}' for i in range(9, 17)]

    # from time_series_array extract the columns that belong to stim_channels
    stim_indices = [channel_names.index(ch) for ch in stim_channels if ch in channel_names] 
    stim_data = time_series_array[:, stim_indices]

    # from time_series_array extract the columns that belong to response_channels
    resp_indices = [channel_names.index(ch) for ch in response_channels if ch in channel_names]
    resp_channel_names = [f'new{channel_names[i]}' for i in resp_indices]
    resp_data = time_series_array[:, resp_indices]

    sti101_new = np.sum(stim_data.T, axis=0, keepdims=True)

    event_data = np.concatenate([sti101_new, resp_data.T], axis=0)

    # Create dummy info and raw object to use find_events
    ch_names = ['newSTI101'] + resp_channel_names
    info = mne.create_info(ch_names, raw.info['sfreq'], 'stim')

    raw_temp = mne.io.RawArray(event_data, info, first_samp=raw.first_samp, verbose="WARNING")
    
    events = mne.find_events(
      raw_temp, 
      stim_channel=ch_names, 
      min_duration=0.002, 
      consecutive='increasing', 
      uint_cast=True, 
      verbose='WARNING')  

    event_counts = mne.count_events(events, ids=None)

    if verbose:
      print("\nEvent Counts")
      for event, count in event_counts.items():
        print(f"    Event {event}: {count} occurrences")  

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