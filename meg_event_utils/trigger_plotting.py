import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

#===============================================================================
def create_basic_plot(data, times, channel_labels, show_channels=None, title="STI Channels Time Course"):
    """
    Create a basic plot of the STI channel time courses.
    
    Parameters:
    -----------
    data : numpy.ndarray (n_channels, n_samples)
        Time courses of the STI channels.
    times : numpy.ndarray (n_samples,)
        Time points corresponding to the data samples.
    channel_labels : list
        List of channel names corresponding to the columns in the data array.
    show_channels : list or None
        List of channel names to display. If None, all channels are displayed.
        
    Returns:
    --------
    plt : matplotlib.pyplot object
        Matplotlib plot object.
    """
    
    data = (data != 0).astype(int)  # Convert to binary values

    # Ensure data is in (n_samples, n_channels) shape
    if data.shape[0] < data.shape[1]:  # If channels are rows and samples are columns, transpose it
        data = data.T  # Convert to (n_samples, n_channels)
    
    if show_channels is not None:
        # Find indices of selected channels in the original list
        selected_indices = [i for i, ch in enumerate(channel_labels) if ch in show_channels]
        
        # Use these indices to filter both channel names and data
        channel_labels = [channel_labels[i] for i in selected_indices]
        data = data[:, selected_indices] if data.ndim > 1 else data[selected_indices].reshape(-1, 1)

    # Ensure data is always 2D
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    num_channels = data.shape[1]  # Get the actual number of channels after filtering

    # Define vertical offset for spacing
    offset_step = 2  # Adjust for more or less spacing
    y_offsets = np.arange(0, len(channel_labels) * offset_step, offset_step)

    # Get the default Matplotlib color cycle
    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    channel_colors = {ch: default_colors[i % len(default_colors)] for i, ch in enumerate(channel_labels)}

    # Plot each active STI channel with offset
    for i, (channel, offset) in enumerate(zip(channel_labels, y_offsets)):
        plt.plot(times, data[:, i] + offset, label=channel, color=channel_colors[channel], alpha=0.7)

    # Set custom y-ticks and labels
    plt.yticks(y_offsets, channel_labels)

    plt.xlabel("Time (s)")
    # plt.ylabel("STI Channels")
    plt.title(title)

    return plt

#===============================================================================
def create_removed_event_plot(time_series, times, event_names, removed_events, y_offset=2, title="Event Time Courses"):
  
  binary_time_series = (time_series != 0).astype(int)

  plt = create_basic_plot(binary_time_series, times, event_names, title=title)

  channel_positions = {ch: i * y_offset for i, ch in enumerate(event_names)}  # Map channels to y-axis positions

  # Overlay markers for removed events
  removed_times = [times[t] for t, _ in removed_events]  # Convert time indices to actual times
  removed_y_positions = [channel_positions[event_names[ch]] for _, ch in removed_events]  # Get y-axis positions
  # Add markers for removed events
  plt.scatter(removed_times, removed_y_positions, color='red', marker='x', alpha=0.5, s=50, label="Removed Events")
  
  return plt

#===============================================================================
def create_interactive_plot(data, times, channel_lables, show_channels=None, downsampling_factor=None, title="STI Channels Time Course", hovermode="x unified"):
    """
    Create an interactive plot of the STI channel time courses using Plotly.

    Parameters:
    -----------
    data : numpy.ndarray (n_samples, n_channels)
        Time courses of the STI channels.
    times : numpy.ndarray (n_samples,)
        Time points corresponding to the data samples.
    channel_lables : list
        List of channel names corresponding to the columns in the data array.
    show_channels : list or None
        List of channel names to display. If None, all channels are displayed.
    downsampling_factor : int or None
        Downsampling factor to reduce the number of time points. If None, no downsampling is applied.
    title : str
        Title of the plot.
    hovermode : str pr False
        Hover mode for the plotly figure

    Returns:
    --------
    fig : plotly.graph_objects.Figure object
        Plotly figure object.
    """

    data = (data != 0).astype(int)  # Convert to binary values
    
    if show_channels is not None:
        # Find indices of selected channels in the original list
        selected_indices = [i for i, ch in enumerate(
            channel_lables) if ch in show_channels]

        # Use these indices to filter both channel names and data
        channel_lables = [channel_lables[i] for i in selected_indices]
        data = data[:, selected_indices]

    if downsampling_factor is not None:
        times = times[::downsampling_factor]
        data = data[::downsampling_factor, :]

    
    # Reverse the order to match the y-axis in the plot
    channel_lables = channel_lables[::-1]
    data = data[:, ::-1]  # Reverse data to match the new order

    # Define vertical offset for spacing (Reverse order)
    offset_step = 2
    y_offsets = np.arange(0, len(channel_lables) * offset_step, offset_step)[::-1]
    
    # Get the default Matplotlib color cycle
    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    channel_colors = {ch: default_colors[i % len(
        default_colors)] for i, ch in enumerate(channel_lables)}

    fig = go.Figure()

    # Add each STI channel as a separate trace
    for i, (channel, offset) in enumerate(zip(channel_lables, y_offsets)):
      hover_values = np.where(data[:, i] != 0, f"{channel} ON", "")  # Show '1' only if active, otherwise empty
      
      fig.add_trace(go.Scatter(
          x=times,
          y=data[:, i] + offset,  # Apply vertical offset
          mode='lines',
          name=channel,
          line=dict(color=channel_colors[channel], width=1),
          # hovertemplate="%{y}" + "%{y:.1f}" + "<extra></extra>",  # Show None when y=0
          # hovertemplate="%{y}" + "<extra></extra>", 
          hovertemplate="%{customdata}<extra></extra>", 
          customdata=hover_values, 
          showlegend=False 
      ))

    # Customize layout
    fig.update_layout(
        title=title,
        xaxis_title="Time (s)",
        yaxis_title="STI Channels",
        yaxis=dict(
            tickmode="array",
            tickvals=y_offsets,
            ticktext=channel_lables  # Label each channel at the offset
        ),
        template="plotly_white",
        hovermode=hovermode,  # "x unified",
        # hoverlabel=dict(namelength=0),  # Hide hover label names
    )
    
    return fig