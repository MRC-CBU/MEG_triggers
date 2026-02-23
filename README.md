# Utility functions for extracting, inspecting, and cleaning MEG trigger events on Neuromag Vectorview and MEGIN Neo MEG systems

**Author:** Dace Apšvalka, MRC CBU  
**Date:** February 2026  
**Requirements:** MNE-Python, `meg_event_utils`

---

> 💡 The functions in this package were developed specifically for **Neuromag Vectorview and MEGIN Neo MEG systems** at the MRC Cognition and Brain Sciences Unit (CBU). This is **not a polished, general-purpose package**: it grew out of practical work on two legacy datasets acquired at the CBU.

## Background

In most MEG experiments, the stimulus delivery program (e.g. PsychoPy, Presentation, E-Prime) sends event triggers to the MEG system, which are recorded on one or more trigger channels. These triggers are the backbone of epoching and condition labelling: they tell us what happened when.

Depending on the MEG system, triggers may be saved as multiple separate "bit" channels (each representing one digital line) and/or as a single summed channel, which encodes the binary state of all bit channels as an integer. For example, in the Neuromag Vectorview and MEGIN Neo MEG systems, there are 16 bit channels labelled **STI001** to **STI016**, and a summed channel labelled **STI101**. The value at each sample in STI101 is the weighted sum of the bit channels (STI001 × 1, STI002 × 2, STI003 × 4, etc.). Typically, the first 8 bit channels are used to encode **stimulus events** (allowing unique codes from 1 to 255 in STI101), while the remaining 8 channels are used to encode **button presses**. This can lead to values in STI101 ranging from 512 up to 65,535 for responses, including unique codes for simultaneous stimulus–response events and for multiple simultaneous button presses.

In practice, however, trigger recordings are often messy. Common problems include:

- **Only the summed channel was saved**, with no individual bit channels
- **Overlapping triggers** - stimulus and keypress occurring simultaneously, creating unexpected combined event codes
- **Spurious pulses** - false triggers that inflate event counts and contaminate analyses

Libraries such as MNE-Python already provide excellent tools for reading and extracting events from trigger channels in standard cases. However, in real-world datasets with spurious triggers or overlapping codes, additional inspection and custom handling are often needed. The functions in `meg_event_utils` are designed to complement existing workflows by making these edge cases easier to diagnose and fix.

---

## Repository Structure

```python
meg_event_utils/             # Python package
    __init__.py
    trigger_cleaning.py      # Decompose STI101, clean spurious/overlapping triggers, etc.
    trigger_misc.py          # Extract events, inspect STI channels, etc.
    trigger_plotting.py      # Plot STI channel time courses, etc.
    trigger_reporting.py     # Generate HTML reports, etc.
meg_trigger_utilities.ipynb  # Tutorial notebook demonstrating the usage
meg_trigger_utilities.html   # Tutorial notebook in HTML format
meg_trigger_utilities.pdf    # Tutorial notebook on PDF format
reports/                     # Example HTML reports generated in the tutorial
```

---

## Installation

### Option 1: Add to Python path

```python
import sys
sys.path.append('/path/to/meg_event_utils')
```

### Option 2: Install in editable mode

```bash
pip install -e /path/to/meg_event_utils
```

### Then import as:

```python
import meg_event_utils as meu
```

---

## Usage

See the tutorial notebook [meg_trigger_utilities.ipynb](meg_trigger_utilities.ipynb) for example usage.


