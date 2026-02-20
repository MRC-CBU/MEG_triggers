# MEG Trigger Utilities

**Author:** Dace Apšvalka, MRC CBU  
**Date:** February 2026  
**Requirements:** MNE-Python, `meg_event_utils`

---

A Python toolkit for extracting, inspecting, and cleaning event triggers from MEG recordings. Built to complement MNE-Python for real-world datasets where trigger channels are messy, incomplete, or contain artefacts.

## Background

In most MEG experiments, the stimulus delivery program sends event triggers to the MEG system, which are recorded on one or more trigger channels (e.g. `STI001`–`STI016`, `STI101`). These triggers are the backbone of epoching and condition labelling.

In practice, however, trigger recordings are often messy. Common problems include:

- **Only the summed `STI101` channel was saved**, with no individual bit channels
- **Overlapping triggers** — stimulus and keypress occurring simultaneously, creating ambiguous combined event codes
- **Spurious pulses** — false triggers that inflate event counts and contaminate analyses

This package provides targeted solutions for all of the above.

> 💡 This is not a polished, general-purpose package. It grew out of practical work on two legacy datasets acquired at the MRC CBU. That said, the inspection and quality-control steps should be relevant for most MEG datasets.

---

## Repository Structure

```
meg_event_utils/           # Python package
    __init__.py
    trigger_cleaning.py    # Decompose STI101, clean spurious/overlapping triggers, etc.
    trigger_misc.py        # Extract events, inspect STI channels, etc.
    trigger_plotting.py    # Plot STI channel time courses, etc.
    trigger_reporting.py   # Generate HTML reports, etc.
meg_trigger_utilities.ipynb  # Tutorial notebook demonstrating the usage
reports/                   # Example HTML reports generated in the tutorial
```

---

## Installation

**Option 1: Add to Python path**

```python
import sys
sys.path.append('/path/to/meg_event_utils')
```

**Option 2: Install in editable mode**

```bash
pip install -e /path/to/meg_event_utils
```

Then import as:

```python
import meg_event_utils as meu
```

---

## Usage

See the tutorial notebook [meg_trigger_utilities.ipynb](meg_trigger_utilities.ipynb) for example usage.


