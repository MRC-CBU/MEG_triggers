import io
import contextlib
import logging
import matplotlib.pyplot as plt
import base64

#===============================================================================
# -----------------------------------------------------------------------------------------------
# Function to print event counts in a table format
# -----------------------------------------------------------------------------------------------
def print_event_counts_table(event_counts):
    events = list(event_counts.keys())
    counts = list(event_counts.values())

    print("Event Counts:")
    print("-" * (len(events) * 8))
    print(" | ".join(f"{e:^6}" for e in events))  # Centered event IDs
    print("-" * (len(events) * 8))
    print(" | ".join(f"{c:^6}" for c in counts))  # Centered counts
    print("-" * (len(events) * 8))
    
#===============================================================================    
# -----------------------------------------------------------------------------------------------
# Function to capture standard print output and logging
# -----------------------------------------------------------------------------------------------
def capture_print_output(func, *args, **kwargs):
    buffer = io.StringIO()  # String buffer for capturing
    log_stream = io.StringIO()  # Separate stream for logging
    
    # Temporarily redirect stdout and logging output
    handler = logging.StreamHandler(log_stream)
    logging.getLogger().addHandler(handler)
    
    try:
        with contextlib.redirect_stdout(buffer):
            result = func(*args, **kwargs)  # Capture function output
    finally:
        logging.getLogger().removeHandler(handler)  # Restore logging
    
    output_text = buffer.getvalue()
    log_text = log_stream.getvalue()
    
    return output_text + "\n" + log_text, result  # Combine both outputs and return function result

#===============================================================================
# -----------------------------------------------------------------------------------------------
# Function to generate an HTML report
# -----------------------------------------------------------------------------------------------
def generate_html_report(subjects_data, phase, task, output_html="report.html", title="MEG Event Triggers"):
    html_content = f"""
    <html>
    <head>
        <title>{phase} {task} {title}</title>
        <style>
            body {{ font-family: Arial, sans-serif; }}
            .subject {{ margin-bottom: 20px; }}
            .subject h2 {{ color: #333; font-size: 1em; }}
            img {{ max-width: 100%; height: auto; }}
            pre {{ background-color: #f4f4f4; padding: 2px; border-radius: 2px; }}
        </style>
    </head>
    <body>
        <h1>{phase} {task} {title}</h1>
    """
    
    for subject, (task_fname, output_text, img_data) in subjects_data.items():
        html_content += f"""
        <div class='subject'>
            <h2>Subject: {subject}</h2>
            <pre>{output_text}</pre>
            <img src="data:image/png;base64,{img_data}" alt="{subject} Plot">
        </div>
        """
    
    html_content += "</body></html>"
    
    with open(output_html, "w") as f:
        f.write(html_content)