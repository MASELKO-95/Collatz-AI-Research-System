import requests
import json
import os
import time

WEBHOOK_URL = "https://discord.com/api/webhooks/1443702583309893683/ESj5ptdBSE0EQ3NPY4LHFxMiXDToxnC3kP0rKF6tl7PDvGqxEzJsoiPTovWJS0kVha8y"

def send_message(content, embeds=None, files=None):
    """
    Sends a message to the Discord webhook.
    """
    data = {
        "content": content,
        "username": "Collatz AI",
        "avatar_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a2/Collatz-graph-all-30-no-labels.svg/1200px-Collatz-graph-all-30-no-labels.svg.png"
    }
    
    if embeds:
        data["embeds"] = embeds
        
    # If sending files, we need to use multipart/form-data
    if files:
        # For files, the JSON payload must be passed as a separate field 'payload_json'
        response = requests.post(
            WEBHOOK_URL,
            data={"payload_json": json.dumps(data)},
            files=files
        )
    else:
        response = requests.post(
            WEBHOOK_URL,
            json=data
        )
        
    if response.status_code not in [200, 204]:
        print(f"Failed to send Discord message: {response.status_code} - {response.text}")

def send_status_update(step, loss, stopping_loss, seq_loss):
    embed = {
        "title": f"Training Status - Step {step}",
        "color": 3447003, # Blue
        "fields": [
            {"name": "Total Loss", "value": f"{loss:.4f}", "inline": True},
            {"name": "Stopping Time Loss", "value": f"{stopping_loss:.4f}", "inline": True},
            {"name": "Sequence Loss", "value": f"{seq_loss:.4f}", "inline": True}
        ],
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())
    }
    send_message("", embeds=[embed])

def send_anomaly_alert(number, actual_stop, predicted_stop, error, image_path=None):
    embed = {
        "title": "🚨 Anomaly Detected! 🚨",
        "description": f"The model found a number that behaves unexpectedly.",
        "color": 15158332, # Red
        "fields": [
            {"name": "Number", "value": str(number), "inline": True},
            {"name": "Actual Stopping Time", "value": str(actual_stop), "inline": True},
            {"name": "Predicted Stopping Time", "value": f"{predicted_stop:.2f}", "inline": True},
            {"name": "Prediction Error", "value": f"{error:.2f}", "inline": True},
            {"name": "Analysis", "value": f"The model expected this number to stop in ~{int(predicted_stop)} steps, but it took {actual_stop}. This deviation of {error:.2f} suggests a pattern the AI hasn't learned yet."}
        ],
        "footer": {"text": "Training paused for analysis."}
    }
    
    files = {}
    if image_path and os.path.exists(image_path):
        files["file"] = (os.path.basename(image_path), open(image_path, "rb"))
        
    send_message("", embeds=[embed], files=files)

def send_analysis_report(summary_text, image_paths=[]):
    files = {}
    for i, path in enumerate(image_paths):
        if os.path.exists(path):
            files[f"file{i}"] = (os.path.basename(path), open(path, "rb"))
            
    send_message(f"**Analysis Report**\n{summary_text}", files=files)
