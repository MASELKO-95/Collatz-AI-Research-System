import requests
import json
import os
import time

# Try to import config, fall back to defaults
try:
    from config import (
        DISCORD_WEBHOOK_URL,
        DISCORD_ALERT_WEBHOOK_URL,
        SEND_TRAINING_START,
        SEND_STATUS_UPDATES,
        SEND_ANOMALIES,
        SEND_CYCLE_ALERTS,
        SEND_MILESTONES
    )
except ImportError:
    print("⚠️  No config.py found. Discord notifications disabled.")
    print("💡 Copy config.example.py to config.py and add your webhook URL.")
    DISCORD_WEBHOOK_URL = ""
    DISCORD_ALERT_WEBHOOK_URL = ""
    SEND_TRAINING_START = False
    SEND_STATUS_UPDATES = False
    SEND_ANOMALIES = False
    SEND_CYCLE_ALERTS = False
    SEND_MILESTONES = False

def send_message(content, embeds=None, files=None, alert=False):
    """
    Sends a message to the Discord webhook.
    
    Args:
        content: Message text
        embeds: Discord embed objects
        files: Files to attach
        alert: If True, send to ALERT webhook (critical notifications only)
    """
    # Choose webhook based on alert flag
    webhook_url = DISCORD_ALERT_WEBHOOK_URL if alert else DISCORD_WEBHOOK_URL
    
    # Skip if webhook not configured
    if not webhook_url:
        return
        
    data = {
        "content": content,
        "username": "Collatz AI" + (" 🚨 ALERT" if alert else ""),
        "avatar_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a2/Collatz-graph-all-30-no-labels.svg/1200px-Collatz-graph-all-30-no-labels.svg.png"
    }
    
    if embeds:
        data["embeds"] = embeds
        
    # If sending files, we need to use multipart/form-data
    if files:
        response = requests.post(
            webhook_url,
            data={"payload_json": json.dumps(data)},
            files=files
        )
    else:
        response = requests.post(
            webhook_url,
            json=data
        )
        
    if response.status_code not in [200, 204]:
        print(f"Failed to send Discord message: {response.status_code} - {response.text}")

def send_status_update(step, loss, stopping_loss, seq_loss):
    """Send periodic training status update (to main webhook)"""
    if not SEND_STATUS_UPDATES:
        return
        
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
    send_message("", embeds=[embed], alert=False)

def send_anomaly_alert(number, actual_stop, predicted_stop, error, image_path=None):
    """Send anomaly detection alert (to main webhook)"""
    if not SEND_ANOMALIES:
        return
        
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
        
    send_message("", embeds=[embed], files=files, alert=False)

def send_cycle_alert(number_high, number_low, cycle_value_high, cycle_value_low):
    """Send CRITICAL alert when non-trivial cycle is found (to ALERT webhook)"""
    if not SEND_CYCLE_ALERTS:
        return
        
    embed = {
        "title": "🚨🚨🚨 NON-TRIVIAL CYCLE FOUND! 🚨🚨🚨",
        "description": "**COLLATZ CONJECTURE DISPROVEN!**",
        "color": 16711680, # Bright red
        "fields": [
            {"name": "Starting Number (high, low)", "value": f"{number_high}, {number_low}", "inline": False},
            {"name": "Cycle Value (high, low)", "value": f"{cycle_value_high}, {cycle_value_low}", "inline": False},
            {"name": "Status", "value": "⚠️ IMMEDIATE VERIFICATION REQUIRED", "inline": False}
        ],
        "footer": {"text": "This is a historic mathematical discovery!"},
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())
    }
    
    # Send to BOTH webhooks for critical alert
    send_message("@everyone **CRITICAL DISCOVERY**", embeds=[embed], alert=True)
    send_message("@everyone **CRITICAL DISCOVERY**", embeds=[embed], alert=False)

def send_milestone_alert(step, loss):
    """Send milestone alert (100k, 200k, etc.) to ALERT webhook"""
    if not SEND_MILESTONES:
        return
        
    milestones = [100000, 200000, 500000, 1000000]
    if step in milestones:
        embed = {
            "title": f"🎯 Milestone Reached: {step:,} Steps!",
            "description": f"Training has reached {step:,} steps.",
            "color": 3066993, # Green
            "fields": [
                {"name": "Current Loss", "value": f"{loss:.4f}", "inline": True},
                {"name": "Progress", "value": f"{step/1000000*100:.1f}% of 1M", "inline": True}
            ],
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())
        }
        send_message("", embeds=[embed], alert=True)

def send_analysis_report(summary_text, image_paths=[]):
    """Send analysis report (to main webhook)"""
    files = {}
    for i, path in enumerate(image_paths):
        if os.path.exists(path):
            files[f"file{i}"] = (os.path.basename(path), open(path, "rb"))
            
    send_message(f"**Analysis Report**\n{summary_text}", files=files, alert=False)

