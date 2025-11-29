# Discord Webhook Configuration
# Copy this file to config.py and add your webhook URL

# Main webhook - receives ALL notifications (training progress, status updates)
# Leave empty to disable
DISCORD_WEBHOOK_URL = ""

# Alert webhook - receives ONLY critical alerts (cycles found, major milestones)
# This is for the project maintainer
DISCORD_ALERT_WEBHOOK_URL = ""

# Notification settings
SEND_TRAINING_START = True      # Notify when training starts
SEND_STATUS_UPDATES = True      # Send periodic status updates (every 500 steps)
SEND_ANOMALIES = True           # Send anomaly detection alerts
SEND_CYCLE_ALERTS = True        # Send alerts when cycles are found (CRITICAL)
SEND_MILESTONES = True          # Send milestone alerts (100k, 200k steps, etc.)

# Update frequency
STATUS_UPDATE_INTERVAL = 500    # Send status every N steps
