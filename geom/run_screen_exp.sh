#!/bin/bash

# Start the Python script in a new screen session
screen -dmS Experiment bash -c "python3 run_experiments.py; exec bash"

echo "Experiment session started in screen. Use 'screen -r experiment_session' to reattach."
