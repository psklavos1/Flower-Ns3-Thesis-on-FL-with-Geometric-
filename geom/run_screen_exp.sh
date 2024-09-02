#!/bin/bash

# Terminate any existing screen sessions named Experiment
screen -S Experiment -X quit

# Start the Python script in a new screen session
screen -dmS Experiment bash -c "python3 run_experiments.py; exec bash"

echo "Experiment session started in screen. Use 'screen -r Experiment' to reattach."


# * NEW

# #!/bin/bash

# # Check if the correct number of arguments are provided
# if [ "$#" -ne 1 ]; then
#   echo "Usage: $0 <CPU_LIST>"
#   echo "Example: $0 0,1,2,3"
#   exit 1
# fi

# # Get the CPU list from the command line argument
# CPU_LIST=$1

# # Start the Python script in a new screen session with specified CPUs
# screen -S Experiment -X quit
# screen -dmS Experiment bash -c "taskset -c $CPU_LIST python3 run_experiments.py; exec bash"
# echo "Experiment session started in screen with CPUs $CPU_LIST. Use 'screen -r Experiment' to reattach."


