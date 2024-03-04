#!/bin/bash

# Define the number of clients
NUM_CLIENTS=$(python get_num_clients.py)
gnome-terminal --title="Server" -- bash -c "source /home/psklavos/miniconda3/etc/profile.d/conda.sh; conda activate flower; python server_main.py ; exec bash"
echo "!! If Ns3 not built before execution let it build, and re-run !!"

sleep 5
# Loop and start each client with a unique partition_id and the total num_clients
for (( i=0; i<NUM_CLIENTS; i++ ))
do
    echo "Starting client $((i+1))"
    gnome-terminal --title="Client $((i+1))" -- bash -c "source /home/psklavos/miniconda3/etc/profile.d/conda.sh; conda activate flower; python client_main.py partition_id=$i num_clients=$NUM_CLIENTS; exec bash"
    # Add a delay if needed to prevent issues with too many simultaneous starts
    sleep .25
done