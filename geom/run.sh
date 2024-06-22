#!/bin/bash

# Define the number of clients
NUM_CLIENTS=$(python get_num_clients.py)
gnome-terminal --title="Server" -- bash -c "source /home/psklavos/miniconda3/etc/profile.d/conda.sh; conda activate geom_aio; python server_main.py ; exec bash"
echo "!! If Ns3 not built before execution let it build, and re-run !!"

sleep 10
# Loop and start each client with a unique partition_id and the total num_clients
for (( i=0; i<NUM_CLIENTS; i++ ))
do
    echo "Starting client $((i+1))"
    if [ $i -eq 0 ]; then
        gnome-terminal --title="Client $((i+1))" -- bash -c "source /home/psklavos/miniconda3/etc/profile.d/conda.sh; conda activate geom_aio; python client_main.py partition_id=$i num_clients=$NUM_CLIENTS keep_log=True; exec bash"
    else
        gnome-terminal --title="Client $((i+1))" -- bash -c "source /home/psklavos/miniconda3/etc/profile.d/conda.sh; conda activate geom_aio; python client_main.py partition_id=$i num_clients=$NUM_CLIENTS; exec bash"
    fi    # Add a delay if needed to prevent issues with too many simultaneous starts
    sleep .25
done

#Screen Management
# #!/bin/bash

# # Define the number of clients
# NUM_CLIENTS=$(python get_num_clients.py)

# # Start the server in a new screen session
# screen -S Server -dm bash -c "source /home/psklavos/miniconda3/etc/profile.d/conda.sh; conda activate geom_aio; python server_main.py; exec bash"
# echo "!! If Ns3 not built before execution let it build, and re-run !!"

# sleep 10

# # Create a screen session for all clients
# screen -S Clients -dm

# # Loop and start each client with a unique partition_id and the total num_clients
# for (( i=0; i<NUM_CLIENTS; i++ ))
# do
#   echo "Starting client $((i+1))"
#   if [ $i -eq 0 ]; then
#       screen -S Clients -X screen -t "Client $((i+1))" bash -c "source /home/psklavos/miniconda3/etc/profile.d/conda.sh; conda activate geom_aio; python client_main.py partition_id=$i num_clients=$NUM_CLIENTS keep_log=True; exec bash"
#   else
#       screen -S Clients -X screen -t "Client $((i+1))" bash -c "source /home/psklavos/miniconda3/etc/profile.d/conda.sh; conda activate geom_aio; python client_main.py partition_id=$i num_clients=$NUM_CLIENTS; exec bash"
# fi    
# Add a delay if needed to prevent issues with too many simultaneous starts
#     sleep .25
# done