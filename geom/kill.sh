#!/bin/bash

# This script closes all instances of GNOME Terminal.

# Find the process IDs of all GNOME Terminal instances and kill them.
# pkill gnome-terminal


#!/bin/bash

# List all screen sessions
screen_sessions=$(screen -ls | awk '/\t/ {print $1}')

# Loop through each session and terminate it
for session in $screen_sessions
do
    screen -S "$session" -X quit
done

echo "All screen sessions have been terminated."




