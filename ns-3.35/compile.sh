#!/bin/bash

# Maximum number of attempts
max_attempts=10
attempt=1

# Loop until the command succeeds or we reach the maximum number of attempts
while true; do
    # echo "Attempt $attempt of $max_attempts"
    ./waf build

    # Check the exit status of the command
    if [ $? -eq 0 ]; then
        echo "Build completed successfully."
        break
    else
        echo "Build failed. Attempting again..."
        ((attempt++))
        # Check if we have reached the maximum number of attempts
        if [ $attempt -gt $max_attempts ]; then
            echo "Maximum number of attempts reached. Exiting."
            exit 1
        fi
    fi

    # Optional: sleep between attempts
    sleep .5
done
