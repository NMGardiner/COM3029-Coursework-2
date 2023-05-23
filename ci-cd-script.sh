#!/bin/bash

echo "--- Running CI/CD script ---"

# Build and save the model to the ./model directory.
echo "Training and saving model..."
#python3 ./build-and-save-model.py

# Begin the webserver in the background.
echo "Starting webservice..."
waitress-serve --listen=localhost:5000 webserver:app &

# Allow time for the webserver to start up.
sleep 3

# Run unit tests to ensure the new model is functioning.
echo "Running unit tests..."
python3 -m unittest

# Kill the webserver.
echo "Exiting..."
kill %%

git_datetime=$(date +"%Y-%m-%d %T")

# Push the model changes to Git.
git add .
git commit -m "[CI/CD]: ${git_datetime}"
git push