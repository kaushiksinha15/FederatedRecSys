#!/bin/bash
source venv/bin/activate
echo "Starting Federated Recommendation System Simulation..."

# Start the server in the background
python3 server.py &
SERVER_PID=$!
echo "Server started with PID: $SERVER_PID"

# Give the server a moment to initialize
sleep 3

# Start 4 clients in the background
for i in {1..4}
do
  echo "Starting client $i..."
  python3 client.py --client_id $i &
  CLIENT_PIDS="$CLIENT_PIDS $!"
done

echo "Clients started with PIDs: $CLIENT_PIDS"

# Wait for the server to finish (simulation complete)
wait $SERVER_PID

# Kill clients if they haven't exited
echo "Simulation finished. Cleaning up clients..."
kill $CLIENT_PIDS 2>/dev/null

echo "Done."
