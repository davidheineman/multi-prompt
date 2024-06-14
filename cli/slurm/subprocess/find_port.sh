#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <number_of_ports>"
    exit 1
fi

number_of_ports=$1

# Select a starting port randomly within the range
start_port=8000
range=$((8900 - start_port + 1))
start_port=$((RANDOM % range + start_port))

end_port=8999

found_ports=()

while [ ${#found_ports[@]} -lt $number_of_ports ]; do
    for port in $(seq $start_port $end_port); do
        if ! lsof -i :$port > /dev/null; then
            found_ports+=($port)
            break
        fi
    done
    ((start_port++))
done

if [ ${#found_ports[@]} -eq 0 ]; then
    echo "No open ports found in the range $start_port-$end_port"
else
    echo "${found_ports[@]}"
fi
