#!/bin/bash

source MCP_Demo/bin/activate

if [ $# -ne 1 ]; then
    echo "Usage: $0 [stdio|sse]"
    exit 1
fi

case $1 in
    "stdio")
        python math_server.py
        ;;
    "sse")
        python math_server2.py --host 0.0.0.0 --port 8080
        ;;
    *)
        echo "Invalid mode: $1. Use 'stdio' or 'sse'"
        exit 1
        ;;
esac