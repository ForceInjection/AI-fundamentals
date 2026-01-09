#!/bin/bash

source MCP_Demo/bin/activate

if [ $# -ne 1 ]; then
    echo "Usage: $0 [stdio|sse]"
    exit 1
fi

case $1 in
    "stdio")
        python client.py
        ;;
    "sse")
        python client2.py http://127.0.0.1:8080/sse
        ;;
    *)
        echo "Invalid mode: $1. Use 'stdio' or 'sse'"
        exit 1
        ;;
esac