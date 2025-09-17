#!/bin/bash

# Ask user for directory name
echo -n "Enter directory name: "
read dir

# Check if directory exists
if [ -d "$dir" ]; then
    echo "C source files in $dir:"
    
    # List files ending with .c
    files=$(ls "$dir"/*.c 2>/dev/null)

    if [ -z "$files" ]; then
        echo "No .c files found."
    else
        echo "$files"
    fi
else
    echo "Directory does not exist."
fi
