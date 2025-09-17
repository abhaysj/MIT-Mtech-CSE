#!/bin/bash

# Iterate through all files in the current directory
for file in *; do
    if [ -f "$file" ]; then
        # Check if the user has read, write, and execute permissions
        if [ -r "$file" ] && [ -w "$file" ] && [ -x "$file" ]; then
            echo "File with read, write, and execute permissions: $file"
        fi
    fi
done

