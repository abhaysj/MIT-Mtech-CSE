#!/bin/bash

# Initialize the counter
count=0

# Iterate through all files in the current directory
for file in *; do
    if [ -f "$file" ]; then
        # Check if the file has full permissions for owner, group, and others
        if [ -r "$file" ] && [ -w "$file" ] && [ -x "$file" ]; then
            if [ -r "$file" ] && [ -w "$file" ] && [ -x "$file" ] && [ -r "$(ls -l "$file" | awk '{print $4}')" ] && [ -w "$(ls -l "$file" | awk '{print $4}')" ] && [ -x "$(ls -l "$file" | awk '{print $4}')" ] && [ -r "$(ls -l "$file" | awk '{print $5}')" ] && [ -w "$(ls -l "$file" | awk '{print $5}')" ] && [ -x "$(ls -l "$file" | awk '{print $5}')" ]; then
                count=$((count+1))
            fi
        fi
    fi
done

echo "Number of files with full permissions: $count"

