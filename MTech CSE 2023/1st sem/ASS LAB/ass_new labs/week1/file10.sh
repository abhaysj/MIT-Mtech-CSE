#!/bin/bash

# Check if at least two arguments are provided
if [ $# -lt 2 ]; then
    echo "Usage: $0 <word> <file1> [<file2> ...]"
    exit 1
fi

word="$1"
shift  # Shift arguments to remove the word from the list

# Iterate through the remaining arguments (filenames)
for file in "$@"; do
    # Check if the file exists
    if [ -f "$file" ]; then
        # Print line numbers containing the specified word
        grep -n "$word" "$file" | cut -d ':' -f 1
    else
        echo "File $file not found."
    fi
done

