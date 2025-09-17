#!/bin/bash

# Check if three arguments are provided
if [ $# -ne 3 ]; then
    echo "Usage: $0 <file1> <file2> <file3>"
    exit 1
fi

file1="$1"
file2="$2"
file3="$3"

# Check if all provided files exist
if [ ! -f "$file1" ] || [ ! -f "$file2" ] || [ ! -f "$file3" ]; then
    echo "One or more input files do not exist."
    exit 1
fi

# Concatenate text from all three files
cat "$file1" "$file2" "$file3" > concatenated.txt
echo "Concatenated text saved to concatenated.txt"

# Sort lines and save to a new file
sort -u "$file1" "$file2" "$file3" > sorted_lines.txt
echo "Sorted lines saved to sorted_lines.txt"

