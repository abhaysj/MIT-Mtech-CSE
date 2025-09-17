#!/bin/bash

if [ $# -ne 2 ]; then
    echo "Usage: $0 file1 file2"
    exit 1
fi

file1="$1"
file2="$2"
output_file="merged_sorted_unique.txt"

if [ ! -f "$file1" ] || [ ! -f "$file2" ]; then
    echo "Error: Both input files must exist."
    exit 1
fi

sort -m -u "$file1" "$file2" > "$output_file"

echo "Merged and sorted unique contents saved to $output_file"

