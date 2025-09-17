#!/bin/bash

# Ask for the filename
echo -n "Enter filename: "
read file

# Check if file exists in current directory
if [ -f "$file" ]; then
    echo "Lines containing 'manipal' in $file:"
    grep "manipal" "$file"
else
    echo "File does not exist in current directory."
fi
