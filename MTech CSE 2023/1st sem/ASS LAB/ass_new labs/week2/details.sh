#!/bin/bash

if [ $# -eq 0 ]; then
  echo "Error: No filenames provided"
  exit 1
fi

echo "Permissions | Size | Filename | Last Modified | Last Accessed"
echo "------------------------------------------------------------"

for file in "$@"; do
  if [ -e "$file" ]; then
    permissions=$(ls -l "$file" | cut -d " " -f 1)
    size=$(ls -lh "$file" | awk '{print $5}')
    filename=$(basename "$file")
    last_modified=$(stat -c "%y" "$file")
    last_accessed=$(stat -c "%x" "$file")
    
    printf "%-12s | %-5s | %-5s | %-16s | %s\n" "$permissions" "$size" "$filename" "$last_modified" "$last_accessed"
  else
    echo "Error: $file does not exist"
  fi
done

