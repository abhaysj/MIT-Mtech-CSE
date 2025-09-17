#!/bin/bash

# touch a b c B

if [ $# -gt 0 ]
then
  for file in "$@"
  do
    if [ -f "$file" ]
    then
      upper_file=$(echo "$file" | tr '[:lower:]' '[:upper:]')
      if [ ! -f "$upper_file" ]
      then
        mv "$file" "$upper_file"
      else
        echo "Error: $upper_file already exists"
      fi
    else
      echo "Error: $file does not exist"
    fi
  done
else
  echo "Error: No file names provided"
fi