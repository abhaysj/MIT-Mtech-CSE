#!/bin/bash
# file1.sh: Uses command line arguments

echo "Program: $0"
echo "The number of arguments specified is $#"
echo "The arguments are: $*"

grep "$1" "$2"

echo -e "\nJob Over"
