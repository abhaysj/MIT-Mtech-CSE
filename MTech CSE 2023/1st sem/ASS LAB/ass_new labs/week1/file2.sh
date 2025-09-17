#!/bin/bash
echo "Enter directory name"
read dir
find "$dir" -type f -name '*.c'
