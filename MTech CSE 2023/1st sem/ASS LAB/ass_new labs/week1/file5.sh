#!/bin/bash


echo "Enter the filename"
read filename
echo "Enter the starting line"
read start_line
echo "Enter the ending line"
read end_line




echo "Lines between $start_line and $end_line in $filename:"
sed -n "${start_line},${end_line}p" "$filename"

