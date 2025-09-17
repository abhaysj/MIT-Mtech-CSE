#!/bin/bash

echo "Enter first and second dir "
read first
read second

uniqFile=$(comm -12 <(ls "$first") <(ls "$second"))
echo "The unique files are:"
echo $uniqFile


for fname in $uniqFile
do
    echo "Deleting $fname from $second" 
    rm "$second/$fname"
done
