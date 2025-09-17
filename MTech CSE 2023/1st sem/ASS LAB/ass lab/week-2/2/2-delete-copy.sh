#!/bin/bash

echo "Enter first and second dir "
read first
read second

uniqFile=$(comm -12 <(ls "$first") <(ls "$second"))

echo $uniqFile

for fname in $uniqFile
do
    echo $fname 
    rm "$second/$fname"
done