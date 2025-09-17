#!/bin/bash

count=0

for file in *
do
    # check if it is a file
    if [ -f $file ]
    then
    # check if it has all permissions
        if [ -r $file -a -w $file -a -x $file ]
        then
            # print the complete file name with -l option
            count=$((count+1))
        # closing second if statement
        fi
    # closing first if statement
    fi
done

echo "no of files with full permission :" $count