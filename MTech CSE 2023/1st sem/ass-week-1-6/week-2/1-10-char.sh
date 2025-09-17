#!/bin/bash

read -p "enter the string : " string

len=`expr length "$string"` 

case $len in
    1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 ) echo "less than 10";; 
    *) echo "greater than 10";;
esac
