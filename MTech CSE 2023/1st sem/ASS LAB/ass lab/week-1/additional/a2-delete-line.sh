#!/bin/bash

echo "pattern : "
read pattern

for var in ${@: 1}
do 
    sed -i "/$pattern/d" $var
done