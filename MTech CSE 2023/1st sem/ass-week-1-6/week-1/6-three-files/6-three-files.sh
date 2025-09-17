#!/bin/bash

touch new.txt
touch sort.txt  

cat $1 >> new.txt
echo -e "\n" >> new.txt
cat $2 >> new.txt
echo -e "\n" >> new.txt
cat $3 >> new.txt

cat new.txt | sort > sort.txt