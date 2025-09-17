#!/bin/bash
name=$(grep "manipal" -r -H . | cut -d: -f 1)

cat $name 
printf "\n"