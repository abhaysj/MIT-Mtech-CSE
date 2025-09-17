#!/bin/bash
# A shell script to find the factorial of a number
 
echo "Enter a number " 
read num

fact=1
 
for((i=2;i<=num;i++))
{
  fact=$((fact*i))
}
 
echo $fact