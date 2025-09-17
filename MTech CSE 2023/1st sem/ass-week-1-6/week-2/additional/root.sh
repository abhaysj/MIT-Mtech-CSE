#!/bin/bash

echo "Enter a value: "
read a
echo "Enter a value: "
read b
echo "Enter a value: "
read c

echo $"(-$b+sqrt($b^2-4*$a*$c))/(2*$a)" | bc
echo $"(-$b-sqrt($b^2-4*$a*$c))/(2*$a)" | bc