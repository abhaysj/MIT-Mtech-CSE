#!/bin/bash

echo "enter the directory : "
read directory

find $directory -type f -name "*.c"