#!/bin/bash

read -p "Enter the filename : " name

stat -L -c 'permission %A' $name
stat -L -c 'size : %s bytes' $name
stat -L -c 'Name : %n' $name
stat -L -c 'Last modified : %y' $name
stat -L -c 'last access : %x' $name