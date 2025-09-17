#!/bin/sh
# file.sh: Interactive version - uses read to take two inputs

echo -n "Enter the pattern to be searched: "
read pname

echo -n "Enter the file to be used: "
read flname

echo "Searching for $pname from file $flname"
grep  --color=always -i "$pname" "$flname"

echo "Selected records shown above"
