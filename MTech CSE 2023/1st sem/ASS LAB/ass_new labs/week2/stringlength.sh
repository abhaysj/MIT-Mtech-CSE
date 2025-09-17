#!/bin/bash

echo -n "Enter a string: "
read input_string

string_length=$(expr length "$input_string")

case $string_length in
    [0-9])
        echo "The string has less than 10 characters."
        ;;
    *)
        echo "The string has 10 or more characters."
        ;;
esac

