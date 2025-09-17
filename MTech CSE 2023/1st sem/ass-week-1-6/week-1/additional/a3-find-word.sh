#!/bin/bash

echo "enter the word : "
read word

grep $word -rHn $1 