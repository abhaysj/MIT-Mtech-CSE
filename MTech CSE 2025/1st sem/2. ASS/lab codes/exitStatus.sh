grep "$1" "$2" > /dev/null
if [ $? -eq 0 ]; then
    echo "Pattern found"
elif [ $? -eq 1 ]; then
    echo "Pattern not found"
else
    echo "file error"
fi
