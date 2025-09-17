echo "Enter the filename: " 
read filename
echo "Lines containing 'manipal' in $filename:"
grep -i "manipal" "$filename"
