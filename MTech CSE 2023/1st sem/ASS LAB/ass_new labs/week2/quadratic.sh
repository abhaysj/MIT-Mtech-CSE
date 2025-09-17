#!/bin/bash

echo "Enter the coefficients of the quadratic equation (ax^2 + bx + c = 0):"
read a b c

discriminant=$((b*b - 4*a*c))

case $discriminant in
    0)
        root=-$(bc <<< "scale=2; -$b / (2*$a)")
        echo "There is one real root: x = $root"
        ;;
    [1-9]*)
        root1=$(bc <<< "scale=2; (-$b + sqrt($discriminant)) / (2*$a)")
        root2=$(bc <<< "scale=2; (-$b - sqrt($discriminant)) / (2*$a)")
        echo "There are two real roots: x1 = $root1 and x2 = $root2"
        ;;
    *)
        real_part=$(bc <<< "scale=2; -$b / (2*$a)")
        imaginary_part=$(bc <<< "scale=2; sqrt(-$discriminant) / (2*$a)")
        echo "There are two complex roots: x1 = $real_part + ${imaginary_part}i and x2 = $real_part - ${imaginary_part}i"
        ;;
esac

