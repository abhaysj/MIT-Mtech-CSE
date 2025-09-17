#include <stdio.h>

int main() {
    // Integers
    int integerNumber = 42;
    printf("Decimal: %d\n", integerNumber);       // Decimal
    printf("Octal: %o\n", integerNumber);         // Octal
    printf("Hexadecimal: %x\n", integerNumber);   // Hexadecimal (lowercase)
    printf("Hexadecimal: %X\n", integerNumber);   
    // Hexadecimal (uppercase)

    // Floating-point numbers
    float floatValue = 3.14159;
    printf("Float: %f\n", floatValue);           // Standard notation
    printf("Scientific: %e\n", floatValue);      // Scientific notation (lowercase)
    printf("Scientific: %E\n", floatValue);      // Scientific notation (uppercase)

    // Characters
    char character = 'A';
    printf("Character: %c\n", character);

    // Strings
    const char *string = "Hello, World!";
    printf("String: %s\n", string);

    // Pointers
    int *pointerValue = &integerNumber;
    printf("Pointer address: %p\n", (void *)pointerValue);

    // Width and Precision
    printf("Width and Precision: %10.3f\n", floatValue);

    return 0;
}

