#include <stdio.h>

int main() {
    for (int i = 0; i < 100000; ++i) {
        for (int j = 0; j < 100000; ++j) {
            // Nested loops with a large number of iterations
        }
    }
    return 0;
}
gcc -o nested_loops nested_loops.c

./nested_loops &
pid=$!
echo "Process ID: $pid"
kill $pid

