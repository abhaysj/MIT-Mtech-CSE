//#include <iostream>
// #include <stack>
// using namespace std;

// class CountedStack {
//     stack<int> S;
//     long long pushCount = 0;
//     long long popCount = 0;
//     long long emptyCheckCount = 0;

// public:
    
//     bool StackEmpty() {
//         emptyCheckCount++;
//         return S.empty();
//     }

//     void Push(int x) {
//         S.push(x);
//         pushCount++;
//     }

//     int Pop() {
//         if (S.empty()) return -1; // or throw exception
//         int top = S.top();
//         S.pop();
//         popCount++;
//         return top;
//     }

//     void printCounts() {
//         cout << "Total Push operations: " << pushCount << "\n";
//         cout << "Total Pop operations: " << popCount << "\n";
//     }
// };

// int main() {
//     CountedStack cs;
//     cs.Push(10);
//     cs.Push(20);
//     cs.Push(30);

//     cout << "Pop: " << cs.Pop() << "\n";
//     cout << "Pop: " << cs.Pop() << "\n";

//     cs.printCounts();
//     return 0;
// }

#include <iostream>
using namespace std;

class CountedStack {
    int *arr;              // dynamic array for stack storage
    int topIndex;          // index of the top element
    int capacity;          // max size of stack
    long long pushCount;   // count push operations
    long long popCount;    // count pop operations

public:
    CountedStack(int size = 100) { // default capacity 100
        capacity = size;
        arr = new int[capacity];
        topIndex = -1;
        pushCount = 0;
        popCount = 0;
    }

    ~CountedStack() {
        delete[] arr; // free memory
    }

    bool StackEmpty() {
        return (topIndex == -1);
    }

    bool StackFull() {
        return (topIndex == capacity - 1);
    }

    void Push(int x) {
        if (StackFull()) {
            cout << "Stack overflow! Cannot push " << x << "\n";
            return;
        }
        arr[++topIndex] = x;
        pushCount++;
    }

    int Pop() {
        if (StackEmpty()) {
            cout << "Stack underflow! Cannot pop\n";
            return -1; // or throw exception
        }
        popCount++;
        return arr[topIndex--];
    }

    void printCounts() {
        cout << "Total Push operations: " << pushCount << "\n";
        cout << "Total Pop operations: " << popCount << "\n";
    }
};

int main() {
    CountedStack cs(10); // stack with capacity 10

    cs.Push(10);
    cs.Push(20);
    cs.Push(30);

    cout << "Pop: " << cs.Pop() << "\n";
    cout << "Pop: " << cs.Pop() << "\n";

    cs.printCounts();
    return 0;
}
