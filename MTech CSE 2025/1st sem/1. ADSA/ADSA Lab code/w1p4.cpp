// #include <iostream>
// #include <queue>
// using namespace std;

// class CountedQueue {
//     queue<int> Q;
//     long long enqueueCount = 0;
//     long long dequeueCount = 0;

// public:
//     void Enqueue(int x) {
//         Q.push(x);
//         enqueueCount++;
//     }

//     int Dequeue() {
//         if (Q.empty()) return -1; // or throw exception
//         int front = Q.front();
//         Q.pop();
//         dequeueCount++;
//         return front;
//     }

//     bool QueueEmpty() {
//         return Q.empty();
//     }

//     void printCounts() {
//         cout << "Total Enqueue operations: " << enqueueCount << "\n";
//         cout << "Total Dequeue operations: " << dequeueCount << "\n";
//     }
// };

// int main() {
//     CountedQueue cq;
//     cq.Enqueue(1);
//     cq.Enqueue(2);
//     cq.Enqueue(3);

//     cout << "Dequeue: " << cq.Dequeue() << "\n";
//     cout << "Dequeue: " << cq.Dequeue() << "\n";

//     cq.printCounts();
//     return 0;
// }

#include <iostream>
using namespace std;

class CountedQueue {
    int *arr;              // array to store elements
    int capacity;          // maximum capacity
    int front;             // index of the front element
    int rear;              // index of the last element
    int size;              // current number of elements
    long long enqueueCount;
    long long dequeueCount;

public:
    CountedQueue(int cap = 100) {
        capacity = cap;
        arr = new int[capacity];
        front = 0;
        rear = -1;
        size = 0;
        enqueueCount = 0;
        dequeueCount = 0;
    }

    ~CountedQueue() {
        delete[] arr;
    }

    bool QueueEmpty() {
        return size == 0;
    }

    bool QueueFull() {
        return size == capacity;
    }

    void Enqueue(int x) {
        if (QueueFull()) {
            cout << "Queue overflow! Cannot enqueue " << x << "\n";
            return;
        }
        rear = (rear + 1) % capacity; // circular increment
        arr[rear] = x;
        size++;
        enqueueCount++;
    }

    int Dequeue() {
        if (QueueEmpty()) {
            cout << "Queue underflow! Cannot dequeue\n";
            return -1;
        }
        int value = arr[front];
        front = (front + 1) % capacity; // circular increment
        size--;
        dequeueCount++;
        return value;
    }

    void printCounts() {
        cout << "Total Enqueue operations: " << enqueueCount << "\n";
        cout << "Total Dequeue operations: " << dequeueCount << "\n";
    }
};

int main() {
    CountedQueue cq(10);

    cq.Enqueue(1);
    cq.Enqueue(2);
    cq.Enqueue(3);

    cout << "Dequeue: " << cq.Dequeue() << "\n";
    cout << "Dequeue: " << cq.Dequeue() << "\n";

    cq.printCounts();
    return 0;
}
