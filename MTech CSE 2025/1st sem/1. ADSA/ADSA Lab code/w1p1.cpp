#include<bits/stdc++.h>
using namespace std;

int selectionSort(vector<int> s, int n){
//Selection sort always performs:
//(n-1) + (n-2) + ... + 1 = n*(n-1)/2 comparisons
    int cnts = 0;
    cout << "array before selection sort\n";
    for(int i=0; i<n; i++){
        cout << s[i] << " ";
    }
    cout << endl;
    for (int i = 0; i < n - 1; i++) {
        int minIndex = i;
        for (int j = i + 1; j < n; j++) {
            cnts++; // count one comparison
            if (s[j] < s[minIndex]) {
                minIndex = j;
            }
        }
        swap(s[i], s[minIndex]);
    }
    cout << "Sorted array after selection sort\n";
    for(int i=0; i<n; i++){
        cout << s[i] << " ";
    }
    cout << endl;
    return cnts;
}

int bubbleSort(vector<int> b, int n){
//Bubble sort compares adjacent pairs in multiple passes.
//For each pass i from 0 to n-2, it compares (n - 1 - i) pairs.
    int cntb = 0;
    bool swapped;
    cout << "array before bubble sort:\n";
    for(int i=0; i<n; i++){
        cout << b[i] << " ";
    }
    cout << endl;
    for(int i=0; i<n-1; i++){
        swapped = false; // reset swapped for each pass
        for(int j=0; j<n-i-1; j++){
            cntb++; // count comparison
            if(b[j] > b[j+1]){ // compare adjacent elements
                swap(b[j], b[j+1]);
                swapped = true;
            }
        }
        if(!swapped){
        break;
        }
    }
    cout << "Sorted array after bubble sort:\n";
    for(int i=0; i<n; i++){
        cout << b[i] << " ";
    }
    cout << endl;
    return cntb;
}


int main(){

    int n;
    cout << "Enter the number of array items\n";
    cin >> n;
    vector<int> a(n);
    cout << "Enter the array items\n";
    for(int i=0; i<n; i++){
        cin >> a[i];
    }
    cout << endl;
    int cntb = bubbleSort(a, n);
    cout << "Number of operations for bubble sort: " << cntb << endl;
    cout << endl;
    int cnts = selectionSort(a, n);
    cout << "Number of operations for selection sort: " << cnts << endl;
    /*Both algorithms do roughly proportional to n² comparisons in the worst case.
    Selection sort always does exactly n*(n-1)/2 comparisons (quadratic).
    Bubble sort can stop early if the list is sorted early, so fewer comparisons are done in practice (but worst-case is also O(n²)).*/
    return 0;


}