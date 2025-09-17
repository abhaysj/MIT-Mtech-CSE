#include<bits/stdc++.h>
using namespace std;

int binarySearch(const vector<int>& arr, int target) {
    int count = 0;
    int left = 0, right = (int)arr.size() - 1;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        count++; // count a comparison
        if (arr[mid] == target) {
            return count; // return comparisons done
        } else if (arr[mid] < target) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    return count; // target not found, still return comparisons done
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
    cout << "Enter the number to search: ";
    int target;
    cin >> target;
    long long count = binarySearch(a, target);
    cout << "Number of opertations for binary search: " << count << endl;
}
