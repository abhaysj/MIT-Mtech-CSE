#!/bin/bash

# Display current date and time
echo "Current Date and Time: $(date)"

# Display number of users logged in
echo "Number of users currently logged in: $(who | wc -l)"

