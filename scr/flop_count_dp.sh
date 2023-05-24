#!/bin/bash

# This script uses nvprof to profile the performance of a CUDA application and
# prints the number of double-precision floating point operations performed.

# Get the name of the CUDA application.
# app_name=$1
app_name="matrix_multiplication.out 100 100"


# Run the CUDA application with nvprof.
nvprof -o nvprof.log ./$app_name

# Parse the nvprof output to get the number of double-precision floating point operations.
flop_count=$(grep -i "flop_count_dp" nvprof.log | awk '{print $2}')

# Print the number of double-precision floating point operations.
echo "Number of double-precision floating point operations: $flop_count"
