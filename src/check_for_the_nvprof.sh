
for m in `cat cuda_modules.txt`; do
    module load $m ; 
    # if nvprof raises a warning, then the module does not have access to float point operations tracking

    nvprof --metrics flop_count_dp -o matrix_multiplication.nvvp ./matrix_multiplication.out 100 100 > nvprof_${m:5:20}.txt 2>&1;

    rm matrix_multiplication.nvvp ;
    # if nvprof --metrics flop_count_dp -f -o matrix_multiplication.nvvp ./matrix_multiplication.out 100 100  | grep -q "Warning: ERR_NVGPUCTRPERM - The user does not have permission to profile on the target device. See the following link for instructions to enable permissions and get more information: https://developer.nvidia.com/ERR_NVGPUCTRPERM"; then
    #     echo "module $m does not have access enable to "
    #     rm matrix_multiplication.nvvp
    # else
    #     echo "module $m has flop_count_dp enabled"
    # fi
    module unload $m;
done


