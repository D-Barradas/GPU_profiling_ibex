#make a list with all the cuda modules 
module av 2>&1 | grep cuda/ | awk '{print $1}' > cuda_modules.txt
# load the cuda modules and test the nsys command
while read -r line; do
    module load $line 2>&1 > /dev/null
    if nsys --version 2>&1 | grep -q "command not found"; then
        echo "module $line does not have nsys"
    else
        echo "module $line has nsys"
    fi
    # nsys --version
    module unload $line 2>&1 > /dev/null
done < cuda_modules.txt
