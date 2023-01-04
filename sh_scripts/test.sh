declare -a StringArray=("3" "6" "8" "10")
for ((i=0; i<${#StringArray[@]}; i++)); do
  # Access the current element of the array using the index variable
    element=${StringArray[$i]}
    j=$((i+1))
    for l in {1..5}
    do
        args=$(python sh_scripts/python_helpers/launch_job_from_grid.py -n $element --exp 10$j$l)
        echo $args
        echo 
    done
done