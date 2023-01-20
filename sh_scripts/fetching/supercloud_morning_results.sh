# Description: Fetches the results from the supercloud morning run

# Get the experiments from ktwo experiments
python sh_scripts/python_helpers/get_results.py --ktwo

# Get the experiments from seed experiments
    # No Sparsity
python sh_scripts/python_helpers/get_results.py --advanced
    # Sparsity
python sh_scripts/python_helpers/get_results.py --advanced --sparsity

# Get the experiments from moebert perm experiments
python sh_scripts/python_helpers/get_results.py --perm