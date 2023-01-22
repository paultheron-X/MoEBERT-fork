# Description: Fetches the results from the supercloud morning run

# Get the experiments from ktwo experiments
python sh_scripts/python_helpers/get_results.py --ktwo

# Get the experiments from seed experiments
    # No Sparsity
python sh_scripts/python_helpers/get_results.py --advanced
    # Sparsity
python sh_scripts/python_helpers/get_results.py --advanced --sparsity


python sh_scripts/python_helpers/preprocess_sparsity_error.py
# Get the experiments from moebert perm experiments
    # No Sparsity
python sh_scripts/python_helpers/get_results.py --perm
    # Sparsity
python sh_scripts/python_helpers/get_results.py --perm --sparsity