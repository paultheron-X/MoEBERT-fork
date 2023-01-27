# Description: Fetches the results from the supercloud morning run

# Get the experiments from ktwo experiments
    # No Sparsity
python sh_scripts/python_helpers/get_results.py --ktwo
    # Sparsity
python sh_scripts/python_helpers/get_results.py --ktwo --sparsity


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

# Get the experiments from hash
python sh_scripts/python_helpers/get_results.py --hash

python sh_scripts/python_helpers/get_results.py --hashp