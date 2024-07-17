#!/bin/bash
if [ -z "$1" ]; then
	    echo "Usage: $0 <env_name>"
	        exit 1
	fi

	ENV_NAME=$1
	BASE_DIR="$SCRATCH/env/${ENV_NAME}"
	SET_ENV_SCRIPT="set_${ENV_NAME}.sh"

	# Create the directory for the environment
	mkdir -p $BASE_DIR

	# Create the environment setup script
	cat <<EOL > $SET_ENV_SCRIPT
module purge
module load cpuarch/amd
module load pytorch-gpu/py3/2.2.0

export PYTHONUSERBASE="${BASE_DIR}"
export PYTHONPATH=\$PYTHONPATH:\$PYTHONUSERBASE
export PATH=\$PATH:\$PYTHONUSERBASE/bin
EOL

# Source the environment setup script
source $SET_ENV_SCRIPT

# Install the current directory as a package
pip install -e .[metrics]


echo "Environment setup complete. Use 'source $SET_ENV_SCRIPT' to activate the environment."

