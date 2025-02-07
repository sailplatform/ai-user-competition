#!/usr/bin/env bash

set -euxo pipefail

# NOTE: If you get build errors related to python interpreter check under Linux then run the following command:
# export WITH_PYTHON_INTERP_CHECK=ON

# ====================================================
# import the utils 
. bash_utils.sh 

# ====================================================

#set -e

#echo ROOT_DIR: $ROOT_DIR
cd $ROOT_DIR  # from bash_utils.sh

# ====================================================
# check if we have external options
EXTERNAL_OPTIONS=$@
if [[ -n "$EXTERNAL_OPTIONS" ]]; then
    echo "external option: $EXTERNAL_OPTIONS" 
fi



OpenCV_DIR="$ROOT_DIR/thirdparty/opencv/install/lib/cmake/opencv4"
if [[ -d "$OpenCV_DIR" ]]; then
    EXTERNAL_OPTIONS="$EXTERNAL_OPTIONS -DOpenCV_DIR=$OpenCV_DIR"
fi 

echo "EXTERNAL_OPTIONS: $EXTERNAL_OPTIONS"


print_blue '================================================'
print_blue "Building and installing cpp ..."

CURRENT_USED_PYENV=$(get_virtualenv_name)
print_blue "currently used pyenv: $CURRENT_USED_PYENV"

cd cpp 

# build utils 
cd utils 
. build.sh $EXTERNAL_OPTIONS       # use . in order to inherit python env configuration 
cd ..

cd .. 


# NOTE: If you get build errors related to python interpreter check under Linux then run the following command:
# export WITH_PYTHON_INTERP_CHECK=ON
