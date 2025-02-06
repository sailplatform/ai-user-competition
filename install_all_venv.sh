#!/usr/bin/env bash

set -euxo pipefail

#N.B: this install script allows you to run main_slam.py and all the scripts 

# import the utils 
. bash_utils.sh 

set -e


case "$OSTYPE" in
  darwin*)
    echo "macOS"
    . install_all_mac_venv.sh
    ;;
  linux*)
    echo "Linux"
    . install_all_linux_venv.sh
    ;;
  *)
    echo "Unknown OS"
    ;;
esac
