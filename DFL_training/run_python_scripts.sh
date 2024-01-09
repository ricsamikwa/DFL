#!/bin/bash

# Call Python scripts in a specific order and run them in the background using nohup
nohup python3 run_DFL_unit.py --split False --group True &

# Return immediately after calling the Python scripts
exit
