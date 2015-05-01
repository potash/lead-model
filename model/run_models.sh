#!/bin/bash
# Usage: run_models.sh <datadir> <outputdir> [paramsfile]
mkdir -p $2

# if specified, move the params file to the subdir
if [ -n "$3" ]; then
    cp $3 $2/params.yaml 
fi

./n_models.py $2/params.yaml

rm -r $2/*/

./get_params.py $1 $2 $2/params.yaml | parallel --delay 5 --joblog $2/log ./run_model.py
