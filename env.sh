
#! /bin/bash

export GRAPHHMUMU=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd -P)
export DEEPJETCORE_SUBPACKAGE=$GRAPHHMUMU

cd $GRAPHHMUMU
export PYTHONPATH=$GRAPHHMUMU/modules:$PYTHONPATH
export PYTHONPATH=$GRAPHHMUMU/modules/datastructures:$PYTHONPATH
export PATH=$GRAPHHMUMU/scripts:$PATH

export LD_LIBRARY_PATH=$GRAPHHMUMU/modules/compiled:$LD_LIBRARY_PATH
export PYTHONPATH=$GRAPHHMUMU/modules/compiled:$PYTHONPATH

