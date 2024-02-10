#!/bin/bash

SAVE_PATH=$1

if [[ -d ${1}/robot-team/unsupervised/data ]]; then
   echo "${1}/robot-team/unsupervised/adata exists"
else
    mkdir -p ${1}/robot-team/unsupervised/data
fi

if [[ -d ${1}/toy-datasets/iris ]]; then
    echo "${1}/toy-datasets/iris exists"
else
    mkdir -p ${1}/toy-datasets/iris
fi



curl -o ${1}/robot-team/unsupervised/data/df_targets.csv https://ncsa.osn.xsede.org/ees230012-bucket01/RobotTeam/unsupervised/df_targets.csv
curl -o ${1}/robot-team/unsupervised/data/df_features.csv https://ncsa.osn.xsede.org/ees230012-bucket01/RobotTeam/unsupervised/df_features.csv
curl -o ${1}/toy-datasets/iris/df_iris.csv https://ncsa.osn.xsede.org/ees230012-bucket01/mintsML/toy-datasets/iris/iris.csv
