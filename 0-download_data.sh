#!/usr/bin/bash

SAVE_PATH=$1


curl -o "${1}/df_targets.csv" https://ncsa.osn.xsede.org/ees230012-bucket01/RobotTeam/unsupervised/df_targets.csv
curl -o "${1}/df_features.csv" https://ncsa.osn.xsede.org/ees230012-bucket01/RobotTeam/unsupervised/df_features.csv
