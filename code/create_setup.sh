#!/usr/bin/env sh

WORK_PATH=$1

if [ -d $WORK_PATH ]; then
  echo "Path $WORK_PATH already exists"
  exit 1
else
  mkdir $WORK_PATH
fi

mkdir $WORK_PATH/work
mkdir $WORK_PATH/output
mkdir $WORK_PATH/alias

ln -s $WORK_PATH/work work
ln -s $WORK_PATH/output output
ln -s $WORK_PATH/alias alias
