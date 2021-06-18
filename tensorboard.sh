#!/bin/bash

PORT=$((($UID-6025) % 65274))
echo $PORT
HOST=$(hostname -s)
echo $HOST
module load languages/anaconda3/2019.07-3.6.5-tflow-1.14
echo "In local terminal:"
echo "ssh -N -L 6006:$HOST:$PORT bcp4"
tensorboard --logdir logs --port "$PORT"
