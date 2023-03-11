#!/bin/bash

if [ ! -d './glove' ]; then
    mkdir glove
else
    echo './glove exist!'
fi

if [ ! -d './knn' ]; then
    mkdir knn
else
    echo './knn exist!'
fi

if [ ! -d './result' ]; then
    mkdir result
else
    echo './result exist!'
fi

python main.py

