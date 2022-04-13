#!/bin/sh
find . -type f -name ".DS_Store" -print | xargs rm -f
rm -rf __pycache__ */__pycache__
