#!/bin/sh
rm -rf log tb images
find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf
find . -name '.DS_Store' -type f -ls -delete
