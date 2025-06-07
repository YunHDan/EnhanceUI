@echo off
export PYTHONPATH="$PWD:PYTHONPATH"
python basicsr/test.py -opt options/test.yml
pause