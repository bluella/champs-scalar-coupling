#!/bin/bash
set -e
unameOut="$(uname -s)"

case "${unameOut}" in
    Linux*)     machine=Linux;;
    Darwin*)    machine=Mac;;
    CYGWIN*)    machine=Cygwin;;
    MINGW*)     machine=MinGw;;
    *)          machine="UNKNOWN:${unameOut}"
esac

echo "your machine is $machine"
echo $[$machine == "Linux"]
if [ $machine == "Linux" ]; then
    virtualenv --no-site-packages -p /usr/bin/python3.7 ds_env
    echo "virtualenv created"
elif [ $machine == "Mac" ]; then
    virtualenv --no-site-packages -p /usr/local/bin/python3.7 ds_env
    echo "virtualenv created"
else
    echo "Setup on your $machine is not supported, please edit setup.sh"
fi

source ds_env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
