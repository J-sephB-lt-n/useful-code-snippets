<<'###BLOCK-COMMENT'
TAGS: cloud|compute engine|ec2|gcp|google|google cloud|install|pyenv|python|setup|set up|virtual machine|vm
DESCRIPTION: Bash code for setting up python on a google cloud platform EC2 compute engine virtual machine (using pyenv)
###BLOCK-COMMENT

################
# QUICK METHOD #
################
# This is quicker to get up and running, but makes it harder when you want to install a new version of python
sudo apt-get install -y python3-pip
python3 --version
sudo apt-get install -y python3.11-venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

################
# PYENV METHOD #
################
# This allows easy management of python versions
sudo apt update
sudo apt install curl git-all -y
sudo apt install build-essential libssl-dev zlib1g-dev \
  libbz2-dev libreadline-dev libsqlite3-dev curl \
  libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev -y

sudo curl https://pyenv.run | bash

echo '
export PYENV_ROOT="$HOME/.pyenv"
[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"' | tee -a ~/.profile ~/.bashrc

echo 'eval "$(pyenv virtualenv-init -)"' >>~/.bashrc

exec "$SHELL" # restart shell

pyenv install --list | grep " 3\.12"
pyenv install -v 3.12.3
pyenv versions
pyenv global 3.12.3
pip install --upgrade pip
