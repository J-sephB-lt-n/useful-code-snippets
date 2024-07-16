<<'###BLOCK-COMMENT'
TAGS: browser|chrome|compute engine|ec2|instance|stable|vm|virtual machine|web
DESCRIPTION: Terminal code for installing latest stable chrome browser on an ubuntu machine
DESCRIPTION: e.g. on a gcloud compute engine VM or Amazon EC2 instance  
###BLOCK-COMMENT

wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
sudo dpkg -i google-chrome-stable_current_amd64.deb
sudo apt --fix-broken install -y
