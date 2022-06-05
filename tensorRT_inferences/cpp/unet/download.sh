# bin/sh
wget https://developer.download.nvidia.com/devblogs/speeding-up-unet.7z
apt-get install p7zip-full
7z x speeding-up-unet.7z
python3 create_network.py
