## RKNN DEMO

```bash
# python
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt install python3.9 python3.9-distutils python3.9-dev
sudo apt install python3-pip python-pip virtualenv

# init
virtualenv venv --python=python3.9
pip3 install opencv-python ffmpeg-python

# rknn
git clone https://github.com/rockchip-linux/rknpu2.git
git clone https://github.com/rockchip-linux/rknn-toolkit2.git
sudo cp rknn_toolkit_lite2/* /usr/lib
pip3 install rknn_toolkit_lite2/rknn_toolkit_lite2-1.4.0-cp39-cp39-linux_aarch64.whl
```