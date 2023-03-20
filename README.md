## RKNN DEMO

```bash
# python
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt install python3.7 python3.7-distutils python3.7-dev
sudo apt install python3-pip python-pip virtualenv autopep8

# init
virtualenv venv --python=python3.7
pip3 install opencv-python ffmpeg-python

# rknn
git clone https://github.com/rockchip-linux/rknpu2.git
git clone https://github.com/rockchip-linux/rknn-toolkit2.git
sudo cp rknpu2/runtime/RK3588/Linux/librknn_api/aarch64/* /usr/lib
pip3 install rknn-toolkit2/rknn_toolkit_lite2/packages/rknn_toolkit_lite2-1.4.0-cp37-cp37m-linux_aarch64.whl
```