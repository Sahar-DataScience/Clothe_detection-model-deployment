# adapted from: https://github.com/facebookresearch/detectron2/blob/master/docker/Dockerfile

FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

ENV DEBIAN_FRONTEND noninteractive
RUN apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-get update && apt-get install -y \
	python3-opencv ca-certificates python3-dev git wget sudo curl && \
  rm -rf /var/lib/apt/lists/*

# create a non-root user
ARG USER_ID=1000
RUN useradd -m --no-log-init --system  --uid ${USER_ID} appuser -g sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER appuser
WORKDIR /home/appuser

ENV PATH="/home/appuser/.local/bin:${PATH}"
RUN curl https://bootstrap.pypa.io/pip/3.6/get-pip.py --output get-pip.py && \
	 python3 get-pip.py --user && \
	 python3 -m pip install --upgrade pip && \
	 rm get-pip.py
#RUN wget https://bootstrap.pypa.io/pip/3.6/get-pip.py && \
#	python3 get-pip.py --user && \
#	rm get-pip.py

# install dependencies
# See https://pytorch.org/ for other options if you use a different version of CUDA
RUN pip3 install --user tensorboard
RUN pip3 install --user torch==1.8.0 torchvision==0.9.0 -f https://download.pytorch.org/whl/cu102/torch_stable.html

RUN pip3 install --user 'git+https://github.com/facebookresearch/fvcore'
# install detectron2
RUN git clone https://github.com/facebookresearch/detectron2 return_img_repo
# set FORCE_CUDA because during `docker build` cuda is not accessible
ENV FORCE_CUDA="1"
# This will by default build detectron2 for all common cuda architectures and take a lot more time,
# because inside `docker build`, there is no way to tell which architecture will be used.
ARG TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"
#RUN pip3 install --upgrade setuptools
RUN pip3 install --user -e return_img_repo

# add dir
COPY requirements.txt /home/appuser/return_img_repo
RUN pip3 install --user -r /home/appuser/return_img_repo/requirements.txt
RUN pip3 install --user 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'


# Set a fixed model cache directory.
ENV FVCORE_CACHE="/tmp"
WORKDIR /home/appuser/return_img_repo
ENV PILLOW_VERSION=7.0.0

# add dir
COPY . /home/appuser/return_img_repo

# Make port 8080 available to the world outside the container
ENV PORT 8080
EXPOSE 8080

CMD ["python3", "app.py"]