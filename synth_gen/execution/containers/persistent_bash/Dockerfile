# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

FROM ubuntu:latest

RUN apt-get update && apt-get install -y python3 python3-pip jupyter python3-requests python3-flask python3-jupyter-client python3-venv

RUN pip install --break-system-packages bash_kernel
RUN python3 -m bash_kernel.install

RUN apt-get install -y sudo
RUN echo 'ubuntu ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers


COPY . /app
WORKDIR /app

CMD ["python3", "--version"]
