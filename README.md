# Playground 

Playground tasks for newcomers!

## 1. Environments

Fork this repository to your own github account:

![image](./docs/imgs/fork.png)

Start a docker from image `playground:v1.0-cuda12.2-cudnn8-ubuntu22`:

```bash
# Params:
#   <container-name>: Your container name. Can be anything.
#   <container-port>: A port that nobody has used. For example: 1145.
#   <project-home>: A directory that stores all your projects. This directory should be shared by all containers.
#   <proxy-addr>: Your proxy address.
docker run --gpus all --name <container-name> -it \
    -p <container-port>:22 --entrypoint /bin/bash \
    -v /proc:/proc \
    -v /sys:/sys \
    -v /dev:/dev \
    -v /var/lock:/var/lock \
    -v /var/log:/var/log \
    -v ~/<project-home>:/root/<project-home>  \
    -e HTTP_PROXY=<proxy-addr>  \
    -e HTTPS_PROXY=<proxy-addr>  \
    -e http_proxy=<proxy-addr>  \
    -e https_proxy=<proxy-addr>  \
    playground:v1.0-cuda12.2-cudnn8-ubuntu22
```

After you are inside the docker, clone this repo:

```bash
cd /root/<project-home>
git clone https://github.com/<your-github-account-name>/playground.git
```

Remember to change your git config:

```bash
git config --global user.name "<your-name>"
git config --global user.email "<your-email>"
```

## 2. Tasks

| No. | Discription | Directory | Document |
|:---:|:---:|:---:|:---:|
| 1 | CUDA Programming: GEMM | [./task-1](./task-1) | [READEME.md](./task-1/README.md) |

