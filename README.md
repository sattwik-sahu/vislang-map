# VisLang Map

Vision-and-Language-Based Mapping for Offroad Vehicles.

## Installation

Clone the repo first, then follow the steps below.

### Setup Docker container

1. In `Dockerfile`, change the `USERNAME` on line 23 to your required username
    ```Dockerfile
    ARG USERNAME=<your_username>
    ```

2. Change the image name in `scripts/build_container.sh` on line 3
    ```bash
    docker build -t <your_image_name>:latest .
    ```

3. Change the container name and image name in `scripts/start_container.sh` (line 17, 18) and `scripts/connect_container.sh` (line 7).

4. Run the commands
    ```bash
    sudo chmod +x ./scripts/*
    ./scripts/build_image.sh
    ./scripts/start_container.sh
    ./scripts/connect_container.sh
    ```

### Setup Environment

1. Install OhMyZsh plugin for zsh *(OPTIONAL)*
    ```bash
    sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"    # via curl

    # OR

    sh -c "$(wget https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh -O -)"    # via wget
    ```

2. Setup `pyenv` for Python version management
    - Install `pyenv`
        ```bash
        curl https://pyenv.run | bash
        ```
    - Configure `pyenv` for `zsh`
        ```bash
        echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
        echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
        echo 'eval "$(pyenv init -)"' >> ~/.zshrc
        ```
    > The above instructions have been taken from the [`pyenv` documentation](https://github.com/pyenv/pyenv)

3. Install Python `3.10` using `pyenv` and activate it inside the project folder
    ```bash
    pyenv install 3.10
    pyenv local 3.10
    ```

### Setup Python Environment

1. Create virtual environment
    ```bash
    python -m venv .venv --prompt=vlmap
    source .venv/bin/activate
    which python                        # Verify the environment activation
    python -m pip install -U pip        # Update pip version
    
2. Setup poetry for dependency management
    ```bash
    python -m pip install poetry
    poetry install                      # Installs deps automatically from pyproject.toml
    ```

## Usage

### Launch ROSBag

ROSBags are stored in `data/rosbags` folder run them using

```bash
rosbag play <path_to_rosbag>
```

> ROSBags are not included in the repo. Download them from [RELLIS-3D ROSBags](https://github.com/unmannedlab/RELLIS-3D?tab=readme-ov-file#ros-bag-download)

### Run CLIPSeg segmentation on RELLIS-3D camera feed

```bash
poetry run python vislang_map
```

### Create pointcloud from stereo image

- Edit `launch/nerian_transform.launch` to publish the transform frames for the camera.
- Edit `launch/stereo_pointcloud.launch` to tweak the tf for the cameras.
- These scripts can be generalized to use any stereo camera topics you publish.

```bash
roslaunch launch/stereo_pointcloud.launch
```
