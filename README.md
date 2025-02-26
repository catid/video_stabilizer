# Halide Accelerated Robust Image Alignment

Uses method described here: https://catid.io/posts/lk40years/

There are some differences and improvements, primarily I've changed how I select the sparse set of pixels to use as keypoints: Now I select about 1000 keypoints with the largest X gradient and 1000 keypoints with the largest Y gradient, basically doubling the work but the result is about 10x fewer frame alignment errors.  My theory is that sometimes only large Y gradient pixels are selected so there is not enough X gradient information to do a good alignment.  By selecting the best pixels for x and y gradient across the whole image, it converges much more often.

Phase correlation initialization is available but is disabled since it seems to make things worse (maybe a bug?)

One thing that is worse is that I haven't managed to get a UKF working to estimate the camera intentional motion so there's a large amount of black pull-in artifacts that can be improved further.


## Setup

I've tested this on an Ubuntu Intel machine and in an Aarch64 Ubuntu VM on MacOS.

```bash
git clone https://github.com/catid/video_stabilizer/
cd video_stabilizer

sudo apt install cmake clang llvm
sudo apt install libeigen3-dev
sudo apt install libopencv-dev
```

Set up Halide:

```bash
mkdir distros
cd distros

wget https://github.com/halide/Halide/releases/download/v19.0.0/Halide-19.0.0-x86-64-linux-5f17d6f8a35e7d374ef2e7e6b2d90061c0530333.tar.gz
tar vzxf *.tar.gz

# Or the release for your platform from: https://github.com/halide/Halide/releases/

cd ..
```

## Build

This requires a very recent version of CMake from https://cmake.org/download/

```bash
mkdir build
cd build
cmake -DHALIDE_VERSION=19.0.0 ..
make -j
./align_test
./video_test
```
