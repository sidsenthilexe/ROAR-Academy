# ROAR-Academy

ROAR Academy was created by Dr. Allen Y. Yang. The material has been taught at the University of California, Berkeley, as part of Berkeley ROAR Racing curriculum to students who want to learn introduction-level skills about:

* Python Programming;
* Scientific Programming using NUMPY
* Gradient-Descent Algorithms
* Deep Neural Networks (DNN)

The material has been made open source FOR NON-COMMERCIAL USE only. Please contact the author for any questions regarding commercial licensing: <allen@intelligentracing.com>

## Basic Installation

Users of this course are recommended to install the following software packages. The packages have been tested on Windows 10 and above, Mac OSX, and Ubuntu Linux systems.

* Python 3.11 via Miniconda (conda): <https://docs.conda.io/en/latest/miniconda.html>
* Git: <https://git-scm.com/book/en/v2/Getting-Started-Installing-Git>
* Visual Studio Code (code): <https://code.visualstudio.com/download>

**NOTE: If you are running conda and code in Windows, launching code from Anaconda Prompt will make sure that the conda environment available inside the integrated Terminal window. Otherwise you may encounter error that conda is not installed**

Once Python 3.11 is installed, create a dedicated conda environment for the course and install every required Python module in a single step using the `requirements.txt` file at the repo root:
~~~
    conda create -n roar python=3.11
    conda activate roar
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
~~~
The `requirements.txt` file pins NumPy (1.x, up to 1.26), matplotlib, TensorFlow 2.15 (Keras 2 bundled), Gymnasium, pygame, and the Jupyter notebook tooling to versions that have been tested together for this course. If you prefer to install modules one at a time, the sections below list the individual `pip install` commands.

For the basic Python and NumPy exercises (Part One and the early Part Two notebooks), the minimum set is:
~~~
    python -m pip install numpy matplotlib
~~~

## Tensorflow Installation for Windows and Linux

For the DNN portion, we use tensorflow 2 and keras. The installation shall reference the official documentation: <https://www.tensorflow.org/install>. 

**If your PC comes with supported NVidia GPU for accelerating DNN code, please follow carefully the setup of GPU support for Linux and Windows. For MacOS, please read the instruction at the end**

After you have setup your system for an installation on either CPU or GPU, run the following pip script to install tensorflow 2.15. TensorFlow 2.15 is the last release that ships Keras 2 by default, which is the Keras API used throughout the course material.
~~~
    python -m pip install --upgrade pip
    python -m pip install "tensorflow==2.15.*"
~~~

Finally, you may verify tensorflow has been properly set up by running the following test
~~~
    python -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
~~~

If you have install tensorflow with GPU support, you may verify by running the following test
~~~
    python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
~~~

## Tensorflow Installation with Metal accelerated backend for MacOS

Apple ships a Metal GPU plugin that lets TensorFlow use the GPU on Apple Silicon Macs (M1/M2/M3/M4). With TensorFlow 2.15 the regular `tensorflow` package on PyPI already ships native arm64 macOS wheels, so the install reduces to two commands — first TensorFlow itself, then the Metal plugin:
~~~
    python -m pip install "tensorflow==2.15.*"
    python -m pip install tensorflow-metal
~~~
This is the same two-line install Apple recommends at <https://developer.apple.com/metal/tensorflow-plugin/>. The Metal plugin only ships arm64 wheels, so it is skipped automatically on Intel Macs and on Linux/Windows.

If you run `pip install -r requirements.txt` the marker `sys_platform == "darwin" and platform_machine == "arm64"` already takes care of installing `tensorflow-metal` for you on Apple Silicon — no extra step needed.
