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
* [Cursor](https://cursor.com/download) (recommended IDE with integrated AI for this course), or Visual Studio Code: <https://code.visualstudio.com/download>

**NOTE: If you are running conda and Cursor/VS Code on Windows, launch the editor from Anaconda Prompt (or ensure your shell initializes conda) so the `roar` environment is available in the integrated Terminal. Otherwise you may encounter an error that conda is not installed.**

Once Python 3.11 is installed, create a dedicated conda environment for the course and install every required Python module in a single step using the `requirements.txt` file at the repo root:
~~~
    conda create -n roar python=3.11
    conda activate roar
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
~~~
The `requirements.txt` file pins NumPy (1.x, up to 1.26), matplotlib, TensorFlow 2.15 (Keras 2 bundled), Gymnasium, pygame, and the Jupyter notebook tooling to versions that have been tested together for this course.

For the basic Python and NumPy exercises (Part One and the early Part Two notebooks), the minimum set is:
~~~
    python -m pip install numpy matplotlib
~~~

### For macOS

**Minimum macOS version**

| Mac type | Minimum macOS | Notes |
|---|---|---|
| Intel Mac | 10.15 (Catalina) | TensorFlow runs on CPU only |
| Apple Silicon (M1/M2/M3/M4) | 12.0 (Monterey) | Required for TensorFlow 2.15 arm64 wheels and Metal GPU acceleration |

**Install Xcode Command Line Tools**

Apple recommends the Xcode Command Line Tools for building native Python packages and for TensorFlow Metal GPU support on Apple Silicon. Install them once before running `pip install -r requirements.txt`:
~~~
    xcode-select --install
~~~
If prompted, accept the license and wait for the download to finish. You can verify the install with:
~~~
    xcode-select -p
~~~

**Apple Silicon**

* Use **native arm64 Python**, not an x86_64 build running under Rosetta. After activating the `roar` environment, check with:
  ~~~
      python -c "import platform; print(platform.machine())"
  ~~~
  The output should be `arm64`.
* `pip install -r requirements.txt` automatically installs `tensorflow-metal` for Metal GPU acceleration — no extra step needed.

**Intel Mac**

TensorFlow installs and runs on CPU. The Metal GPU plugin (`tensorflow-metal`) is Apple Silicon only and is skipped automatically by `requirements.txt`.

**NVIDIA GPU on Windows or Linux**

For GPU-accelerated TensorFlow on Windows or Linux, set up CUDA and cuDNN per the [official TensorFlow install guide](https://www.tensorflow.org/install) before running `pip install -r requirements.txt`.

## Cursor AI for Teaching and Learning

ROAR Academy notebooks and sample code work in any Jupyter-compatible editor. **Cursor** is the recommended editor for this course because it combines VS Code's notebook support with an AI assistant that can explain lecture concepts, help debug code, and guide exercises — while you stay in the same workspace as the course material.

### Install Cursor

1. Download Cursor for your platform: <https://cursor.com/download>
2. Install and launch Cursor.
3. Sign in when prompted (a free tier is available; some AI features may require a paid plan depending on Cursor's current offering).

Cursor is built on VS Code. If you already use VS Code, your familiar keyboard shortcuts, extensions, and settings largely carry over.

### Open the course repository

From a terminal:
~~~
    git clone https://github.com/learnwithallen/ROAR-Academy.git
    cd ROAR-Academy
~~~
In Cursor, choose **File → Open Folder** and select the `ROAR-Academy` directory.

### Install Python modules for Cursor

Complete the **Basic Installation** steps above first — they apply whether you use Cursor or another editor. In Cursor's integrated terminal (**View → Terminal**), with the `roar` environment activated:

~~~
    conda activate roar
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
~~~

The `requirements.txt` file installs everything needed for the full curriculum:

| Package group | Used for |
|---|---|
| `numpy`, `matplotlib`, `scipy`, `pandas`, `pillow`, `opencv-python` | Part One Python basics; Part Two NumPy, plotting, and file/image samples |
| `tensorflow==2.15.*` (+ `tensorflow-metal` on Apple Silicon) | Deep neural network lectures and MNIST/CNN samples |
| `gymnasium`, `pygame` | Reinforcement learning and CartPole DQN samples |
| `stable-baselines3`, `highway-env` | Advanced RL sample (`parking_sac_her.py`) |
| `jupyter`, `ipykernel`, `notebook`, `ipywidgets` | Running `.ipynb` lecture notebooks inside Cursor |

**Part One only (no TensorFlow yet):** the minimum install above (`numpy`, `matplotlib`) is enough to start. Install the full `requirements.txt` before Part Two DNN and RL notebooks. On macOS, complete **For macOS** under Basic Installation first (Xcode tools and version requirements).

Verify the environment from Cursor's terminal:
~~~
    python -c "import numpy, matplotlib; print('numpy', numpy.__version__)"
    python -c "import tensorflow as tf; print('tensorflow', tf.__version__)"
~~~
Skip the TensorFlow line until you reach the DNN portion of the course.

### Connect Cursor to the `roar` kernel

So notebook cells run with the correct packages:

1. **Select the Python interpreter:** open the Command Palette (**Cmd+Shift+P** on Mac, **Ctrl+Shift+P** on Windows/Linux), run **Python: Select Interpreter**, and choose the `roar` conda environment.
2. **Register the kernel** (once per machine):
   ~~~
       conda activate roar
       python -m ipykernel install --user --name roar --display-name "Python (roar)"
   ~~~
3. Open any notebook under `Part One/` or `Part Two/notebooks/`.
4. Click the kernel picker in the top-right of the notebook and select **Python (roar)**.

Run a single cell with the play button on the left of the cell, or run all cells from the notebook toolbar.

### Using Cursor AI in the course

Cursor provides several ways to get help while working through lectures and exercises:

| Feature | When to use it |
|---|---|
| **Chat** (AI panel) | Ask concept questions, request step-by-step explanations, or clarify lecture Keywords |
| **Inline Edit** (select code → ask for changes) | Refactor a small block, fix a syntax error, or add comments to your own code |
| **Agent** | Multi-step tasks: debug a failing cell, trace an import error, or walk through an exercise with terminal commands |

**Tips for effective prompts:**

* **Point Cursor at the repo.** With the `ROAR-Academy` folder open, the AI can read notebooks and samples in `Part One/`, `Part Two/notebooks/`, and `Part Two/samples/`.
* **Reference the lecture.** Name the notebook (e.g. `1-6-functions.ipynb`) or paste the exercise text from the notebook.
* **Ask for teaching, not just answers.** Phrases like "explain why this fails" or "give me a hint without the full solution" keep the learning goal clear.
* **Include errors verbatim.** Paste the full traceback from a notebook cell or terminal when debugging.

### Example prompts for students

**Understanding lecture material**
~~~
    I'm on Part One, lecture 1-4-lists.ipynb. Explain the difference between
    append() and extend() on a Python list, with a short example I can run
    in a new cell.
~~~

**Debugging notebook code**
~~~
    This cell in 2-2-numpy.ipynb raises "ValueError: shapes not aligned".
    Explain what went wrong and suggest a fix, but let me apply the change myself.
~~~

**Exercise help (hint-first)**
~~~
    Part Two exercise: write a function that computes the dot product of two
    NumPy arrays without using np.dot. Give me a hint about which loop structure
    to use — don't write the full solution.
~~~

**Connecting concepts to samples**
~~~
    How does gradient_descent.py in Part Two/samples relate to the math in
    2-7-deep-neural-networks.ipynb? Summarize the shared ideas in plain language.
~~~

### Example prompts for instructors and TAs

**Lesson prep**
~~~
    Review Part One/1-8-sets-and-hashing.ipynb and list the three most common
    mistakes beginners make with Python sets. Suggest one in-notebook check
    question for each.
~~~

**Sample code walkthrough**
~~~
    Walk through perceptron.py in Part Two/samples and outline a 10-minute
    live-coding demo that connects it to lecture 2-6.
~~~

**Environment troubleshooting**
~~~
    A student on Windows 11 reports "No module named tensorflow" when running
    2-7-deep-neural-networks.ipynb in Cursor. List the most likely causes and
    the commands to verify their conda env and kernel selection.
~~~

### Academic integrity and responsible AI use

Cursor is a **learning assistant**, not a substitute for doing the work. Course policy may vary by institution; by default:

* **Do** use AI to clarify definitions, debug errors, and explore alternative explanations.
* **Do** write and run code yourself so you understand every line you submit.
* **Do not** paste entire exercise or exam solutions generated by AI without attribution where your instructor requires it.
* **Do not** submit AI-generated code you cannot explain in your own words.

When in doubt, ask your instructor how AI tools are allowed for assignments and exams.
