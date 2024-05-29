## pyFFD - A fast finite difference solver for Maxwell's wave equation in biological tissue

This is an implementation of the fast finite difference solver presented in (TODO:add paper link).

### Installation

This code has been tested on Windows, Mac and Linux with Python 3.10.12, numpy 1.25.0, matplotlib 3.7.1 and scipy 1.11.0. To leverage multiple CPU cores to speed up computation, you'll need the multiprocessing module. This module likely comes with your python installation, but may not work with Jupyter notebooks - you'll have to run the code from a terminal.

Consider using a numpy package that is optimized for your processor. In most cases, you'll see up to a 2x improvement in performance over a standard numpy installation.

### Usage

simple_example.py is a minimal demonstration of the algorithm. It is single-threaded and does not use an adaptive resolution. The first time you run the code, it peroforms a benchmark to identify matrix sizes that are fastest for FFT in your processor.

STED_donut_LG_HG_simulation.py is the code used to generate the results in the paper. Data_visualization.py calculates the STED resolution from the donut profiles calculated by STED_donut_LG_HG_simulation.py

### Known Issues

When using the adaptive step size algorithm, the depth has to be a multiple of section_depth.

If the discretization is too large, the power of the beam blows up. i.e., the total optical power at the focus may be greater than the power sent in.

There is currently no check to make sure there is sufficient RAM. If you set num_processes too high, you will run out of RAM and the simulation will slow to a crawl.
