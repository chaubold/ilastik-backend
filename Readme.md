# ilastik-backend prototype

The ilastik backend prototype was initially developed at a hackathon in Jan/Feb 2017.
It was an experiment to use Intel's Threading Building Blocks (TBB) from C++ to perform [ilastik](https://ilastik.org)'s pixel classification batch prediction mode.
Based on TBB's parallelized data flow graph, blocks of the input volume are processed. These graph operators can be found in `include/ilastik-backend/operators` and `include/ilastik-backend/flowgraph`, and a command line executable to run it in `bin`.

## Dependencies

* a C++ compiler supporting at least C++11
* Boost
* vigra
* libhdf5
* optional: Intel's TBB

## Microservice architecture (microservice branch)

To be able to scale to multiple machines e.g. in the cloud, the utilities of repository got a second use: as a C++ backend for little python microservices that can perform the computations needed by pixel classification and provide a simple REST Api.
The computational backend of pixel classification can be found in `include/ilastik-backend/tasks`. The Python wrapper is built in the `python` folder using `externals/pybind11`. 

To build the project, do:

```sh
git clone https://github.com/chaubold/ilastik-backend.git
cd ilastik-backend
git checkout microservice
git submodule init
git submodule update --recursive
mkdir build
cd build
cmake .. # and configure everything properly, including WITH_PYTHON=TRUE, you'll need boost, vigra and libhdf5!
make install
```

See the `services` folder for the Python based microservices.