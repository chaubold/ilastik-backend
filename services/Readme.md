# ilastik backend microservices

This is a prototype for running ilastik as a service (*IaaS?*). Python 3 only!


## Requirements

* Flask: pip install flask
* Autodoc (will probably replaced in the future): pip install Flask-Autodoc
* Requests: conda install requests
* Redis-py: conda install redis-py

Additionally, the C++ part of this project must be built with its python bindings:

```sh
git clone https://github.com/chaubold/ilastik-backend.git
cd ilastik-backend
git checkout microservice
git submodule init
git submodule update --recursive
mkdir build
cd build
cmake .. # and configure everything properly, including WITH_PYTHON=TRUE, you need vigra and libhdf5!
make install
```

You also need to run a redis server on the default port which is used to communicate the results.
Install docker and run the latest redis in a linux container as follows: 
    
```sh
docker run -d -p 6379:6379 --name redis bitnami/redis:latest
```

Where the parameters mean:
* `-d` = run as daemon
* `-p` port:port = forward this specific port from the container to localhost (6379 is redis default)
* `--name` = provide a name for the container that can be used with start/stop,...


## Running a pixel classification prediction server

To start the pixel classification prediction server for a little test project and dataset, which listens on the default port `8888`, run:

```sh
cd pixelclassificationservice
python service.py --project ../test/pc.ilp --raw-data-file ../test/raw.h5 --raw-data-path exported_data --use-caching
```

To see the API the client offers, navigate your webbrowser to `localhost:8888/doc`, for instance retrieve the predictions for a roi by

```sh
http://localhost:8888/prediction/png/roi?extents_min=10_10&extents_max=100_150
```