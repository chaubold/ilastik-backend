#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vigra/multi_array.hxx>
#include <vigra/multi_math.hxx>
#include <exception>

namespace py = pybind11;

template<int DIM, typename DTYPE>
vigra::MultiArrayView<DIM, DTYPE> numpyArrayToVigra(py::array_t<DTYPE, py::array::c_style | py::array::forcecast> py_array)
{
    py::buffer_info info = py_array.request();
    /**
    struct buffer_info {
        void *ptr;
        size_t itemsize;
        std::string format;
        int ndim;
        std::vector<size_t> shape;
        std::vector<size_t> strides;
    };
    */

    if(info.ndim != DIM)
        throw std::runtime_error("Dimension mismatch between function and argument");

    vigra::TinyVector<int64_t, DIM> shape;
    vigra::TinyVector<int64_t, DIM> strides;

    for(int i = 0; i < DIM; i++)
    {
        shape[i] = info.shape[i];
        strides[i] = info.strides[i] / sizeof(DTYPE); // vigra uses stride = num elements, pybind11 num bytes
    }

    auto ptr = static_cast<DTYPE *>(info.ptr);
    vigra::MultiArrayView<DIM, DTYPE> out(shape, strides, ptr);
    return out;
}

template<int DIM, typename DTYPE>
py::array_t<DTYPE, py::array::c_style | py::array::forcecast> vigraToNumpyArray(vigra::MultiArrayView<DIM, DTYPE> vigra_array)
{
    std::vector<size_t> strides;
    std::vector<size_t> shape;

    for(int i = 0; i < DIM; i++)
    {
        strides.push_back(vigra_array.stride()[i] * sizeof(DTYPE));
        shape.push_back(vigra_array.shape()[i]);
    }

    return py::array(py::buffer_info(vigra_array.data(), sizeof(DTYPE),
                   py::format_descriptor<DTYPE>::value,
                   DIM, shape, strides));
}

template<int DIM, typename DTYPE>
py::array_t<DTYPE, py::array::c_style | py::array::forcecast> add(py::array_t<DTYPE, py::array::c_style | py::array::forcecast> a,
                                                                  py::array_t<DTYPE, py::array::c_style | py::array::forcecast> b)
{
    using namespace vigra::multi_math;
    auto vigraA = numpyArrayToVigra<DIM, DTYPE>(a);
    auto vigraB = numpyArrayToVigra<DIM, DTYPE>(b);
    vigra::MultiArray<DIM, DTYPE> vigraResult(vigraA + vigraB);
    return vigraToNumpyArray<DIM, DTYPE>(vigraResult);
}

PYBIND11_PLUGIN(pyilastikbackend) {
    py::module m("pyilastikbackend", "python ilastik backend module providing feature computation "
                                     "and pixel classification methods on blocks");

    m.def("add_2d_Float", &add<2,double>, "A function which adds two arrays", py::arg().noconvert(), py::arg().noconvert());
    m.def("add_2d_Uint8", &add<2,uint8_t>, "A function which adds two arrays", py::arg().noconvert(), py::arg().noconvert());

    return m.ptr();
}
