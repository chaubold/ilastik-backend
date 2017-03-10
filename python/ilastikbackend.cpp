#include <pybind11/pybind11.h>
#include <vigra/multi_math.hxx>

#include "vigraconverter.h"
#include "pyblocking.h"
#include "pypixelclassification.h"

namespace py = pybind11;

template<int DIM, typename DTYPE>
py::array_t<DTYPE, py::array::c_style | py::array::forcecast> add(py::array_t<DTYPE, py::array::c_style | py::array::forcecast> a,
                                                                  py::array_t<DTYPE, py::array::c_style | py::array::forcecast> b)
{
    using namespace vigra::multi_math;
    auto vigraA = numpy_to_vigra<DIM, DTYPE>(a);
    auto vigraB = numpy_to_vigra<DIM, DTYPE>(b);
    vigra::MultiArray<DIM, DTYPE> vigraResult(vigraA + vigraB);
    return vigra_to_numpy<DIM, DTYPE>(vigraResult);
}

void export_pixel_classification(pybind11::module& m)
{
    export_pixel_classificationT<2, uint8_t, float>(m, "uint8");
    export_pixel_classificationT<3, uint8_t, float>(m, "uint8");

    export_pixel_classificationT<2, uint16_t, float>(m, "uint16");
    export_pixel_classificationT<3, uint16_t, float>(m, "uint16");

    export_pixel_classificationT<2, float, float>(m, "float32");
    export_pixel_classificationT<3, float, float>(m, "float32");
}

void export_blocking(py::module& m)
{
    export_blockingT<2>(m);
    export_blockingT<3>(m);
}


PYBIND11_PLUGIN(pyilastikbackend) {
    py::module m("pyilastikbackend", "python ilastik backend module providing feature computation "
                                     "and pixel classification methods on blocks");

    m.def("add_2d_Float", &add<2,float>, "A function which adds two arrays", py::arg().noconvert(), py::arg().noconvert());
    m.def("add_2d_Uint8", &add<2,uint8_t>, "A function which adds two arrays", py::arg().noconvert(), py::arg().noconvert());

    export_pixel_classification(m);
    export_blocking(m);

    return m.ptr();
}
