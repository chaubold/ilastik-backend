#include <pybind11/pybind11.h>
#include <vigra/multi_math.hxx>

#include "vigraconverter.h"
#include "pyblocking.h"
#include "pypixelclassification.h"

namespace py = pybind11;

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
//    export_blockingT<2>(m);
//    export_blockingT<3>(m);
//    export_blockingT<4>(m);
    // we only use 5D blocking!
    export_blockingT<5>(m);
}


PYBIND11_PLUGIN(pyilastikbackend) {
    py::module m("pyilastikbackend", "python ilastik backend module providing feature computation "
                                     "and pixel classification methods on blocks");

    export_pixel_classification(m);
    export_blocking(m);

    return m.ptr();
}
