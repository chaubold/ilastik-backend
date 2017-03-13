#ifndef PYBLOCKING_H
#define PYBLOCKING_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "vigraconverter.h"

#include "ilastik-backend/utils/blocking.h"

using coordinate_type = int64_t;
using coordinate_array = pybind11::array_t<coordinate_type, pybind11::array::f_style | pybind11::array::forcecast>;

// ------------------------------------------------------------------------------------
template<int DIM>
class PyBlock{
public:
    PyBlock(coordinate_array begin, coordinate_array end):
        block_(numpy_to_tiny_vector<DIM, coordinate_type>(begin),
               numpy_to_tiny_vector<DIM, coordinate_type>(end))
    {}

    PyBlock(const ilastikbackend::utils::Block<DIM>& block):
        block_(block)
    {}

    const coordinate_array begin() const
    { return tiny_vector_to_numpy<DIM, coordinate_type>(block_.begin()); }

    const coordinate_array end() const
    { return tiny_vector_to_numpy<DIM, coordinate_type>(block_.end()); }

    const coordinate_array shape() const
    { return tiny_vector_to_numpy<DIM, coordinate_type>(block_.shape()); }

private:
    ilastikbackend::utils::Block<DIM> block_;
};

// ------------------------------------------------------------------------------------
template<int DIM>
class PyBlockWithHalo{
public:
    PyBlockWithHalo(coordinate_array outerBlock, coordinate_array innerBlock):
        block_(numpy_to_tiny_vector<DIM, coordinate_type>(outerBlock),
               numpy_to_tiny_vector<DIM, coordinate_type>(innerBlock))
    {}

    PyBlockWithHalo(const ilastikbackend::utils::BlockWithHalo<DIM>& block):
        block_(block)
    {}

    const PyBlock<DIM> outerBlock() const
    { return PyBlock<DIM>(block_.outerBlock()); }

    const PyBlock<DIM> innerBlock() const
    { return PyBlock<DIM>(block_.innerBlock()); }

    const PyBlock<DIM> innerBlockLocal() const
    { return PyBlock<DIM>(block_.innerBlockLocal()); }
private:
    ilastikbackend::utils::BlockWithHalo<DIM> block_;
};

// ------------------------------------------------------------------------------------
template<int DIM>
class PyBlocking{
public:
    PyBlocking(coordinate_array roiBegin,
               coordinate_array roiEnd,
               coordinate_array blockShape):
        blocking_(numpy_to_tiny_vector<DIM, coordinate_type>(roiBegin),
                  numpy_to_tiny_vector<DIM, coordinate_type>(roiEnd),
                  numpy_to_tiny_vector<DIM, coordinate_type>(blockShape))
    {}

    PyBlocking(coordinate_array roiBegin,
               coordinate_array roiEnd,
               coordinate_array blockShape,
               coordinate_array blockShift):
        blocking_(numpy_to_tiny_vector<DIM, coordinate_type>(roiBegin),
                  numpy_to_tiny_vector<DIM, coordinate_type>(roiEnd),
                  numpy_to_tiny_vector<DIM, coordinate_type>(blockShape),
                  numpy_to_tiny_vector<DIM, coordinate_type>(blockShift))
    {}

    PyBlocking(ilastikbackend::utils::Blocking<DIM> blocking):
        blocking_(blocking)
    {}

    PyBlock<DIM> getBlock(const uint64_t blockIndex) const
    { return PyBlock<DIM>(blocking_.getBlock(blockIndex)); }

    PyBlockWithHalo<DIM> getBlockWithHalo(const uint64_t blockIndex, coordinate_array halo) const
    { return PyBlockWithHalo<DIM>(blocking_.getBlockWithHalo(blockIndex, numpy_to_tiny_vector<DIM, coordinate_type>(halo))); }

    const uint64_t getSurroundingBlockIndex(coordinate_array coordinate) const {
        return blocking_.getSurroundingBlockIndex(numpy_to_tiny_vector<DIM, coordinate_type>(coordinate));
    }

    const coordinate_array roiBegin() const
    { return tiny_vector_to_numpy<DIM, coordinate_type>(blocking_.roiBegin()); }

    const coordinate_array roiEnd() const
    { return tiny_vector_to_numpy<DIM, coordinate_type>(blocking_.roiEnd()); }

    const coordinate_array blockShape() const
    { return tiny_vector_to_numpy<DIM, coordinate_type>(blocking_.blockShape()); }

    const coordinate_array blockShift() const
    { return tiny_vector_to_numpy<DIM, coordinate_type>(blocking_.blockShift()); }

    const coordinate_array blocksPerAxis() const
    { return tiny_vector_to_numpy<DIM, coordinate_type>(blocking_.blocksPerAxis()); }

    const coordinate_array blocksPerAxisStrides() const
    { return tiny_vector_to_numpy<DIM, coordinate_type>(blocking_.blocksPerAxisStrides()); }

    const size_t numberOfBlocks() const
    { return blocking_.numberOfBlocks(); }

    ilastikbackend::utils::Blocking<DIM> getBlocking() const
    { return blocking_; }

private:
    ilastikbackend::utils::Blocking<DIM> blocking_;
};

// ------------------------------------------------------------------------------------
template<int DIM>
void export_blockingT(pybind11::module& m)
{
    const auto dim_str = std::string("_") + std::to_string(DIM) + std::string("d");

    const auto block_class_name = std::string("Block") + dim_str;
    pybind11::class_<PyBlock<DIM>>(m, block_class_name.c_str())
        .def(pybind11::init<coordinate_array, coordinate_array>())
        .def_property_readonly("begin",&PyBlock<DIM>::begin)
        .def_property_readonly("end",&PyBlock<DIM>::end)
        .def_property_readonly("shape",&PyBlock<DIM>::shape)
    ;

    const auto block_halo_class_name= std::string("BlockWithHalo") + dim_str;
    pybind11::class_<PyBlockWithHalo<DIM>>(m, block_halo_class_name.c_str())
        .def(pybind11::init<coordinate_array, coordinate_array>())
        .def_property_readonly("outerBlock",&PyBlockWithHalo<DIM>::outerBlock)
        .def_property_readonly("innerBlock",&PyBlockWithHalo<DIM>::innerBlock)
        .def_property_readonly("innerBlockLocal",&PyBlockWithHalo<DIM>::innerBlockLocal)
    ;

    const auto blocking_class_name = std::string("Blocking") + dim_str;
    pybind11::class_<PyBlocking<DIM>>(m, blocking_class_name.c_str())
        .def(pybind11::init<coordinate_array, coordinate_array, coordinate_array, coordinate_array>())
        .def(pybind11::init<coordinate_array, coordinate_array, coordinate_array>())
        .def("getBlock",&PyBlocking<DIM>::getBlock)
        .def("getBlockWithHalo",&PyBlocking<DIM>::getBlockWithHalo)
        .def("getSurroundingBlockIndex",&PyBlocking<DIM>::getSurroundingBlockIndex)
        .def_property_readonly("roiBegin",&PyBlocking<DIM>::roiBegin)
        .def_property_readonly("roiEnd",&PyBlocking<DIM>::roiEnd)
        .def_property_readonly("blockShape",&PyBlocking<DIM>::blockShape)
        .def_property_readonly("blockShift",&PyBlocking<DIM>::blockShift)
        .def_property_readonly("blocksPerAxis",&PyBlocking<DIM>::blocksPerAxis)
        .def_property_readonly("blocksPerAxisStrides",&PyBlocking<DIM>::blocksPerAxisStrides)
        .def_property_readonly("numberOfBlocks",&PyBlocking<DIM>::numberOfBlocks)
    ;
}

#endif // PYBLOCKING_H
