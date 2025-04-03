// BINDING C++ ENV CLASS TO PYTHON

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "Environment.h"

namespace py = pybind11;

PYBIND11_MODULE(cpp_env, m) {

    m.doc() = "C++ ABM Environment Module";

    py::class_<Environment>(m, "Environment")
        .def(py::init<>())
        .def("reset", &Environment::reset)
        .def("step", &Environment::step)
        .def("action_space", &Environment::getActionSpace)
        .def("observation_space", &Environment::getObservationSpace)
        .def("isDone", &Environment::getDone)
        .def("getTime", &Environment::getTime)
        .def("outputTxt", &Environment::outputTxt);
}