# PyroWave (Vulkan-compute, intra-only wavelet codec) integration.
#
# PyroWave is vendored as a git submodule at third-party/pyrowave and hard-requires Granite +
# volk. It MUST be linked as its shared library (libpyrowave-shared), not statically: the shared
# library uses a version script (link.T) that exports only the pyrowave_* C API and hides its
# bundled volk. Linking Granite/volk statically instead would leak volk's strong `vkCreateInstance`
# (etc.) symbols and override the real Vulkan loader Sunshine uses, crashing all direct Vulkan
# calls.
#
# Build the shared library before configuring Sunshine (its checkout_granite.sh also fetches the
# Granite and volk sources that PyroWave needs):
#   git submodule update --init third-party/pyrowave
#   cd third-party/pyrowave && ./checkout_granite.sh
#   cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build

set(PYROWAVE_DIR "${CMAKE_SOURCE_DIR}/third-party/pyrowave")
set(PYROWAVE_SHARED "${PYROWAVE_DIR}/build/libpyrowave-shared.so")

if(NOT EXISTS "${PYROWAVE_DIR}/pyrowave.h")
    message(FATAL_ERROR "SUNSHINE_ENABLE_PYROWAVE is ON but third-party/pyrowave is missing. "
            "Run: git submodule update --init third-party/pyrowave")
endif()
if(NOT EXISTS "${PYROWAVE_SHARED}")
    message(FATAL_ERROR "PyroWave shared library not built. Run:\n"
            "  (cd third-party/pyrowave && ./checkout_granite.sh && "
            "cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build)")
endif()

add_library(pyrowave_c SHARED IMPORTED)
set_target_properties(pyrowave_c PROPERTIES
        IMPORTED_LOCATION "${PYROWAVE_SHARED}"
        IMPORTED_SONAME "libpyrowave-shared.so.0"
        INTERFACE_INCLUDE_DIRECTORIES "${PYROWAVE_DIR}"
        # Bake the shared-library location into the runtime search path so the binary finds it.
        INTERFACE_LINK_OPTIONS "-Wl,-rpath,${PYROWAVE_DIR}/build")

set(PYROWAVE_LIBRARIES pyrowave_c)
