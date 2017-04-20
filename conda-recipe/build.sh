mkdir build
cd build

LINKER_FLAGS="-L${PREFIX}/lib"
export DYLIB="dylib"
if [ `uname` != "Darwin" ]; then
    LINKER_FLAGS="-Wl,-rpath-link,${PREFIX}/lib ${LINKER_FLAGS}"
    export DYLIB="so"
fi

cmake .. \
    -DCMAKE_C_COMPILER=${PREFIX}/bin/gcc \
    -DCMAKE_CXX_COMPILER=${PREFIX}/bin/g++ \
    -DCMAKE_OSX_DEPLOYMENT_TARGET=10.7 \
    -DCMAKE_INSTALL_PREFIX=${PREFIX} \
    -DCMAKE_BUILD_TYPE=Release \
    -DPYTHON_EXECUTABLE=${PYTHON} \
    -DVIGRA_INCLUDE_DIR=${PREFIX}/include \
    -DPYTHON_LIBRARY=${PREFIX}/lib/libpython3.5m.${DYLIB} \
    -DPYTHON_INCLUDE_DIR=${PREFIX}/include/python3.5 \
    -DPYTHON_INCLUDE_DIR2=${PREFIX}/include/python3.5 \
    -DWITH_LOG=OFF

make -j${CPU_COUNT} VERBOSE=1
make install