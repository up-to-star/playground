#!/bin/bash
echo "[@~@] Build Start >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
export CC="clang"
export CXX="clang++"
export CUDA_CC="clang"
export CUDA_DIR="/usr/local/cuda"

PROJ_HOME=$(pwd)
TaskNo="1"

BuildType="Release"
CleanFirst="false"
CleanAll="false"
TestKernelVersion="0"
TestDataType="float32"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        Release|Debug)
            BuildType=$1
            ;;
        -cf|--clean-first)
            CleanFirst="true"
            ;;
        -ca|--clean-all)
            CleanAll="true"
            ;;
        -v*)
            TestKernelVersion="${1#*v}"
            ;;
        --test-kernel-version=*)
            TestKernelVersion="${1#*=}"
            ;;
        -f32|--float32)
            TestDataType="float32"
            ;;
        -f16|--float16)
            TestDataType="float16"
            ;;
        --help)
            cat ./docs/scripts/build-task1-help.txt
            exit 1
            ;;
        *)
            echo "build fatal: Invalid argument '$1'. Use --help for more information."
            exit 1
            ;;
    esac
    shift
done


if [ "$CleanAll" = "true" ] && [ -d "$PROJ_HOME/build" ]; then
    echo "Cleaning all build files..."
    rm -rf $PROJ_HOME/build
fi

cmake -S . -B ./build -G "Ninja" \
    -DTASK_NO=$TaskNo \
    -DCMAKE_BUILD_TYPE=$BuildType \
    -DTEST_KERNEL_VERSION=$TestKernelVersion \
    -DTEST_DATA_TYPE=$TestDataType

if [ "$CleanFirst" = "true" ]; then
    cmake --build ./build --parallel $(nproc) --clean-first
else
    cmake --build ./build --parallel $(nproc)
fi

echo "[@v@] Build Finished <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
