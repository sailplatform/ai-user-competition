#!/usr/bin/env bash

set -euxo pipefail

# NOTE: If you get build errors related to python interpreter check under Linux then run the following command:
# export WITH_PYTHON_INTERP_CHECK=ON

# ====================================================
# import the utils 
. bash_utils.sh 

# ====================================================

print_blue '================================================'
print_blue "Building Thirdparty"
print_blue '================================================'

#echo ROOT_DIR: $ROOT_DIR
cd $ROOT_DIR  # from bash_utils.sh

STARTING_DIR=`pwd`  # this should be the main folder directory of the repo

# ====================================================
# check if we have external options
EXTERNAL_OPTIONS=$@
if [[ -n "$EXTERNAL_OPTIONS" ]]; then
    echo "external option: $EXTERNAL_OPTIONS" 
fi


OpenCV_DIR="$ROOT_DIR/thirdparty/opencv/install/lib/cmake/opencv4"
if [[ -d "$OpenCV_DIR" ]]; then
    EXTERNAL_OPTIONS="$EXTERNAL_OPTIONS -DOpenCV_DIR=$OpenCV_DIR"
fi 

echo "EXTERNAL_OPTIONS: $EXTERNAL_OPTIONS"

# ====================================================

CURRENT_USED_PYENV=$(get_virtualenv_name)
print_blue "currently used pyenv: $CURRENT_USED_PYENV"

print_blue "=================================================================="
print_blue "Configuring and building thirdparty/orbslam2_features ..."
cd thirdparty/orbslam2_features
. build.sh $EXTERNAL_OPTIONS
cd $STARTING_DIR

print_blue '================================================'
print_blue "Configuring and building thirdparty/Pangolin ..."

make_dir thirdparty

INSTALL_PANGOLIN_ORIGINAL=0
cd thirdparty
if [ $INSTALL_PANGOLIN_ORIGINAL -eq 1 ] ; then
    # N.B.: pay attention this will generate a module 'pypangolin' ( it does not have the methods dcam.SetBounds(...) and pangolin.DrawPoints(points, colors)  )
    if [ ! -d pangolin ]; then
        if [[ "$OSTYPE" == "linux-gnu"* ]]; then
            sudo apt-get install -y libglew-dev
        fi     
        git clone https://github.com/stevenlovegrove/Pangolin.git pangolin
        cd pangolin
        git submodule init && git submodule update
        cd ..
    fi
    cd pangolin
    make_dir build 
    if [ ! -f build/src/libpangolin.so ]; then
        cd build
        cmake ../ -DAVFORMAT_INCLUDE_DIR="" -DCPP11_NO_BOOST=ON $EXTERNAL_OPTIONS
        make -j8
        cd build/src
        ln -s pypangolin.*-linux-gnu.so  pangolin.linux-gnu.so
    fi
else
    # N.B.: pay attention this will generate a module 'pangolin' 
    if [ ! -d pangolin ]; then
        if [[ "$OSTYPE" == "linux-gnu"* ]]; then    
            sudo apt-get install -y libglew-dev
            # git clone https://github.com/uoip/pangolin.git
            # cd pangolin
            # PANGOLIN_UOIP_REVISION=3ac794a
            # git checkout $PANGOLIN_UOIP_REVISION
            # cd ..      
            # # copy local changes 
            # rsync ./pangolin_changes/python_CMakeLists.txt ./pangolin/python/CMakeLists.txt 
            git clone --recursive https://gitlab.com/luigifreda/pypangolin.git pangolin
        fi 
        if [[ "$OSTYPE" == "darwin"* ]]; then
            git clone --recursive https://gitlab.com/luigifreda/pypangolin.git pangolin 
        fi 
        cd pangolin
        git apply ../pangolin.patch
        cd ..
    fi
    cd pangolin
    if [ ! -f pangolin.cpython-*.so ]; then   
        make_dir build   
        cd build
        cmake .. -DBUILD_PANGOLIN_LIBREALSENSE=OFF -DBUILD_PANGOLIN_LIBREALSENSE2=OFF \
                 -DBUILD_PANGOLIN_OPENNI=OFF -DBUILD_PANGOLIN_OPENNI2=OFF \
                 -DBUILD_PANGOLIN_FFMPEG=OFF -DBUILD_PANGOLIN_LIBOPENEXR=OFF $EXTERNAL_OPTIONS # disable realsense 
        make -j8
        cd ..
        #python setup.py install
    fi
fi
cd $STARTING_DIR


print_blue "=================================================================="
print_blue "Configuring and building thirdparty/g2o ..."

cd thirdparty
if [ ! -d g2opy ]; then
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        sudo apt-get install -y libsuitesparse-dev libeigen3-dev
    fi     
	git clone https://github.com/uoip/g2opy.git
    cd g2opy
    G2OPY_REVISION=5587024
    git checkout $G2OPY_REVISION
    git apply ../g2opy.patch
    cd ..     
fi
cd g2opy
if [ ! -f lib/g2o.cpython-*.so ]; then  
    make_buid_dir
    cd build
    cmake .. $EXTERNAL_OPTIONS
    make -j8
    cd ..
    #python3 setup.py install --user
fi    
cd $STARTING_DIR

print_blue "=================================================================="
print_blue "Configuring and building thirdparty/pydbow3 ..."

cd thirdparty/pydbow3
./build.sh $EXTERNAL_OPTIONS

cd $STARTING_DIR

print_blue "=================================================================="
print_blue "Configuring and building thirdparty/pydbow2 ..."

cd thirdparty/pydbow2
./build.sh $EXTERNAL_OPTIONS

cd $STARTING_DIR


print_blue "=================================================================="
print_blue "Configuring and building thirdparty/pyibow ..."

cd thirdparty/pyibow
./build.sh $EXTERNAL_OPTIONS

cd $STARTING_DIR


if [[ "$OSTYPE" == "darwin"* ]]; then
    print_blue "=================================================================="
    print_blue "Configuring and building thirdparty/open3d ..."

    # NOTE: Under mac I got segmentation faults when trying to use open3d python bindings
    #       This happends when trying to load the open3d dynamic library.
    ./install_open3d_python.sh

    cd $STARTING_DIR
fi 


print_blue "=================================================================="
print_blue "Configuring and building thirdparty/ml_depth_pro ..."

cd thirdparty
if [ ! -d ml_depth_pro ]; then
    git clone https://github.com/apple/ml-depth-pro.git ml_depth_pro
    cd ml_depth_pro
    #git checkout b2cd0d51daa95e49277a9f642f7fd736b7f9e91d # use this commit if you hit any problems

    git apply ../ml_depth_pro.patch

    source get_pretrained_models.sh   # Files will be downloaded to `ml_depth_pro/checkpoints` directory. 
fi

cd $STARTING_DIR



print_blue "=================================================================="
print_blue "Configuring and building thirdparty/depth_anything_v2 ..."

cd thirdparty
if [ ! -d depth_anything_v2 ]; then
    git clone https://github.com/DepthAnything/Depth-Anything-V2.git depth_anything_v2
    cd depth_anything_v2
    #git checkout 31dc97708961675ce6b3a8d8ffa729170a4aa273 # use this commit if you hit any problems

    git apply ../depth_anything_v2.patch

    ./download_metric_models.py
fi

cd $STARTING_DIR

print_blue "=================================================================="
print_blue "Configuring and building thirdparty/raft_stereo ..."

cd thirdparty
if [ ! -d raft_stereo ]; then
    git clone https://github.com/princeton-vl/RAFT-Stereo.git raft_stereo
    cd raft_stereo
    #git checkout 6068c1a26f84f8132de10f60b2bc0ce61568e085 # use this commit if you hit any problems

    git apply ../raft_stereo.patch
    
    ./download_models.sh
fi

cd $STARTING_DIR


print_blue "=================================================================="
print_blue "Configuring and building thirdparty/crestereo ..."

cd thirdparty
if [ ! -d crestereo ]; then
    git clone https://github.com/megvii-research/CREStereo.git crestereo
    cd crestereo
    #git checkout ad3a1613bdedd88b93247e5f002cb7c80799762d # use this commit if you hit any problems

    git apply ../crestereo.patch
    
    ./download_models.py
fi

cd $STARTING_DIR

print_blue "=================================================================="
print_blue "Configuring and building thirdparty/crestereo_pytorch ..."

cd thirdparty
if [ ! -d crestereo_pytorch ]; then
    git clone https://github.com/ibaiGorordo/CREStereo-Pytorch.git crestereo_pytorch
    cd crestereo_pytorch
    #git checkout b6c7a9fe8dc2e9e56ba7b96f4677312309282d15 # use this commit if you hit any problems

    git apply ../crestereo_pytorch.patch
    
    ./download_models.py
fi

cd $STARTING_DIR
echo "...done with thirdparty"


# NOTE: If you get build errors related to python interpreter check under Linux then run the following command:
# export WITH_PYTHON_INTERP_CHECK=ON
