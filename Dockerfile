# ================== Etapa de construcción ================== #
FROM python:3.12-slim-bookworm AS builder

ARG FOAM_VERSION=2406

# Instalar dependencias de compilación
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        autoconf \
        autotools-dev \
        bison \
        cmake \
        build-essential \
        flex \
        g++ \
        gawk \
        gfortran \
        git \
        gnuplot \
        libarpack2-dev \
        libboost-all-dev \
        libcgal-dev \
        libeigen3-dev \
        libfl-dev \
        libgmp-dev \
        liblapack-dev \
        libmpc-dev \
        libmpfr-dev \
        libncurses-dev \
        libreadline-dev \
        libscotch-dev \
        libspooles-dev \
        libxml2-dev \
        libxt-dev \
        libyaml-cpp-dev \
        make \
        tar \
        wget \
    && rm -rf /var/lib/apt/lists/*

RUN git clone -b release https://gitlab.com/petsc/petsc.git --depth=1 /tmp/petsc-build && \
    cd /tmp/petsc-build && \
    git pull && \
    ./configure \
    --prefix=/usr/local \
    --with-shared-libraries=1 \
    --with-debugging=0 \
    --with-fortran-bindings=1 \
    --with-cxx-dialect=C++11 \
    --download-fblaslapack \
    --download-hdf5 \
    --download-openmpi \
    --download-metis \
    --download-hypre \
    --download-ptscotch \
    --download-scalapack \
    --download-mumps \
    --download-superlu_dist \
    --download-fftw \
    --download-cmake \
    --download-parmetis \
    --download-zlib && \
    make && \
    make install && \
    rm -rf /tmp/petsc-build

RUN git clone -b release https://gitlab.com/slepc/slepc.git --depth=1 /tmp/slepc-build && \
    cd /tmp/slepc-build && \
    PETSC_DIR=/usr/local SLEPC_DIR=/tmp/slepc-build \
    ./configure --prefix=/usr/local --with-scalapack && \
    make SLEPC_DIR=/tmp/slepc-build PETSC_DIR=/usr/local && \
    make SLEPC_DIR=/tmp/slepc-build PETSC_DIR=/usr/local install && \
    rm -rf /tmp/petsc-build /tmp/slepc-build

WORKDIR /tmp/precice
RUN pip3 install --no-cache-dir numpy && \
    wget https://github.com/precice/precice/archive/v3.2.0.tar.gz && \
    tar -xzf v3.2.0.tar.gz --strip-components=1 && \
    cmake -B build -S . \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON \
        -DPRECICE_CONFIGURE_PACKAGE_GENERATION=OFF && \
    cmake --build build --parallel $(nproc) && \
    cmake --install build && \
    rm -rf /tmp/precice

WORKDIR /tmp/ccx
RUN wget http://www.dhondt.de/ccx_2.20.src.tar.bz2 && \
    wget https://github.com/precice/calculix-adapter/archive/refs/heads/master.tar.gz && \
    tar xvjf ccx_2.20.src.tar.bz2 && \
    tar -xzf master.tar.gz && \
    cd calculix-adapter-master && \
    sed -i 's/\$(HOME)\/CalculiX\/ccx_\$(CCX_VERSION)\/src/\/tmp\/ccx\/CalculiX\/ccx_2.20\/src/g' Makefile && \
    sed -i 's#FFLAGS = -Wall -O3 -fopenmp \$(INCLUDES)#FFLAGS = -Wall -O3 -fopenmp -fallow-argument-mismatch \$(INCLUDES)#g'  Makefile && \
    make clean && \
    make && \
    cp bin/ccx_preCICE /usr/local/bin && \
    chmod +x /usr/local/bin/ccx_preCICE
    

ARG SOURCES_DIR=/sources
RUN mkdir -p ${SOURCES_DIR} && \
    cd ${SOURCES_DIR} && \
    wget -c https://develop.openfoam.com/Development/openfoam/-/archive/OpenFOAM-v${FOAM_VERSION}/openfoam-OpenFOAM-v${FOAM_VERSION}.tar.gz && \
    wget -c https://dl.openfoam.com/source/v${FOAM_VERSION}/ThirdParty-v${FOAM_VERSION}.tgz

ENV FOAM_INST_DIR=/opt
ENV WM_PROJECT=${FOAM_INST_DIR}/OpenFOAM-v${FOAM_VERSION}
ENV WM_PROJECT_DIR=${FOAM_INST_DIR}/OpenFOAM-v${FOAM_VERSION}

RUN cd ${SOURCES_DIR} && \
    mkdir -p ${WM_PROJECT_DIR} && \
    tar -xf openfoam-OpenFOAM-v${FOAM_VERSION}.tar.gz -C ${FOAM_INST_DIR} && \
    tar -xf ThirdParty-v${FOAM_VERSION}.tgz -C ${FOAM_INST_DIR} && \
    cp -rf ${FOAM_INST_DIR}/openfoam-OpenFOAM-v${FOAM_VERSION}/* ${WM_PROJECT_DIR}/ && \
    rm -rf ${FOAM_INST_DIR}/openfoam-OpenFOAM-v${FOAM_VERSION}

# Clonar complementos
RUN git clone --depth=1 https://develop.openfoam.com/Community/integration-cfmesh.git ${WM_PROJECT_DIR}/plugins/cfmesh && \
    git clone --depth=1 https://github.com/precice/openfoam-adapter.git ${WM_PROJECT_DIR}/plugins/precice-adapter && \
    git clone --depth=1 -b v2412 https://github.com/unicfdlab/libAcoustics.git ${WM_PROJECT_DIR}/plugins/libAcoustics

    
RUN cd ${FOAM_INST_DIR}/ThirdParty-v${FOAM_VERSION} && \
    wget -c https://bitbucket.org/petsc/pkg-metis/get/v5.1.0-p12.tar.gz && \
    mkdir -p sources/metis/metis-5.1.0 && \
    tar -xzf v5.1.0-p12.tar.gz -C sources/metis/metis-5.1.0 --strip-components=1

# Limpieza de componentes innecesarios
RUN rm -rf /opt/ThirdParty-v${FOAM_VERSION}/sources/{paraview,openmpi,cgal,boost}

RUN cd /opt/OpenFOAM-v${FOAM_VERSION}/applications && \
    rm -rf solvers/{acoustic,basic/laplacianFoam,basic/scalarTransportFoam,combustion,compressible,discreteMethods,DNS,electromagnetics,financial,finiteArea,heatTransfer,incompressible/{adjoint*,boundaryFoam,nonNewtonian*,shallow*},lagrangian,multiphase,stressAnalysis} test

RUN cd $WM_PROJECT_DIR && \
    bin/tools/foamConfigurePaths \
        -project-path $WM_PROJECT_DIR \
        gmp-system \
        mpfr-system \
        -metis metis-5.1.0 \
        -int64 -dp 

RUN echo "CPUs disponibles: $(nproc)"

RUN cd ${WM_PROJECT_DIR} && \
    . etc/bashrc && \
    ./Allwmake -s -j 3


RUN cd ${WM_PROJECT_DIR} && \
    . etc/bashrc && \
    export FOAM_EXTRA_CFLAGS="-I/usr/local/include" && \
    export FOAM_EXTRA_CXXFLAGS="-I/usr/local/include" && \
    export FOAM_EXTRA_LDFLAGS="\
    -L/usr/local/lib \
    -L/usr/local/lib64 \
    -L${WM_PROJECT_DIR}/platforms/linux64GccDPInt64Opt/lib/sys-openmpi \
    -Wl,-rpath,/usr/local/lib:/usr/local/lib64:${WM_PROJECT_DIR}/platforms/linux64GccDPInt64Opt/lib/sys-openmpi \
    -Wl,-rpath-link,${WM_PROJECT_DIR}/platforms/linux64GccDPInt64Opt/lib/sys-openmpi" && \
    export FOAM_USER_LIBBIN=${WM_PROJECT_DIR}/platforms/linux64GccDPInt64Opt/lib && \
    cd ${WM_PROJECT_DIR}/plugins/cfmesh && ./Allwmake && \
    cd ${WM_PROJECT_DIR}/plugins/libAcoustics && ./makeLibrary.sh &&\
    cd ${WM_PROJECT_DIR}/plugins/precice-adapter && ./Allwmake


# Preparar instalación final de OpenFOAM
RUN mkdir -p /opt/openfoam && \
    cd ${WM_PROJECT_DIR} && \
    cp -rf platforms/linux64GccDPInt64Opt/* /opt/openfoam && \
    cp -rf etc bin wmake /opt/openfoam && \
    cd /opt/ThirdParty-v${FOAM_VERSION} && \
    cp -rf platforms/linux64Gcc/*/* /opt/openfoam && \
    cp -rf platforms/linux64GccDPInt64/lib/* /opt/openfoam/lib/  && \
    cp -rf platforms/linux64GccDPInt64/metis-5.1.0/* /opt/openfoam/  && \
    cp -rf platforms/linux64GccDPInt64/scotch_6.1.0/* /opt/openfoam/ && \
    rm -rf ${WM_PROJECT_DIR} /opt/ThirdParty-v${FOAM_VERSION} ${SOURCES_DIR}


RUN find /opt/openfoam -type f -exec strip --strip-unneeded {} \; || true && \
    find /opt/openfoam -name '*.a' -delete && \
    rm -rf /opt/openfoam/{doc,man,test-tutorials}

# Instalar paquete Python
WORKDIR /workspace
COPY src /workspace/
COPY pyproject.toml /workspace/
RUN pip install --no-cache-dir -e . && rm -rf *
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir \
    ipython \
    mpi4py \
    petsc4py \
    polars \
    pyprecice \
    precice-cli \
    trimesh \
    slepc4py

RUN rm -rf /root/.cache/pip /tmp/*

# # ================== Etapa de ejecución ================== #
FROM python:3.12-slim-bookworm

ARG FOAM_VERSION=2406

# Dependencias de tiempo de ejecución
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        libgfortran5 \
        libarpack2 \
        libgomp1 \
        libblas3 \
        libboost-filesystem1.74.0 \
        libboost-log1.74.0 \
        libboost-program-options1.74.0 \
        libboost-system1.74.0 \
        libboost-thread1.74.0 \
        libdbus-1-3 \
        libegl1-mesa \
        libgl1 \
        libglib2.0-0 \
        libglu1-mesa \
        libmetis5 \
        libnl-3-200 \
        libscotch-7.0 \
        libspooles2.2 \
        libxcursor1 \
        libxft2 \
        libxinerama1 \
        libxkbcommon0 \
        libxml2 \
        libxrender1 \
        libxt6 \ 
        libyaml-cpp0.7 \
        && rm -rf /var/lib/apt/lists/*

# Copiar instalaciones desde el builder
COPY --from=builder /opt/openfoam /opt/openfoam
COPY --from=builder /usr/local /usr/local

# Configurar entorno y usuario
WORKDIR /workspace
RUN useradd --user-group --create-home --shell /bin/bash foam && \
    echo "foam ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers && \
    chown -R foam:foam /workspace

USER foam

ENV WM_PROJECT_DIR=/opt/openfoam
ENV WM_OPTIONS=linux64GccDPInt64Opt
ENV PATH=${WM_PROJECT_DIR}/bin:${WM_PROJECT_DIR}/bin/tools:${WM_PROJECT_DIR}/wmake:${PATH}
ENV LD_LIBRARY_PATH=/lib:/usr/local/lib:/lib64:${WM_PROJECT_DIR}/lib:${WM_PROJECT_DIR}/lib/sys-openmpi
ENV OMPI_MCA_btl_vader_single_copy_mechanism=none
ENV WM_PROJECT_USER_DIR=/workspace/foam-${FOAM_VERSION}

CMD ["bash"]
