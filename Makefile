#-------------------------------------------------------------------------------
# Configuration Section
#-------------------------------------------------------------------------------

# Version configurations
BOOST_VERSION      := 1.81.0
CGNS_VERSION       := 4.5.0
CGNSUTILS_VERSION  := 2.8.1
EIGEN_VERSION      := 3.4.0
FOAM_VERSION       := 2406
LIBFFI_VERSION     := 3.4.2
LIBXML2_VERSION    := 2.13.6
NCURSES_VERSION    := 6.3
OMPI_VERSION       := 4.1.8
OMPI_SHORT_VERSION := $OMPI_VERSION := 4.1.8
OMPI_SHORT_VERSION := $(word 1,$(subst ., ,$(OMPI_VERSION))).$(word 2,$(subst ., ,$(OMPI_VERSION)))
OPENSSL_VERSION    := 3.4.1
PETSC_VERSION      := 3.24.2
SLEPC_VERSION      := 3.24.1
PRECICE_VERSION    := 3.2.0
PYTHON_VERSION     := 3.12.9
READLINE_VERSION   := 8.2

# Dependency file names
BOOST_TAR          := boost_$(subst .,_,$(BOOST_VERSION)).tar.bz2
BZ2_TAR            := bzip2-master.tar.gz
CGNS_TAR           := v$(CGNS_VERSION).tar.gz
CGNSUTILS_TAR      := v$(CGNSUTILS_VERSION).tar.gz
EIGEN_TAR          := eigen-$(EIGEN_VERSION).tar.gz
FOAM_TAR           := openfoam-OpenFOAM-v$(FOAM_VERSION).tar.gz
LIBFFI_TAR         := libffi-$(LIBFFI_VERSION).tar.gz
LIBXML2_TAR        := libxml2_v$(LIBXML2_VERSION).tar.gz
NCURSES_TAR        := ncurses-$(NCURSES_VERSION).tar.gz
OMPI_TAR           := openmpi-$(OMPI_VERSION).tar.gz
OPENSSL_TAR        := openssl-$(OPENSSL_VERSION).tar.gz
PETSC_TAR          := petsc-$(PETSC_VERSION).tar.gz
PRECICE_TAR        := v$(PRECICE_VERSION).tar.gz
PYTHON_TAR         := Python-$(PYTHON_VERSION).tgz
READLINE_TAR       := readline-$(READLINE_VERSION).tar.gz
SLEPC_TAR          := slepc-$(SLEPC_VERSION).tar.gz

# Directory configurations
export PATH            := ${VENV_DIR}/bin:${VENV_DIR}/sbin:/bin:/usr/bin
include .env
export $(bash sed -n 's/^export //p' .env)

VENV_DIR    := $(PWD)/.venv
SOURCES_DIR := $(PWD)/.sources
BUILD_DIR   := $(SOURCES_DIR)/build

# Build Commands
NPROC         := 3
DOWNLOAD      := wget -nc
MAKE_CMD      := LD_LIBRARY_PATH=$(VENV_DIR)/lib:$(VENV_DIR)/lib64 \
				 make -j$(NPROC) && \
				 LD_LIBRARY_PATH=$(VENV_DIR)/lib:$(VENV_DIR)/lib64 \
				 make install PREFIX=$(VENV_DIR)
CONFIGURE_CMD := LD_LIBRARY_PATH=$(VENV_DIR)/lib:$(VENV_DIR)/lib64 \
				 LDFLAGS="-L$(VENV_DIR)/lib -L$(VENV_DIR)/lib64 -Wl,-rpath,$(VENV_DIR)/lib:$(VENV_DIR)/lib64" \
				 CPPFLAGS="-I$(VENV_DIR)/include" \
				 ./configure --prefix=$(VENV_DIR)
PIP_INSTALL   := PETSC_DIR=$(VENV_DIR) LD_LIBRARY_PATH=$(VENV_DIR)/lib:$(VENV_DIR)/lib64 \
				 $(VENV_DIR)/bin/pip3 install --no-cache-dir --force-reinstall
CMAKE_CMD     := LD_LIBRARY_PATH=$(VENV_DIR)/lib:$(VENV_DIR)/lib64 \
				cmake \
				 -DCMAKE_BUILD_TYPE=Release \
				 -DCMAKE_INSTALL_PREFIX=$(VENV_DIR) \
				 -DCMAKE_EXE_LINKER_FLAGS="-L$(VENV_DIR)/lib -L$(VENV_DIR)/lib64 -Wl,-rpath,$(VENV_DIR)/lib:$(VENV_DIR)/lib64" \
				 -DCMAKE_LIBRARY_PATH="$(VENV_DIR)/lib:$(VENV_DIR)/lib64"

# Compiler flags
export PATH            := ${VENV_DIR}/bin:${VENV_DIR}/sbin:/bin:/usr/bin
export LIBDIR          := $(VENV_DIR)/lib:$(VENV_DIR)/lib64
export PKG_CONFIG_PATH := $(VENV_DIR)/lib/pkgconfig:$(VENV_DIR)/lib64/pkgconfig
export CPPFLAGS        := -I$(VENV_DIR)/include
export LDFLAGS         := -L$(VENV_DIR)/lib -L$(VENV_DIR)/lib64 \
                          -Wl,-rpath,$(VENV_DIR)/lib:$(VENV_DIR)/lib64
export CFLAGS          := -O2
export CXXFLAGS        := -O2

export MPI_HOME        := $(VENV_DIR)
export MPI_ROOT        := $(VENV_DIR)
export FOAM_INST_DIR   := $(VENV_DIR)/opt
export WM_PROJECT_DIR  := $(FOAM_INST_DIR)/OpenFOAM-v$(FOAM_VERSION)
export LD_LIBRARY_PATH := $(VENV_DIR)/lib:$(VENV_DIR)/lib64:$(LD_LIBRARY_PATH)

#-------------------------------------------------------------------------------
# Main Targets
#-------------------------------------------------------------------------------
.PHONY: all clean pip docker docker-build docker-build-low-memory

all: download_sources python petsc slepc precice openfoam python_env pyhyp
	pip freeze > requirements.lock
	@echo "====================================="
	@echo "# All components built successfully #"
	@echo "====================================="

clean:
	rm -rf $(VENV_DIR) $(SOURCES_DIR)/build $(SOURCES_DIR)/download.done

# Docker targets
docker-build-low-memory:
	@echo "Building Docker image with low memory optimizations..."
	@./build-docker-low-memory.sh

docker-build: docker-build-low-memory

docker:
	@echo "Starting Docker container..."
	@docker run --rm -it fem-shell:latest bash

#-------------------------------------------------------------------------------
# Minimal Targets
#-------------------------------------------------------------------------------
download_sources: $(SOURCES_DIR)/download.done

openfoam: $(VENV_DIR)/.openfoam.done

petsc: $(VENV_DIR)/.petsc.done

precice: $(VENV_DIR)/.precice.done

python_env: $(VENV_DIR)/.python_env.done

python: $(VENV_DIR)/.python.done

slepc: $(VENV_DIR)/.slepc.done

pyhyp: $(VENV_DIR)/.pyhyp.done

#-------------------------------------------------------------------------------
# Download sources
#-------------------------------------------------------------------------------
$(SOURCES_DIR)/download.done:
	@mkdir -p $(SOURCES_DIR)
	@chmod -R u+w $(SOURCES_DIR)
	@echo "Downloading source packages..."
	@cd $(SOURCES_DIR) && $(DOWNLOAD) \
		https://archives.boost.io/release/$(BOOST_VERSION)/source/$(BOOST_TAR) \
		https://develop.openfoam.com/Development/openfoam/-/archive/OpenFOAM-v$(FOAM_VERSION)/$(FOAM_TAR) \
		https://dl.openfoam.com/source/v$(FOAM_VERSION)/ThirdParty-v$(FOAM_VERSION).tgz \
		https://download.open-mpi.org/release/open-mpi/v$(OMPI_SHORT_VERSION)/$(OMPI_TAR) \
		https://ftp.gnu.org/gnu/ncurses/$(NCURSES_TAR) \
		https://ftp.gnu.org/gnu/readline/$(READLINE_TAR) \
		https://github.com/libffi/libffi/releases/download/v$(LIBFFI_VERSION)/$(LIBFFI_TAR) \
		https://github.com/CGNS/CGNS/archive/refs/tags/$(CGNS_TAR) \
		https://github.com/mdolab/cgnsutilities/archive/refs/tags/$(CGNSUTILS_TAR) \
		https://github.com/precice/precice/archive/$(PRECICE_TAR) \
		https://gitlab.com/bzip2/bzip2/-/archive/master/$(BZ2_TAR) \
		https://gitlab.com/libeigen/eigen/-/archive/$(EIGEN_VERSION)/$(EIGEN_TAR) \
		https://gitlab.gnome.org/GNOME/libxml2/-/archive/v$(LIBXML2_VERSION)/$(LIBXML2_TAR) \
		https://slepc.upv.es/download/distrib/$(SLEPC_TAR) \
		https://sqlite.org/2025/sqlite-autoconf-3490100.tar.gz \
		https://web.cels.anl.gov/projects/petsc/download/release-snapshots/$(PETSC_TAR) \
		https://www.openssl.org/source/$(OPENSSL_TAR) \
		https://www.python.org/ftp/python/$(PYTHON_VERSION)/$(PYTHON_TAR)
	@touch $@

#-------------------------------------------------------------------------------
# Pattern rule for building libraries
#-------------------------------------------------------------------------------

define build-library
$(VENV_DIR)/.$(1).done: $(SOURCES_DIR)/download.done
	@echo "Building $(1)..."
	@mkdir -p $(BUILD_DIR)/$(1)
	@tar -xzf $(SOURCES_DIR)/$(2) -C $(BUILD_DIR)/$(1) --strip-components=1
	@cd $(BUILD_DIR)/$(1) && \
		$(CONFIGURE_CMD) $(3) && \
		$(MAKE_CMD)
	@touch $$@
endef

#-------------------------------------------------------------------------------
# Python Build
#-------------------------------------------------------------------------------
$(eval $(call build-library,libffi,$(LIBFFI_TAR),--disable-dependency-tracking))
# ncurses: disable C++ bindings to avoid libstdc++ conflicts on modern compilers
# Also skip ADA to speed up and reduce toolchain requirements
$(eval $(call build-library,ncurses,$(NCURSES_TAR),--with-shared --with-termlib --without-debug --without-cxx --without-cxx-binding --without-ada))

$(VENV_DIR)/.openssl.done: $(SOURCES_DIR)/download.done
	@echo "Building OpenSSL..."
	@mkdir -p $(BUILD_DIR)/openssl
	@tar -xzf $(SOURCES_DIR)/$(OPENSSL_TAR) -C $(BUILD_DIR)/openssl --strip-components=1
	@cd $(BUILD_DIR)/openssl && \
		./config --prefix=$(VENV_DIR) --openssldir=/etc/ssl && \
		make -j$(NPROC) depend && \
		make install_sw
	@touch $@

$(VENV_DIR)/.bz2.done:
	@echo "Building Bzip2..."
	@mkdir -p $(BUILD_DIR)/bz2/build
	@tar -xzf $(SOURCES_DIR)/$(BZ2_TAR) -C $(BUILD_DIR)/bz2 --strip-components=1
	@cd $(BUILD_DIR)/bz2/build && \
		$(CMAKE_CMD) -DENABLE_SHARED_LIB=ON -DENABLE_STATIC_LIB=OFF .. && \
		$(MAKE_CMD)
	@touch $@		

$(VENV_DIR)/.readline.done: $(SOURCES_DIR)/download.done $(VENV_DIR)/.ncurses.done
	@echo "Building Readline..."
	@mkdir -p $(BUILD_DIR)/readline
	@tar -xzf $(SOURCES_DIR)/$(READLINE_TAR) -C $(BUILD_DIR)/readline --strip-components=1
	@cd $(BUILD_DIR)/readline && \
		CC=gcc \
		LD_LIBRARY_PATH=$(VENV_DIR)/lib:$(VENV_DIR)/lib64 \
		LDFLAGS="-L$(VENV_DIR)/lib -L$(VENV_DIR)/lib64 -Wl,-rpath,$(VENV_DIR)/lib:$(VENV_DIR)/lib64" \
		CPPFLAGS="-I$(VENV_DIR)/include -I$(VENV_DIR)/include/ncurses" \
		TERMCAP_LIB=-ltinfo \
		./configure --prefix=$(VENV_DIR) --with-curses --enable-shared && \
		LD_LIBRARY_PATH=$(VENV_DIR)/lib:$(VENV_DIR)/lib64 CPPFLAGS="-I$(VENV_DIR)/include -I$(VENV_DIR)/include/ncurses" make -j$(NPROC) && \
		LD_LIBRARY_PATH=$(VENV_DIR)/lib:$(VENV_DIR)/lib64 make -C shlib clean && \
		LD_LIBRARY_PATH=$(VENV_DIR)/lib:$(VENV_DIR)/lib64 make -C shlib -j$(NPROC) SHLIB_LIBS="-ltinfo -lncurses" && \
		LD_LIBRARY_PATH=$(VENV_DIR)/lib:$(VENV_DIR)/lib64 make -C shlib DESTDIR= install && \
		LD_LIBRARY_PATH=$(VENV_DIR)/lib:$(VENV_DIR)/lib64 make install PREFIX=$(VENV_DIR)
	@touch $@
	
$(VENV_DIR)/.sqlite.done: $(SOURCES_DIR)/download.done $(VENV_DIR)/.readline.done
	@echo "Building SQlite..."
	@mkdir -p $(BUILD_DIR)/sqlite
	@tar -xzf $(SOURCES_DIR)/sqlite-autoconf-3490100.tar.gz -C $(BUILD_DIR)/sqlite --strip-components=1
	@cd $(BUILD_DIR)/sqlite && \
		LD_LIBRARY_PATH=$(VENV_DIR)/lib:$(VENV_DIR)/lib64 \
		CPPFLAGS="-I$(VENV_DIR)/include -I$(VENV_DIR)/include/ncurses" \
		LDFLAGS="-L$(VENV_DIR)/lib -L$(VENV_DIR)/lib64 -Wl,-rpath,$(VENV_DIR)/lib:$(VENV_DIR)/lib64" \
		LIBS="-lreadline -lncurses -ltinfo -lpthread -ldl -lz -lm" \
		./configure --prefix=$(VENV_DIR) --enable-shared --enable-readline && \
		LD_LIBRARY_PATH=$(VENV_DIR)/lib:$(VENV_DIR)/lib64 make -j$(NPROC) && \
		LD_LIBRARY_PATH=$(VENV_DIR)/lib:$(VENV_DIR)/lib64 make install PREFIX=$(VENV_DIR)
	@touch $@

$(VENV_DIR)/.python.done: $(SOURCES_DIR)/download.done \
	$(VENV_DIR)/.readline.done \
	$(VENV_DIR)/.bz2.done \
	$(VENV_DIR)/.libffi.done \
	$(VENV_DIR)/.sqlite.done
	@echo "Building Python $(PYTHON_VERSION)..."
	@mkdir -p $(BUILD_DIR)/python
	@tar -xzf $(SOURCES_DIR)/$(PYTHON_TAR) -C $(BUILD_DIR)/python --strip-components=1
	@cd $(BUILD_DIR)/python && \
		LD_LIBRARY_PATH=$(VENV_DIR)/lib:$(VENV_DIR)/lib64 \
		LDFLAGS="-L$(VENV_DIR)/lib -L$(VENV_DIR)/lib64 -Wl,-rpath,$(VENV_DIR)/lib:$(VENV_DIR)/lib64" \
		CPPFLAGS="-I$(VENV_DIR)/include" \
		./configure --prefix=$(VENV_DIR) \
			--enable-optimizations \
			--enable-shared \
			--disable-test-modules \
			--with-readline=readline \
			--with-ensurepip=install
	@cd $(BUILD_DIR)/python && $(MAKE_CMD)
	@ln -sf $(VENV_DIR)/bin/python3 $(VENV_DIR)/bin/python
	@touch $@

#-------------------------------------------------------------------------------
# Virtual Environment Setup
#-------------------------------------------------------------------------------
$(VENV_DIR)/.python_env.done: $(VENV_DIR)/.python.done

	@ln -sf $(VENV_DIR)/bin/pip3 $(VENV_DIR)/bin/pip
	# $(PIP_INSTALL) --upgrade pip setuptools wheel ipython ipykernel jupyterlab pytest pytest-cov
	$(PIP_INSTALL) mpi4py "petsc4py==$(PETSC_VERSION)" "slepc4py==$(SLEPC_VERSION)" pyprecice==$(PRECICE_VERSION)
	$(PIP_INSTALL) -e .

	@touch $@

#-------------------------------------------------------------------------------
# PETSc Installation
#-------------------------------------------------------------------------------
$(eval $(call build-library,openmpi,$(OMPI_TAR),))
$(VENV_DIR)/.boost.done: $(SOURCES_DIR)/download.done
	@echo "Installing BOOST..."
	@mkdir -p $(BUILD_DIR)/boost
	@tar -xjf $(SOURCES_DIR)/$(BOOST_TAR) -C $(BUILD_DIR)/boost --strip-components=1
	@cd $(BUILD_DIR)/boost && \
		./bootstrap.sh --with-libraries=all --prefix=$(VENV_DIR) && \
		./b2 install --prefix=$(VENV_DIR)
	@touch $@

# --with-metis-dir=$(FOAM_INST_DIR)/openfoam \
# --with-scotch-dir=$(FOAM_INST_DIR)/openfoam \

$(VENV_DIR)/.petsc.done: $(VENV_DIR)/.openmpi.done $(VENV_DIR)/.boost.done
	@echo "Installing PETSc..."
	@mkdir -p $(BUILD_DIR)/petsc
	@tar -xzf $(SOURCES_DIR)/$(PETSC_TAR) -C $(BUILD_DIR)/petsc --strip-components=1
	@cd $(BUILD_DIR)/petsc && \
		$(CONFIGURE_CMD) \
			LDFLAGS=$$LDFLAGS \
			--with-shared-libraries=1 \
			--with-mpi-dir=$(VENV_DIR) \
			--with-debugging=0 \
			--with-boost-dir=$(VENV_DIR) \
			--with-fortran-bindings=1 \
			--with-cxx-dialect=C++11 \
			--download-fblaslapack \
			--download-metis \
			--download-fftw \
			--download-ptscotch \
			--download-zlib && \
		make LD_LIBRARY_PATH=$(VENV_DIR)/lib:$(VENV_DIR)/lib64 -j$(NPROC) PETSC_DIR=$(BUILD_DIR)/petsc all && \
		make LD_LIBRARY_PATH=$(VENV_DIR)/lib:$(VENV_DIR)/lib64 PETSC_DIR=$(BUILD_DIR)/petsc install
	@touch $@

$(VENV_DIR)/.slepc.done: $(VENV_DIR)/.petsc.done
	@echo "Installing SLEPc..."
	@mkdir -p $(BUILD_DIR)/slepc
	@tar -xzf $(SOURCES_DIR)/$(SLEPC_TAR) -C $(BUILD_DIR)/slepc --strip-components=1
	@cd $(BUILD_DIR)/slepc && \
		PETSC_DIR=$(VENV_DIR) SLEPC_DIR=$(BUILD_DIR)/slepc $(CONFIGURE_CMD) && \
		make  LD_LIBRARY_PATH=$(VENV_DIR)/lib:$(VENV_DIR)/lib64 -j$(NPROC) PETSC_DIR=$(VENV_DIR) SLEPC_DIR=$(BUILD_DIR)/slepc all && \
		make  LD_LIBRARY_PATH=$(VENV_DIR)/lib:$(VENV_DIR)/lib64 -j$(NPROC) PETSC_DIR=$(VENV_DIR) SLEPC_DIR=$(BUILD_DIR)/slepc install
	@touch $@

#-------------------------------------------------------------------------------
# preCICE Dependencies and Installation
#-------------------------------------------------------------------------------
$(VENV_DIR)/.eigen.done:
	@echo "Installing Eigen..."
	@mkdir -p $(BUILD_DIR)/eigen/build
	@tar -xzf $(SOURCES_DIR)/$(EIGEN_TAR) -C $(BUILD_DIR)/eigen --strip-components=1
	@cd $(BUILD_DIR)/eigen/build && \
		$(CMAKE_CMD) .. && \
		$(MAKE_CMD)
	@touch $@


$(VENV_DIR)/.libxml2.done:
	@echo "Installing libxml2..."
	@mkdir -p $(BUILD_DIR)/libxml2
	@tar -xf $(SOURCES_DIR)/$(LIBXML2_TAR) -C $(BUILD_DIR)/libxml2 --strip-components=1
	@cd $(BUILD_DIR)/libxml2 && \
		LD_LIBRARY_PATH=$(VENV_DIR)/lib:$(VENV_DIR)/lib64 ./autogen.sh && \
		$(CONFIGURE_CMD) --without-python && \
		$(MAKE_CMD)
	@touch $@

$(VENV_DIR)/.precice.done: $(VENV_DIR)/.python.done $(VENV_DIR)/.petsc.done $(VENV_DIR)/.boost.done $(VENV_DIR)/.eigen.done $(VENV_DIR)/.libxml2.done
	@echo "Installing preCICE..."
	@mkdir -p $(BUILD_DIR)/precice
	@tar -xf $(SOURCES_DIR)/$(PRECICE_TAR) -C $(BUILD_DIR)/precice --strip-components=1
	$(PIP_INSTALL) polars "numpy>=2"
	@cd $(BUILD_DIR)/precice && \
		$(CMAKE_CMD) --preset=production \
			-DEIGEN3_INCLUDE_DIR=$(VENV_DIR)/include/eigen3 \
			-DPython3_EXECUTABLE=$(VENV_DIR)/bin/python3 \
			-DMPI_C_COMPILER=$(VENV_DIR)/bin/mpicc \
			-DMPI_CXX_COMPILER=$(VENV_DIR)/bin/mpicxx \
			-DBUILD_TESTING=OFF \
			-DPRECICE_CONFIGURE_PACKAGE_GENERATION=OFF \
			-DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON && \
			cd build && \
		$(MAKE_CMD)
	@touch $@


#-------------------------------------------------------------------------------
# OpenFOAM Installation
#-------------------------------------------------------------------------------


$(VENV_DIR)/.openfoam.done:
	@echo "Installing OpenFOAM..."
	@mkdir -p $(WM_PROJECT_DIR)
	@tar -xf $(SOURCES_DIR)/$(FOAM_TAR) -C $(FOAM_INST_DIR)
	@tar -xf $(SOURCES_DIR)/ThirdParty-v$(FOAM_VERSION).tgz -C $(FOAM_INST_DIR)

	cp -rf $(FOAM_INST_DIR)/openfoam-OpenFOAM-v$(FOAM_VERSION)/* $(WM_PROJECT_DIR)/

	rm -rf $(FOAM_INST_DIR)/openfoam-OpenFOAM-v$(FOAM_VERSION) 
	git clone --depth=1 https://develop.openfoam.com/Community/integration-cfmesh.git $(WM_PROJECT_DIR)/plugins/cfmesh || true
	git clone --depth=1 https://github.com/precice/openfoam-adapter.git $(WM_PROJECT_DIR)/plugins/precice-adapter || true
	git clone --depth=1 -b v2412 https://github.com/unicfdlab/libAcoustics.git $(WM_PROJECT_DIR)/plugins/libAcoustics || true

	source $(WM_PROJECT_DIR)/etc/bashrc && \
		cd $(FOAM_INST_DIR)/ThirdParty-v$(FOAM_VERSION) && \
			wget -c https://bitbucket.org/petsc/pkg-metis/get/v5.1.0-p12.tar.gz && \
			mkdir -p sources/metis/metis-5.1.0 && \
			tar -xzf v5.1.0-p12.tar.gz -C sources/metis/metis-5.1.0 --strip-components=1 || true

	cd $(FOAM_INST_DIR)/ThirdParty-v$(FOAM_VERSION) && \
		rm -rf sources/paraview \
				sources/openmpi \
				sources/cgal \
				sources/boost 

	@cd $(WM_PROJECT_DIR)/applications && \
		rm -rf \
			solvers/acoustic \
			solvers/basic/laplacianFoam \
			solvers/basic/scalarTransportFoam \
			solvers/combustion \
			solvers/compressible \
			solvers/compressible \
			solvers/discreteMethods \
			solvers/DNS \
			solvers/electromagnetics \
			solvers/financial \
			solvers/finiteArea \
			solvers/heatTransfer \
			solvers/incompressible/adjointOptimisationFoam \
			solvers/incompressible/adjointShapeOptimizationFoam \
			solvers/incompressible/boundaryFoam \
			solvers/incompressible/nonNewtonianIcoFoam \
			solvers/incompressible/shallowWaterFoam \
			solvers/lagrangian \
			solvers/multiphase \
			solvers/stressAnalysis \
			test

	@cd $(WM_PROJECT_DIR) && \
		bin/tools/foamConfigurePaths \
			gmp-system \
			mpfr-system \
			-metis metis-5.1.0 \
			-int64 -dp \
			;

	@cd $(WM_PROJECT_DIR) && \
		source ./etc/bashrc && \
		./Allwmake -s -q -l -j && \
		FOAM_EXTRA_CFLAGS="-I$(VENV_DIR)/include" FOAM_EXTRA_CXXFLAGS="-I$(VENV_DIR)/include" \
		FOAM_EXTRA_LDFLAGS="-L$(VENV_DIR)/lib -L$(VENV_DIR)/lib64 -Wl,-rpath,$(VENV_DIR)/lib:$(VENV_DIR)/lib64" \
		FOAM_USER_LIBBIN=$(WM_PROJECT_DIR)/platforms/linux64GccDPInt64Opt/lib \
		./Allwmake-plugins -s -q -l -j 

	mkdir -p $(FOAM_INST_DIR)/openfoam
	@cd $(WM_PROJECT_DIR) && \
		cp -rf platforms/linux64GccDPInt64Opt/* $(FOAM_INST_DIR)/openfoam && \
		cp -rf etc $(FOAM_INST_DIR)/openfoam && \
		cp -rf bin $(FOAM_INST_DIR)/openfoam && \
		cp -rf wmake $(FOAM_INST_DIR)/openfoam

	@cd $(FOAM_INST_DIR)/ThirdParty-v$(FOAM_VERSION) && \
		cp -rf platforms/linux64Gcc/*/* $(FOAM_INST_DIR)/openfoam && \
		cp -rf platforms/linux64GccDPInt64/lib/* $(FOAM_INST_DIR)/openfoam/lib/ && \
		cp -rf platforms/linux64GccDPInt64/metis*/* $(FOAM_INST_DIR)/openfoam && \
		cp -rf platforms/linux64GccDPInt64/scotch*/* $(FOAM_INST_DIR)/openfoam

	rm -rf $(WM_PROJECT_DIR) $(FOAM_INST_DIR)/ThirdParty-v$(FOAM_VERSION)
	@touch $@


#-------------------------------------------------------------------------------
# pyHyp
#-------------------------------------------------------------------------------

$(VENV_DIR)/.cgns.done:
	@echo "Installing CGNS..."
	@mkdir -p $(BUILD_DIR)/cgns/build
	@tar -xzf $(SOURCES_DIR)/$(CGNS_TAR) -C $(BUILD_DIR)/cgns --strip-components=1
	@cd $(BUILD_DIR)/cgns/build && \
		$(CMAKE_CMD) .. \
			-DCGNS_ENABLE_FORTRAN=ON \
			-DCGNS_ENABLE_64BIT=ON \
			-D CGNS_ENABLE_HDF5=ON \
			-DCGNS_BUILD_CGNSTOOLS=OFF \
			-DCMAKE_C_FLAGS="-fPIC" \
			-DCMAKE_Fortran_FLAGS="-fPIC" && \
		$(MAKE_CMD)
	@touch $@
	
$(VENV_DIR)/.cgnsutils.done: $(VENV_DIR)/.cgns.done $(VENV_DIR)/.python.done $(VENV_DIR)/.petsc.done
	@echo "Installing cgnsutils..."
	@mkdir -p $(BUILD_DIR)/cgnsutils
	@tar -xzf $(SOURCES_DIR)/$(CGNSUTILS_TAR) -C $(BUILD_DIR)/cgnsutils --strip-components=1
	@cd $(BUILD_DIR)/cgnsutils && \
		cp config/defaults/config.LINUX_GFORTRAN.mk config/config.mk && \
		CGNS_HOME=$(VENV_DIR) make
	@cd $(BUILD_DIR)/cgnsutils && \
		CGNS_HOME=$(VENV_DIR) $(PIP_INSTALL) .
	@touch $@
	
$(VENV_DIR)/.pyhyp.done: $(VENV_DIR)/.cgns.done $(VENV_DIR)/.cgnsutils.done $(VENV_DIR)/.python.done $(VENV_DIR)/.petsc.done
	@echo "Installing pyhyp..."
	@mkdir -p $(BUILD_DIR)/pyhyp
	git clone --depth=1 https://github.com/mdolab/pyhyp.git $(BUILD_DIR)/pyhyp || true
	$(PIP_INSTALL) mdolab-baseclasses
	@cd $(BUILD_DIR)/pyhyp && \
		cp config/defaults/config.LINUX_GFORTRAN.mk config/config.mk && \
		CGNS_HOME=$(VENV_DIR) PETSC_DIR=$(VENV_DIR) make
	@cd $(BUILD_DIR)/pyhyp && \
		pip install .
	@touch $@