# Makefile for compiling Python with full module support
# Includes dependencies for common optional modules and development tools

CHECK_DEPS := $(shell command -v make cmake git unzip wget)
ifndef CHECK_DEPS
  $(error "Missing required tools: make, cmake, git, unzip, wget")
endif

#-------------------------------------------------------------------------------
# Configuration Section
#-------------------------------------------------------------------------------

# Version configurations
BOOST_VERSION      := 1.81.0
EIGEN_VERSION      := 3.3.7
FOAM_VERSION       := 2406
GDBM_VERSION       := 1.23
LIBFFI_VERSION     := 3.4.2
LIBUUID_VERSION    := 1.0.3
LIBXML2_VERSION    := 2.13.6
NCURSES_VERSION    := 6.3
OMPI_VERSION       := 4.1.8
OMPI_SHORT_VERSION := $OMPI_VERSION := 4.1.8
OMPI_SHORT_VERSION := $(word 1,$(subst ., ,$(OMPI_VERSION))).$(word 2,$(subst ., ,$(OMPI_VERSION)))
OPENSSL_VERSION    := 3.4.1
PETSC_VERSION      := 3.22.2
PRECICE_VERSION    := 3.1.2
PYTHON_VERSION     := 3.12.9
READLINE_VERSION   := 8.2
SQLITE_VERSION     := 3490000
KHIP_VERSION      := 3.18


# Dependency file names
BOOST_TAR          := boost_$(subst .,_,$(BOOST_VERSION)).tar.bz2
BZ2_TAR            := bzip2-master.tar.gz
EIGEN_TAR          := eigen-$(EIGEN_VERSION).tar.gz
FOAM_TAR           := openfoam-OpenFOAM-v$(FOAM_VERSION).tar.gz
GDBM_TAR           := gdbm-$(GDBM_VERSION).tar.gz
KHIP_ZIP           := v$(KHIP_VERSION).zip
LIBFFI_TAR         := libffi-$(LIBFFI_VERSION).tar.gz
LIBUUID_TAR        := libuuid-$(LIBUUID_VERSION).tar.gz
LIBXML2_TAR        := libxml2_v$(LIBXML2_VERSION).tar.gz
NCURSES_TAR        := ncurses-$(NCURSES_VERSION).tar.gz
OMPI_TAR           := openmpi-$(OMPI_VERSION).tar.gz
OPENSSL_TAR        := openssl-$(OPENSSL_VERSION).tar.gz
PETSC_TAR          := petsc-$(PETSC_VERSION).tar.gz
PRECICE_TAR        := v$(PRECICE_VERSION).tar.gz
PYTHON_TAR         := Python-$(PYTHON_VERSION).tgz
READLINE_TAR       := readline-$(READLINE_VERSION).tar.gz
SLEPC_TAR          := slepc-$(PETSC_VERSION).tar.gz
SQLITE_TAR         := sqlite-autoconf-$(SQLITE_VERSION).tar.gz

# Directory configurations
include .env
export $(shell sed -n 's/^export //p' .env)

VENV_DIR    := $(PWD)/.venv
SOURCES_DIR := $(PWD)/.sources
BUILD_DIR   := $(SOURCES_DIR)/build

# Build Commands
NPROC         := $(shell nproc)
DOWNLOAD      := wget -nc
MAKE_CMD      := LD_LIBRARY_PATH=$(VENV_DIR)/lib:$(VENV_DIR)/lib64 \
				 make -j$(NPROC) && \
				 LD_LIBRARY_PATH=$(VENV_DIR)/lib:$(VENV_DIR)/lib64 \
				 make install PREFIX=$(VENV_DIR)
CONFIGURE_CMD := LD_LIBRARY_PATH=$(VENV_DIR)/lib:$(VENV_DIR)/lib64 \
				 ./configure --prefix=$(VENV_DIR)
PIP_INSTALL   := LD_LIBRARY_PATH=$(VENV_DIR)/lib:$(VENV_DIR)/lib64 \
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
.PHONY: all clean pip
.NOTPARALLEL: $(VENV_DIR)/.python.done $(VENV_DIR)/.petsc.done

all: download_sources python petsc slepc precice openfoam python_env
	pip freeze > requirements.lock
	@echo "====================================="
	@echo "# All components built successfully #"
	@echo "====================================="

clean:
	rm -rf $(VENV_DIR) $(SOURCES_DIR)/build

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
		https://download.open-mpi.org/release/open-mpi/v$(OMPI_SHORT_VERSION)/$(OMPI_TAR) \
		https://ftp.gnu.org/gnu/gdbm/$(GDBM_TAR) \
		https://ftp.gnu.org/gnu/ncurses/$(NCURSES_TAR) \
		https://ftp.gnu.org/gnu/readline/$(READLINE_TAR) \
		https://github.com/KaHIP/KaHIP/archive/refs/tags/$(KHIP_ZIP) \
		https://github.com/libffi/libffi/releases/download/v$(LIBFFI_VERSION)/$(LIBFFI_TAR) \
		https://github.com/precice/precice/archive/$(PRECICE_TAR) \
		https://gitlab.com/libeigen/eigen/-/archive/$(EIGEN_VERSION)/$(EIGEN_TAR) \
		https://gitlab.gnome.org/GNOME/libxml2/-/archive/v$(LIBXML2_VERSION)/$(LIBXML2_TAR) \
		https://slepc.upv.es/download/distrib/$(SLEPC_TAR) \
		https://sourceforge.net/projects/libuuid/files/$(LIBUUID_TAR) \
		https://web.cels.anl.gov/projects/petsc/download/release-snapshots/$(PETSC_TAR) \
		https://www.openssl.org/source/$(OPENSSL_TAR) \
		https://www.python.org/ftp/python/$(PYTHON_VERSION)/$(PYTHON_TAR) \
		https://gitlab.com/bzip2/bzip2/-/archive/master/$(BZ2_TAR) \
		https://www.sqlite.org/2025/$(SQLITE_TAR)
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
$(eval $(call build-library,libffi,$(LIBFFI_TAR),--disable-static))
$(eval $(call build-library,sqlite,$(SQLITE_TAR),))
$(eval $(call build-library,ncurses,$(NCURSES_TAR),--with-shared --with-termlib --without-debug))
$(eval $(call build-library,gdbm,$(GDBM_TAR),))
$(eval $(call build-library,libuuid,$(LIBUUID_TAR),))

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
		$(CONFIGURE_CMD) LDFLAGS="-L$(VENV_DIR)/lib -lncurses" --with-curses --with-shared-termcap-library --with-shared --with-termlib && \
		$(MAKE_CMD)
	@touch $@

$(VENV_DIR)/.python.done: $(SOURCES_DIR)/download.done \
	$(VENV_DIR)/.bz2.done \
	$(VENV_DIR)/.libffi.done \
	$(VENV_DIR)/.openssl.done \
	$(VENV_DIR)/.ncurses.done \
	$(VENV_DIR)/.gdbm.done \
	$(VENV_DIR)/.readline.done
	@echo "Building Python $(PYTHON_VERSION)..."
	@mkdir -p $(BUILD_DIR)/python
	@tar -xzf $(SOURCES_DIR)/$(PYTHON_TAR) -C $(BUILD_DIR)/python --strip-components=1
	@cd $(BUILD_DIR)/python && \
		$(CONFIGURE_CMD) \
			--enable-optimizations \
			--enable-shared && \
		$(MAKE_CMD)
	@ln -sf $(VENV_DIR)/bin/python3 $(VENV_DIR)/bin/python
	@touch $@

# Virtual Environment Setup

$(VENV_DIR)/.python_env.done: $(VENV_DIR)/.python.done

	@ln -sf $(VENV_DIR)/bin/pip3 $(VENV_DIR)/bin/pip
	$(PIP_INSTALL) --upgrade pip setuptools wheel Cython
	$(PIP_INSTALL) mpi4py petsc4py slepc4py
	$(PIP_INSTALL) pyprecice==3.1.0
	$(PIP_INSTALL) -e .

	@touch $@

#-------------------------------------------------------------------------------
# PETSc Installation
#-------------------------------------------------------------------------------
$(eval $(call build-library,openmpi,$(OMPI_TAR),))

$(VENV_DIR)/.petsc.done: $(VENV_DIR)/.openmpi.done
	@echo "Installing PETSc..."
	@mkdir -p $(BUILD_DIR)/petsc
	@tar -xzf $(SOURCES_DIR)/$(PETSC_TAR) -C $(BUILD_DIR)/petsc --strip-components=1
	@cd $(BUILD_DIR)/petsc && \
		$(CONFIGURE_CMD) \
			LDFLAGS=$$LDFLAGS \
			--with-shared-libraries=1 \
			--with-mpi-dir=$(VENV_DIR) \
			--with-debugging=0 \
			--download-fblaslapack \
			--download-hdf5 \
			--download-hypre \
			--download-metis \
			--download-scalapack \
			--download-mumps \
			--download-ptscotch \
			--download-superlu_dist \
			--download-zlib && \
		make LD_LIBRARY_PATH=$(VENV_DIR)/lib:$(VENV_DIR)/lib64 -j$(NPROC) PETSC_DIR=$(BUILD_DIR)/petsc all && \
		make LD_LIBRARY_PATH=$(VENV_DIR)/lib:$(VENV_DIR)/lib64 PETSC_DIR=$(BUILD_DIR)/petsc install
	@touch $@

$(VENV_DIR)/.slepc.done: $(VENV_DIR)/.petsc.done
	@echo "Installing SLEPc..."
	@mkdir -p $(BUILD_DIR)/slepc
	@tar -xzf $(SOURCES_DIR)/$(SLEPC_TAR) -C $(BUILD_DIR)/slepc --strip-components=1
	@cd $(BUILD_DIR)/slepc && \
		PETSC_DIR=$(VENV_DIR) SLEPC_DIR=$(BUILD_DIR)/slepc $(CONFIGURE_CMD) --with-scalapack && \
		make  LD_LIBRARY_PATH=$(VENV_DIR)/lib:$(VENV_DIR)/lib64 -j$(NPROC) PETSC_DIR=$(VENV_DIR) SLEPC_DIR=$(BUILD_DIR)/slepc all && \
		make  LD_LIBRARY_PATH=$(VENV_DIR)/lib:$(VENV_DIR)/lib64 -j$(NPROC) PETSC_DIR=$(VENV_DIR) SLEPC_DIR=$(BUILD_DIR)/slepc install
	@touch $@

#-------------------------------------------------------------------------------
# preCICE Dependencies and Installation
#-------------------------------------------------------------------------------

$(VENV_DIR)/.eigen.done:
	@echo "Installing Eigen..."
	@tar -xzf $(SOURCES_DIR)/$(EIGEN_TAR) -C $(VENV_DIR)/include/
	@mv $(VENV_DIR)/include/eigen-$(EIGEN_VERSION) $(VENV_DIR)/include/eigen
	@touch $@

$(VENV_DIR)/.boost.done: $(SOURCES_DIR)/download.done
	@echo "Installing BOOST..."
	@mkdir -p $(BUILD_DIR)/boost
	@tar -xjf $(SOURCES_DIR)/$(BOOST_TAR) -C $(BUILD_DIR)/boost --strip-components=1
	@cd $(BUILD_DIR)/boost && \
		./bootstrap.sh --with-libraries=all --prefix=$(VENV_DIR) && \
		./b2 install --prefix=$(VENV_DIR)
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
	$(PIP_INSTALL) "polars" "numpy>=2"
	@cd $(BUILD_DIR)/precice && \
		$(CMAKE_CMD) --preset=production \
			-DEIGEN3_INCLUDE_DIR=$(VENV_DIR)/include/eigen \
			-DPython_EXECUTABLE=$(VENV_DIR)/bin/python \
			-DPython3_EXECUTABLE=$(VENV_DIR)/bin/python3 \
			-DMPI_C_COMPILER=$(VENV_DIR)/bin/mpicc \
			-DMPI_CXX_COMPILER=$(VENV_DIR)/bin/mpicxx && \
			cd build && \
				$(MAKE_CMD)
	@touch $@


#-------------------------------------------------------------------------------
# OpenFOAM Installation
#-------------------------------------------------------------------------------

$(VENV_DIR)/.khip.done: $(VENV_DIR)/.openmpi.done
	@echo "Installing KHIP..."
	@unzip $(SOURCES_DIR)/$(KHIP_ZIP) -d $(BUILD_DIR)
	@mkdir -p $(BUILD_DIR)/KaHIP-$(KHIP_VERSION)/build
	@cd $(BUILD_DIR)/KaHIP-$(KHIP_VERSION)/build && \
		$(CMAKE_CMD) .. && \
		$(MAKE_CMD)
	@touch $@

$(VENV_DIR)/.openfoam.done:
	@echo "Installing OpenFOAM..."
	@mkdir -p $(FOAM_INST_DIR)
	@tar -xf $(SOURCES_DIR)/$(FOAM_TAR) -C $(FOAM_INST_DIR)
	mv $(FOAM_INST_DIR)/openfoam-OpenFOAM-v$(FOAM_VERSION) $(WM_PROJECT_DIR)

	rm -rf $(WM_PROJECT_DIR)/plugins/cfmesh $(WM_PROJECT_DIR)/plugins/precice-adapter $(WM_PROJECT_DIR)/plugins/libAcoustics
	git clone --depth=1 https://develop.openfoam.com/Community/integration-cfmesh.git $(WM_PROJECT_DIR)/plugins/cfmesh
	git clone --depth=1 https://github.com/precice/openfoam-adapter.git $(WM_PROJECT_DIR)/plugins/precice-adapter
	git clone --depth=1 https://github.com/unicfdlab/libAcoustics.git $(WM_PROJECT_DIR)/plugins/libAcoustics

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
			test \
			tools \
			utilities/surface \
			utilities/thermophysical

	@cd $(WM_PROJECT_DIR) && \
		bin/tools/foamConfigurePaths \
			-project-path "$(WM_PROJECT_DIR)" \
			-boost boost-system \
			-kahip kahip-system \
			-scotch scotch-system \
			-boost-path $(VENV_DIR)/lib \
			-kahip-path $(VENV_DIR)/lib \
			-scotch-path $(VENV_DIR)/lib \
			-cgal  cgal-none \
			-fftw  fftw-none \
			;

	@cd $(WM_PROJECT_DIR) && \
		WM_ARCH_OPTION=64 \
		WM_COMPILE_OPTION=Opt \
		WM_COMPILER_TYPE=system \
		WM_COMPILER=Gcc \
		WM_LABEL_SIZE=64 \
		WM_MPLIB=OPENMPI \
		WM_PRECISION_OPTION=DP \
		WM_PROJECT_VERSION=v$(FOAM_VERSION) \
		WM_PROJECT=OpenFOAM \
		source $(WM_PROJECT_DIR)/etc/config.sh/setup && \
		MPI_ROOT=$(VENV_DIR) MPI_ARCH_PATH=$(VENV_DIR) \
		FOAM_EXT_LIBBIN=$(VENV_DIR)/lib \
		FOAM_EXT_INCLUDE=$(VENV_DIR)/include \
		LD_LIBRARY_PATH=$(VENV_DIR)/lib:$(VENV_DIR)/lib64 \
		CPPFLAGS="-I$(VENV_DIR)/include" CXXFLAGS="-I$(VENV_DIR)/include" \
		FOAM_MODULE_PREFIX=false \
		./Allwmake -j$(NPROC)

	@cd $(WM_PROJECT_DIR) && \
		LD_LIBRARY_PATH=$(VENV_DIR)/lib:$(VENV_DIR)/lib64 \
		FOAM_USER_LIBBIN=$(VENV_DIR)/lib:$(VENV_DIR)/lib64 \
		./Allwmake-plugins -j$(NPROC)
		cd plugins/libAcoustics && \
			LD_LIBRARY_PATH=$(VENV_DIR)/lib:$(VENV_DIR)/lib64 ./makeLibrary.sh

	@cd $(WM_PROJECT_DIR) && \
		cp -rf platforms/linux64GccDPInt64Opt/* $(VENV_DIR) && \
		cp -rf etc $(VENV_DIR) && \
		cp -rf bin $(VENV_DIR)

	@touch $@