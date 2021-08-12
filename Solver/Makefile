# Enda Carroll
# June 2021
# Makfile to compile programmes for Pseudopectral Solver for the 2D Ekman Navier Stokes eqns

 
# ---------------------------------------------------------------------
#  Compiler flags
# ---------------------------------------------------------------------
# Get the operating system:
# Need to strip leading and/or trailing whitespaces otherwise if statement wont work
OS = $(strip $(shell lsb_release -si))

# Set the operating system dependent compiler variables
ifeq ($(OS), Ubuntu)
	# CHIRP
	GCC := mpicc
	CCFLAG := -O3 -W -Wall -g
	LIBFLAG := -lfftw3_mpi -lfftw3 -lm -lhdf5 -lhdf5_hl #-fopenmp -lpthread -lfftw3_omp
	# GSLFLAG := -lgsl -lgslcblas
	# LAPACKFLAG := -llapacke -llapack -lblas -lgfortran
	INCDIR = -I/usr/include/hdf5/openmpi/  #-I/usr/include/gsl/
	LIBDIR = -L/usr/lib/x86_64-linux-gnu/hdf5/openmpi 
else ifeq ($(OS), RedHatEnterpriseServer)
	# SONIC
	GCC := mpicc
	CCFLAG := -O3 -W -Wall -g
	LIBFLAG := -lfftw3_mpi -lfftw3 -lm -lhdf5 -lhdf5_hl #-fopenmp -lpthread -lfftw3_omp
	INCDIR = -I/usr/include/hdf5/openmpi/ 
	LIBDIR = -L/usr/lib/x86_64-linux-gnu/hdf5/openmpi 
else ifeq ($(OS), CentOS)
	# KAY
	GCC := mpicc
	CCFLAG := -O3 -W -Wall -g
	LIBFLAG := -lfftw3_mpi -lfftw3 -lm -lhdf5 -lhdf5_hl #-fopenmp -lpthread -lfftw3_omp
	INCDIR = -I/ichec/packages/hdf5/gcc_mpi/1.12.0/include
	LIBDIR = -L/ichec/packages/hdf5/gcc_mpi/1.12.0/lib 
else
	$(error Unknown OS found, please check OS and add appropriate flags in Makefile)
endif


DEPS = $(wildcard *.h) 
# ---------------------------------------------------------------------
#  Directory creation
# ---------------------------------------------------------------------
OBJBIN := obj
OBJDIR := $(shell mkdir -p $(OBJBIN))

BIN := bin
BINDIR := $(shell mkdir -p $(BIN))

OUT := output
OUTDIR := $(shell mkdir -p $(OUT))

SRCDIR := src
SRCFILE := $(wildcard $(SRCDIR)/*.c)
SRCOBJ := $(patsubst $(SRCDIR)/%.c, $(OBJBIN)/%.o, $(SRCFILE))


# ---------------------------------------------------------------------
#  Builds
# ---------------------------------------------------------------------
############
.PHONY: all
############


all: $(BIN)/main


############
#  Solver
############
solver: $(BIN)/main 

$(BIN)/main: $(SRCOBJ) $(DEPS)
		$(GCC) $(CCFLAG) -o $@ $^ ${INCDIR} ${LIBDIR} $(LIBFLAG) $(GSLFLAG)

$(OBJBIN)/%.o: $(SRCDIR)/%.c $(DEPS)
	$(GCC) $(CCFLAG) -c $^ -o $@ ${INCDIR} ${LIBDIR} $(LIBFLAG) $(GSLFLAG)



# ---------------------------------------------------------------------
#  Clean up
# ---------------------------------------------------------------------
clean:
	rm -rf $(OBJBIN)
	rm -rf $(BIN)
	rm -rf $(OUT)