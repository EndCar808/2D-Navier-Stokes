# Enda Carroll
# June 2021
# Makfile to compile programmes for Pseudopectral Solver for the 2D Ekman Navier Stokes eqns

 
# ---------------------------------------------------------------------
#  Compiler flags
# ---------------------------------------------------------------------
GCC := mpicc
CCFLAG := -O3 -W -Wall #-g
LIBFLAG := -lfftw3_mpi -lfftw3 -lm -lhdf5 -lhdf5_hl #-fopenmp -lpthread -lfftw3_omp
# GSLFLAG := -lgsl -lgslcblas
# LAPACKFLAG := -llapacke -llapack -lblas -lgfortran
INCDIR = -I/usr/include/hdf5/openmpi/  #-I/usr/include/gsl/
LIBDIR = -L/usr/lib/x86_64-linux-gnu/hdf5/openmpi 


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