# Enda Carroll
# Spet 2021
# Makfile to compile post processing code for the solver data

 
# ---------------------------------------------------------------------
#  Compiler flags
# ---------------------------------------------------------------------
# Get the operating system:
# Need to strip leading and/or trailing whitespaces otherwise if statement wont work
OS = $(strip $(shell lsb_release -si))

# Set the operating system dependent compiler variables
ifeq ($(OS), Ubuntu)
	# CHIRP
	GCC     := gcc
	CCFLAG  := -O3 -W -Wall -g
	LIBFLAG := -fopenmp -lpthread -lfftw3_omp -lfftw3 -lm -lhdf5 -lhdf5_hl 
	GSLFLAG := -lgsl -lgslcblas
	# LAPACKFLAG := -llapacke -llapack -lblas -lgfortran
	INCDIR   = -I/usr/include/hdf5/serial  -I/usr/include/gsl/
	LIBDIR   = -L/usr/lib/x86_64-linux-gnu/hdf5/serial
else ifeq ($(OS), RedHatEnterpriseServer)
	# SONIC
	GCC     := gcc
	CCFLAG  := -O3 -W -Wall -g
	LIBFLAG := -lfftw3 -lm -lhdf5 -lhdf5_hl #-fopenmp -lpthread -lfftw3_omp
	INCDIR   = -I/opt/software/hdf5/1.10.5/include -I/opt/software/fftw/3.3.8/include
	LIBDIR   = -L/opt/software/hdf5/1.10.5/lib -L/opt/software/fftw/3.3.8/lib
else ifeq ($(OS), CentOS)
	# KAY
	GCC     := gcc
	CCFLAG  := -O3 -W -Wall -g
	LIBFLAG := -lfftw3 -lm -lhdf5 -lhdf5_hl #-fopenmp -lpthread -lfftw3_omp
	INCDIR   = -I/ichec/packages/hdf5/gcc/1.12.0/include -I/ichec/packages/fftw/3.3.8/gcc/double/include
	LIBDIR   = -L/ichec/packages/hdf5/gcc/1.12.0/lib -L/ichec/packages/fftw/3.3.8/gcc/double/lib
else
	$(error Unknown OS found, please check OS and add appropriate flags in Makefile)
endif


# ---------------------------------------------------------------------
#  Directory creation
# ---------------------------------------------------------------------
OBJBIN := obj
OBJDIR := $(shell mkdir -p $(OBJBIN))

BIN    := bin
BINDIR := $(shell mkdir -p $(BIN))

OUT    := output
OUTDIR := $(shell mkdir -p $(OUT))

SRCDIR  := src
SRCFILE := $(wildcard $(SRCDIR)/*.c)
SRCOBJ  := $(patsubst $(SRCDIR)/%.c, $(OBJBIN)/%.o, $(SRCFILE))
# ---------------------------------------------------------------------
#  Builds
# ---------------------------------------------------------------------
############
.PHONY: all
############


all: $(BIN)/main

$(BIN)/main: $(SRCOBJ) $(DEPS)
		$(GCC) $(CCFLAG) $(SOLVER_FLAGS) -o $@ $^ ${INCDIR} ${LIBDIR} $(LIBFLAG) $(GSLFLAG)

$(OBJBIN)/%.o: $(SRCDIR)/%.c $(DEPS)
	$(GCC) $(CCFLAG) $(SOLVER_FLAGS) -c $^ -o $@ ${INCDIR} ${LIBDIR} $(LIBFLAG) $(GSLFLAG)

# ---------------------------------------------------------------------
#  Clean up
# ---------------------------------------------------------------------
clean:
	rm -rf $(OBJBIN)
	rm -rf $(BIN)
	rm -rf $(OUT)
# ---------------------------------------------------------------------
#  End of File
# ---------------------------------------------------------------------