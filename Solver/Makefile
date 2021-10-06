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

# ---------------------------------------------------------------------
#  Executable flags
# ---------------------------------------------------------------------
# Flags to be passed to the solver for normal runs
SOLVER_FLAGS := -D__NAVIER -D__ADAPTIVE_STEP -D__CFL_STEP # -D__HYPER -D__EKMN_DRAG

# Dataset Flags to be passed to the solver
DSET_FLAGS := -D__VORT_REAL # -D__SPEC -D__MODES -D__REALSPACE

# Flags to be passed to the solver for testing
TG_TEST_FLAGS    := -D__NAVIER -D__TESTING -D__ADAPTIVE_STEP -D__CFL_STEP
DT_TEST_FLAGS    := -D__NAVIER -D__TESTING -D__ADAPTIVE_STEP -D__CFL_STEP
DB_SL_TEST_FLAGS := -D__EULER -D__TESTING -D__ADAPTIVE_STEP -D__CFL_STEP
# ---------------------------------------------------------------------
#  Directory creation
# ---------------------------------------------------------------------
OBJBIN := obj
OBJDIR := $(shell mkdir -p $(OBJBIN))

BIN    := bin
BINDIR := $(shell mkdir -p $(BIN))

OUT    := output
OUTDIR := $(shell mkdir -p $(OUT))

SRCDIR     := src
SRCFILE    := $(wildcard $(SRCDIR)/*.c)
SRCOBJ     := $(patsubst $(SRCDIR)/%.c, $(OBJBIN)/%.o, $(SRCFILE))
SRCOBJ_TG  := $(patsubst $(SRCDIR)/%.c, $(OBJBIN)/%_TG_test.o, $(SRCFILE))
SRCOBJ_DSL := $(patsubst $(SRCDIR)/%.c, $(OBJBIN)/%_DB_SL_test.o, $(SRCFILE))
SRCOBJ_DT  := $(patsubst $(SRCDIR)/%.c, $(OBJBIN)/%_DT_test.o, $(SRCFILE))

DEPS = $(wildcard *.h) 

# ---------------------------------------------------------------------
#  Builds
# ---------------------------------------------------------------------
############
.PHONY: all
############


all: $(BIN)/main $(BIN)/main_TG_test $(BIN)/main_DB_SL_test $(BIN)/main_DT_test


####--------
#  Solver
####--------
solver: $(BIN)/main 

$(BIN)/main: $(SRCOBJ) $(DEPS)
		$(GCC) $(CCFLAG) $(SOLVER_FLAGS) -o $@ $^ ${INCDIR} ${LIBDIR} $(LIBFLAG) $(GSLFLAG)

$(OBJBIN)/%.o: $(SRCDIR)/%.c $(DEPS)
	$(GCC) $(CCFLAG) $(SOLVER_FLAGS) -c $^ -o $@ ${INCDIR} ${LIBDIR} $(LIBFLAG) $(GSLFLAG)


####--------
#  Testing
####--------
test: $(BIN)/main_TG_test $(BIN)/main_DB_SL_test $(BIN)/main_DT_test

test_run: $(BIN)/main_TG_test $(BIN)/main_DB_SL_test $(BIN)/main_DT_test
	mpirun -n 4 $(BIN)/main_TG_test -o "../Data/Test/" -n 128 -n 128 -s 0.0 -e 1.0 -h 0.0001 -i "TG_VORT" -v 1.0 -t "TG-TEST"
	mpirun -n 4 $(BIN)/main_DB_SL_test -o "../Data/Test/" -n 128 -n 128 -s 0.0 -e 2.0 -h 0.001 -i "DOUBLE_SHEAR_LAYER" -v 0.0 -t "DBL-TEST"
	mpirun -n 4 $(BIN)/main_DT_test -o "../Data/Test/" -n 128 -n 128 -s 0.0 -e 1.0 -h 0.001 -i "DECAY_TURB" -v 0.01 -t "DECAY-TEST"

# Taylor Green IC
$(BIN)/main_TG_test: $(SRCOBJ_TG) $(DEPS)
		$(GCC) $(CCFLAG) $(TG_TEST_FLAGS) $(DSET_FLAGS) -o $@ $^ ${INCDIR} ${LIBDIR} $(LIBFLAG) $(GSLFLAG)

$(OBJBIN)/%_TG_test.o: $(SRCDIR)/%.c $(DEPS)
	$(GCC) $(CCFLAG) $(TG_TEST_FLAGS) $(DSET_FLAGS) -c $^ -o $@ ${INCDIR} ${LIBDIR} $(LIBFLAG) $(GSLFLAG)

# Double Shear Layor IC
$(BIN)/main_DB_SL_test: $(SRCOBJ_DSL) $(DEPS)
		$(GCC) $(CCFLAG) $(DB_SL_TEST_FLAGS) $(DSET_FLAGS) -o $@ $^ ${INCDIR} ${LIBDIR} $(LIBFLAG) $(GSLFLAG)

$(OBJBIN)/%_DB_SL_test.o: $(SRCDIR)/%.c $(DEPS)
	$(GCC) $(CCFLAG) $(DB_SL_TEST_FLAGS) $(DSET_FLAGS) -c $^ -o $@ ${INCDIR} ${LIBDIR} $(LIBFLAG) $(GSLFLAG)

# Decaying Turbulence IC
$(BIN)/main_DT_test: $(SRCOBJ_DT) $(DEPS)
		$(GCC) $(CCFLAG) $(DT_TEST_FLAGS) $(DSET_FLAGS) -o $@ $^ ${INCDIR} ${LIBDIR} $(LIBFLAG) $(GSLFLAG)

$(OBJBIN)/%_DT_test.o: $(SRCDIR)/%.c $(DEPS)
	$(GCC) $(CCFLAG) $(DT_TEST_FLAGS) $(DSET_FLAGS) -c $^ -o $@ ${INCDIR} ${LIBDIR} $(LIBFLAG) $(GSLFLAG)

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