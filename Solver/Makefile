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
SOLVER_FLAGS := -D__NAVIER #-D__HYPER -D__ADAPTIVE_STEP -D__CFL_STEP #-D__TRANSIENTS  -D__EKMN_DRAG

# Phase only flags to be passed to solver
PHASE_ONLY_FLAGS := -D__NAVIER -D__PHASE_ONLY #-D__HYPER -D__TRANSIENTS

# Dataset Flags to be passed to the solver
DSET_FLAGS := -D__VORT_FOUR -D__VORT_REAL -D__SYS_MEASURES -D__ENST_FLUX -D__ENRG_FLUX -D__ENST_SPECT -D__ENRG_SPECT -D__ENST_FLUX_SPECT -D__ENRG_FLUX_SPECT #-D__PHASE_SYNC  # -D__VORT_REAL #-D__REALSPACE # -D__MODES  
 
# Flags to be passed to the solver for testing
TG_TEST_FLAGS    := -D__NAVIER -D__TESTING -D__DEBUG #-D__ADAPTIVE_STEP -D__CFL_STEP -D__TRANSIENTS
DT_TEST_FLAGS    := -D__NAVIER -D__TESTING -D__DEBUG #-D__HYPER #-D__ADAPTIVE_STEP -D__CFL_STEP -D__TRANSIENTS
DT2_TEST_FLAGS   := -D__NAVIER -D__TESTING -D__DEBUG #-D__HYPER #-D__ADAPTIVE_STEP -D__CFL_STEP -D__TRANSIENTS
DB_SL_TEST_FLAGS := -D__EULER -D__TESTING -D__DEBUG #-D__ADAPTIVE_STEP -D__CFL_STEP -D__TRANSIENTS
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
SRCOBJ_PO  := $(patsubst $(SRCDIR)/%.c, $(OBJBIN)/%_phase_only.o, $(SRCFILE))
SRCOBJ_TG  := $(patsubst $(SRCDIR)/%.c, $(OBJBIN)/%_TG_test.o, $(SRCFILE))
SRCOBJ_DSL := $(patsubst $(SRCDIR)/%.c, $(OBJBIN)/%_DB_SL_test.o, $(SRCFILE))
SRCOBJ_DT  := $(patsubst $(SRCDIR)/%.c, $(OBJBIN)/%_DT_test.o, $(SRCFILE))
SRCOBJ_DT2 := $(patsubst $(SRCDIR)/%.c, $(OBJBIN)/%_DT2_test.o, $(SRCFILE))

DEPS = $(wildcard *.h) 

# ---------------------------------------------------------------------
#  Builds
# ---------------------------------------------------------------------
############
.PHONY: all
############


all: $(BIN)/solver $(BIN)/solver_phase_only $(BIN)/solver_TG_test $(BIN)/solver_DB_SL_test $(BIN)/solver_DT_test $(BIN)/solver_DT2_test


####--------
#  Solver
####--------
solver: $(BIN)/solver 

$(BIN)/solver: $(SRCOBJ) $(DEPS)
	$(GCC) $(CCFLAG) $(SOLVER_FLAGS) $(DSET_FLAGS) -o $@ $^ ${INCDIR} ${LIBDIR} $(LIBFLAG) $(GSLFLAG)

$(OBJBIN)/%.o: $(SRCDIR)/%.c $(DEPS)
	$(GCC) $(CCFLAG) $(SOLVER_FLAGS) $(DSET_FLAGS) -c $^ -o $@ ${INCDIR} ${LIBDIR} $(LIBFLAG) $(GSLFLAG)

####-----------
#  Phase Only 
####-----------
solver_phase_only: $(BIN)/solver_phase_only 

$(BIN)/solver_phase_only: $(SRCOBJ_PO) $(DEPS)
	$(GCC) $(CCFLAG) $(PHASE_ONLY_FLAGS) $(DSET_FLAGS) -o $@ $^ ${INCDIR} ${LIBDIR} $(LIBFLAG) $(GSLFLAG)

$(OBJBIN)/%_phase_only.o: $(SRCDIR)/%.c $(DEPS)
	$(GCC) $(CCFLAG) $(PHASE_ONLY_FLAGS) $(DSET_FLAGS) -c $^ -o $@ ${INCDIR} ${LIBDIR} $(LIBFLAG) $(GSLFLAG)

####--------
#  Testing
####--------
test: $(BIN)/solver_TG_test $(BIN)/solver_DB_SL_test $(BIN)/solver_DT_test $(BIN)/solver_DT2_test

test_run: $(BIN)/solver_TG_test $(BIN)/solver_DB_SL_test $(BIN)/solver_DT_test $(BIN)/solver_DT2_test
	mpirun -n 4 $(BIN)/solver_TG_test -o "../Data/Test/" -n 128 -n 128 -s 0.0 -e 1.0 -h 0.0001 -i "TG_VORT" -v 1.0 -t "TG-TEST"
	mpirun -n 4 $(BIN)/solver_DB_SL_test -o "../Data/Test/" -n 128 -n 128 -s 0.0 -e 2.0 -h 0.001 -i "DOUBLE_SHEAR_LAYER" -v 0.0 -t "DBL-TEST"
	mpirun -n 4 $(BIN)/solver_DT_test -o "../Data/Test/" -n 512 -n 512 -s 0.0 -e 11.0 -h 0.001 -i "DECAY_TURB" -v 0.01 -t "DECAY-TEST" -p 100
	mpirun -n 4 $(BIN)/solver_DT2_test -o "../Data/Test/" -n 512 -n 512 -s 0.0 -e 11.0 -h 0.001 -i "DECAY_TURB_II" -v 0.01 -t "DECAY2-TEST" -p 100

# Taylor Green IC
$(BIN)/solver_TG_test: $(SRCOBJ_TG) $(DEPS)
	$(GCC) $(CCFLAG) $(TG_TEST_FLAGS) $(DSET_FLAGS) -o $@ $^ ${INCDIR} ${LIBDIR} $(LIBFLAG) $(GSLFLAG)

$(OBJBIN)/%_TG_test.o: $(SRCDIR)/%.c $(DEPS)
	$(GCC) $(CCFLAG) $(TG_TEST_FLAGS) $(DSET_FLAGS) -c $^ -o $@ ${INCDIR} ${LIBDIR} $(LIBFLAG) $(GSLFLAG)

# Double Shear Layor IC
$(BIN)/solver_DB_SL_test: $(SRCOBJ_DSL) $(DEPS)
	$(GCC) $(CCFLAG) $(DB_SL_TEST_FLAGS) $(DSET_FLAGS) -o $@ $^ ${INCDIR} ${LIBDIR} $(LIBFLAG) $(GSLFLAG)

$(OBJBIN)/%_DB_SL_test.o: $(SRCDIR)/%.c $(DEPS)
	$(GCC) $(CCFLAG) $(DB_SL_TEST_FLAGS) $(DSET_FLAGS) -c $^ -o $@ ${INCDIR} ${LIBDIR} $(LIBFLAG) $(GSLFLAG)

# Decaying Turbulence IC
$(BIN)/solver_DT_test: $(SRCOBJ_DT) $(DEPS)
	$(GCC) $(CCFLAG) $(DT_TEST_FLAGS) $(DSET_FLAGS) -o $@ $^ ${INCDIR} ${LIBDIR} $(LIBFLAG) $(GSLFLAG)

$(OBJBIN)/%_DT_test.o: $(SRCDIR)/%.c $(DEPS)
	$(GCC) $(CCFLAG) $(DT_TEST_FLAGS) $(DSET_FLAGS) -c $^ -o $@ ${INCDIR} ${LIBDIR} $(LIBFLAG) $(GSLFLAG)

# Decaying Turbulence IC with Hyperviscosity
$(BIN)/solver_DT2_test: $(SRCOBJ_DT2) $(DEPS)
	$(GCC) $(CCFLAG) $(DT2_TEST_FLAGS) $(DSET_FLAGS) -o $@ $^ ${INCDIR} ${LIBDIR} $(LIBFLAG) $(GSLFLAG)

$(OBJBIN)/%_DT2_test.o: $(SRCDIR)/%.c $(DEPS)
	$(GCC) $(CCFLAG) $(DT2_TEST_FLAGS) $(DSET_FLAGS) -c $^ -o $@ ${INCDIR} ${LIBDIR} $(LIBFLAG) $(GSLFLAG)

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