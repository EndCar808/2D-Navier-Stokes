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
SRCOBJ_STATS := $(patsubst $(SRCDIR)/%.c, $(OBJBIN)/%_stats.o, $(SRCFILE))
SRCOBJ_FULL  := $(patsubst $(SRCDIR)/%.c, $(OBJBIN)/%_full_field.o, $(SRCFILE))
SRCOBJ_SYNC  := $(patsubst $(SRCDIR)/%.c, $(OBJBIN)/%_phase_sync.o, $(SRCFILE))

# ---------------------------------------------------------------------
#  Executable flags
# ---------------------------------------------------------------------
# Flags to be passed for stats runs
STATS_FLAGS := -D__POST_STATS

# Flags to be passed for full field data
FULL_FIELD_FLAGS := -D__POST_FIELD

# Flags to be passed for phase sync data
PHASE_SYNC_FLAGS := -D__POST_SYNC

# Flags to be passed for general postprocessing
POST_FLAGS := -D__POST_STATS -D__POST_FIELD

# Data sets to save
DSETS = -D__TIME -D__COLLOC_PTS -D__WAVELIST
# DSETS += -D__VORT_FOUR
# DSETS += -D__VORT_REAL
# DSETS += -D__MODES
# DSETS += -D__REALSPACE
# DSETS += -D__NONLIN
# DSETS += -D__SPECTRA
# DSETS += -D__ENST_FLUX -D__ENRG_FLUX



# ---------------------------------------------------------------------
#  Builds
# ---------------------------------------------------------------------
############
.PHONY: all
############


all: $(BIN)/PostProcess $(BIN)/PostProcess_stats $(BIN)/PostProcess_full_field $(BIN)/PostProcess_phase_sync
stats: $(BIN)/PostProcess_stats
field: $(BIN)/PostProcess_full_field
sync: $(BIN)/PostProcess_phase_sync

###########
# All Post
###########
$(BIN)/PostProcess: $(SRCOBJ) $(DEPS)
		$(GCC) $(CCFLAG) $(POST_FLAGS) $(DSETS) -o $@ $^ ${INCDIR} ${LIBDIR} $(LIBFLAG) $(GSLFLAG)

$(OBJBIN)/%.o: $(SRCDIR)/%.c $(DEPS)
	$(GCC) $(CCFLAG) $(POST_FLAGS) $(DSETS) -c $^ -o $@ ${INCDIR} ${LIBDIR} $(LIBFLAG) $(GSLFLAG)

###########
# Stats
###########
$(BIN)/PostProcess_stats: $(SRCOBJ_STATS) $(DEPS)
		$(GCC) $(CCFLAG) $(STATS_FLAGS) $(DSETS) -o $@ $^ ${INCDIR} ${LIBDIR} $(LIBFLAG) $(GSLFLAG)

$(OBJBIN)/%_stats.o: $(SRCDIR)/%.c $(DEPS)
	$(GCC) $(CCFLAG) $(STATS_FLAGS) $(DSETS) -c $^ -o $@ ${INCDIR} ${LIBDIR} $(LIBFLAG) $(GSLFLAG)

##############
# Full Field
##############
$(BIN)/PostProcess_full_field: $(SRCOBJ_FULL) $(DEPS)
		$(GCC) $(CCFLAG) $(FULL_FIELD_FLAGS) $(DSETS) -o $@ $^ ${INCDIR} ${LIBDIR} $(LIBFLAG) $(GSLFLAG)

$(OBJBIN)/%_full_field.o: $(SRCDIR)/%.c $(DEPS)
	$(GCC) $(CCFLAG) $(FULL_FIELD_FLAGS) $(DSETS) -c $^ -o $@ ${INCDIR} ${LIBDIR} $(LIBFLAG) $(GSLFLAG)

##############
# Phase Sync
##############
$(BIN)/PostProcess_phase_sync: $(SRCOBJ_SYNC) $(DEPS)
		$(GCC) $(CCFLAG) $(PHASE_SYNC_FLAGS) $(DSETS) -o $@ $^ ${INCDIR} ${LIBDIR} $(LIBFLAG) $(GSLFLAG)

$(OBJBIN)/%_phase_sync.o: $(SRCDIR)/%.c $(DEPS)
	$(GCC) $(CCFLAG) $(PHASE_SYNC_FLAGS) $(DSETS) -c $^ -o $@ ${INCDIR} ${LIBDIR} $(LIBFLAG) $(GSLFLAG)

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