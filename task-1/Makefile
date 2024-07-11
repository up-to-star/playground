# for testing different versions of matrix multiplication
 TESTFILE := test_fp32
#   TESTFILE := test_fp16

# for testing cuBLAS
# TESTFILE := test_cuBLAS
# TESTFILE := test_cuBLAS_cudacore_fp32
# TESTFILE := test_cuBLAS_tensorcore_fp16

# cuda core matmul version
MATMUL := MMult_v0
# MATMUL := MMult_v1

# tensor core matmul version
# MATMUL := MMult_t_v0
# MATMUL := MMult_t_v1

# If set to compile in debug mode
DEBUG_FLAG := off

SMS ?= 80 86
CC := nvcc
LINKER := $(CC) #.o -> executable file

OUTPUT_DIR := output_files
TARGET := ./$(OUTPUT_DIR)/output_$(MATMUL)_$(TESTFILE).txt

$(foreach sm, $(SMS), $(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

CFLAGS := -std=c++17 -O2 #used in compilation
LDFLAGS := -lm -lcublas -lopenblas #used in link

UTIL := comp_mat.o\
		gen_mat.o \
		MMult_ref.o \
		tran_mat.o

TEST_OBJS := $(TESTFILE).o $(MATMUL).o

#generate .o files
ifeq ($(DEBUG_FLAG), on)
%.o: %.cpp
	$(CC) $(CFLAGS)$(GENCODE_FLAGS) -lineinfo -c $< -o $@

%.o: %.cu
	$(CC) $(CFLAGS)$(GENCODE_FLAGS) -lineinfo -c $< -o $@
else
%.o: %.cpp
	$(CC) $(CFLAGS)$(GENCODE_FLAGS) -c $< -o $@

%.o: %.cu
	$(CC) $(CFLAGS)$(GENCODE_FLAGS) -c $< -o $@
endif

all: $(TARGET)
	make clean
	make test.exe

#generate executable files
test.exe: $(TEST_OBJS) $(UTIL) parameters.h
	$(LINKER) $(TEST_OBJS) $(UTIL) $(LDFLAGS) -o $@

#create directory if not exist
$(TARGET): | $(OUTPUT_DIR)

$(OUTPUT_DIR):
	mkdir -p $@

#run
run:
	make all
	echo "version = '$(MATMUL) + $(TESTFILE)';" > $(TARGET)
	./test.exe >> $(TARGET)

clean:
	rm -f *.o *~ core *.exe

