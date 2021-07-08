NVCC        = nvcc
NVCC_FLAGS  = -Xcompiler -fPIC -O3 -I/usr/local/cuda/include  -lcublas_static -lcublasLt_static -lculibos -lcudart_static -lpthread -ldl
NVCC_SPEC_FLAGS = --default-stream per-thread
LD_FLAGS    = -I/usr/local/cuda/include -L/usr/local/cuda/lib64/ -lcublas_static -lcublasLt_static  -lculibos -lcudart_static -lpthread -ldl

EXE5            = split
EXE6            = 1_split

OBJ5            = split.o support.o common.o
OBJ6            = split_1.o support.o common.o
default: $(EXE5)

sp:$(EXE5)

sp1:$(EXE6)

support.o: support.cu support.h
	$(NVCC) -c -o $@ support.cu $(NVCC_FLAGS)
common.o: common.h common.cu
	$(NVCC) -c -o $@ common.cu $(NVCC_FLAGS)
split.o:  split.cu support.h common.h
	$(NVCC) -c -o $@ split.cu $(NVCC_FLAGS)

split_1.o:  split_1.cu common.h support.h
	$(NVCC) -c -o $@ split_1.cu $(NVCC_FLAGS)

$(EXE5): $(OBJ5)
	$(NVCC) $(OBJ5) -o $(EXE5) $(LD_FLAGS)
$(EXE6): $(OBJ6)
	$(NVCC) $(OBJ6) -o $(EXE6) $(LD_FLAGS)
clean:
	rm -rf *.o $(EXE1) $(EXE2) $(EXE3) $(EXE4) $(EXE5) $(EXE6) 
