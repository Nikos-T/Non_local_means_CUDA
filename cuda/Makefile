SHELL := /bin/bash # Use bash syntax

CC = sm_20

NVCC	= nvcc -arch=$(CC)
STEP1	= nlm_step1
STEP2	= nlm_step2
ARGS	= -ptx

all: comp

comp:
	$(NVCC) $(ARGS) $(STEP1).cu $(STEP2).cu

clean:
	rm -rf *~ *.ptx
