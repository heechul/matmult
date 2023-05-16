CXX=clang++
OPTS=-O3 -march=native -flto 
CXXFLAGS=-I/opt/homebrew/Cellar/eigen/3.4.0_1/include/eigen3 $(OPTS) -std=c++17
CFLAGS=$(OPTS) -std=c11

matrix: matrix.o
	$(CXX) $< -o $@
matmult-eigen: matmult-eigen.o
	$(CXX) $< -o $@
clean:
	rm *.o
