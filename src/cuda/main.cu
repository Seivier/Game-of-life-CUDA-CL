#include <chrono>
#include <iostream>
#include <vector>
#include <random>
#include <fstream>

const int ITER = 500;


using cell = unsigned char;

using namespace std;

__global__ void iteration(const cell* in, cell* out, int n, int m) {
    int size = n * m;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
        int x1 = i % n;
        int y1 = i - x1;
        int x0 = (x1 + n - 1) % n;
        int x2 = (x1 + 1) % n;
        int y0 = (y1 + n) % (size);
        int y2 = (y1 + size - n) % (size);
        int alive_neighbours = in[x0 + y0] + in[x0 + y1] + in[x0 + y2] + in[x1 + y0] + in[x1 + y2] + in[x2 + y0] + in[x2 + y1] + in[x2 + y2];
        out[x1 + y1] = (alive_neighbours == 3 || (alive_neighbours == 2 && in[x1 + y1])) ? 1 : 0;
    }
}

__global__ void iteration_if(const cell* in, cell* out, int n, int m) {
	int size = n * m;
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
		int x1 = i % n;
		int y1 = i - x1;
		int x0 = (x1 + n - 1) % n;
		int x2 = (x1 + 1) % n;
		int y0 = (y1 + n) % (size);
		int y2 = (y1 + size - n) % (size);
		int alive_neighbours = 0;
		if (in[x0 + y0]) alive_neighbours++;
		if (in[x0 + y1]) alive_neighbours++;
		if (in[x0 + y2]) alive_neighbours++;
		if (in[x1 + y0]) alive_neighbours++;
		if (in[x1 + y2]) alive_neighbours++;
		if (in[x2 + y0]) alive_neighbours++;
		if (in[x2 + y1]) alive_neighbours++;
		if (in[x2 + y2]) alive_neighbours++;
		out[x1 + y1] = (alive_neighbours == 3 || (alive_neighbours == 2 && in[x1 + y1])) ? 1 : 0;
	}
}

void random_data(vector<cell>& data) {

    // Init seed
	seed_seq seq{ 123456 };
	random_device rd;
  	default_random_engine eng(seq);
	uniform_int_distribution<int> distr(0, 1);

    for (cell& c : data)
        c = distr(eng);
}

void display(vector<cell> data, int m, int n) {
    int lives = 0;
    for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) 
			lives += data[i * n + j] ? 1 : 0;
	}
    cout << "Live cells: " << lives << endl;
    cout << "Dead cells: " << m * n - lives << endl;
}

void simulate(const int N, const int M, const int BLOCK_SIZE, bool if_version) {
	const long long size = sizeof(cell) * N * M;
	vector<cell> h_in(N * M), h_out(N * M);
	cell* d_in, * d_out;

	cudaMalloc((void**)&d_in, size);
	cudaMalloc((void**)&d_out, size);

	// Inicializar la data
	auto t_start = chrono::high_resolution_clock::now();
	random_data(h_in);
	auto t_end = chrono::high_resolution_clock::now();
	auto t_data = chrono::duration<double, milli>(t_end - t_start).count();

//	cout << "INITIAL STATE:" << endl;
//	display(h_in, N, M);
//	cout << endl;

	// Copiar host a device
	t_start = chrono::high_resolution_clock::now();
	cudaMemcpy(d_in, h_in.data(), size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_out, h_out.data(), size, cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
	t_end = chrono::high_resolution_clock::now();
	auto t_host_to_device = chrono::duration<double, milli>(t_end - t_start).count();


	// Correr la simulacion
	t_start = chrono::high_resolution_clock::now();
	for (int i = 0; i < ITER; i++)
	{
		if (if_version)
			iteration_if <<< (N * M) / BLOCK_SIZE, BLOCK_SIZE >>>(d_in, d_out, N, M);
		else
			iteration <<< (N * M) / BLOCK_SIZE, BLOCK_SIZE >>>(d_in, d_out, N, M);
		swap(d_in, d_out);
	}
	cudaDeviceSynchronize();
	t_end = chrono::high_resolution_clock::now();
	auto t_exec = chrono::duration<double, milli>(t_end - t_start).count();


	// Copiar device a host
	t_start = chrono::high_resolution_clock::now();
	cudaMemcpy(h_in.data(), d_in, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_out.data(), d_out, size, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	t_end = chrono::high_resolution_clock::now();
	auto t_device_to_host = chrono::duration<double, milli>(t_end - t_start).count();

//	cout << "FINAL STATE: \n";
//	display(h_in, N, M);
//	cout << endl;

	cudaFree(d_in);
	cudaFree(d_out);

//	cout << "Time to create the data: " << t_data << " ms\n";
//	cout << "Time to copy data from host to device: " << t_host_to_device << " ms\n";
//	cout << "Time to execute the simulation: " << t_exec << " ms\n";
//	cout << "Time to copy data from device to host: " << t_device_to_host << " ms" << endl;
//	cout << "Total time: " << t_data + t_exec + t_device_to_host + t_host_to_device << " ms" << endl;

	string filename = if_version? "cuda_if.csv" : "cuda.csv";

	fstream file;
	file.open(filename, ios::out | ios::app);
	file << N << "," << M << "," << BLOCK_SIZE << "," << t_data << "," << t_host_to_device << "," << t_exec << ","
		 << t_device_to_host << "," << t_data + t_exec + t_device_to_host + t_host_to_device << endl;
	file.close();

	display(h_in, N, M);
}

int main(int argc, char* argv[]) {
	if (argc != 5) {
		cout << "Usage: " << argv[0] << " N M BLOCK_SIZE VERSION" << endl;
		return 1;
	}

	int N = atoi(argv[1]);
	int M = atoi(argv[2]);
	int BLOCK_SIZE = atoi(argv[3]);
	bool if_version = atoi(argv[4]);
	simulate(N, M, BLOCK_SIZE, if_version);
}