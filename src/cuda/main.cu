
#include <chrono>
#include <iostream>
#include <vector>
#include <random>



const int ITER = 10000;
using cell = unsigned char;
using namespace std;
#define BLOCK_SIZE 128

__global__ void iteration(cell* in, cell* out, int n, int m) {
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

__global__ void iteration_local(cell* in, cell* out, int n, int m) {
    int size = n * m; // blockDim=4, gridDim=400, n=100, m=100, size = 10000
    // iter = 10000 / 400 * 4 = 6.25
    int ln = BLOCK_SIZE + 2; // 6 (18)
    int lm = size / blockDim.x * gridDim.x; // 1
    __shared__ cell shared_mem[(3*BLOCK_SIZE+2)*100]; // 2 extra para las cells de arrriba y abajo y 2 para los que quedan a al izq y der del borde (
    int li = threadIdx.x + 1 + ln; // 1 + 6 = 7
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
        // i = 1
        // li = 7
        int x1 = i % n; // 1
        int y1 = i - x1; // 0
        int y0 = (y1 + n) % (size); // 100
        int y2 = (y1 + size - n) % (size); // 9900

        int lx1 = li % ln; // 1
        int ly1 = li - lx1; // 6
        int ly0 = (ly1 + ln); // 12
        int ly2 = (ly1 - ln); // 0

        shared_mem[lx1 + ly1] = in[x1 + y1]; // [7] = [1]
        shared_mem[lx1 + ly0] = in[x1 + y0]; // [13] = [9900]
        shared_mem[lx1 + ly2] = in[x1 + y2]; // [1] = [100]
        
        if (threadIdx.x == 0) {
            int x0 = (x1 + n - 1) % n;
            int x2 = (x1 + BLOCK_SIZE) % n; 


            int lx_init = (lx1 - 1); // 0
            int lx_fin = (lx1 + BLOCK_SIZE); // 5
            
            shared_mem[lx_init + ly0] = in[x0 + y0];
            shared_mem[lx_init + ly1] = in[x0 + y1];
            shared_mem[lx_init + ly2] = in[x0 + y2];

            shared_mem[lx_fin + ly0] = in[x2 + y0];
            shared_mem[lx_fin + ly1] = in[x2 + y1];
            shared_mem[lx_fin + ly2] = in[x2 + y2];
        }

        li += 3*ln;
    }
    __syncthreads();

    li = threadIdx.x + ln + 1; // 5
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {

        int x1 = i % n;
        int y1 = i - x1;

        int lx1 = li % ln; // 5
        int ly1 = li - lx1; // 14
        int lx0 = lx1 - 1;
        int lx2 = lx1 + 1;
        int ly0 = ly1 + ln;
        int ly2 = ly1 - ln;
        int alive_neighbours = shared_mem[lx0 + ly0] + shared_mem[lx0 + ly1] + shared_mem[lx0 + ly2] + shared_mem[lx1 + ly0] + shared_mem[lx1 + ly2] + shared_mem[lx2 + ly0] + shared_mem[lx2 + ly1] + shared_mem[lx2 + ly2];
        out[x1 + y1] = (alive_neighbours == 3 || (alive_neighbours == 2 && shared_mem[lx1 + ly1])) ? 1 : 0;
        //out[10000*10000+1] = 0;
        li += 3*ln;
    } 
}


void random_data(vector<cell>& data) {

    // Init seed
    seed_seq seq{ 123456 };
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

int main() {
    const long long N = 10000;
    const long long M = 10000;
    const long long size = sizeof(cell) * N * M;
    vector<cell> h_in(N*M), h_out(N*M);
    cell* d_in, * d_out;

    cudaMalloc((void**)&d_in, size);
    cudaMalloc((void**)&d_out, size);

    // Inicializar la data
    auto t_start = chrono::high_resolution_clock::now();
    random_data(h_in);
    auto t_end = chrono::high_resolution_clock::now();
    auto t_data = chrono::duration<double, milli>(t_end - t_start).count();

    cout << "INITIAL STATE:" << endl;
    display(h_in, N, M);
    cout << endl;

    // Copiar host a device
    t_start = chrono::high_resolution_clock::now();
    cudaMemcpy(d_in, h_in.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_out, h_out.data(), size, cudaMemcpyHostToDevice);
    t_end = chrono::high_resolution_clock::now();
    auto t_host_to_device = chrono::duration<double, milli>(t_end - t_start).count();


    // Correr la simulacion
    t_start = chrono::high_resolution_clock::now();
    for (int i = 0; i < ITER; i++) {
        iteration_local <<< (N*M) / BLOCK_SIZE, BLOCK_SIZE >> > (d_in, d_out, N, M);
        swap(d_in, d_out);
	}
    t_end = chrono::high_resolution_clock::now();
    auto t_exec = chrono::duration<double, milli>(t_end - t_start).count();


    // Copiar device a host
    t_start = chrono::high_resolution_clock::now();
    cudaMemcpy(h_in.data(), d_in, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_out.data(), d_out, size, cudaMemcpyDeviceToHost);
    t_end = chrono::high_resolution_clock::now();
    auto t_device_to_host = chrono::duration<double, milli>(t_end - t_start).count();

    cout << "FINAL STATE: \n";
    display(h_in, N, M);
    cout << endl;


    cudaFree(d_in); cudaFree(d_out);

	cout << "Time to create the data: " << t_data << " ms\n";
    cout << "Time to copy data from host to device: " << t_host_to_device << " ms\n";
	cout << "Time to execute the simulation: " << t_exec << " ms\n";
    cout << "Time to copy data from device to host: " << t_device_to_host << " ms" << endl;
    cout << "Total time: " << t_data + t_exec + t_device_to_host + t_host_to_device << " ms" << endl;

	return 0;
}