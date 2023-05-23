#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 300
#define CL_TARGET_OPENCL_VERSION 300

#include <chrono>
#include <iostream>
#include <vector>
#include <random>
#include <fstream>
#include <CL/cl.hpp>

#define BLOCK_SIZE 512
#define N 5120
#define M 5120

const int ITER = 500;
using cell = unsigned char;
using namespace std;


const char* normSource = R"(
typedef unsigned char cell;

__kernel void iteration(__global cell* in, __global cell* out, int n, int m) {
    int size = n * m;
    for (int i = get_global_id(0); i < size; i += get_global_size(0)) {
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
)";

const char* ifSource = R"(
typedef unsigned char cell;

__kernel void iteration_if(__global cell* in,__global cell* out, int n, int m) {
	int size = n * m;
	for (int i = get_global_id(0); i < size; i += get_global_size(0)) {
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
)";

void random_data(vector<cell>& data) {
	random_device rd;
	seed_seq seq{123456};
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

int main()
{
	for (int tries = 0; tries < 1; tries++)
	{
		cl_int err;

		vector<cl::Platform> platforms;
		cl::Platform::get(&platforms);
		if (platforms.empty())
		{
			cout << "No platforms found!" << endl;
			return 1;
		}
		cl::Platform platform = platforms[0];
		cout << "Using platform: " << platform.getInfo<CL_PLATFORM_NAME>() << endl;

		vector<cl::Device> devices;
		platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
		if (devices.empty())
		{
			cout << "No devices found!" << endl;
			return 1;
		}
		cl::Device device = devices[0];
		cout << "Using device: " << device.getInfo<CL_DEVICE_NAME>() << endl;

		cl::Context context(device);

		cl::CommandQueue queue(context, device);


		cl::Program iter(context, normSource);
		cl::Program iter_if(context, ifSource);
		err = iter.build();
		if (err != CL_SUCCESS)
		{
			cout << "Build error: " << err << endl;
			cout << iter.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << endl;
			return 1;
		}

		err = iter_if.build();
		if (err != CL_SUCCESS)
		{
			cout << "Build error: " << err << endl;
			cout << iter_if.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << endl;
			return 1;
		}


		cl::Kernel kernel(iter, "iteration", &err);
		if (err != CL_SUCCESS)
		{
			cout << "Kernel error: " << err << endl;
			return 1;
		}

		vector<cell> in(N * M);
		vector<cell> out(N * M);
		cl::Buffer inBuf(context, CL_MEM_READ_WRITE, N * M * sizeof(cell));
		cl::Buffer outBuf(context, CL_MEM_READ_WRITE, N * M * sizeof(cell));

		// Init
		auto t_start = chrono::high_resolution_clock::now();
		random_data(in);
		auto t_end = chrono::high_resolution_clock::now();
		auto t_data = chrono::duration_cast<chrono::milliseconds>(t_end - t_start).count();

		cout << "INITIAL STATE:" << endl;
		display(in, N, M);
		cout << endl;

		// Copy host to device
		t_start = chrono::high_resolution_clock::now();
		err = queue.enqueueWriteBuffer(inBuf, CL_TRUE, 0, N * M * sizeof(cell), in.data());
		err |= queue.enqueueWriteBuffer(outBuf, CL_TRUE, 0, N * M * sizeof(cell), out.data());
		queue.finish();
		t_end = chrono::high_resolution_clock::now();
		auto t_host_to_device = chrono::duration_cast<chrono::milliseconds>(t_end - t_start).count();

		cl::NDRange globalSize((N * M) / BLOCK_SIZE);
		cl::NDRange localSize(BLOCK_SIZE);

		// Exec
		t_start = chrono::high_resolution_clock::now();
		for (int i = 0; i < ITER; i++)
		{
			err = kernel.setArg(i % 2, inBuf);
			err |= kernel.setArg((i + 1) % 2, outBuf);
			err |= kernel.setArg(2, N);
			err |= kernel.setArg(3, M);
			err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalSize, localSize);
		}
		queue.finish();
		t_end = chrono::high_resolution_clock::now();
		auto t_exec = chrono::duration_cast<chrono::milliseconds>(t_end - t_start).count();

		// Copy device to host
		t_start = chrono::high_resolution_clock::now();
		err = queue.enqueueReadBuffer(inBuf, CL_TRUE, 0, N * M * sizeof(cell), in.data());
		err = queue.enqueueReadBuffer(outBuf, CL_TRUE, 0, N * M * sizeof(cell), out.data());
		queue.finish();
		t_end = chrono::high_resolution_clock::now();
		auto t_device_to_host = chrono::duration_cast<chrono::milliseconds>(t_end - t_start).count();

		cout << "FINAL STATE:" << endl;
		display(in, N, M);
		cout << endl;

		cout << "Time to create the data: " << t_data << " ms\n";
		cout << "Time to copy data from host to device: " << t_host_to_device << " ms\n";
		cout << "Time to execute the simulation: " << t_exec << " ms\n";
		cout << "Time to copy data from device to host: " << t_device_to_host << " ms" << endl;
		cout << "Total time: " << t_data + t_exec + t_device_to_host + t_host_to_device << " ms" << endl;
	}
}
