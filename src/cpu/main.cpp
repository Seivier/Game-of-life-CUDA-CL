#include <chrono>
#include <iostream>
#include <vector>
#include <random>
#include <fstream>

const int ITER = 500;
using cell = unsigned char;
using namespace std;


int countNeighbours(vector<cell>& in, int x0, int x1, int x2, int y0, int y1, int y2) {
    return in[y0 + x0] + in[y0 + x1] + in[y0 + x2] + in[y1 + x0] + in[y1 + x2] + in[y2 + x0] + in[y2 + x1] + in[y2 + x2];
}


void iteration(vector<cell>& in, vector<cell>& out, int n, int m) {
    for (int i = 0; i < m; i++) {
        int y0 = ((i + m - 1) % m) * n;
        int y1 = i * n;
        int y2 = ((i + 1) % m) * n;
        for (int j = 0; j < n; j++) {
            int x0 = (j + n - 1) % n;
            int x2 = (j + 1) % n;
            int alive_neighbours = countNeighbours(in, x0, i, x2, y0, y1, y2);
            out[y1 + j] = alive_neighbours == 3 || (alive_neighbours == 2 && in[y1 + j]) ? 1 : 0;
        }
    }
}

void iteration_if(vector<cell>& in, vector<cell>& out, int n, int m) {
	for (int i = 0; i < m; i++) {
		int y0 = ((i + m - 1) % m) * n;
		int y1 = i * n;
		int y2 = ((i + 1) % m) * n;
		for (int j = 0; j < n; j++) {
			int x0 = (j + n - 1) % n;
			int x1 = i;
			int x2 = (j + 1) % n;
			int alive_neighbours = 0;
			if (in[x0 + y0]) alive_neighbours++;
			if (in[x0 + y1]) alive_neighbours++;
			if (in[x0 + y2]) alive_neighbours++;
			if (in[x1 + y0]) alive_neighbours++;
			if (in[x1 + y2]) alive_neighbours++;
			if (in[x2 + y0]) alive_neighbours++;
			if (in[x2 + y1]) alive_neighbours++;
			if (in[x2 + y2]) alive_neighbours++;
			out[y1 + j] = alive_neighbours == 3 || (alive_neighbours == 2 && in[y1 + j]) ? 1 : 0;
		}
	}
}



void random_data(vector<cell>& data) {
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


void simulate(const int N, const int M, bool if_version) {
	vector<cell> in(N*M), out(N*M);

	// Inicializar la data
	auto t_start = chrono::high_resolution_clock::now();
	random_data(in);
	auto t_end = chrono::high_resolution_clock::now();
	auto t_data = chrono::duration<double, milli>(t_end - t_start).count();

//	cout << "INITIAL STATE:" << endl;
//	display(in, N, M);
//	cout << endl;


	// Correr la simulacion
	t_start = chrono::high_resolution_clock::now();
	for (int i = 0; i < ITER; i++) {
		if (if_version)
			iteration_if(in, out, N, M);
		else
			iteration(in, out, N, M);
		swap(in, out);
	}
	t_end = chrono::high_resolution_clock::now();
	auto t_sim = chrono::duration<double, milli>(t_end - t_start).count();

//	cout << "FINAL STATE: \n";
//	display(in, N, M);
//	cout << endl;

	string filename = "cpu.csv";

	fstream file;
	file.open(filename, ios::out | ios::app);

	file << N << "," << M << "," << t_data << "," << t_sim << "," << t_data + t_sim << ","
		<< if_version << endl;
	file.close();

//	cout << "Time to create the data: " << t_data << " ms\n";
//	cout << "Time to execute the simulation: " << t_sim << " ms\n";
//	cout << "Total time: " << t_data + t_sim << " ms\n";
	display(in, N, M);
}

int main(int argc, char* argv[]) {
	if (argc != 4) {
		cout << "Usage: " << argv[0] << " N M VERSION" << endl;
		return 1;
	}

	int N = atoi(argv[1]);
	int M = atoi(argv[2]);
	bool if_version = atoi(argv[3]);
	simulate(N, M, if_version);
}