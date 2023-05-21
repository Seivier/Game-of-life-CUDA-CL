#include <chrono>
#include <iostream>
#include <vector>
#include <random>

const int ITER = 1000;
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


void random_data(vector<cell>& data) {
	random_device rd;
	default_random_engine eng(rd());
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
    int const N = 1000;
    int const M = 1000;
    vector<cell> in(N*M), out(N*M);

    // Inicializar la data
    auto t_start = chrono::high_resolution_clock::now();
    random_data(in);
    auto t_end = chrono::high_resolution_clock::now();
    auto t_data = chrono::duration<double, milli>(t_end - t_start).count();

    cout << "INITIAL STATE:" << endl;
    display(in, N, M);
    cout << endl;


    // Correr la simulacion
    t_start = chrono::high_resolution_clock::now();
    for (int i = 0; i < ITER; i++) {
		iteration(in, out, N, M);
		swap(in, out);
	}
    t_end = chrono::high_resolution_clock::now();
	auto t_sim = chrono::duration<double, milli>(t_end - t_start).count();

    cout << "FINAL STATE: \n";
    display(in, N, M);
    cout << endl;


	cout << "Time to create the data: " << t_data << " ms\n";
	cout << "Time to execute the simulation: " << t_sim << " ms\n";
	cout << "Total time: " << t_data + t_sim << " ms\n";

   
	return 0;
}