
#include <iostream>
#include <fstream>
#include <thread>
#include <unistd.h>
#include <tgmath.h>

/*
@author ysbecca
*/
using namespace std;

const int INTERVALS = 12;
const int TAYLOR_TERMS = 20;
const int NUM_THREADS = 12;
#define PI 3.14159265

float *angles;

// function declarations
void compute_sine(int index);

int main(int argc, char **argv) {

  cout << "Starting threads..." << endl;

  // threads and the angles array
  thread ids[NUM_THREADS];
  angles = float[INTERVALS]();

	for (int i = 0; i < NUM_THREADS; ++i) {
		ids[i] = thread(compute_sine, i);
	}

  // wait for them all to finish
	for (int i = 0; i < NUM_THREADS; ++i) {
		ids[i].join();
	}

  // print table of sines
  for (int i = 0; i < INTERVALS; i++) {
    cout << "Sin(" << "pi/"
  }

  return 0;
}

// 0 1 2 3 4 5
// 1 3 5 7 9 11
// num = 2i+1
void compute_sine(int index {

  float total = 0.0;
  float angle = index * PI / 6.0;
  int computed_i;
  int factorial = 1;

  for (int i = 0; i < TAYLOR_TERMS; i++) {
    computed_i = 2*i + 1;
    if(i == 0)
      factorial = 1;
    else
      factorial *= computed_i * (computed_i - 1);

    if(i % 2 == 0) {
      // even term - positive
      total += pow(angle, computed_i) / factorial;
    } else {
      // odd term - negative
      total -= pow(angle, computed_i) / factorial;
    }
  }

  angles[index] = total;
}
