
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
const int TAYLOR_TERMS = 9;
const int NUM_THREADS = 12;
#define PI 3.14159265

float *angles;

// function declarations
void compute_sine(int index);

int main(int argc, char **argv) {

  cout << "Starting threads..." << endl;

  // threads and the angles array
  thread ids[NUM_THREADS];
  angles = new float[INTERVALS]();

	for (int i = 0; i < NUM_THREADS; ++i) {
		ids[i] = thread(compute_sine, i);
	}

  // wait for them all to finish
	for (int i = 0; i < NUM_THREADS; ++i) {
		ids[i].join();
	}

  // print table of sines
  double angle;
  for (int i = 0; i < INTERVALS; i++) {
    angle = i * PI / 6.0;
    cout << "Sin(" << i << " * pi / 6) = " << angles[i]*180 / PI << " degrees" << endl;
  }

  return 0;
}


void compute_sine(int index) {

  float total = 0.0;
  float angle = index * PI / 6.0;
  int computed_i;
  long factorial = 1;

  for (int i = 1; i <= TAYLOR_TERMS; i++) {
    computed_i = 2*i + 1;

    if(computed_i != 1)
      factorial *= computed_i * (computed_i - 1);

    if(i % 2 == 0) {
      // even term - negative
      total -= pow(angle, computed_i) / factorial;
    } else {
      // odd term - positive
      total += pow(angle, computed_i) / factorial;
    }
  }

  angles[index] = total;
}
