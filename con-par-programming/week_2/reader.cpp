
#include <iostream>
#include <fstream>
#include <thread>
#include <unistd.h>
#include <tgmath.h>

/*
@author ysbecca
*/

using namespace std;

void counter(int index, unsigned char data[], int *total_count);

const int NUM_THREADS = 8;

int main(int argc, char **argv)
{
  std::ifstream src("data.txt");
  unsigned int val;
  unsigned char data[1 << 18];
  int size = 0;

  if (!src) {
      std::cerr << "Cannot open data file." << std::endl;
      exit(1);
  }
  while (src >> val)
      data[size++] = (unsigned char)val;
  std::cout << "Read " << size << " values from file." << std::endl;

  cout << "Starting threads..." << endl;

  // threads and the counter array
  thread ids[NUM_THREADS];
  int *total_count = new int[NUM_THREADS];

	for (int i = 0; i < NUM_THREADS; ++i) {
    // pass total_count by reference so they all alter the same data
		ids[i] = thread(counter, i, ref(data), ref(total_count));
	}

	for (int i = 0; i < NUM_THREADS; ++i) {
		ids[i].join();
	}

  int final_sum = 0;
  for (int i = 0; i < NUM_THREADS; i++) {
    final_sum += total_count[i];
  }

  cout << "FINAL SUM: " << final_sum << endl;
  cout << "PERCENT 0's IN FILE: " << float(final_sum) / float(size) * 100 << endl;
  return 0;
}


void counter(int index, unsigned char data[], int *total_count) {

  int start = index * pow(2, 18) / NUM_THREADS;
  int end = (index + 1) * pow(2, 18) / NUM_THREADS;

  int partial_sum = 0;
  for (int i = start; i < end; i++) {
    if(data[i] == 0) {
      partial_sum++;
    }
  }

  total_count[index] = partial_sum;
}
