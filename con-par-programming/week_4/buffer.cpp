#include <iostream>
#include <thread>
#include <unistd.h>
#include <condition_variable>
#include <queue>

/*
@author ysbecca
*/
using namespace std;

mutex m;
queue<int> *buffer;
condition_variable cv;

// Upper bound on number of values queue can hold
int BUFFER_SIZE = 5;
int NUM_THREADS = 1; // We will have 2 * NUM_THREADS total, consumers and producers

// Next number to generate for producers
int next_val;
// Expected counter for consumers
int expected_val;

void producer_thread();
void consumer_thread();

int main(int argc, char **argv) {

  thread p_ids[NUM_THREADS];
  thread c_ids[NUM_THREADS];

  next_val, expected_val = 0;
  buffer = new queue<int>;
  for (int i = 0; i < NUM_THREADS; ++i) {
    p_ids[i] = thread(producer_thread);
    c_ids[i] = thread(consumer_thread);
  }

  for (int i = 0; i < NUM_THREADS; ++i) {
    p_ids[i].join();
    c_ids[i].join();
  }

  return 0;
}

void producer_thread() {
  while (true) {
    lock_guard<mutex> gate(m);
    next_val++;
    buffer.push(next_val);
    cv.notify_one();
  }
}

void consumer_thread() {
  // while(true) {
  //   unique_lock<mutex> gate(m);
  //   cv.wait(gate, [] { return !data.empty(); });
  //   data.pop();
  //   gate.unlock();
  //   // do something with (v) now
  // }
}
