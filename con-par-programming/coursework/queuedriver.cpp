/*
@author ysbecca Rebecca Stone (Young) [sc16rsmy]
@description Thread-safe priority queue driver
*/

#include <iostream>
#include <unistd.h>
#include <stdlib.h>
#include <thread>
#include <time.h>
#include "pqueue.h"


using namespace std;

// Number of threads to run.
const int NUM_THREADS = 2;
// Max capacity of the heap.
const int MAX_HEAP_CAPACITY = 16; // 250000;

int main(int argc, char **argv) {

	cout << "Priority Queue [sc16rsmy] Driver" << endl;

	srand ( time(NULL) );
	PriorityQueue<int> *p = new PriorityQueue<int>(MAX_HEAP_CAPACITY);
	thread ids[NUM_THREADS];

	int block_size = MAX_HEAP_CAPACITY / NUM_THREADS;

	// Task threads with inserting large arrays of value, priority pairs.
	for(int i = 0; i < NUM_THREADS; i++) {
		// Generating the data
		int *values = new int[block_size];
		int *priorities = new int[block_size];

		for(int i = 0; i < block_size; ++i) {
			values[i] = i; // TODO replace with the same random value as priority
			priorities[i] = rand() % MAX_HEAP_CAPACITY;
		}

		ids[i] = thread(&PriorityQueue<int>::mass_insert, p, priorities, values, block_size);
	}

	for(int i = 0; i < NUM_THREADS; i++) {
		ids[i].join();
	}

	// p->print(); // Just for testing, or when you really want 50 000 nodes printed to your terminal.
	p->isCorrect();
	
	// Now sequentially delete all the nodes (one thread).
	p->mass_delete(MAX_HEAP_CAPACITY);

	p->print();
	p->isCorrect();
	return 0;
}











