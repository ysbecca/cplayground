/*
@author ysbecca Rebecca Young [sc16rsmy]
@description Sorting a large array in-memory using concurrent threads.

COMP5811 Parallel & Concurrent Programming
University of Leeds 2016-2017
Prof. David Duke
*/

#include <iostream>
#include <unistd.h>
#include <tgmath.h>
#include <stdlib.h>
#include <time.h>
#include <thread>

using namespace std;

// FUNCTION HEADERS
void sorter(int index, int data[]);
void merger(int index, int parts, int data[], int merged[]);
void print_data(int data[]); // For when you want to see printed output.

// Number of threads to run.
const int NUM_THREADS = 32;
// Default size of array to sort.
const int ARRAY_SIZE = pow(2, 19);

int main(int argc, char **argv) {
	// Create an array and populate it.
	int *data = new int[ARRAY_SIZE];
	int *merged = new int[ARRAY_SIZE];

	srand(time(NULL)); // Pseudo-random time.
	for (int i = 0; i < ARRAY_SIZE; ++i) {
		data[i] = rand() % 100;
	}
	cout << "Created an array of size " << ARRAY_SIZE << ", full of random ints." << endl;
	cout << "Creating " << NUM_THREADS << " threads to work on sorting it..." << endl;

	// Create threads and task them with sorting, then rejoin.
	thread ids[NUM_THREADS];
	for (int i = 0; i < NUM_THREADS; ++i) {
		ids[i] = thread(sorter, i, ref(data));
	}
	for (int i = 0; i < NUM_THREADS; ++i) {
		ids[i].join();
	}

	cout << "Merging..." << endl;

	// For as long as we have more than one part, set threads merging.
	for(int parts = NUM_THREADS / 2; parts >= 1; parts = parts / 2) {
		thread *merging_ids = new thread[parts];
		for (int i = 0; i < parts; ++i) {
			merging_ids[i] = thread(merger, i, parts, ref(data), ref(merged));
		}
		for (int i = 0; i < parts; ++i) {
			merging_ids[i].join();
		}
		delete [] merging_ids;
		for (int i = 0; i < ARRAY_SIZE; ++i) {
			data[i] = merged[i]; // Update data array.
		}
	}

	cout << "Checking result..." << endl;
	// Check result and display.
	int is_correct = 1;
	for (int i = 0; i < ARRAY_SIZE - 1; ++i) {
		if(data[i] > data[i + 1]) {
			is_correct = 0;
			break;
		}
	}
	if(is_correct)
		cout << "Whoopie! Array is sorted." << endl;
	else
		cout << "Oups! ERROR: the array is not sorted." << endl;

	delete [] merged; // Tidy up.
	delete [] data;
	return 0;
}

/*
Sorts the designated portion of the array by a simple in-place insertion sort.
- Parameters:
	- index: thread number
	- data: reference to the main data structure to sort
*/
void sorter(int index, int data[]) {
	int start = index * ARRAY_SIZE / NUM_THREADS;
	int end = (index + 1) * ARRAY_SIZE / NUM_THREADS;

	for (int i = start; i < end; ++i) {
		int j = i; // Moving index.
		int temp;
		while(j > start && data[j - 1] > data[j]) {
			temp = data[j - 1];
			data[j - 1] = data[j]; // Swap.
			data[j] = temp;
			j--;
		}
	}
}

/*
Merge sorts two designated portions of the array.
- Parameters:
	- index: thread number
	- parts: how many parts the threads are currently merging
	- data: reference to the main data structure
	- merged: reference to the copied array to put merge into
*/
void merger(int index, int parts, int data[], int merged[]) {
	int part_size = ARRAY_SIZE / parts; // These values are mostly to help me think straight.
	int end = index * part_size + part_size;
	int start = index * part_size;
	int middle = start + (part_size / 2);

	int ptr = start; // Location of ptr to new merged array.
	int p1 = start;
	int p2 = middle;
	while(ptr < end && p1 < middle && p2 < end) {
		if(data[p1] < data[p2]) {
			merged[ptr] = data[p1];
			p1++;
		} else {
			merged[ptr] = data[p2];
			p2++;
		}
		ptr++;
	}
	if(p1 < middle) {
		for (int i = p1; i < middle; ++i) {
			merged[ptr] = data[i];
			ptr++;
		}
	} else if(p2 < end) {
		for (int i = p2; i < end; ++i) {
			merged[ptr] = data[i];
			ptr++;
		}
	} // If both reached the end, then merged[] is full.
}

// Printing function for testing.
void print_data(int data[]) {
	for (int i = 0; i < ARRAY_SIZE; ++i) {
		cout << data[i] << ' ';
	}
	cout << endl;
}
