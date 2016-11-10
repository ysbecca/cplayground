/*
@author ysbecca Rebecca Stone (Young) [sc16rsmy]
@description Thread-safe priority queue
*/

#ifndef ADD_PQUEUE
#define ADD_PQUEUE

#include <unistd.h>
#include <stdlib.h>
#include <time.h>
#include <tgmath.h>

using namespace std;
template <class T>


class PriorityQueue {
	private:
    	class Node {
			public:
				T value;
				int priority;
				int tag = -1; // -1 = EMPTY, 0 = AVAILABLE; otherwise PID
				int lock; // Index of mutex lock.

				Node() {
					priority = -1;
					tag = -1;
				}

				Node(T val, int p) {
					value = val;
					priority = p;
				}
		};

		struct ReversedBitCounter {
			unsigned int counter = 0;
			unsigned int reversed = 0;
		};

		ReversedBitCounter heap_size;
		Node *items; // List of all the items in the heap.
		mutex *locks; // All locks. Index 0 lock is heap size lock.
		int max_size;
		
		/* Updates the bit reversed counter by one.
		- Parameters:
			- inc: true to increment, false to decrement the counter.
		- Returns:
			- heap_size->reversed: the bit reversal of the updated counter.
		*/
		int bit_reversed_step(bool inc) {
			int ctr;
			if(inc)
				ctr = ++heap_size.counter;
			else
				ctr = --heap_size.counter;
		    int bits = log2(ctr); // Depth of our heap.
		    int shift = 1 << bits;
		    heap_size.reversed = ctr;
		    for(int i = 1; i < bits; i++) {
		        ctr >>= 1;
		        heap_size.reversed <<= 1;
		        heap_size.reversed |= ctr & 1;
		    }
		    heap_size.reversed &= shift - 1;
		    heap_size.reversed |= 1 << bits; // Add the most significant bit back in.
		    return heap_size.reversed;
		}

	public:
		/* Basic intitialiser */
		PriorityQueue(int size) {  // Root at index 1.
			max_size = size + 1;
			locks = new mutex[max_size];
			items = new Node[max_size];
			for (int i = 0; i < max_size; ++i){
				items[i].lock = i;
			}
		}

		/* Inserts a new item at the bottom of the heap then moves it up until
			heap structure is satisfied. Supports concurrency.
		*/
		void insert(int priority, const T& item) {
			unique_lock<mutex> heap_guard(locks[0]);
			srand(time(NULL)*priority); // Randomly generate a PID.
			int PID = rand();
			int i = bit_reversed_step(true); // Point of insertion.
			
			unique_lock<mutex> node_guard(locks[items[i].lock]); // Acquire lock of node i.
			heap_guard.unlock();
			items[i].value = item;
			items[i].tag = PID;
			items[i].priority = priority;
			node_guard.unlock();

			// Move item towards top of heap while it has higher priority than its parent.
			while(i > 1) {
				int parent = i/2;
				unique_lock<mutex> parent_guard(locks[items[parent].lock]);
				unique_lock<mutex> node_guard(locks[items[i].lock]);

				if(items[parent].tag == 0 && items[i].tag == PID) {
					if(items[i].priority > items[parent].priority) {
						Node temp = items[i]; // Swap item and parent.
						items[i] = items[parent];
						items[parent] = temp;
					} else {
						items[i].tag = 0;
						i = 0;
					}
				} else if(items[parent].tag == -1) {
					i = 0;
				} else if(items[i].tag != PID) {
					i = parent;
				}
				node_guard.unlock();
				parent_guard.unlock();
			}
			if(i == 1) {
				unique_lock<mutex> node_guard(locks[items[i].lock]);
				if(items[i].tag == PID) {
					items[i].tag = 0;
				}
				node_guard.unlock();
			}
		}

		/* Removes the first item from the heap if possible, and replaces the item with one
			from the bottom of the heap, swapping it into a position that makes the heap valid.
			Returns whether the delete was successful or not.
		*/
		bool deleteMin(T& item) {
			if(isEmpty()) // Don't bother if the heap is empty.
				return false;
			unique_lock<mutex> heap_guard(locks[0]); // Heap size lock.
			unique_lock<mutex> root_guard(locks[items[1].lock]);
			item = items[1].value;
			if(heap_size.counter == 1) { // Only 1 item; just return it.
				// cout << "Only one item left. Removing it." << endl;
				items[1].tag = -1;
				--heap_size.counter;
				root_guard.unlock();
				heap_guard.unlock();
				return true;
			}
			// Otherwise, we need a replacement.
			heap_guard.unlock(); 
			int i = bit_reversed_step(false); // Node to replace the root.
			unique_lock<mutex> node_guard(locks[items[i].lock]);
			
			// Swap the top item with the item stored at the bottom. We have root and i locks.
			int root_lock = items[1].lock;
			items[1] = items[i];
			items[1].tag = 0;
			items[i].tag = -1;
			items[i].lock = root_lock; // Swap the locks.
			root_guard.unlock(); // Node_guard holds lock to first item now.
			// Adjust heap by swapping new node into correct position.
			i = 1;
			int left, right, child;
			while(i < max_size/2) {
				left = i*2;
				right = i*2 + 1;
				unique_lock<mutex> left_guard(locks[items[left].lock]);
				unique_lock<mutex> right_guard(locks[items[right].lock]);
				if(items[left].tag == -1) {
					left_guard.unlock();
					right_guard.unlock();
					node_guard.unlock();
					break;
				} else if(items[right].tag == -1 || items[left].priority > items[right].priority) {
					right_guard.unlock();
					child = left;
				} else {
					left_guard.unlock();
					child = right;
				}
				// If child has greater priority than parent, swap; otherwise done.
				bool swap = items[child].priority > items[i].priority;
				if(swap) {
					Node temp = items[child];
					items[child] = items[i];
					items[i] = temp; // Lock gets copied; node_guard still holds lock i.
					i = child; 
				} // Unlock everything for this iteration.
				if(child == left)
					left_guard.unlock();
				else
					right_guard.unlock();
				if(!swap) {
					node_guard.unlock();
					break;
				}
			}
			// cout << "=== State of tree after removing " << item << ": ====" << endl;
			print();
			return true;
		}

		/* Returns whether the heap is empty or not. */
		bool isEmpty() {
			bool isEmpty = false;
			unique_lock<mutex> heap_guard(locks[0]);
				if(heap_size.counter == 0)
					isEmpty = true;
			heap_guard.unlock();
			return isEmpty;
		}

		/* Helper function which one by one inserts n value, priority pairs. */
		void mass_insert(int priorities[], T values[], int n) {
			for (int i = 0; i < n; ++i) {
				insert(priorities[i], values[i]);
			}
		}

		/* Helper function which one by one deletes n values. */
		void mass_delete(int n) {
			T *values = new T[n];
			bool *results = new bool[n];
			for(int i = 0; i < n; ++i) {
				results[i] = deleteMin(values[i]);
			}
		}

		/* A very detailed print of the heap for testing AFTER all threads have been joined. */
		void print() {
			for(int i = 1; i < max_size; ++i) {
				if(items[i].tag != -1) // Last level may be not completely full; don't show blanks.
					cout << items[i].value << ", priority " << items[i].priority 
						<< ", tag " << items[i].tag << ", lock " << items[i].lock << endl;
			}
		}

		/* Determines whether the priority queue is in a correct state and prints detailed results.
			Not intended to be run concurrently; intended to be run after all the operations
			have taken place, since clearly during an insert the heap may not be in a valid state.
		*/
		bool isCorrect() {
			cout << "==== Priority Queue analysis =====" << endl;
			cout << "Heap size: " << heap_size.counter << endl;
			if(heap_size.counter == 0) {
				cout << "Empty heap." << endl;
				return true;
			}
			int depth = floor(log2(heap_size.counter));
			cout << "Depth: " << depth << endl;

			int i = 1; // 1. Check that the priority of the parents >= priority of children.
			bool isOrdered = true;
			while((2*i + 1) < max_size) {
				if(items[i].priority < items[2*i].priority || items[i].priority < items[2*i+1].priority) {
					isOrdered = false;
					cout << "Not ordered! " << items[i].priority << " at index " << i << endl;
					break;
				}
				i++;
			}
			bool isFilled = true; // 2. Check that every level except last is filled.
			for(int i = 1; i < pow(2, depth - 1); ++i) {
				if(items[i].tag == -1) {
					isFilled = false;
					cout << "Not filled! Item " << i << " tag == 0." << endl;
					break;
				}
			}
			if(isOrdered && isFilled) {
				cout << "The priority of every node is greater than that of its children." << endl;
				cout << "Every level except the last layer of the heap is filled." << endl;
				cout << "The priority queue is in a good state!" << endl;
			} else
				cout << "The priority queue is in an INVALID state. Oups." << endl;
			return isOrdered && isFilled;
		}

};

#endif
