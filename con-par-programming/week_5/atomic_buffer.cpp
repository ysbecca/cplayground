#include <atomic>
#include <thread>
#include <unistd.h>
#include <queue>
#include <iostream>


using namespace std;

const int NUM_THREADS = 1;


struct Buffer {
    atomic<int> count;
    atomic<int> expected;

    queue<int> *buffer;

    void push_next(){
    	// get count

        next = count.load();
        while(count.compare_exchange_weak(count, next, ++next)) {
        	
        }
    }
    void read_next(){
        // get expected
        // check and compare that expected = what we expected
        // when it does, update expected.
    }
    int get_count(){
        return count.load();
    }
    int get_expected() {
    	return expected.load();
    }
};

int main(int argc, char **argv) {

	Buffer *buffer = new Buffer();


	thread ids[NUM_THREADS];
    for(int i = 0; i < NUM_THREADS; ++i) {
    	if(i % 2 == 0)
	    	ids[i] = thread(&Buffer::push_next, buffer);
	    else
	    	ids[i] = thread(&Buffer::read_next, buffer);
    }

    for (int i = 0; i < NUM_THREADS; ++i) {
        ids[i].join();
    }

    return 0;
}










