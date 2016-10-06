#include <iostream>
#include <thread>
#include <unistd.h>

using namespace std;

/*
@author ysbecca
1. Forking (creates copy of entire process memory)

ParentID = PID
ChildID = 0

2. Threads (separate execution path, shared code and data)

*/

void counter(int index) {
	cout << "Hello from thread number " << index << "." << endl;
}

int main() {

	thread ids[5];
	
	for (int i = 0; i < 5; ++i)
	{
		ids[i] = thread(counter, i);
	}

	cout << "Hello from the parent." << endl;

	for (int i = 0; i < 5; ++i)
	{
		ids[i].join();
	}

	/*
	pid_t child;

	child = fork();
	if(child == 0)
		cout << "Child process running" << endl;
	else
		cout << "Parent process running" << endl;
	*/
	return 0;

}