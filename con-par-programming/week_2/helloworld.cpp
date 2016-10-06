#include <iostream>
#include <thread>
#include <unistd.h>

using namespace std;

/*
@author ysbecca
*/

void hello() {
	cout << "Hello world from child thread." << endl;
}

int main() {

	thread id = thread(hello);


  id.join();
	cout << "Hello from the parent." << endl;



	return 0;

}
