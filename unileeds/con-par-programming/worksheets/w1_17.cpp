#include <iostream>
/*
@author ysbecca Rebecca Stone
*/

#include "w1_17.h"

using namespace std;

// driver to test List class
int main() {

	List list;
	
	// insert values into head of list
	list.insert(3);
	list.insert(5);
	list.insert(9);
	list.insert(21);
	list.insert(24);
	list.insert(74);
	list.insert(100);
	list.print();

	// try adding to the end
	list.add(1);
	list.add(101);
	list.print();

	// find a value
	cout << list.find(21) << endl;
	list.print_remembered_node();

	// insert into remembered position
	list.insert_remembered(66);
	list.print();

	// delete the one we just added in remembered position
	list.delete_remembered();
	list.print();

	// one more add
	list.add(88);
	list.print();

	return 0;
}

