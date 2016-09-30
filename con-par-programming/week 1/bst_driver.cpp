#include <iostream>
#include "binarysearchtree.h"


using namespace std;

/*
@author ysbecca Rebecca Stone

A driver that interactively prompts the user for an operation and any relevant data,
performs the operation, and displays the result.
*/


// example: looking for 7
int sorter(int val) {
	if(val > 7) {
		return -1;
	} else if(val < 7) {
		return 1;
	} else {
		return 0;
	}
}

int main() {

	// integer tree
	BinarySearchTree<int> tree;
	tree.insert(8);
	tree.insert(7);
	tree.insert(3);
	tree.insert(10);
	tree.insert(11);
	tree.insert(9);
	tree.insert(4);
	tree.insert(1);
	tree.print();

	tree.search(sorter);

	tree.deleteTreeNode(7);
	tree.print();

	// char tree
	
	BinarySearchTree<char> secondtree;
	// secondtree.insert('c');
	// secondtree.insert('a');
	// secondtree.insert('b');
	// secondtree.insert('f');
	// secondtree.insert('z');
	// secondtree.insert('w');
	// secondtree.print();
	
	return 0;
}

