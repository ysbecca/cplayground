// This is start of the header guard.  
#ifndef ADD_LIST
#define ADD_LIST
 
using namespace std;

/*
@author ysbecca Rebecca Stone

Implement a generic list class that supports the following operations:
• create an empty list
• insert a value at the front of the list
• add a value to the end of the list
• find a value in the list, remembering its position
• insert a value after the last remembered position; if the last value searched for was not found, then the
new value is added to the end of the list.
• delete the value at the last remembered position; if the last value searched for was not found, then delete
the last value in the list.
• print the list
*/

class List {
    private:
    	class Node {
			public:
				int value;
				Node *next;
		};

    	int size;
    	int remembered_node;
    	Node *first;


	public:
		List() {
			size = 0;
			first = NULL;
		}
		// memory deallocation for nodes should be automatic bc we have ptrs?
		~List() {}

		// insert at front
		void insert(int val) {
			Node *newNode = new Node;
			newNode->value = val;
			newNode->next = first;

			first = newNode;
			size++;
		};

		// insert at end
		void add(int val) {
			Node *newNode = new Node;
			newNode->value = val;
			newNode->next = NULL;

			if(!first) {
				insert(val);
			} else {
				Node *curr = first;
				// get to last node
				while(curr->next) {
					curr = curr->next;
				}
				curr->next = newNode;
				size++;
			}
		};

		// find value and remember the position
		int find(int val) {
			int pos = size;
			Node *curr = first;

			if(!first ) { return -1; }
			int index = 0;

			while(curr) {
				if(curr->value == val) {
					cout << "Found value at : " << index << endl;
					pos = index;
					break;
				}
				index++;
				curr = curr->next;
			}

			remembered_node = pos;
			return pos;
		};

		// insert after remembered position
		int insert_remembered(int val) {
			if(!first ) { return -1; }
			int index = 0;
			Node *curr = first;

			while(index < remembered_node - 1) {
				curr = curr->next;
				index++;
			}
			// now curr points at remembered node
			Node *newNode = new Node;
			newNode->value = val;
			newNode->next = curr->next;
			curr->next = newNode;
			size++;
			return remembered_node;
		};

		// delete node at remembered position
		int delete_remembered() {
			if(!first ) { return -1; }
			int index = 0;
			Node *curr = first;

			while(index < remembered_node - 1) {
				curr = curr->next;
				index++;
			}
			Node *temp = curr->next;
			curr->next = temp->next;
			delete temp;

			return size--;
		}

		void print_remembered_node() {
			cout << "Remembered node: " << remembered_node << endl;
		}

		void print() {
			Node *curr = first;
			do
			{
				cout << curr->value << " ";
				curr = curr->next;

			} while(curr);
			cout << endl;
		};
};


#endif