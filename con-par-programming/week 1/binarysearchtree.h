#ifndef ADD_BST
#define ADD_BST
 
using namespace std;

/*
@author ysbecca Rebecca Stone


Write a generic C++ to implement a binary search tree data structure, providing operations to
• create an empty tree
• insert a value into a tree
• delete a value from a tree
• search the tree for a given value by passing in a user-defined function to be applied to values in the tree; 
the function when applied to a value in the tree should return -1, 0, or 1 depending on whether the tree value 
is less than, equal to, or greater than the target,
• print a representation of the tree suitable for debugging, i.e. which should show both the content and 
structure of the tree.

*/

template <typename T>

class BinarySearchTree {
    private:
    	class TreeNode {
			public:
				T value;
				TreeNode *rightNode;
				TreeNode *leftNode;
				
				TreeNode(T val) {
					value = val;
					rightNode = NULL;
					leftNode = NULL;
				}

		};

		void insertHelper(TreeNode *&node, T val) {
			if(!node) {
				node = new TreeNode(val);
			} else {

				if(val < node->value) {
    				insertHelper(node->leftNode, val);
    			} else { // val >= root.value
    				insertHelper(node->rightNode, val);
    			}
    		}
		}

		void insertSubTree(TreeNode *&node) {}

		void deleteHelper(TreeNode *&node, T val) {
			if(node->leftNode) {
				if(node->leftNode->value == val) {
					TreeNode *rtemp = node->leftNode->rightNode;

					// if node to delete has a left tree, move it up
					if(node->leftNode->leftNode) {
						TreeNode *ltemp = node->leftNode->leftNode;
						node->leftNode = ltemp;	
					}

					if(rtemp) {
						insertSubTree(rtemp);
					}
				} else {
					deleteHelper(node->leftNode, val);
				}
			}
			if(node->rightNode) {
				if(node->rightNode->value == val) {
					TreeNode *rtemp = node->rightNode->rightNode;

					// if node to delete has a left tree, move it up
					if(node->rightNode->leftNode) {
						TreeNode *ltemp = node->rightNode->leftNode;
						node->rightNode = ltemp;	
					}
					
					if(rtemp) {
						insertSubTree(rtemp);
					}
				} else {
					deleteHelper(node->rightNode, val);
				}
			}
		}

		void printHelper(string code, TreeNode *&node, int spaces) {
			if(node) {
				
				for (int i = 0; i < spaces; ++i) {
					cout << "  ";
				}
				cout << code << endl;;
				for (int i = 0; i < spaces; ++i) {
					cout << "  ";
				}
				cout << "( " << node->value << " )" << endl;
				printHelper("/", node->leftNode, spaces-3);
				printHelper("\\", node->rightNode, spaces+3);
			}
		}

		T searchHelper(TreeNode *&node, int (*comp)(T)) {
			if(!node) { 
				cout << "Returning NULL." << endl;
				return T(NULL); 
			}

			int res = comp(node->value);
			cout << "Comparing node->value " << node->value << " with comp function." << endl;
    		if (res < 0) {
    			cout << "Going left" << endl;
    			return searchHelper(node->leftNode, comp);
    		} else if(res > 0) {
    			cout << "Going right" << endl;
    			return searchHelper(node->rightNode, comp);
    		} else {
    			cout << "Searched and found value: " << node->value << endl;
    			return node->value;
    		}
		}

    public:

    	TreeNode *root;

    	BinarySearchTree() {
    		root = NULL;
    	}
    	~BinarySearchTree() {}

    	void insert(T val) {
    		insertHelper(root, val);
    	}

    	void deleteTreeNode(T val) {
    		// TODO what if delete the root node???
  			deleteHelper(root, val);
    	}

    	T search(int (*comp)(T)) {
    		return searchHelper(root, comp);
    	}

    	void print() {
    		printHelper("", root, 15);
    	}

};



#endif