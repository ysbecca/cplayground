#include <iostream>
#include "w1.h"

using namespace std;

/*
@author ysbecca Rebecca Stone
*/

int main()
{
    std::cout << "Worksheet 1" << std::endl;

    // for running each problem individually
    // p1();
    // p2();
    int numbers[8] = {4, 2, 5, 1, 0, 6, 3, 7};
    // p3(numbers);
    // p5(numbers);
    int array[] = {1, 2, 1, 3, 1, 2, 1, 4, 1, 2, 1, 3, 1, 2, 6, 1, 5};
    // p7(array, sizeof(array) / sizeof(array[0]));
    

    return 0;
}

/*
Write a program that removes duplicates from an array, and prints the output. 
For example, if the initial array is 1 2 1 3 1 2 1 4 1 2 1 3 1 2 1 1 5, 
the output should be 1 2 3 4 5.
*/
void p7(int *array, int size)
{
    int max_size = size;

    // TOTAL BYTES: sizeof(array) & ELEMENT SIZE: sizeof(array[0])
    int *dups = new int[max_size]();
    int index = 0;
    int found = 0;

    for (int i = 0; i < max_size; ++i)
    {
        for (int j = 0; j <= index; ++j)
        {
            // see if array[i] is in dups.
            if (array[i] == dups[j])
            {
                found = 1;
                break;
            }
        }
        if(!found)
        {
            // add to dups
            dups[index] = array[i];
            index++;
        }
        found = 0; // reset
    }

    for (int i = 0; i < index; ++i)
    {
        std::cout << dups[i] << " ";
    }
    std::cout << std::endl;

}

/*
Write a program that will write out the contents and memory address of the values held in a one dimensional 
array of integers. It should produce a neatly formatted table showing: array index, value at that index, and 
address of that array entry.
*/

void p5(int numbers[8]) 
{

    for (int i = 0; i < 8; ++i)
    {
        std::cout << "Array contents: " << std::endl;
        std::cout << i << " | " << numbers[i] << " | " << &numbers[i] << std::endl;
    }

}


/*
Implement insertion sorting on an array. Here is a reminder of how it works:
• if the array has length <= 1, it is sorted.
• To sort an array where elements 0 .. i−1 are sorted, and the remaining elements i..n−1 are unsorted,
• take the ith element (call it “temp”), and set j = i.
• If the (j−1)st element is smaller than temp, we are done.
• Else, we copy the (j−1) st element up one position, storing it at position j, and then decrease j by 1.
• Continue doing this until either you find the position for temp, or j reaches 0, in which case we are storing temp at the front of the list.
*/
void p3(int numbers[8]) 
{

    for (int i = 1; i < 8; ++i)
    {
        // look at element i
        int prev = i-1;
        int curr = numbers[i];
        int curr_pt = i;

        // compare to previous
        while(curr < numbers[prev])
        {
            // while it's bigger, swap it with the previous number
            int temp = numbers[prev];
            numbers[prev] = curr;
            numbers[curr_pt] = temp;
            curr_pt--;
            if (prev == 0)
            {
                break;
            } else
            {
                prev--;
            }
        }
    }

    // print array contents
    for (int i = 0; i < 8; ++i)
    {
        std::cout << numbers[i] << " ";
    }
    std::cout << std::endl;

}


/*
Write a program that prints a string, then prints the reverse of that string, e.g. output might be
Hello, world! !dlrow ,olleH
*/
void p2() 
{

    std::string name;
    std::cout << "Enter your name: " << std::endl;
    std::cin >> name;

    for (int i = name.size() - 1; i >= 0; --i)
    {
        std::cout << name[i];
    }
    std::cout << std::endl;

}

/* 
Write a program that produces the following output:   
*
**
***
****
***** 
 
Ensure that you are able to implement this using two nested for-loops.
*/
void p1() 
{

    int dim = 5;

    for (int i = 0; i < dim; ++i)
    {
        for(int j = 0; j < i + 1; ++j)
        {
            std::cout << "*";
        }

        std::cout << "" << std::endl;
    }

}
