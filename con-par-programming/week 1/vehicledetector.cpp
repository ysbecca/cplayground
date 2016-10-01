#include <iostream>

using namespace std;


/*
@author ysbecca Rebecca Stone

A roadside vehicle detector has been installed in a busy road to monitor traffic. The detector itself is a
 simple device that emits a sequence of numeric signals:
0, for every hour that has passed with the device operational; 1, when it detects the passage of a car
2, when it detects the passage of a cycle
3, when it detects a bus
-1, when it shuts down from detecting.
So, one possible stream of signals is the following (spacing carries no meaning, it is just to make it easier 
to read):
1,1,2,1,3,1,0, 3,3,1,1,1,2,0, 1,1,0, 0, 3,3,1,-1
Your task is to write a program that receives signals and carries out simple statistical analysis. For 
this exercise, the detector itself will be replaced by the user; the program should prompt the user for 
signal values until it receives the shut-down signal. Thereafter, the program should generate two 
summary reports:
For each hour of operation, the number of cars, buses and cycles detected. 
Over the entire operating period, the average numbers of cars, buses and cycles per hour. So for 
example, processing the above stream should result in the following output:

Hour Cars Buses Cycles 1411 2321 3200 4000 5120
Daily averages: Cars: 2.0 Buses: 1.0 Cycles: 0.4

*/


struct HourStats {
	int hour;
	int cars;
	int cycles;
	int buses;
	HourStats *next;
};

int main() {

	int cars = 0, cycles = 0, buses = 0, hours = 0;
	int input;
	HourStats *first_hour;
	HourStats *last_hour;
	
	cout << "Ready to receive input..." << endl;
	// loop to take user input
	do {
		cin >> input;

		switch(input) {
			case 0:
			case -1:
			{
				HourStats *newHour = new HourStats();
				newHour->hour = hours;
				newHour->cars = cars;
				newHour->cycles = cycles;
				newHour->buses = buses;
				
				if(hours == 0) {
					first_hour = newHour;
					last_hour = first_hour;
				} else {
					last_hour->next = newHour;
					last_hour = newHour;
				}
				cars = 0;
				cycles = 0;
				buses = 0;

				hours++;
				break;
			}
			case 1:
				cars++;
				break;
			case 2:
				cycles++;
				break;
			case 3:
				buses++;
				break;
			default:
				cout << "That was not a valid signal." << endl;
		}


	} while(input >= 0);

	HourStats *curr = first_hour;

	int total_cars = 0, total_buses = 0, total_cycles = 0;
	cout << "Hour  Cars  Buses  Cycles" << endl;
	while(curr) {
		// print each struct
		cout << curr->hour << "     " << curr->cars << "     ";
		cout << curr->buses << "     " << curr->cycles << endl;
		total_cycles += curr->cycles;
		total_buses += curr->buses;
		total_cars += curr->cars;

		curr = curr->next;
	}
	curr = first_hour;
	cout << "Daily averages" << endl;

	cout << "Cars: " << total_cars/hours << endl;
	cout << "Buses: " << total_buses/hours << endl;
	cout << "Cycles: " << total_cycles/hours << endl;
}




