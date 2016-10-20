#include <iostream>
#include <fstream>
#include <cstdlib>
#include <thread>
#include <unistd.h>
#include <tgmath.h>
#include <map>
#include <mutex>
#include <queue>
#include <time.h>
#include "bank.cpp"

int NUM_THREADS = 5;
int NUM_TRANSACTIONS = 10;

int main(int argc, char **argv) {
    cout << "New bank." << endl;
    Bank bank;
    thread ids[NUM_THREADS];

  queue<int> transactions;
    for (int i = 0; i < NUM_TRANSACTIONS; i++) {
      srand (time(NULL));
      transactions.push((rand() % 5) + 1);
    }
    // Create five accounts and do stuff to them
    cout << transactions << endl;

    for (int i = 0; i < NUM_THREADS; i++) {
      switch (transactions.pop()) {
        case 0:
          ids[i] = thread(&Bank::create_account, bank, 100);
          break;
        case 1:
          ids[i] = thread(&Bank::query_account, bank, (rand() % 5));
          break;
        case 2:
          ids[i] = thread(&Bank::deposit, bank, (rand() % 5), 100);
          break;
        case 3:
          ids[i] = thread(&Bank::transfer, bank, (rand() % 5), (rand() % 5), 50);
          break;
        case 4:
          ids[i] = thread(&Bank::withdraw, bank, (rand() % 5), 100);
          break;
        default:
          ids[i] = thread(&Bank::get_num_accounts, bank);
      }
    }

    for (int i = 0; i < NUM_THREADS; i++) {
      ids[i].join();
    }

    cout << "Closing bank." << endl;
    return 0;
}
