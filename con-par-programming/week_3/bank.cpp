#include <iostream>
#include <fstream>
#include <thread>
#include <unistd.h>
#include <tgmath.h>
#include <map>
#include <mutex>
#include <queue>

/*
@author ysbecca
*/
using namespace std;
mutex m;

class Bank {

  private:
    map<int, int> accounts = {};
    int num_accounts = 0;

  public:
    int create_account(int amount = 0) {
      lock_guard<mutex> hold(m);
      accounts[num_accounts++] = amount;
      cout << "Creating account " << num_accounts << " with amount " << amount << endl;
      return num_accounts;
    }

    int query_account(int id) {
      lock_guard<mutex> hold(m);
      return accounts[id];
    }

    int deposit(int id, int amount) {
      lock_guard<mutex> hold(m);
      cout << "Depositing into account " << id << " with amount " << amount << endl;
      accounts[id] += amount;
      return accounts[id];
    }

    int transfer(int id_A, int id_B, int amount) {
      if(id_A == id_B)
        return 0;
      lock_guard<mutex> hold(m);
      if(accounts[id_A] - amount >= 0) {
        accounts[id_A] -= amount;
        accounts[id_B] += amount;
        cout << "Transfering " << amount << " from account " << id_A << " to " << id_B <<  endl;
        return amount;
      } else {
        return 0;
      }
    }

    int withdraw(int id, int amount) {
      lock_guard<mutex> hold(m);
      if(accounts[id] - amount >= 0) {
        accounts[id] -= amount;
        return amount;
      } else {
        return 0;
      }
    }

    int get_num_accounts() {
      lock_guard<mutex> hold(m);
      return num_accounts;
    }
};
