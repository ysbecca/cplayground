/*

@author ysbecca

An ATM (Automatic Teller Machine) is stocked with notes and coins of the following denominations: £50, £20, £10, £5, £2 and £1. ATMs implement an algorithm to ensure that the machine dispenses the fewest notes/ coins when servicing a request for cash. Suppose a machine has the following:
 
£50 x 2, £20 x 1, £10 x 3, £5 x 2, £2 x 1, £1 x 2  
If a customer requests £35, it could dispense £10 x 3 + £2 x 2 + £1 x 1, but this require 6 notes/coins. A better outcome is to dispense £20 x 1 + £10 x 1 + £5 x 1, using only 3 notes/coins. As the request has been satisfied, the machine’s stock has been reduced, and its state must be updated to reflect this: the machine now has £50 x 2, £20 x 0, £10 x 2, £5 x 1, £2 x 1, £1 x 2.
Further requests will reduce the machine’s stock again, and it will reach a point where it cannot satisfy some requests. For example, although the machine above has over £50 in cash, it now cannot dispense £40. If a customer requests £40, the machine must work out that it can’t produce the required cash, and inform the customer. It never provides “partial” amounts.
An algorithm for minimizing the number of notes/coins used is known, and informally works as follows. To make up an amount £X from a set of denominations, you start from the highest denomination (call it H), and see how many of these go into X. Suppose that to dispense £X, you can use up to h units of £H. However,
your machine may not have h × £H’s. In this case, the number of £H’s you dispense, dh, should be the maximum of h and however many £H’s the machine has available. Once you’ve worked out dh, the customer still requires an amount equal to X − (dh × H). So now you consider the next currency unit in decreasing value (e.g. £20), and repeat the process. Continue until either the request has been satisfied, or you find that the request cannot be met by the stock in the machine.
Implement an ATM simulator in C. It should repeatedly prompt the user to enter an amount to be dispensed, and then either show how it will make up that amount from the currency units in stock, or inform the user that it has insufficient funds. Interaction should be terminated by entering end-of-stream (^D), at which point the machine displays its final stock levels. Keep error checking to a minimum; you can assume that the user always enters a positive integer as an amount.


/*