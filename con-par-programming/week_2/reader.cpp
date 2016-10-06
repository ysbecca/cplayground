
#include <iostream>
#include <fstream>

int main(int argc, char **argv)
{
  std::ifstream src("data.txt");
  unsigned char data[1 << 18];
  unsigned int val;
  int size = 0;

  if (!src) { 
      std::cerr << "Cannot open data file." << std::endl; 
      exit(1); 
  }
  while (src >> val) 
      data[size++] = (unsigned char)val;
  std::cout << "Read " << size << " values from file." << std::endl;

  return 0;
}
