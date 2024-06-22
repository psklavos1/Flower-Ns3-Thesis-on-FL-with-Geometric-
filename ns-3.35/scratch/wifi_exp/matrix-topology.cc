#include "matrix-topology.h"
#include <fstream>
#include <sstream>
#include <iostream>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cmath> // for sqrt, atan2, and M_PI
#include <iomanip>

using namespace std;

vector<vector<double>>
readCoordinates (string node_coordinates_file_name, int numEntries)
{
  ifstream node_coordinates_file (node_coordinates_file_name.c_str (), ios::in);
  if (node_coordinates_file.fail ())
    {
      throw runtime_error ("File " + node_coordinates_file_name + " not found");
    }
  vector<vector<double>> coord_array;

  string line;
  int readEntries = 0; // Counter for the number of entries read

  // Read lines from the file up to numEntries or until no more lines are available
  while (getline (node_coordinates_file, line) && readEntries < numEntries)
    {
      if (line.empty ())
        {
          break; // Exit if an empty line is encountered
        }

      istringstream iss (line);
      double coordinate;
      vector<double> row;

      while (iss >> coordinate)
        {
          row.push_back (coordinate);
        }

      if (row.size () != 2)
        {
          throw runtime_error ("A line in the coordinate file does not have exactly 2 elements.");
        }
      else
        {
          coord_array.push_back (row);
          ++readEntries; // Increment the counter since a valid entry was added
        }
    }

  node_coordinates_file.close ();
  return coord_array;
}

void
printCoordinates (vector<vector<double>> coord_array)
{
  printf ("===================== Coordinates ====================\n");
  if (!coord_array.empty ())
    {
      cout << fixed << setprecision (2);
      for (size_t i = 0; i < coord_array.size (); ++i)
        {
          if (!coord_array[i].empty ())
            {
              if (i == 0)
                {
                  printf ("Server:   (%.2f, %.2f)\n", coord_array[i][0], coord_array[i][1]);
                }
              else
                {
                  cout << "Client_" << i - 1 << ": (" << setw (6) << left << coord_array[i][0]
                       << ", " << setw (6) << left << coord_array[i][1] << ") " << setw (20)
                       << right << "-> Distance From Server: " << setw (6) << left
                       << getDistance (coord_array[i], coord_array[0]) << "\n";
                }
            }
        }
    }
  printf ("======================================================\n");
}

double
getDistance (std::vector<double> node, std::vector<double> refNode)
{
  double deltaX = node[0] - refNode[0];
  double deltaY = node[1] - refNode[1];
  return sqrt (deltaX * deltaX + deltaY * deltaY);
}

std::vector<std::string>
readDataRates (std::string filename, int numEntries)
{
  std::vector<std::string> dataRates;
  std::ifstream file (filename);
  std::string buffer;

  if (!file)
    {
      std::perror ("Error opening file");
      return {}; // Return an empty vector
    }

  // Read strings from the file and store them in the vector
  while (numEntries-- > 0 && file >> buffer)
    {
      // Push each read string into the vector
      dataRates.push_back (buffer);
    }

  return dataRates; // Return the vector of strings
}

void
printDataRates (std::vector<std::string> dataRates)
{
  std::cout << "===================== Data Rates =====================\n";

  for (size_t i = 0; i < dataRates.size (); ++i)
    {
      if (i == 0)
        {
          std::cout << "Server: " << dataRates[i] << "\n";
        }
      else
        {
          std::cout << "Client_" << i - 1 << ": " << dataRates[i] << "\n";
        }
    }

  std::cout << "======================================================\n";
}

// Obsolete

vector<vector<bool>>
readNxNMatrix (string adj_mat_file_name)
{
  ifstream adj_mat_file;
  adj_mat_file.open (adj_mat_file_name.c_str (), ios::in);
  if (adj_mat_file.fail ())
    {
      throw runtime_error ("File " + adj_mat_file_name + " not found");
    }
  vector<vector<bool>> array;
  int i = 0;
  int n_nodes = 0;

  while (!adj_mat_file.eof ())
    {
      string line;
      getline (adj_mat_file, line);
      if (line.empty ())
        {
          break;
        }

      istringstream iss (line);
      bool element;
      vector<bool> row;
      int j = 0;

      while (iss >> element)
        {
          row.push_back (element);
          j++;
        }

      if (i == 0)
        {
          n_nodes = j;
        }

      if (j != n_nodes)
        {
          throw runtime_error ("The number of elements in a row does not match the expected size.");
        }
      else
        {
          array.push_back (row);
        }
      i++;
    }

  if (i != n_nodes)
    {
      throw runtime_error ("The number of rows does not match the expected size.");
    }

  adj_mat_file.close ();
  return array;
}

// Function to read coordinates from a file and convert them to polar coordinates
std::vector<PolarCoordinate>
readCoordinatesFileToPolar (std::string &filename)
{
  std::ifstream file (filename);
  std::vector<PolarCoordinate> polarCoordinates;
  std::string line;
  double refX = 0.0, refY = 0.0;
  bool firstLine = true;

  if (!file.is_open ())
    {
      std::cerr << "Unable to open file: " << filename << std::endl;
      return polarCoordinates; // Return an empty vector in case of failure
    }

  while (std::getline (file, line))
    {
      std::istringstream iss (line);
      double x, y;
      if (!(iss >> x >> y))
        {
          break; // Error or end of file
        }

      if (firstLine)
        {
          // The first line represents the reference point (e.g., server location)
          refX = x;
          refY = y;
          firstLine = false;
          polarCoordinates.push_back (
              {0.0, 0.0}); // The reference point is at (0,0) in polar coordinates
        }
      else
        {
          // Convert and store the polar coordinates
          polarCoordinates.push_back (cartesianToPolar (x, y, refX, refY));
        }
    }

  file.close ();
  return polarCoordinates;
}

// Function to convert Cartesian coordinates to polar, relative to a reference point
PolarCoordinate
cartesianToPolar (double x, double y, double refX, double refY)
{
  double deltaX = x - refX;
  double deltaY = y - refY;
  PolarCoordinate polar;
  polar.radius = sqrt (deltaX * deltaX + deltaY * deltaY);
  polar.theta = atan2 (deltaY, deltaX) * (180.0 / M_PI); // Convert to degrees
  return polar;
}

void
printMatrix (vector<vector<bool>> array)
{
  cout << "===================== Adjacency Matrix =====================" << endl;
  for (const auto &row : array)
    {
      for (bool val : row)
        {
          cout << val << ' ';
        }
      cout << endl;
    }
}