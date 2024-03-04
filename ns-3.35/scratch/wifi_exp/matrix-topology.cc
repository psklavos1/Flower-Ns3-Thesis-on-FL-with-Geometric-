#include "matrix-topology.h"
#include <fstream>
#include <sstream>
#include <iostream>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cmath> // for sqrt, atan2, and M_PI

using namespace std;

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

vector<vector<double>>
readCordinatesFile (string node_coordinates_file_name)
{
  ifstream node_coordinates_file;
  node_coordinates_file.open (node_coordinates_file_name.c_str (), ios::in);
  if (node_coordinates_file.fail ())
    {
      throw runtime_error ("File " + node_coordinates_file_name + " not found");
    }
  vector<vector<double>> coord_array;

  while (!node_coordinates_file.eof ())
    {
      string line;
      getline (node_coordinates_file, line);
      if (line.empty ())
        {
          break;
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
        }
    }

  node_coordinates_file.close ();
  return coord_array;
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

void
printCoordinateArray (vector<vector<double>> coord_array)
{
  cout << "===================== Coordinates Array =====================" << endl;
  for (const auto &row : coord_array)
    {
      for (double val : row)
        {
          cout << val << ' ';
        }
      cout << endl;
    }
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

// Function to read coordinates from a file and convert them to polar coordinates
std::vector<PolarCoordinate>
readCoordinatesFileToPolar (const std::string &filename)
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

#define MAX_STRING_LENGTH 10

char **
read_dataRates (const char *filePath, int numEntries)
{
  char **dataRates =
      (char **) malloc (numEntries * sizeof (char *)); // Allocate array of string pointers
  FILE *file;
  char buffer[MAX_STRING_LENGTH];
  int i = 0;

  // Open the file
  file = fopen (filePath, "r");
  if (file == NULL)
    {
      perror ("Error opening file");
      free (dataRates); // Clean up allocated memory on error
      return NULL;
    }

  // Read strings from the file and store them in the array
  while (i < numEntries && fscanf (file, "%9s", buffer) == 1)
    { // Read up to STRING_LENGTH - 1 characters
      dataRates[i] =
          (char *) malloc (MAX_STRING_LENGTH * sizeof (char)); // Allocate memory for each string
      if (dataRates[i] == NULL)
        {
          perror ("Memory allocation failed for dataRates string");
          for (int j = 0; j < i; j++)
            { // Free any previously allocated strings
              free (dataRates[j]);
            }
          free (dataRates);
          fclose (file);
          return NULL;
        }

      strcpy (dataRates[i], buffer); // Copy the string from buffer to the array
      i++;
    }
  fclose (file);

  return dataRates; // Return the array of strings
}

void
print_dataRates (char **dataRates, int numEntries)
{
  printf("===================== Data Rates =====================\n");
  if (dataRates != NULL)
    {
      for (int i = 0; i < numEntries; i++)
        {
          printf ("%s\n", dataRates[i]);
        }
    }
}

void
clean_dataRates (char **dataRates, int numEntries)
{
  if (dataRates != NULL)
    {
      for (int i = 0; i < numEntries; i++)
        {
          free (dataRates[i]);
        }
      free (dataRates);
    }
}