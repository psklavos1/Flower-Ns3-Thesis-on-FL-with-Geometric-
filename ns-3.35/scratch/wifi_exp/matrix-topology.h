// topology-matrix.h
#ifndef TOPOLOGY_MATRIX_H
#define TOPOLOGY_MATRIX_H

#include <vector>
#include <string>
#include <utility> // for std::pair

struct PolarCoordinate {
    double radius;
    double theta; // Angle in degrees
};

// Function prototypes
PolarCoordinate cartesianToPolar(double x, double y, double refX, double refY);
std::vector<std::vector<bool>> readNxNMatrix(std::string adj_mat_file_name);
std::vector<std::vector<double>> readCoordinatesFile(std::string node_coordinates_file_name);
char** read_dataRates (const char* filename, int numEntries);
std::vector<PolarCoordinate> readCoordinatesFileToPolar(const std::string& filename);
void printCoordinateArray(std::vector<std::vector<double>> coord_array);
void printMatrix(std::vector<std::vector<bool>> array);
void print_dataRates (char **dataRates, int numEntries);
void clean_dataRates (char **dataRates, int numEntries);

#endif // TOPOLOGY_MATRIX_H
