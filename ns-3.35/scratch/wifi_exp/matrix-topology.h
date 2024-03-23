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

std::vector<std::vector<double>> readCoordinates(std::string node_coordinates_file_name,int numEntries);
std::vector<std::string> readDataRates (std::string filename, int numEntries);
void printCoordinates(const std::vector<std::vector<double>> coord_array);
void printDataRates (const std::vector<std::string>datarates);

std::vector<PolarCoordinate> readCoordinatesFileToPolar(std::string& filename);
double getDistance(std::vector<double> node1, std::vector<double> node2);
void printMatrix(const std::vector<std::vector<bool>> array);

#endif // TOPOLOGY_MATRIX_H
