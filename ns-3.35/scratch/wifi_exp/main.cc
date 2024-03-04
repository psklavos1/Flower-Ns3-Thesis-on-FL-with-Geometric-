/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2022 Emily Ekaireb
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation;
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 * Author: Emily Ekaireb <eekaireb@ucsd.edu>
 */

#include <random>
#include <chrono>
#include <vector>
#include <iostream> // Include the I/O library
#include <unistd.h> // For UNIX/Linux systems
#include <fstream>

#include "fl-experiment.h"
#include "matrix-topology.h"

using sysclock_t = std::chrono::system_clock;

using namespace ns3;
using namespace std;

FlwrProvider g_FlwrProvider (8080);
std::map<int, std::shared_ptr<ClientSession>> g_clients;

NS_LOG_COMPONENT_DEFINE ("Wifi-Adhoc");

int
main (int argc, char *argv[])
{
  // std::ofstream logFile ("./scratch/wifi_exp/log_file.txt");
  // Save the old buffers of std::cout and std::cerr
  // so they can be restored later.
  // auto oldCoutBuf = std::cout.rdbuf ();
  // auto oldCerrBuf = std::cerr.rdbuf ();

  // Redirect std::cout and std::cerr to logFile (log_file.txt).
  // std::cout.rdbuf (logFile.rdbuf ());
  // std::cerr.rdbuf (logFile.rdbuf ());

  FlwrProvider *FlwrProvider = &g_FlwrProvider;

  ns3::CommandLine cmd;

  // Default Values
  int numClients = 10; //when numClients is 50 or greater, packets are not recieved by server
  std::string networkType = "wifi";
  int maxPacketSize = 1024; //bytes
  double TxGain = 0.0; //dB + 30 = dBm
  double modelSize = 1.500 * 10; // kb
  std::string dataRate = "1000kbps"; /* Application layer datarate. */
  std::string learningModel = "sync";
  std::string modelType = "MNIST";
  std::string deviceType = "400";

  cmd.AddValue ("numClients", "Number of clients", numClients);
  cmd.AddValue ("networkType", "Type of network", networkType);
  cmd.AddValue ("maxPacketSize", "Maximum size packet that can be sent", maxPacketSize);
  cmd.AddValue ("txGain", "Power transmitted from clients and server", TxGain);
  cmd.AddValue ("modelSize", "Size of model", modelSize);
  cmd.AddValue ("dataRate", "Application data rate", dataRate);
  cmd.AddValue ("learningModel", "Async or Sync federated learning", learningModel);
  cmd.AddValue (
      "modelType",
      "modelType on which the training is happening(For better computational time calculation)",
      modelType);
  cmd.AddValue ("deviceType", "Type of Devices used as clients", deviceType);

  modelSize = 1.500 * 10; // kb

  cmd.Parse (argc, argv);
  if (modelType.compare ("mnist") == 0)
    {
      modelType = "MNIST";
    }

  else if (modelType.compare ("fashion_mnist") == 0)
    modelType = "FashionMNIST";
  else if (modelType.compare ("cifar10") == 0)
    modelType = "CIFAR-10";

  bool bAsync = false;
  if (learningModel.compare ("async") == 0)
    {
      bAsync = true;
    }

  // NS_LOG_UNCOND ("modelSize: " << modelSize);

  NS_LOG_UNCOND ("{NumClients:" << numClients
                                << ","
                                   "networkType:"
                                << networkType
                                << ","
                                   "maxPacketSize:"
                                << maxPacketSize
                                << ","
                                   "TxGain:"
                                << TxGain
                                << ","
                                   "modelType:"
                                << modelType
                                << ","
                                   "modelSize:"
                                << modelSize
                                << ","
                                   "dataRate:"
                                << dataRate
                                << ","
                                   "learningModel:"
                                << learningModel << "}");
  //Experiment experiment(numClients,networkType,maxPacketSize,TxGain);
  modelSize = modelSize * 1000; // conversion to bytes

  std::time_t now = sysclock_t::to_time_t (sysclock_t::now ());

  char buf[80] = {0};
  std::strftime (buf, sizeof (buf), "%Y-%m-%d_%H-%H-%S.csv", std::localtime (&now));

  char strbuff[100];
  snprintf (strbuff, 99, "%s_%s_%.2f_%s", learningModel.c_str (), networkType.c_str (), TxGain,
            buf);

  FILE *fp = fopen (strbuff, "w");

  std::default_random_engine generator;
  std::uniform_real_distribution<double> r_dist (1.0, 4.0);
  //std::uniform_real_distribution<double> t_dist(0,1.0);

  // Generate Coordinates
  std::string node_coordinates_file_name = "./scratch/wifi_exp/node_coordinates.txt";
  vector<vector<double>> coord_array;
  std::vector<PolarCoordinate> polar_coord_array =
      readCoordinatesFileToPolar (node_coordinates_file_name);
  // printCoordinateArray (node_coordinates_file_name.c_str (),coord_array);

  //  Get DataRates
  const char *dataRates_file_name = "./scratch/wifi_exp/client_datarates.txt";
  char **client_dataRates = read_dataRates (dataRates_file_name, numClients);
  print_dataRates (client_dataRates, numClients);
  // To pause execution
  // std::string stringValue;
  // std::getline(std::cin, stringValue); // Take string input, including spaces

  NS_LOG_UNCOND ("\n===================== Coordinates Around Server =====================");
  for (int j = 0; j < numClients; j++)
    {
      // std::string stringValue;
      // std::getline(std::cin, stringValue); // Take string input, including spaces
      char *client_dataRate = client_dataRates[j];
      PolarCoordinate polar = polar_coord_array[j + 1];
      double radius = polar.radius;
      //double theta = t_dist(generator);
      double theta = polar.theta;

      NS_LOG_UNCOND ("INIT:J=" << j << " r=" << radius << " th=" << theta);
      g_clients[j] =
          std::shared_ptr<ClientSession> (new ClientSession (j, radius, theta, client_dataRate));
    }
  ns3::Time timeOffset (0);

  if (FlwrProvider)
    {
      g_FlwrProvider.waitForConnection ();
    }

  int round = 0;

  while (true)
    {

      round++;

      if (FlwrProvider)
        {
          FlwrProvider::COMMAND::Type type = g_FlwrProvider.recv (g_clients);

          if (type == FlwrProvider::COMMAND::Type::EXIT)
            {
              g_FlwrProvider.Close ();
              break;
            }
        }

      auto experiment = Experiment (numClients, networkType, maxPacketSize, TxGain, modelType,
                                    modelSize, dataRate, deviceType, bAsync, FlwrProvider, fp, round

      );
      NS_LOG_UNCOND (">>>>>>>>>>>>>>>>>>>>>>>>>\nBefore: " << timeOffset
                                                           << "\n"
                                                              ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>");
      auto roundStats = experiment.WeakNetwork (g_clients, timeOffset);

      NS_LOG_UNCOND (">>>>>>>>>>>>>>>>>>>>>>>>>\nTIME_OFFSET:" << timeOffset
                                                               << "\n"
                                                                  ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>");

      if (FlwrProvider && !bAsync)
        {
          g_FlwrProvider.send (roundStats);
        }
      if (!FlwrProvider)
        {
          break;
        }

      fflush (fp);
    }

  fclose (fp);
  clean_dataRates (client_dataRates, numClients);
  NS_LOG_UNCOND ("Exiting c++");

  // Restore the original buffers
  // std::cerr.rdbuf (oldCerrBuf);
  // logFile.close (); // Close the log file
  return 0;
}
