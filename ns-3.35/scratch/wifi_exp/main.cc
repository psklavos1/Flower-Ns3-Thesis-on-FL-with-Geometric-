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
#include <sys/stat.h>

using sysclock_t = std::chrono::system_clock;
using namespace ns3;
using namespace std;

FlwrProvider g_FlwrProvider (9090);
std::map<int, std::shared_ptr<ClientSession>> g_clients;

NS_LOG_COMPONENT_DEFINE ("Wifi-Adhoc");

int
main (int argc, char *argv[])
{
  // Logic to direct error stream to a file for debug
  // relative to the call directory
  // ofstream logFile ("./scratch/wifi_exp/error_log.txt");
  // auto oldCerrBuf = std::cerr.rdbuf ();
  // std::cerr.rdbuf (logFile.rdbuf ());

  FlwrProvider *FlwrProvider = &g_FlwrProvider;
  ns3::CommandLine cmd;

  // Default Values
  int numClients = 10; //when numClients is 50 or greater, packets are not recieved by server
  std::string networkType = "wifi";
  int maxPacketSize = 1024; //bytes
  double TxGain = 0.0; //dB + 30 = dBm
  double modelSize = 1.500 * 10; // kb
  std::string learningModel = "sync";
  std::string modelType = "MNIST";
  std::string deviceType = "400";
  bool moving_clients = false;
  int wifi_net_template = 0;

  cmd.AddValue ("numClients", "Number of clients", numClients);
  cmd.AddValue ("wifiNetTemplate", "Template of Wifi speed", wifi_net_template);
  cmd.AddValue ("wifiNetTemplate", "Template of Wifi speed", wifi_net_template);
  cmd.AddValue ("movingClients", "Assing mobility to clients", moving_clients);
  cmd.AddValue ("networkType", "Type of network", networkType);
  cmd.AddValue ("maxPacketSize", "Maximum size packet that can be sent", maxPacketSize);
  cmd.AddValue ("txGain", "Power transmitted from clients and server", TxGain);
  cmd.AddValue ("modelSize", "Size of model", modelSize);
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
  else if (modelType.compare ("cifar100") == 0)
    modelType = "CIFAR-100";

  bool bAsync = false;
  if (learningModel.compare ("async") == 0)
    {
      bAsync = true;
    }

  NS_LOG_UNCOND ("{NumClients:" << numClients
                                << ","
                                   "networkType:"
                                << networkType
                                << ","
                                   "wifNetTemplate:"
                                << wifi_net_template
                                << ","
                                   "movingClients:"
                                << moving_clients
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
                                   "learningModel:"
                                << learningModel << "}");
  modelSize = modelSize * 1000; // conversion to bytes

  // Generetate file to keep ns3 data
  time_t now = sysclock_t::to_time_t (sysclock_t::now ());

  // Output folder
  const char *outputFolder = "output_logs";

  // Ensure the folder exists or create it if necessary
  struct stat st;
  if (stat (outputFolder, &st) != 0)
    {
      mkdir (outputFolder, 0700); // Create directory if it does not exist
    }

  char buf[80] = {0};
  std::strftime (buf, sizeof (buf), "%Y-%m-%d_%H-%H-%S.csv", std::localtime (&now));
  char strbuff[200];
  snprintf (strbuff, sizeof (strbuff), "%s/%s_%s_%d_%s", outputFolder, networkType.c_str (),
            modelType.c_str (), numClients, buf);
  FILE *fp = fopen (strbuff, "w");

  // Get Coordinates
  string node_coordinates_file_name = "./scratch/wifi_exp/node_coordinates.txt";
  vector<vector<double>> coord_array = readCoordinates (node_coordinates_file_name, numClients + 1);
  printCoordinates (coord_array);
  //  Get DataRates
  std::string dataRates_file_name = "./scratch/wifi_exp/datarates.txt";
  std::vector<std::string> dataRates = readDataRates (dataRates_file_name, numClients + 1);
  // printDataRates (dataRates);

  // Assign Clients with characteristics
  int server = 0;
  for (int j = 0; j < numClients; j++)
    {
      std::string client_dataRate = dataRates[j + 1];
      int x = coord_array[j + 1][0];
      int y = coord_array[j + 1][1];
      g_clients[j] = std::shared_ptr<ClientSession> (new ClientSession (j, x, y, client_dataRate));
    }

  ns3::Time timeOffset (0);

  if (FlwrProvider)
    {
      g_FlwrProvider.waitForConnection ();
    }

  int round = 0;
  std::vector<std::pair<double, double>> updatedPositions;

  while (true)
    { // Repetitive setup as at each round in used Simulator::Destroy that cleans up the setup to avoid leaks
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

      auto experiment =
          Experiment (numClients, wifi_net_template, moving_clients, networkType, maxPacketSize,
                      TxGain, modelType, modelSize, dataRates[server], deviceType, bAsync,
                      FlwrProvider, fp, coord_array[server], round);

      // Print the clients' positions
      NS_LOG_UNCOND (">>>>>>>>>>>>>>>>>>>>>>>>> Start >>>>>>>>>>>>>>>>>>>>>>>>>");
      auto result = experiment.Run_Round (g_clients, timeOffset);
      auto roundStats = result.first;
      updatedPositions = result.second;
      NS_LOG_UNCOND (">>>>>>>>>>>>>>>>>>>>>>>>>> End >>>>>>>>>>>>>>>>>>>>>>>>>");

      // Update client positions with new coordinates after each round
      for (size_t i = 0; i < updatedPositions.size (); ++i)
        {
          g_clients[i]->SetX (updatedPositions[i].first);
          g_clients[i]->SetY (updatedPositions[i].second);
        }

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
  NS_LOG_UNCOND ("Exiting c++");

  // Restore the original buffers
  // std::cerr.rdbuf (oldCerrBuf);
  // logFile.close (); // Close the log file
  return 0;
}
