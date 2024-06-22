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

#include "fl-experiment.h"
#include "fl-client-application.h"
#include "ns3/core-module.h"
#include "ns3/csma-module.h"
#include "ns3/applications-module.h"
#include "ns3/internet-module.h"
#include "fl-server-helper.h"
#include "ns3/reliability-helper.h"
#include "ns3/energy-module.h"
#include "ns3/internet-module.h"
#include "ns3/reliability-module.h"
#include "ns3/yans-error-rate-model.h"
#include "ns3/ssid.h"

// #include "ns3/olsr-helper.h"
// #include "ns3/aodv-module.h"

#include <iomanip> // For std::setprecision and std::fixed
#include <algorithm> // For std::shuffle
#include <random> // For std::default_random_engine
#include <chrono> // For std::chrono::system_clock

namespace ns3 {

Experiment::Experiment (int numClients, int wifi_net_template, bool moving_clients,
                        std::string &networkType, int maxPacketSize, double txGain,
                        std::string &modelType, double modelSize, std::string &dataRate,
                        std::string &deviceType, bool bAsync, FlwrProvider *flwr_provider, FILE *fp,
                        std::vector<double> server_coordinates, int round)
    : m_numClients (numClients),
      m_wifi_net_template (wifi_net_template),
      m_moving_clients (moving_clients),
      m_networkType (networkType),
      m_maxPacketSize (maxPacketSize),
      m_txGain (txGain),
      m_modelType (modelType),
      m_deviceType (deviceType),
      m_modelSize (modelSize),
      m_dataRate (dataRate),
      m_bAsync (bAsync),
      m_flwrProvider (flwr_provider),
      m_fp (fp),
      m_server_coordinates (server_coordinates),
      m_router_coordinates (2),
      m_round (round)
{
  // Assign values to m_router_coordinates based on m_server_coordinates
  m_router_coordinates[0] = m_server_coordinates[0] + 2;
  m_router_coordinates[1] = m_server_coordinates[1] - 2;
}

//>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Experiment Round Run >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
void
Experiment::LogClientCharacteristics (Ptr<Node> clientNode, const Vector &routerPosition,
                                      int clientIndex)
{
  // Get the client's position
  Vector clientPosition = GetPosition (clientNode);
  // Calculate the distance from the server
  double distance = std::sqrt (std::pow (clientPosition.x - routerPosition.x, 2) +
                               std::pow (clientPosition.y - routerPosition.y, 2) +
                               std::pow (clientPosition.z - routerPosition.z, 2));

  // Print the client's position and distance from the server
  NS_LOG_UNCOND ("In Round Client " << clientIndex << " Position: (" << clientPosition.x << ", "
                                    << clientPosition.y << ", " << clientPosition.z
                                    << "), Distance from Server: " << distance << " meters");
}

// Calback for monitoring
void
Monitor (std::string context, Ptr<const Packet> pkt, unsigned short channel, WifiTxVector txVector,
         MpduInfo mpdu, SignalNoiseDbm snr, uint16_t staId)
{
  std::cout << context << std::endl;
  std::cout << "\tChannel: " << channel << "Tx: " << txVector.GetMode ()
            << "\tSignal= " << snr.signal << "\tNoise: " << snr.noise << std::endl;
}

void
Pause ()
{
  std::cout << "Enter any key to continue..." << std::endl;
  std::cin.ignore ();
}
void
Experiment::SetupMobilityForStaNodes (NodeContainer &staNodes,
                                      const std::vector<double> &router_coords, double radius)
{
  MobilityHelper mobility;

  for (uint32_t i = 0; i < staNodes.GetN (); ++i)
    {
      std::cout << "coord_0: " << router_coords[0] << std::endl;
      std::cout << router_coords[1] << std::endl;

      // Set the mobility model to RandomWalk2dMobilityModel
      mobility.SetMobilityModel (
          "ns3::RandomWalk2dMobilityModel", "Bounds",
          RectangleValue (Rectangle (router_coords[0] - radius, router_coords[0] + radius,
                                     router_coords[1] - radius, router_coords[1] + radius)),
          "Distance", DoubleValue (1.0), // Maximum distance the node can move in one step
          "Direction",
          StringValue (
              "ns3::UniformRandomVariable[Min=0.0|Max=6.2830]"), // Random direction in radians
          "Speed", StringValue ("ns3::ConstantRandomVariable[Constant=1.0]") // Constant speed
      );
      mobility.Install (staNodes.Get (i));
    }
}

void
printRoutingTable ()
{
  // Print routing tables
  Ipv4GlobalRoutingHelper g;
  Ptr<OutputStreamWrapper> routingStream = Create<OutputStreamWrapper> (&std::cout);
  g.PrintRoutingTableAllAt (Seconds (1), routingStream);
}

void
runPingExp (NodeContainer &c, Ipv4InterfaceContainer &interfaces, int16_t udp_server_index)
{
  // Install Echo Server on the server node
  UdpEchoServerHelper echoServer (9999);
  ApplicationContainer serverApps;
  serverApps = echoServer.Install (c.Get (udp_server_index));
  std::cout << "Server Node: " << udp_server_index << std::endl;

  serverApps.Start (Seconds (2.0));
  serverApps.Stop (Seconds (100));
  // Install Echo Client on client nodes to ping the server
  for (uint16_t i = 0; i < c.GetN (); ++i)
    {
      if (i != udp_server_index)
        {
          Ptr<Node> clientNode;
          clientNode = c.Get (i);

          if (udp_server_index == 0)
            {
              InetSocketAddress serverAddress =
                  InetSocketAddress (interfaces.GetAddress (udp_server_index), 9999);
              UdpEchoClientHelper echoClient (serverAddress);
              echoClient.SetAttribute ("MaxPackets", UintegerValue (10));
              echoClient.SetAttribute ("Interval", TimeValue (Seconds (1.0)));
              echoClient.SetAttribute ("PacketSize", UintegerValue (1024));

              ApplicationContainer pingApps = echoClient.Install (clientNode);
              std::cout << "Client Node: " << i << "send to " << serverAddress.GetIpv4 ()
                        << std::endl;

              pingApps.Start (Seconds (3.0));
              pingApps.Stop (Seconds (10));
            }
          else if (udp_server_index == 1)
            {
              if (i < 2)
                {
                  InetSocketAddress serverAddress =
                      InetSocketAddress (interfaces.GetAddress (udp_server_index), 9999);
                  UdpEchoClientHelper echoClient (serverAddress);
                  echoClient.SetAttribute ("MaxPackets", UintegerValue (10));
                  echoClient.SetAttribute ("Interval", TimeValue (Seconds (1.0)));
                  echoClient.SetAttribute ("PacketSize", UintegerValue (1024));

                  ApplicationContainer pingApps = echoClient.Install (clientNode);
                  std::cout << "Client Node: " << i << "send to " << serverAddress.GetIpv4 ()
                            << std::endl;
                  pingApps.Start (Seconds (3.0));
                  pingApps.Stop (Seconds (10));
                }
              else
                {
                  InetSocketAddress serverAddress =
                      InetSocketAddress (interfaces.GetAddress (udp_server_index + 1), 9999);
                  UdpEchoClientHelper echoClient (serverAddress);
                  echoClient.SetAttribute ("MaxPackets", UintegerValue (10));
                  echoClient.SetAttribute ("Interval", TimeValue (Seconds (1.0)));
                  echoClient.SetAttribute ("PacketSize", UintegerValue (1024));

                  ApplicationContainer pingApps = echoClient.Install (clientNode);
                  std::cout << "Client Node: " << i << "send to " << serverAddress.GetIpv4 ()
                            << std::endl;
                  pingApps.Start (Seconds (3.0));
                  pingApps.Stop (Seconds (10));
                }
            }
          else
            {
              InetSocketAddress serverAddress =
                  InetSocketAddress (interfaces.GetAddress (udp_server_index + 1), 9999);
              UdpEchoClientHelper echoClient (serverAddress);

              echoClient.SetAttribute ("MaxPackets", UintegerValue (10));
              echoClient.SetAttribute ("Interval", TimeValue (Seconds (1.0)));
              echoClient.SetAttribute ("PacketSize", UintegerValue (1024));

              ApplicationContainer pingApps = echoClient.Install (clientNode);
              std::cout << "Client Node: " << i << "send to " << serverAddress.GetIpv4 ()
                        << std::endl;
              pingApps.Start (Seconds (3.0));
              pingApps.Stop (Seconds (10));
            }
        }
    }
}

std::pair<std::map<int, FlwrProvider::Message>, std::vector<std::pair<double, double>>>
Experiment::Run_Round (std::map<int, std::shared_ptr<ClientSession>> &clients,
                       ns3::Time &timeOffset)
{
  NS_LOG_UNCOND ("Experiment Round Setup");
  double server_start_time = 3.0;
  double stop_time = 1000.0;
  double client_start_time = 5.0;
  uint16_t server_port = 80;

  // 1.Devices Setup
  NodeContainer c;
  // take into account server and router and the rest of the clients
  c.Create (clients.size () + 2);
  NetDeviceContainer devices = SetupDevices (c, clients);

  // 2.Internet Stack Setup
  Ipv4InterfaceContainer interfaces = SetupInternetStack (c, devices);

  Ipv4GlobalRoutingHelper::PopulateRoutingTables ();
  // 3.Server Setup
  Ptr<Server> serverApp =
      SetupServer (c.Get (0), server_port, interfaces, timeOffset, server_start_time, stop_time);

  // 4.Clients Setup
  std::map<Ipv4Address, int> m_addrMap;
  ApplicationContainer clientApps = SetupClients (c, clients, server_port, interfaces, m_addrMap,
                                                  timeOffset, client_start_time, stop_time);

  // printRoutingTable ();

  // runPingExp (c, interfaces, 1);
  // 5.Client Session Manager Setup
  ClientSessionManager client_session_manager (clients);
  serverApp->GetObject<ns3::Server> ()->SetClientSessionManager (&client_session_manager,
                                                                 m_flwrProvider, m_fp, m_round);
  // 6. Netanim Interface Setup
  AnimationInterface anim = ConfigureAnimation (c, client_start_time);
  NS_LOG_UNCOND ("Round Setup Complete");

  LogComponentEnable ("UdpEchoClientApplication", LOG_LEVEL_INFO);
  LogComponentEnable ("UdpEchoServerApplication", LOG_LEVEL_INFO);

  // 7.Run Simulation
  // Config::Connect ("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Phy/MonitorSnifferRx",
  //                  MakeCallback (&Monitor));
  Simulator::Stop (Seconds (stop_time));
  // Pause ();

  NS_LOG_UNCOND ("================= Starting Ns3 Round Simulation ===================");
  Simulator::Run ();

  // 8.Extract Simulation Results
  std::map<Ipv4Address, FlwrProvider::Message> stats;
  ExtractDownlinkResults (clients, server_port, interfaces, stats);
  ExtractUplinkResults (serverApp, server_port, interfaces, stats);

  std::map<int, FlwrProvider::Message> roundStats = ProcessResults (m_addrMap, stats);
  std::vector<std::pair<double, double>> updatedPositions;
  for (uint32_t j = 2; j < c.GetN (); ++j)
    {
      Vector pos = GetPosition (c.Get (j));
      updatedPositions.push_back (std::make_pair (pos.x, pos.y));
    }
  Simulator::Destroy ();
  // Update node coordinates if clients are moving

  return std::make_pair (roundStats, updatedPositions);
}
//>>>>>>>>>>>>>>>>>>>>>>>>>>>> \Experiment Round Run >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Positioning Helpers >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
void
Experiment::SetPositionPolar (Ptr<Node> node, double radius, double theta)
{
  double x = radius * cos (theta);
  double y = radius * sin (theta);
  double z = 0;
  Ptr<MobilityModel> mobility = node->GetObject<MobilityModel> ();
  mobility->SetPosition (Vector (x, y, z));
}

void
Experiment::SetPositionCartesian (Ptr<Node> node, double x, double y)
{
  double z = 0;
  Ptr<MobilityModel> mobility = node->GetObject<MobilityModel> ();
  mobility->SetPosition (Vector (x, y, z));
}

Vector
Experiment::GetPosition (Ptr<Node> node)
{
  Ptr<MobilityModel> mobility = node->GetObject<MobilityModel> ();
  return mobility->GetPosition ();
}
//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> \Positioning Helpers >>>>>>>>>>>>>>>>>>>>>>>>>>>>>

//>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Network Setup Helpers >>>>>>>>>>>>>>>>>>>>>>>>>>>>>

// Network Setup Helpers
NetDeviceContainer
Experiment::MyWifi (NodeContainer &c, std::map<int, std::shared_ptr<ClientSession>> &clients)
{

  // Split the nodes into Server, AP, and Clients
  NodeContainer csmaNodes;
  NodeContainer serverNode;

  NodeContainer apNode;
  NodeContainer staNodes;

  NetDeviceContainer allDevices;

  serverNode.Add (c.Get (0)); // First node as Server
  apNode.Add (c.Get (1)); // Second node as AP
  csmaNodes.Add (serverNode);
  csmaNodes.Add (apNode);

  for (uint32_t i = 2; i < c.GetN (); ++i)
    {
      staNodes.Add (c.Get (i)); // Remaining nodes as Clients
    }

  // First The Ethernet connection between the server and the clients
  // Setup Ethernet connection for the server and the router
  CsmaHelper csma;
  csma.SetChannelAttribute ("DataRate", StringValue (m_dataRate));
  csma.SetChannelAttribute ("Delay", TimeValue (NanoSeconds (6560)));
  NetDeviceContainer csmaDevices = csma.Install (csmaNodes);
  // Store all the Devices created
  allDevices.Add (csmaDevices);

  // Now The wifi Devices of the clients and their AP's

  // 0. Define Helpers
  YansWifiChannelHelper wifiChannel = YansWifiChannelHelper::Default ();
  YansWifiPhyHelper wifiPhy;
  WifiHelper wifi;
  WifiMacHelper wifiMac;

  // 1. Setup Channel
  // wifiChannel.AddPropagationLoss ("ns3::LogDistancePropagationLossModel", "Exponent",
  //                                 DoubleValue (3.0));
  // wifiChannel.AddPropagationLoss ("ns3::NakagamiPropagationLossModel");
  wifiChannel.SetPropagationDelay ("ns3::ConstantSpeedPropagationDelayModel");

  // 2. Setup Physical Layer
  wifiPhy.SetErrorRateModel ("ns3::YansErrorRateModel");
  wifiPhy.SetChannel (wifiChannel.Create ());
  // wifiPhy.Set ("TxGain", DoubleValue (m_txGain));
  // wifiPhy.Set ("RxGain", DoubleValue (0));
  wifiPhy.Set ("TxPowerStart", DoubleValue (20.0)); // Example setting for min power
  wifiPhy.Set ("TxPowerEnd", DoubleValue (20.0)); // Example setting for max power

  // std::string phyMode ("HtMcs2");
  // Config::SetDefault ("ns3::WifiRemoteStationManager::NonUnicastMode", StringValue (phyMode));

  // 3. Setup Wifi for client connectivity
  wifi.SetStandard (WIFI_STANDARD_80211a);
  // wifi.SetRemoteStationManager ("ns3::MinstrelHtWifiManager");
  wifi.SetRemoteStationManager ("ns3::AarfWifiManager");

  // 4. Setup Mac for AP
  Ssid ssid = Ssid ("home-wifi-ssid");
  wifiMac.SetType ("ns3::ApWifiMac", "Ssid", SsidValue (ssid));
  NetDeviceContainer apDevices = wifi.Install (wifiPhy, wifiMac, apNode);

  // 4. Setup Mac for STA
  wifiMac.SetType ("ns3::StaWifiMac", "Ssid", SsidValue (ssid), "ActiveProbing",
                   BooleanValue (false));
  NetDeviceContainer staDevices = wifi.Install (wifiPhy, wifiMac, staNodes);

  allDevices.Add (apDevices);
  allDevices.Add (staDevices);

  MobilityHelper mobility;

  // Setup Mobility
  mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
  mobility.Install (apNode);
  mobility.Install (serverNode);

  // If mobile clients
  if (m_moving_clients == true)
    {
      SetupMobilityForStaNodes (staNodes, m_router_coordinates, 35);
    }
  mobility.Install (staNodes);

  // Setup Devices positions
  SetPositionCartesian (c.Get (0), m_server_coordinates[0], m_server_coordinates[1]);
  // assume they are next to each other in the lab
  SetPositionCartesian (c.Get (1), m_router_coordinates[0], m_router_coordinates[1]);

  for (uint32_t j = 2; j < c.GetN (); ++j)
    {
      SetPositionCartesian (c.Get (j), clients[j - 2]->GetX (), clients[j - 2]->GetY ());
    }

  return allDevices;
}

NetDeviceContainer
Experiment::WeakWifi (NodeContainer &c, std::map<int, std::shared_ptr<ClientSession>> &clients)
{
  NodeContainer csmaNodes;
  NodeContainer serverNode;
  NodeContainer apNode;
  NodeContainer staNodes;
  NetDeviceContainer allDevices;

  serverNode.Add (c.Get (0)); // First node as Server
  apNode.Add (c.Get (1)); // Second node as AP
  csmaNodes.Add (serverNode);
  csmaNodes.Add (apNode);

  for (uint32_t i = 2; i < c.GetN (); ++i)
    {
      staNodes.Add (c.Get (i)); // Remaining nodes as Clients
    }

  // Setup Ethernet connection for the server and the router
  CsmaHelper csma;
  csma.SetChannelAttribute ("DataRate", StringValue (m_dataRate));
  csma.SetChannelAttribute ("Delay", TimeValue (NanoSeconds (6560)));
  NetDeviceContainer csmaDevices = csma.Install (csmaNodes);
  allDevices.Add (csmaDevices);

  // Setup Wi-Fi devices
  YansWifiChannelHelper wifiChannel = YansWifiChannelHelper::Default ();
  YansWifiPhyHelper wifiPhy;
  WifiHelper wifi;
  WifiMacHelper wifiMac;

  // Setup Channel
  wifiChannel.SetPropagationDelay ("ns3::ConstantSpeedPropagationDelayModel");
  // wifiChannel.AddPropagationLoss ("ns3::LogDistancePropagationLossModel", "Exponent",
  //                                 DoubleValue (2.0));

  // Setup Physical Layer
  wifiPhy.SetChannel (wifiChannel.Create ());
  wifiPhy.Set ("TxPowerStart", DoubleValue (10)); // Lower transmit power
  wifiPhy.Set ("TxPowerEnd", DoubleValue (14.0)); // Lower transmit power
  wifiPhy.SetErrorRateModel ("ns3::YansErrorRateModel");

  // Setup Wi-Fi for client connectivity
  wifi.SetStandard (WIFI_STANDARD_80211a);
  wifi.SetRemoteStationManager ("ns3::AarfWifiManager");

  // Setup Mac for AP
  Ssid ssid = Ssid ("weak-wifi-ssid");
  wifiMac.SetType ("ns3::ApWifiMac", "Ssid", SsidValue (ssid));
  NetDeviceContainer apDevices = wifi.Install (wifiPhy, wifiMac, apNode);

  // Setup Mac for STA
  wifiMac.SetType ("ns3::StaWifiMac", "Ssid", SsidValue (ssid), "ActiveProbing",
                   BooleanValue (false));
  NetDeviceContainer staDevices = wifi.Install (wifiPhy, wifiMac, staNodes);

  allDevices.Add (apDevices);
  allDevices.Add (staDevices);

  MobilityHelper mobility;

  // Setup Mobility
  mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
  mobility.Install (apNode);
  mobility.Install (serverNode);

  // If mobile clients
  if (m_moving_clients == true)
    {
      SetupMobilityForStaNodes (staNodes, m_router_coordinates, MAX_DISTANCE_FROM_ROUTER);
    }
  mobility.Install (staNodes);

  // Setup Devices positions
  SetPositionCartesian (c.Get (0), m_server_coordinates[0], m_server_coordinates[1]);
  SetPositionCartesian (c.Get (1), m_router_coordinates[0], m_router_coordinates[1]);
  for (uint32_t j = 2; j < c.GetN (); ++j)
    {
      SetPositionCartesian (c.Get (j), clients[j - 2]->GetX (), clients[j - 2]->GetY ());
    }

  return allDevices;
}

NetDeviceContainer
Experiment::MidWifi (NodeContainer &c, std::map<int, std::shared_ptr<ClientSession>> &clients)
{
  NodeContainer csmaNodes;
  NodeContainer serverNode;
  NodeContainer apNode;
  NodeContainer staNodes;
  NetDeviceContainer allDevices;

  serverNode.Add (c.Get (0)); // First node as Server
  apNode.Add (c.Get (1)); // Second node as AP
  csmaNodes.Add (serverNode);
  csmaNodes.Add (apNode);

  for (uint32_t i = 2; i < c.GetN (); ++i)
    {
      staNodes.Add (c.Get (i)); // Remaining nodes as Clients
    }

  // Setup Ethernet connection for the server and the router
  CsmaHelper csma;
  csma.SetChannelAttribute ("DataRate", StringValue (m_dataRate));
  csma.SetChannelAttribute ("Delay", TimeValue (NanoSeconds (6560)));
  NetDeviceContainer csmaDevices = csma.Install (csmaNodes);
  allDevices.Add (csmaDevices);

  // Setup Wi-Fi devices
  YansWifiChannelHelper wifiChannel = YansWifiChannelHelper::Default ();
  YansWifiPhyHelper wifiPhy;
  WifiHelper wifi;
  WifiMacHelper wifiMac;

  // Setup Channel
  wifiChannel.SetPropagationDelay ("ns3::ConstantSpeedPropagationDelayModel");
  // wifiChannel.AddPropagationLoss ("ns3::LogDistancePropagationLossModel", "Exponent",
  //                                 DoubleValue (3.0));

  // Setup Physical Layer
  wifiPhy.SetChannel (wifiChannel.Create ());
  wifiPhy.Set ("TxPowerStart", DoubleValue (15.0)); // Moderate transmit power
  wifiPhy.Set ("TxPowerEnd", DoubleValue (18.0)); // Moderate transmit power
  wifiPhy.SetErrorRateModel ("ns3::YansErrorRateModel");

  // Setup Wi-Fi for client connectivity
  wifi.SetStandard (WIFI_STANDARD_80211g);
  wifi.SetRemoteStationManager ("ns3::MinstrelWifiManager");

  // Setup Mac for AP
  Ssid ssid = Ssid ("mid-wifi-ssid");
  wifiMac.SetType ("ns3::ApWifiMac", "Ssid", SsidValue (ssid));
  NetDeviceContainer apDevices = wifi.Install (wifiPhy, wifiMac, apNode);

  // Setup Mac for STA
  wifiMac.SetType ("ns3::StaWifiMac", "Ssid", SsidValue (ssid), "ActiveProbing",
                   BooleanValue (false));
  NetDeviceContainer staDevices = wifi.Install (wifiPhy, wifiMac, staNodes);

  allDevices.Add (apDevices);
  allDevices.Add (staDevices);

  MobilityHelper mobility;

  // Setup Mobility
  mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
  mobility.Install (apNode);
  mobility.Install (serverNode);

  // If mobile clients
  if (m_moving_clients == true)
    {
      SetupMobilityForStaNodes (staNodes, m_router_coordinates, MAX_DISTANCE_FROM_ROUTER);
    }
  mobility.Install (staNodes);
  // Setup Devices positions
  SetPositionCartesian (c.Get (0), m_server_coordinates[0], m_server_coordinates[1]);
  SetPositionCartesian (c.Get (1), m_router_coordinates[0], m_router_coordinates[1]);

  for (uint32_t j = 2; j < c.GetN (); ++j)
    {
      SetPositionCartesian (c.Get (j), clients[j - 2]->GetX (), clients[j - 2]->GetY ());
    }

  return allDevices;
}

NetDeviceContainer
Experiment::FastWifi (NodeContainer &c, std::map<int, std::shared_ptr<ClientSession>> &clients)
{
  NodeContainer csmaNodes;
  NodeContainer serverNode;
  NodeContainer apNode;
  NodeContainer staNodes;
  NetDeviceContainer allDevices;

  serverNode.Add (c.Get (0)); // First node as Server
  apNode.Add (c.Get (1)); // Second node as AP
  csmaNodes.Add (serverNode);
  csmaNodes.Add (apNode);

  for (uint32_t i = 2; i < c.GetN (); ++i)
    {
      staNodes.Add (c.Get (i)); // Remaining nodes as Clients
    }

  // Setup Ethernet connection for the server and the router
  CsmaHelper csma;
  csma.SetChannelAttribute ("DataRate", StringValue (m_dataRate));
  csma.SetChannelAttribute ("Delay", TimeValue (NanoSeconds (6560)));
  NetDeviceContainer csmaDevices = csma.Install (csmaNodes);
  allDevices.Add (csmaDevices);

  // Setup Wi-Fi devices
  YansWifiChannelHelper wifiChannel = YansWifiChannelHelper::Default ();
  YansWifiPhyHelper wifiPhy;
  WifiHelper wifi;
  WifiMacHelper wifiMac;

  // Setup Channel
  wifiChannel.SetPropagationDelay ("ns3::ConstantSpeedPropagationDelayModel");
  // wifiChannel.AddPropagationLoss ("ns3::LogDistancePropagationLossModel", "Exponent",
  //                                 DoubleValue (2.0));

  // Setup Physical Layer
  wifiPhy.SetChannel (wifiChannel.Create ());
  wifiPhy.Set ("TxPowerStart", DoubleValue (18.0)); // High transmit power
  wifiPhy.Set ("TxPowerEnd", DoubleValue (20.0)); // High transmit power
  wifiPhy.SetErrorRateModel ("ns3::YansErrorRateModel");

  // Setup Wi-Fi for client connectivity
  wifi.SetStandard (WIFI_STANDARD_80211ac);
  wifi.SetRemoteStationManager ("ns3::MinstrelHtWifiManager");

  // Setup Mac for AP
  Ssid ssid = Ssid ("fast-wifi-ssid");
  wifiMac.SetType ("ns3::ApWifiMac", "Ssid", SsidValue (ssid));
  NetDeviceContainer apDevices = wifi.Install (wifiPhy, wifiMac, apNode);

  // Setup Mac for STA
  wifiMac.SetType ("ns3::StaWifiMac", "Ssid", SsidValue (ssid), "ActiveProbing",
                   BooleanValue (false));
  NetDeviceContainer staDevices = wifi.Install (wifiPhy, wifiMac, staNodes);

  allDevices.Add (apDevices);
  allDevices.Add (staDevices);
  // Setup Mobility
  MobilityHelper mobility;

  // Setup Mobility
  mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
  mobility.Install (apNode);
  mobility.Install (serverNode);

  // If mobile clients
  if (m_moving_clients == true)
    {
      SetupMobilityForStaNodes (staNodes, m_router_coordinates, MAX_DISTANCE_FROM_ROUTER);
    }
  mobility.Install (staNodes);
  // Setup Devices positions
  SetPositionCartesian (c.Get (0), m_server_coordinates[0], m_server_coordinates[1]);
  SetPositionCartesian (c.Get (1), m_router_coordinates[0], m_router_coordinates[1]);

  for (uint32_t j = 2; j < c.GetN (); ++j)
    {
      SetPositionCartesian (c.Get (j), clients[j - 2]->GetX (), clients[j - 2]->GetY ());
    }

  return allDevices;
  return allDevices;
}

NetDeviceContainer
Experiment::Ethernet (NodeContainer &c, std::map<int, std::shared_ptr<ClientSession>> &clients)
{
  CsmaHelper csma;
  csma.SetChannelAttribute ("DataRate", StringValue ("100Mbps"));
  csma.SetChannelAttribute ("Delay", TimeValue (NanoSeconds (6560)));

  NetDeviceContainer csmaDevices;
  csmaDevices = csma.Install (c);

  return csmaDevices;
}
//>>>>>>>>>>>>>>>>>>>>>>>>>>>> \Network Setup Helpers >>>>>>>>>>>>>>>>>>>>>>>>>>>>>

//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Experiment Helpers >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
NetDeviceContainer
Experiment::SetupDevices (NodeContainer &c, std::map<int, std::shared_ptr<ClientSession>> &clients)
{

  // Default Network Topology
  //
  //           WIFI 192.168.3.0
  //        AP
  //        *       *    *    *
  // srvr   |       |    |    |
  //  no   n1      n2   n3   n4
  //   |    |
  //   =======
  // CSMA 192.168.1.0
  //
  if (m_networkType.compare ("wifi") == 0)
    {
      if (m_wifi_net_template == 0) // Slow speed network
        {
          NS_LOG_UNCOND ("Weak Wifi Setup");
          return WeakWifi (c, clients);
        }
      else if (m_wifi_net_template == 1) // Mid speed network
        {
          NS_LOG_UNCOND ("Mid Wifi Setup");
          return MidWifi (c, clients);
        }
      else
        {
          NS_LOG_UNCOND ("Fast Wifi Setup"); // High speed network
          return FastWifi (c, clients);
        }
    }
  else
    { // assume ethernet if not specified
      return Ethernet (c, clients);
    }
}

Ipv4InterfaceContainer
Experiment::SetupInternetStack (NodeContainer &c, NetDeviceContainer &devices)
{
  InternetStackHelper internet;
  internet.Install (c);

  Ipv4AddressHelper ipv4;
  NetDeviceContainer csmaDevices;
  NetDeviceContainer wifiDevices;

  // Assuming the first 2 devices are CSMA and the rest are WiFi
  uint32_t nDevices = devices.GetN ();
  for (uint32_t i = 0; i < 2; ++i)
    {
      csmaDevices.Add (devices.Get (i));
    }
  for (uint32_t i = 2; i < nDevices; ++i)
    {
      wifiDevices.Add (devices.Get (i));
    }

  Ipv4InterfaceContainer csmaInterfaces;
  Ipv4InterfaceContainer wifiInterfaces;

  // ethernet devices
  ipv4.SetBase ("192.168.1.0", "255.255.255.0");
  csmaInterfaces = ipv4.Assign (csmaDevices);

  // wifi devices
  ipv4.SetBase ("192.168.3.0", "255.255.255.0");
  wifiInterfaces = ipv4.Assign (wifiDevices);

  // Combine interfaces
  Ipv4InterfaceContainer interfaces;
  interfaces.Add (csmaInterfaces);
  interfaces.Add (wifiInterfaces);

  // for (uint32_t i = 0; i < interfaces.GetN (); ++i)
  //   {
  //     NS_LOG_UNCOND ("IP: " << interfaces.GetAddress (i));
  //   }

  return interfaces;
}

Ptr<Server>
Experiment::SetupServer (Ptr<Node> server, int16_t server_port, Ipv4InterfaceContainer &interfaces,
                         ns3::Time &timeOffset, double start_time, double stop_time)
{

  NS_LOG_UNCOND ("Setup Server Address: " << interfaces.GetAddress (0) << ":" << server_port);

  ServerHelper server_helper ("ns3::TcpSocketFactory",
                              InetSocketAddress (interfaces.GetAddress (0), server_port));
  server_helper.SetAttribute ("MaxPacketSize", UintegerValue (m_maxPacketSize));
  server_helper.SetAttribute ("BytesModel", UintegerValue (m_modelSize));
  server_helper.SetAttribute ("ModelType", StringValue (m_modelType));
  server_helper.SetAttribute ("DataRate", StringValue (m_dataRate));
  server_helper.SetAttribute ("DeviceType", StringValue (m_deviceType));
  server_helper.SetAttribute ("Async", BooleanValue (m_bAsync));
  server_helper.SetAttribute ("TimeOffset", TimeValue (timeOffset));

  ApplicationContainer sinkApps = server_helper.Install (server);

  sinkApps.Start (Seconds (start_time));
  sinkApps.Stop (Seconds (stop_time));

  return DynamicCast<Server> (sinkApps.Get (0));
}

ApplicationContainer
Experiment::SetupClients (NodeContainer &c, std::map<int, std::shared_ptr<ClientSession>> &clients,
                          int16_t server_port, Ipv4InterfaceContainer &interfaces,
                          std::map<Ipv4Address, int> &m_addrMap, ns3::Time &timeOffset,
                          double start_time, double stop_time)
{
  ApplicationContainer clientApps;
  Ipv4Address serverAddress = interfaces.GetAddress (0);
  Address peerAddress (InetSocketAddress (serverAddress, server_port));

  NS_LOG_UNCOND ("Setup Client, Server Address: " << interfaces.GetAddress (0) << ":"
                                                  << server_port);
  // Get the server position
  Vector routerPosition = GetPosition (c.Get (1));
  for (uint32_t j = 2; j < c.GetN (); ++j)
    {
      if (clients[j - 2]->GetInRound ())
        {

          Ptr<Node> clientNode = c.Get (j);
          // Log the client's position and distance from the server
          LogClientCharacteristics (clientNode, routerPosition, j - 2);
          // Create the client application and set its attributes
          Ptr<ClientApplication> clientApp = CreateObject<ClientApplication> ();
          Ptr<Socket> socket = Socket::CreateSocket (clientNode, TcpSocketFactory::GetTypeId ());
          socket->SetAttribute ("ConnCount", UintegerValue (1000));
          // socket->SetAttribute ("DataRetries", UintegerValue (100));

          Ipv4Address clientAddress = interfaces.GetAddress (j + 1);
          m_addrMap[clientAddress] = j - 2;

          std::string client_datarate = clients[j - 2]->GetDataRate ();
          clientApp->Setup (socket, peerAddress, m_maxPacketSize, m_modelSize,
                            DataRate (client_datarate), m_deviceType, m_modelType);
          clientApp->SetStartTime (Seconds (start_time));
          clientApp->SetStopTime (Seconds (stop_time));

          clientNode->AddApplication (clientApp);
          clients[j - 2]->SetClient (socket);
          clients[j - 2]->SetCycle (0);
          clientApps.Add (clientApp); // Add this app to the container to be returned
        }
    }
  return clientApps;
}

AnimationInterface
Experiment::ConfigureAnimation (NodeContainer &c, double startTime)
{
  AnimationInterface anim ("animation.xml");
  // Set the background image with a scale that includes all nodes
  anim.SetBackgroundImage ("campus.jpg", 0, 0, .165, .145, 0.9);
  anim.EnablePacketMetadata (true); // Optional: Depends on your ns-3 build configuration
  if (m_moving_clients)
    anim.SetMobilityPollInterval (Seconds (1)); // Not moving so no need to update
  else
    anim.SetMobilityPollInterval (Seconds (100000)); // Not moving so no need to update

  anim.SetStartTime (Seconds (startTime));

  // Set the positions for the server and client nodes and configure their properties
  for (uint32_t j = 0; j < c.GetN (); ++j)
    {
      if (j == 0) // The first node is the server
        {
          // Set the server properties
          anim.UpdateNodeDescription (c.Get (j), "Server");
          anim.UpdateNodeColor (c.Get (j), 0, 255, 0); // Green color for the server
          anim.UpdateNodeSize (j, 4.0, 4.0); // Make the server node larger
        }
      else if (j == 1) // The second node is the router
        {
          anim.UpdateNodeDescription (c.Get (j), "Router");
          anim.UpdateNodeColor (c.Get (j), 0, 0, 255); // Blue color for the router
          anim.UpdateNodeSize (j, 2.0, 2.0); // Make the router node medium size
        }
      else
        {
          // Set the client properties
          anim.UpdateNodeDescription (c.Get (j), "Client " + std::to_string (j - 1));
          anim.UpdateNodeColor (c.Get (j), 255, 0, 0); // Red color for the clients
          anim.UpdateNodeSize (j, 3.0, 3.0); // Standard size for client nodes
        }
    }

  return anim;
}

void
Experiment::ExtractDownlinkResults (std::map<int, std::shared_ptr<ClientSession>> &clients,
                                    int16_t server_port, Ipv4InterfaceContainer &interfaces,
                                    std::map<Ipv4Address, FlwrProvider::Message> &stats)
{
  int numClients = clients.size ();
  // interfaces 0-> server, 1-> AP:ethernet, 2-> AP: WiFi
  for (int j = 2; j < numClients + 2; j++)
    {
      if (clients[j - 2]->GetInRound ())
        {
          auto app = clients[j - 2]->GetClient ()->GetNode ()->GetApplication (0);
          UintegerValue sent;
          UintegerValue rec;
          TimeValue beginDownLink;
          TimeValue endDownLink;
          Ipv4Address clientAddress;
          app->GetAttribute ("BytesSent", sent);
          app->GetAttribute ("BytesReceived", rec);
          app->GetAttribute ("BeginDownlink", beginDownLink);
          app->GetAttribute ("EndDownlink", endDownLink);

          clientAddress = InetSocketAddress::ConvertFrom (
                              InetSocketAddress (interfaces.GetAddress (j + 1), server_port))
                              .GetIpv4 ();

          stats[clientAddress].downlinkTime =
              (endDownLink.Get () - beginDownLink.Get ()).GetDouble () / 1000000000.0;

          NS_LOG_UNCOND ("[CLIENT]  "
                         << interfaces.GetAddress (0, 0) << " -> " << clientAddress << std::endl
                         << "  Recv=" << rec.Get () << " bytes" << std::endl
                         << "  Sent=" << sent.Get () << " bytes" << std::endl
                         << "  Begin downlink=" << beginDownLink.Get ().As (Time::S) << std::endl
                         << "  End downlink=" << endDownLink.Get ().As (Time::S) << std::endl
                         << "  Downlink duration=" << stats[clientAddress].downlinkTime
                         << std::endl);
        }
    }
}
void
Experiment::ExtractUplinkResults (Ptr<Server> server, int16_t server_port,
                                  Ipv4InterfaceContainer &interfaces,
                                  std::map<Ipv4Address, FlwrProvider::Message> &stats)
{
  auto sk = server->GetAcceptedSockets ();
  for (auto itr = sk.begin (); itr != sk.end (); itr++)
    {
      auto beginUplink = itr->second->m_timeBeginReceivingModelFromClient;
      auto endUplink = itr->second->m_timeEndReceivingModelFromClient;
      auto clientAddress = InetSocketAddress::ConvertFrom (itr->second->m_address).GetIpv4 ();
      auto serverAddress = interfaces.GetAddress (0, 0);

      NS_LOG_UNCOND (
          "[SERVER]  " << clientAddress << " -> " << serverAddress << std::endl
                       << "  Sent=     " << itr->second->m_bytesSent << " bytes" << std::endl
                       << "  Recv=     " << itr->second->m_bytesReceived << " bytes" << std::endl
                       << "  Begin uplink=" << beginUplink.As (Time::S) << std::endl
                       << "  End uplink=" << endUplink.As (Time::S) << std::endl
                       << "  Uplink duration=" << (endUplink - beginUplink).As (Time::S));

      stats[clientAddress].uplinkTime = (endUplink - beginUplink).GetDouble () / 1000000000.0;
      // To Mbps
      stats[clientAddress].throughput =
          itr->second->m_bytesReceived * 8.0 / 1000000.0 /
          ((endUplink.GetDouble () - beginUplink.GetDouble ()) / 1000000000.0);
    }
}

// Method to combine and process the extracted results
std::map<int, FlwrProvider::Message>
Experiment::ProcessResults (std::map<Ipv4Address, int> m_addrMap,
                            std::map<Ipv4Address, FlwrProvider::Message> &stats)
{
  std::map<int, FlwrProvider::Message> roundStats;
  for (auto itr : stats)
    {

      int id = m_addrMap[itr.first];
      NS_LOG_UNCOND ("ID " << id << "  ,ADDRESS: " << itr.first << "| DownlinkTime= " << std::fixed
                           << std::setprecision (2) << itr.second.downlinkTime << "s"
                           << "| UplinkTime= " << std::fixed << std::setprecision (2)
                           << itr.second.uplinkTime << "s"
                           << "| Round Throughput= " << std::fixed << std::setprecision (2)
                           << itr.second.throughput << "Mbps");

      roundStats[id].throughput = itr.second.throughput;
      roundStats[id].downlinkTime = itr.second.downlinkTime;
      roundStats[id].uplinkTime = itr.second.uplinkTime;
    }
  return roundStats;
}

//>>>>>>>>>>>>>>>>>>>>>>>>>>>>> \Experiment Helpers >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

} // namespace ns3