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
// #include "ns3/olsr-helper.h"
// #include "ns3/aodv-module.h"

#include <iomanip> // For std::setprecision and std::fixed
#include <algorithm> // For std::shuffle
#include <random> // For std::default_random_engine
#include <chrono> // For std::chrono::system_clock

namespace ns3 {

Experiment::Experiment (int numClients, std::string &networkType, int maxPacketSize, double txGain,
                        std::string &modelType, double modelSize, std::string &dataRate,
                        std::string &deviceType, bool bAsync, FlwrProvider *flwr_provider, FILE *fp,
                        std::vector<double> server_coordinates, int round)
    : m_numClients (numClients),
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
      m_round (round)
{
  // Constructor body (if needed)
}

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

NetDeviceContainer
Experiment::Wifi (NodeContainer &c, std::map<int, std::shared_ptr<ClientSession>> &clients)
{
  // 0. Define Helpers
  YansWifiChannelHelper wifiChannel;
  YansWifiPhyHelper wifiPhy;
  WifiHelper wifi;
  WifiMacHelper wifiMac;

  // 1.Setup Channel
  wifiChannel.AddPropagationLoss ("ns3::LogDistancePropagationLossModel", "Exponent",
                                  DoubleValue (3.0), "ReferenceLoss", DoubleValue (40.0));
  wifiChannel.AddPropagationLoss ("ns3::NakagamiPropagationLossModel");
  wifiChannel.SetPropagationDelay ("ns3::ConstantSpeedPropagationDelayModel");

  // 2. Setup Physical Layer
  wifiPhy.SetErrorRateModel ("ns3::YansErrorRateModel");
  wifiPhy.SetChannel (wifiChannel.Create ());
  wifiPhy.Set ("RxGain", DoubleValue (0));
  std::string phyMode ("HtMcs7");
  Config::SetDefault ("ns3::WifiRemoteStationManager::NonUnicastMode", StringValue (phyMode));

  // 3. Setup Wifi
  wifi.SetStandard (WIFI_STANDARD_80211n_5GHZ);
  wifi.SetRemoteStationManager ("ns3::MinstrelHtWifiManager");

  // 4. Setup Mac
  wifiMac.SetType ("ns3::AdhocWifiMac");

  // 5. Setup devices with network settings
  NetDeviceContainer devices = wifi.Install (wifiPhy, wifiMac, c);

  // 6. Setup device mobility
  MobilityHelper mobility;
  mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
  mobility.Install (c);

  int numClients = clients.size ();
  // 7. Setup Devices positions
  Experiment::SetPositionCartesian (c.Get (0), m_server_coordinates[0], m_server_coordinates[1]);
  for (int j = 1; j <= numClients; j++)
    {
      Experiment::SetPositionCartesian (c.Get (j), clients[j - 1]->GetX (),
                                        clients[j - 1]->GetY ());
    }
  return devices;
}
//>>>>>>>>>>>>>>>>>>>>>>>>>>>> \Network Setup Helpers >>>>>>>>>>>>>>>>>>>>>>>>>>>>>

//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Experiment Helpers >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
NetDeviceContainer
Experiment::SetupDevices (NodeContainer &c, std::map<int, std::shared_ptr<ClientSession>> &clients)
{
  if (m_networkType.compare ("wifi") == 0)
    {
      return Wifi (c, clients);
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
  ipv4.SetBase ("10.1.1.0", "255.255.255.0");
  return ipv4.Assign (devices);
}

Ptr<Server>
Experiment::SetupServer (Ptr<Node> server, Ipv4InterfaceContainer &interfaces,
                         ns3::Time &timeOffset, double start_time, double stop_time)
{
  ServerHelper server_helper ("ns3::TcpSocketFactory",
                              InetSocketAddress (Ipv4Address::GetAny (), 80));
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
                          Ipv4InterfaceContainer &interfaces, std::map<Ipv4Address, int> &m_addrMap,
                          ns3::Time &timeOffset, double start_time, double stop_time)
{
  ApplicationContainer clientApps;

  int numClients = clients.size ();
  for (int j = 1; j <= numClients; ++j)
    {
      if (clients[j - 1]->GetInRound ())
        {
          Ptr<Socket> source = Socket::CreateSocket (c.Get (j), TcpSocketFactory::GetTypeId ());
          m_addrMap[c.Get (j)->GetObject<Ipv4> ()->GetAddress (1, 0).GetLocal ()] = j - 1;

          source->SetAttribute ("ConnCount", UintegerValue (1000));
          source->SetAttribute ("DataRetries", UintegerValue (100));

          Ptr<ClientApplication> app = CreateObject<ClientApplication> ();
          char *client_datarate = clients[j - 1]->GetDataRate ();
          app->Setup (source, InetSocketAddress (interfaces.GetAddress (0), 80), m_maxPacketSize,
                      m_modelSize, std::string (client_datarate), m_deviceType, m_modelType);
          c.Get (j)->AddApplication (app);
          app->SetStartTime (Seconds (start_time));
          app->SetStopTime (Seconds (stop_time)); // Example stop time, adjust as needed

          clients[j - 1]->SetClient (source);
          clients[j - 1]->SetCycle (0);
          clientApps.Add (app); // Add this app to the container to be returned
          NS_LOG_UNCOND ("In Round Client " << j - 1 << " Datarate: " << client_datarate);
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
                                    Ipv4InterfaceContainer &interfaces,
                                    std::map<Ipv4Address, FlwrProvider::Message> &stats)
{
  int numClients = clients.size ();
  for (int j = 1; j <= numClients; j++)
    {
      if (clients[j - 1]->GetInRound ())
        {
          auto app = clients[j - 1]->GetClient ()->GetNode ()->GetApplication (0);
          UintegerValue sent;
          UintegerValue rec;
          TimeValue beginDownLink;
          TimeValue endDownLink;
          Ipv4Address clientAddress;
          app->GetAttribute ("BytesSent", sent);
          app->GetAttribute ("BytesReceived", rec);
          app->GetAttribute ("BeginDownlink", beginDownLink);
          app->GetAttribute ("EndDownlink", endDownLink);

          clientAddress =
              InetSocketAddress::ConvertFrom (InetSocketAddress (interfaces.GetAddress (j, 0), 80))
                  .GetIpv4 ();

          stats[clientAddress].downlinkTime =
              (endDownLink.Get () - beginDownLink.Get ()).GetDouble () / 1000000000.0;

          NS_LOG_UNCOND ("[CLIENT]  "
                         << "10.1.1.1 -> " << clientAddress << std::endl
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
Experiment::ExtractUplinkResults (Ptr<Server> server,
                                  std::map<Ipv4Address, FlwrProvider::Message> &stats)
{
  auto sk = server->GetAcceptedSockets ();
  for (auto itr = sk.begin (); itr != sk.end (); itr++)
    {
      auto beginUplink = itr->second->m_timeBeginReceivingModelFromClient;
      auto endUplink = itr->second->m_timeEndReceivingModelFromClient;
      auto clientAddress = InetSocketAddress::ConvertFrom (itr->second->m_address).GetIpv4 ();

      NS_LOG_UNCOND (
          "[SERVER]  " << clientAddress << " -> 10.1.1.1" << std::endl
                       << "  Sent=     " << itr->second->m_bytesSent << " bytes" << std::endl
                       << "  Recv=     " << itr->second->m_bytesReceived << " bytes" << std::endl
                       << "  Begin uplink=" << beginUplink.As (Time::S) << std::endl
                       << "  End uplink=" << endUplink.As (Time::S) << std::endl
                       << "  Uplink duration=" << (endUplink - beginUplink).As (Time::S));

      stats[clientAddress].uplinkTime = (endUplink - beginUplink).GetDouble () / 1000000000.0;
      stats[clientAddress].throughput =
          itr->second->m_bytesReceived * 8.0 / 1000.0 /
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
                           << itr.second.throughput << "kbps");

      roundStats[id].throughput = itr.second.throughput;
      roundStats[id].downlinkTime = itr.second.downlinkTime;
      roundStats[id].uplinkTime = itr.second.uplinkTime;
    }
  return roundStats;
}

//>>>>>>>>>>>>>>>>>>>>>>>>>>>>> \Experiment Helpers >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

//>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Experiment Round Run >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

std::map<int, FlwrProvider::Message>
Experiment::Run_Round (std::map<int, std::shared_ptr<ClientSession>> &clients,
                       ns3::Time &timeOffset)
{
  NS_LOG_UNCOND ("Experiment Round Setup");
  double server_start_time = 3.0;
  double stop_time = 1000000.0;
  double client_start_time = 5.0;

  // 1.Devices Setup
  NodeContainer c;
  c.Create (clients.size () + 1);
  NetDeviceContainer devices = SetupDevices (c, clients);

  // 2.Internet Stack Setup
  Ipv4InterfaceContainer interfaces = SetupInternetStack (c, devices);

  // 3.Server Setup
  Ptr<Server> serverApp =
      SetupServer (c.Get (0), interfaces, timeOffset, server_start_time, stop_time);

  // 4.Clients Setup
  std::map<Ipv4Address, int> m_addrMap;
  ApplicationContainer clientApps =
      SetupClients (c, clients, interfaces, m_addrMap, timeOffset, client_start_time, stop_time);

  // 5.Client Session Manager Setup
  ClientSessionManager client_session_manager (clients);
  serverApp->GetObject<ns3::Server> ()->SetClientSessionManager (&client_session_manager,
                                                                 m_flwrProvider, m_fp, m_round);
  // 6. Netanim Interface Setup
  AnimationInterface anim = ConfigureAnimation (c, client_start_time);
  NS_LOG_UNCOND ("Round Setup Complete");

  // 7.Run Simulation
  Simulator::Stop (Seconds (stop_time));
  NS_LOG_UNCOND ("================= Starting Ns3 Round Simulation ===================");
  Simulator::Run ();

  // 8.Extract Simulation Results
  std::map<Ipv4Address, FlwrProvider::Message> stats;
  ExtractDownlinkResults (clients, interfaces, stats);
  ExtractUplinkResults (serverApp, stats);

  std::map<int, FlwrProvider::Message> roundStats = ProcessResults (m_addrMap, stats);

  Simulator::Destroy ();
  return roundStats;
}
//>>>>>>>>>>>>>>>>>>>>>>>>>>>> \Experiment Round Run >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

} // namespace ns3
