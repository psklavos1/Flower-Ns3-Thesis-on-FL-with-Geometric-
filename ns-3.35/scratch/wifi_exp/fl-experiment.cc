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
#include "fl-server.h"
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
#include "ns3/olsr-helper.h"
#include "ns3/aodv-module.h"
#include "ns3/netanim-module.h"

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
}

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

  YansWifiChannelHelper wifiChannel = YansWifiChannelHelper ();
  YansWifiPhyHelper wifiPhy;
  WifiHelper wifi;
  WifiMacHelper wifiMac;

  wifiChannel.AddPropagationLoss ("ns3::LogDistancePropagationLossModel", "Exponent",
                                  DoubleValue (3.0), "ReferenceLoss", DoubleValue (40.0));
  wifiChannel.AddPropagationLoss ("ns3::NakagamiPropagationLossModel");
  wifiChannel.SetPropagationDelay ("ns3::ConstantSpeedPropagationDelayModel");

  wifiPhy.SetErrorRateModel ("ns3::YansErrorRateModel");
  wifiPhy.SetChannel (wifiChannel.Create ());
  wifiPhy.Set ("RxGain", DoubleValue (0));
  std::string phyMode ("HtMcs7");
  Config::SetDefault ("ns3::WifiRemoteStationManager::NonUnicastMode", StringValue (phyMode));

  wifi.SetStandard (WIFI_STANDARD_80211n_5GHZ);

  // Add a mac and disable rate control
  wifi.SetRemoteStationManager ("ns3::MinstrelHtWifiManager");

  // Set it to adhoc mode
  wifiMac.SetType ("ns3::AdhocWifiMac");

  NetDeviceContainer devices = wifi.Install (wifiPhy, wifiMac, c);

  MobilityHelper mobility;
  mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
  mobility.Install (c);

  int numClients = clients.size ();
  // Set the server too
  // In Polar system Server always in the middle
  Experiment::SetPositionCartesian (c.Get (0), m_server_coordinates[0], m_server_coordinates[1]);
  // clients used in round are positioned accordingly
  for (int j = 1; j <= numClients; j++)
    {
      // if (clients[j - 1]->GetInRound ())
      // {

      Experiment::SetPositionCartesian (c.Get (j), clients[j - 1]->GetX (),
                                        clients[j - 1]->GetY ());
      // }
    }

  return devices;
}

std::map<int, FlwrProvider::Message>
Experiment::WeakNetwork (std::map<int, std::shared_ptr<ClientSession>> &clients,
                         ns3::Time &timeOffset)
{
  NS_LOG_UNCOND ("Experiment Round Setup");
  int server = 0;
  int numClients = clients.size ();

  NodeContainer c;
  c.Create (numClients + 1);

  NetDeviceContainer devices;
  // const char **strings;
  if (m_networkType.compare ("wifi") == 0)
    {
      devices = Wifi (c, clients);
      // strings = wifi_strings;
    }
  else //assume ethernet if not specified
    {
      devices = Ethernet (c, clients);
      // strings = ethernet_strings;
    }

  // Choose best routing practise for my environment
  // Case 1: Static Routing. make static rout for each node
  // Case 2: OLSR maintains found routs. Leading to lower latency than aodv
  // Case 3: Aodv is best fitted for dynamic envinonments finding alwayes new routes
  // cost is latency to find route,
  // OlsrHelper olsr;
  // internet.SetRoutingHelper(olsr); // Use olsr routing
  // AodvHelper aodv;
  // internet.SetRoutingHelper(aodv); // Set AODV as the routing protocol

  InternetStackHelper internet;
  // OlsrHelper olsr;
  // Ipv4StaticRoutingHelper staticRouting;

  // Ipv4ListRoutingHelper list;
  // list.Add (staticRouting, 0);
  // list.Add (olsr, 10);

  // internet.SetRoutingHelper (list);
  // AodvHelper aodv;
  // internet.SetRoutingHelper(aodv); // Set AODV as the routing protocol
  internet.Install (c);
  Ipv4AddressHelper ipv4;
  ipv4.SetBase ("10.1.1.0", "255.255.255.0");
  Ipv4InterfaceContainer interfaces = ipv4.Assign (devices);

  //Setup Server
  ServerHelper server_helper ("ns3::TcpSocketFactory",
                              InetSocketAddress (Ipv4Address::GetAny (), 80));
  server_helper.SetAttribute ("MaxPacketSize", UintegerValue (m_maxPacketSize));
  server_helper.SetAttribute ("BytesModel", UintegerValue (m_modelSize));
  server_helper.SetAttribute ("ModelType", StringValue (m_modelType));
  server_helper.SetAttribute ("DataRate", StringValue (m_dataRate));
  server_helper.SetAttribute ("DeviceType", StringValue (m_deviceType));

  server_helper.SetAttribute ("Async", BooleanValue (m_bAsync));
  server_helper.SetAttribute ("TimeOffset", TimeValue (timeOffset));
  ApplicationContainer sinkApps = server_helper.Install (c.Get (server));

  sinkApps.Start (Seconds (3.));

  Address sinkAddress (InetSocketAddress (interfaces.GetAddress (server, 0), 80));
  std::map<Ipv4Address, int> m_addrMap;
  //initialize clients

  for (int j = 1; j <= numClients; j++)
    {
      if (clients[j - 1]->GetInRound ())
        {

          // Experiment::SetPosition (c.Get (j), clients[j - 1]->radius, clients[j - 1]->theta);
          Ptr<Socket> source = Socket::CreateSocket (c.Get (j), TcpSocketFactory::GetTypeId ());

          m_addrMap[c.Get ((j))->GetObject<Ipv4> ()->GetAddress (1, 0).GetLocal ()] = j - 1;

          source->SetAttribute ("ConnCount", UintegerValue (1000));
          source->SetAttribute ("DataRetries", UintegerValue (100));

          Ptr<ClientApplication> app = CreateObject<ClientApplication> ();
          char *client_datarate = clients[j - 1]->GetDataRate ();
          app->Setup (source, sinkAddress, m_maxPacketSize, m_modelSize,
                      std::string (client_datarate), m_deviceType, m_modelType);
          c.Get (j)->AddApplication (app);
          // Let sometime for Server to be ready. Especially when using OLSR at least 3 s.
          app->SetStartTime (Seconds (5.));
          app->SetStopTime (Seconds (1000000.0));

          clients[j - 1]->SetClient (source);
          clients[j - 1]->SetCycle (0);
          NS_LOG_UNCOND ("In Round Client " << j - 1 << " Datarate: " << client_datarate);
        }
    }
  ClientSessionManager client_session_manager (clients);
  sinkApps.Get (0)->GetObject<ns3::Server> ()->SetClientSessionManager (
      &client_session_manager, m_flwrProvider, m_fp, m_round);

  // ======================== NetAnim Section =========================
  // double imageScale = 100.0; // This is an arbitrary scale;

  AnimationInterface anim ("animation.xml");
  // Set the background image with a scale that includes all nodes

  anim.SetBackgroundImage ("campus.jpg", 0, 0, .165, .145, 0.9);
  anim.EnablePacketMetadata (true); // Optional: Depends on your ns-3 build configuration
  anim.SetMobilityPollInterval (Seconds (10000)); // Not moving so no need to update
  anim.SetStartTime (Seconds (5));

  // Set the positions for the server and client nodes and configure their properties
  for (uint32_t j = 0; j < c.GetN (); ++j)
    {
      // Vector pos = Experiment::GetPosition (c.Get (j));
      // NS_LOG_UNCOND ("Node " << j << ": Position (" << pos.x << ", " << pos.y << ")");

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
  // ========================= \NetAnim Section =========================

  Simulator::Stop (Seconds (1000000.0));
  NS_LOG_UNCOND ("================= Starting Ns3 Round Simulation ===================");
  Simulator::Run ();

  TimeValue endTime;
  sinkApps.Get (0)->GetObject<ns3::Server> ()->GetAttribute ("TimeOffset", endTime);
  timeOffset = endTime.Get ();

  std::map<int, FlwrProvider::Message> roundStats;
  // This part is in reverse order.
  // First are the the stats about the server receiving the model after training
  if (m_bAsync == false)
    {
      std::map<Ipv4Address, FlwrProvider::Message> stats;

      // The below code is for the server sending the model to the clients before the fit has started.
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

              clientAddress = InetSocketAddress::ConvertFrom (
                                  InetSocketAddress (interfaces.GetAddress (j, 0), 80))
                                  .GetIpv4 ();

              stats[clientAddress].downlinkTime =
                  (endDownLink.Get () - beginDownLink.Get ()).GetDouble () / 1000000000.0;
              // temp value to update below. This is the start of the round for the client i
              stats[clientAddress].computationTime = endDownLink.Get ().GetDouble ();

              NS_LOG_UNCOND ("[CLIENT]  "
                             << "10.1.1.1 -> " << clientAddress << std::endl
                             << "  Recv=" << rec.Get () << " bytes" << std::endl
                             << "  Sent=" << sent.Get () << " bytes" << std::endl
                             << "  Begin downlink=" << beginDownLink.Get ().As (Time::S)
                             << std::endl
                             << "  End downlink=" << endDownLink.Get ().As (Time::S) << std::endl
                             << "  Downlink duration=" << stats[clientAddress].downlinkTime
                             << std::endl);

              // double td = endDownLink.Get ().GetDouble () - beginDownLink.Get ().GetDouble ();
              // stats[clientAddress].roundTime = (stats[clientAddress].roundTime + td) / 1000000000.0;
            }
        }

      auto s1 = sinkApps.Get (0)->GetObject<ns3::Server> ();
      auto sk = s1->GetAcceptedSockets ();
      std::map<Ipv4Address, double> uplinkRound;
      for (auto itr = sk.begin (); itr != sk.end (); itr++)
        {
          auto beginUplink = itr->second->m_timeBeginReceivingModelFromClient;
          auto endUplink = itr->second->m_timeEndReceivingModelFromClient;
          auto clientAddress = InetSocketAddress::ConvertFrom (itr->second->m_address).GetIpv4 ();

          NS_LOG_UNCOND ("[SERVER]  "
                         << clientAddress << " -> 10.1.1.1" << std::endl
                         << "  Sent=     " << itr->second->m_bytesSent << " bytes" << std::endl
                         << "  Recv=     " << itr->second->m_bytesReceived << " bytes" << std::endl
                         << "  Begin uplink=" << beginUplink.As (Time::S) << std::endl
                         << "  End uplink=" << endUplink.As (Time::S) << std::endl
                         << "  Uplink duration=" << (endUplink - beginUplink).As (Time::S));

          // Tweaked Implementation so that the computational time being Extracted from flower instead of the simulator
          // stats[clientAddress].roundTime = endUplink.GetDouble ();
          // Initially the roundTime was calculated starting from the moment the server started sending the model
          // to the client to the point where the moment the client sent the whole model back
          stats[clientAddress].uplinkTime = (endUplink - beginUplink).GetDouble () / 1000000000.0;
          stats[clientAddress].computationTime =
              (beginUplink.GetDouble () - stats[clientAddress].computationTime) / 1000000000.0;

          stats[clientAddress].throughput =
              itr->second->m_bytesReceived * 8.0 / 1000.0 /
              ((endUplink.GetDouble () - beginUplink.GetDouble ()) / 1000000000.0);
        }

      for (auto itr : stats)
        {

          int id = m_addrMap[itr.first];

          NS_LOG_UNCOND ("ID " << id << "  ,ADDRESS: " << itr.first
                               << "| DownlinkTime= " << std::fixed << std::setprecision (2)
                               << itr.second.downlinkTime << "s"
                               << "| ComputationTime= " << std::fixed << std::setprecision (2)
                               << itr.second.computationTime << "s"
                               << "| UplinkTime= " << std::fixed << std::setprecision (2)
                               << itr.second.uplinkTime << "s"
                               << "| Round Throughput= " << std::fixed << std::setprecision (2)
                               << itr.second.throughput << "kbps");

          roundStats[id].throughput = itr.second.throughput;
          roundStats[id].downlinkTime = itr.second.downlinkTime;
          roundStats[id].computationTime = itr.second.computationTime;
          roundStats[id].uplinkTime = itr.second.uplinkTime;
        }
    }
  Simulator::Destroy ();
  return roundStats;
}

} // namespace ns3

// NetDeviceContainer
// Experiment::Wifi (NodeContainer &c, std::map<int, std::shared_ptr<ClientSession>> &clients)
// {

//   WifiHelper wifi;
//   WifiMacHelper wifiMac;
//   YansWifiPhyHelper wifiPhy;
//   YansWifiChannelHelper wifiChannel = YansWifiChannelHelper ();

//   // wifiPhy.Set ("TxGain", DoubleValue (m_txGain)); //-23.5) );

//   wifiPhy.SetErrorRateModel ("ns3::YansErrorRateModel");
//   // phyHelper.Set("TxPowerStart", DoubleValue(m_txGain)); // Transmission power
//   // phyHelper.Set("TxPowerEnd", DoubleValue(m_txGain));

//   //90 Weak Network
//   //70 Medium
//   //30 Stong
//   //double trigger = 30.0;

//   Ptr<UniformRandomVariable> expVar = CreateObjectWithAttributes<UniformRandomVariable> (
//       "Min", DoubleValue (m_txGain), "Max", DoubleValue (m_txGain + 30));

//   wifiChannel.AddPropagationLoss ("ns3::RandomPropagationLossModel", "Variable",
//                                   PointerValue (expVar));

//   wifiChannel.SetPropagationDelay ("ns3::ConstantSpeedPropagationDelayModel");
//   //wifiChannel.SetPropagationDelay("ns3::RandomPropagationDelayModel", "Variable", StringValue ("ns3::UniformRandomVariable[Min=0|Max=2]"));

//   // std::string phyMode ("DsssRate11Mbps"); -- slower
//   // Direct Sequence Sepread Spectrum. This modulation technique spreads the signal into  a wider frequency band than the original data bandwidth.
//   // It offers benefits like resistance to interference and improved signal reception quality.
//   // the above may be obsolete try using the following
//   // VhtMcs8 -- faster or HtMc7
//   std::string phyMode (
//       "HtMcs7"); // better throughputs up to 150 Mbs than first(Up to 11Mbps) slower than second(1Gbps). Good for lab environment

//   // Fix non-unicast data rate to be the same as that of unicast
//   Config::SetDefault ("ns3::WifiRemoteStationManager::NonUnicastMode", StringValue (phyMode));
//   // wifi.SetStandard (WIFI_STANDARD_80211b);
//   wifi.SetStandard (WIFI_STANDARD_80211n_5GHZ);

//   // This is one parameter that matters when using FixedRssLossModel
//   // set it to zero; otherwise, gain will be added
//   wifiPhy.Set ("RxGain", DoubleValue (0));

//   // ns-3 supports RadioTap and Prism tracing extensions for 802.11b
//   //   wifiPhy.SetPcapDataLinkType(WifiPhyHelper::DLT_IEEE802_11_RADIO);

//   wifiPhy.SetChannel (wifiChannel.Create ());

//   // Add a mac and disable rate control
//   // The manager below has constant rate. So I try MinstrelHtWifiManager
//   // Designed for high throughput nets and changes the sending rate. Try and see
//   // how it responds.
//   // wifi.SetRemoteStationManager ("ns3::MinstrelHtWifiManager");
//   wifi.SetRemoteStationManager ("ns3::ConstantRateWifiManager", "DataMode", StringValue ("HtMcs7"),
//                                 "ControlMode", StringValue ("HtMcs0"));
//   // Set it to adhoc mode
//   wifiMac.SetType ("ns3::AdhocWifiMac");

//   NetDeviceContainer devices = wifi.Install (wifiPhy, wifiMac, c);

//   MobilityHelper mobility;
//   mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
//   mobility.Install (c);

//   int numClients = clients.size ();
//   for (int j = 1; j <= numClients; j++)
//     {
//       if (clients[j - 1]->GetInRound ())
//         {

//           Experiment::SetPosition (c.Get (j), clients[j - 1]->GetRadius (),
//                                    clients[j - 1]->GetTheta ());
//         }
//     }

//   return devices;
// }