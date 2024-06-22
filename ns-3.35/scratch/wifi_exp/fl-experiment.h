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

#ifndef IOT_EXPERIMENT_H
#define IOT_EXPERIMENT_H

#include "ns3/command-line.h"
#include "ns3/config.h"
#include "ns3/uinteger.h"
#include "ns3/string.h"
#include "ns3/log.h"
#include "ns3/yans-wifi-helper.h"
#include "ns3/mobility-helper.h"
#include "ns3/ipv4-address-helper.h"
#include "ns3/on-off-helper.h"
#include "ns3/yans-wifi-channel.h"
#include "ns3/mobility-model.h"
#include "ns3/packet-socket-helper.h"
#include "ns3/packet-socket-address.h"
#include "flwr-interface.h"
#include "fl-client-session.h"
#include "ns3/netanim-module.h"
#include "fl-server.h"

#include <memory>
#include <string>

#include <cstdio>

namespace ns3 {
    /**
  * \ingroup fl-experiment
  * \brief Sets up and runs fl experiments
  */
    class Experiment {
    public:
        /**
        * \brief Constructs Experiment
        * \param numClients             Number of clients in experiment
        * \param wifi_net_template 
        * \param moving_clients
        * \param networkType            Network type (wifi or ethernet)
        * \param maxPacketSize          Max packet size for network
        * \param txGain                 TX gain for wifi network
        * \param modelType              Name of modelType trained
        * \param modelSize              Model size
        * \param dataRate               Datarate for server
        * \param deviceType             Type of client Device
        * \param bAsync                 If running async experiment, true
        * \param pFlwrProvider          pointer to an flwr-interface (used to communicate with flower)
        * \param server_coordinates     coordinates of server used in NetAnim
        * \param round                  The number of the simulated round 
        */
        Experiment (int numClients, int wifi_net_template, bool moving_clients,
                        std::string &networkType, int maxPacketSize, double txGain,
                        std::string &modelType, double modelSize, std::string &dataRate,
                        std::string &deviceType, bool bAsync, FlwrProvider *flwr_provider, FILE *fp,
                        std::vector<double> server_coordinates, int round);
        /**
        * \brief Runs network experiment
        * \param packetsReceived   map of <client, client sessions>
        * \param timeOffset        Async, make timeline between rounds continious
        * \return                  map of <client id, message>, messages to send back to flower for each client
        */
        std::pair<std::map<int, FlwrProvider::Message>, std::vector<std::pair<double, double>>>
        Run_Round(std::map<int, std::shared_ptr<ClientSession> > &clients, ns3::Time &timeOffset);


    private:
        void LogClientCharacteristics (Ptr<Node> clientNode, const Vector &serverPosition,int clientIndex);

        void
        SetupMobilityForStaNodes (NodeContainer &staNodes,const std::vector<double> &router_coords, double radius);
        
        /**
        * \brief Set position of node in network
        * \param node        Node to set position of
        * \param radius      Radius location of node
        * \param theta       Angular location of node
        */
        void SetPositionPolar(Ptr <Node> node, double radius, double theta);

        /**
        * \brief Set position of node in network
        * \param node        Node to set position of
        * \param x           x location of node
        * \param y           y location of node
        */
        void SetPositionCartesian(Ptr <Node> node, double x, double y);

        /**
        * \brief Gets position of node
        * \param node   Node to get position of
        * \return       Vector of node position
        */
        Vector GetPosition(Ptr <Node> node);

        /**
        * \brief Sets up Wifi network
        * \param c         Container of nodes used in network
        * \param clients   Mapping from ids to client sessions 
        * \return          The Net Device container of all the Wifi devices
        */
        NetDeviceContainer MyWifi (NodeContainer &c, std::map<int, std::shared_ptr<ClientSession>> &clients);
        
        /**
        * \brief Sets up Wifi network
        * \param c         Container of nodes used in network
        * \param clients   Mapping from ids to client sessions 
        * \return          The Net Device container of all the Wifi devices
        */
        NetDeviceContainer WeakWifi (NodeContainer &c, std::map<int, std::shared_ptr<ClientSession>> &clients);

        /**
        * \brief Sets up Wifi network
        * \param c         Container of nodes used in network
        * \param clients   Mapping from ids to client sessions 
        * \return          The Net Device container of all the Wifi devices
        */
        NetDeviceContainer MidWifi (NodeContainer &c, std::map<int, std::shared_ptr<ClientSession>> &clients);

        /**
        * \brief Sets up Wifi network
        * \param c         Container of nodes used in network
        * \param clients   Mapping from ids to client sessions 
        * \return          The Net Device container of all the Wifi devices
        */
        NetDeviceContainer FastWifi (NodeContainer &c, std::map<int, std::shared_ptr<ClientSession>> &clients);




        /**
        * \brief Sets up Etherner network
        * \param c         Container of nodes used in network
        * \param clients   Mapping from ids to client sessions 
        * \return          The Net Device container of all the Ethernet devices
        */
        NetDeviceContainer Ethernet(NodeContainer &c, std::map<int, std::shared_ptr<ClientSession> > &clients);

        /**
        * \brief Sets up the network devices
        * \param c         Container of nodes used in network
        * \param clients   Mapping from ids to client sessions 
        * \return          The Net Device container of all the Network devices
        */
        NetDeviceContainer SetupDevices(NodeContainer& c, std::map<int, std::shared_ptr<ClientSession>>& clients);
        
        /**
        * \brief Sets up the Internet stack
        * \param c         Container of nodes used in network
        * \param devices   The network devices setup for the network 
        * \return          A container of Interfaces assigned to netDevices
        */
        Ipv4InterfaceContainer SetupInternetStack(NodeContainer& c, NetDeviceContainer& devices);
    
        /**
        * \brief Setup the server for simulation with given attributes 
        * \param server     The server Node
        * \param interfaces The Ipv4 interfaces container
        * \param timeOffset Parameter for async setup
        * \param start_time The start time in simulation for server
        * \param stop_time  The stop time in simulation for server
        * \return           The Server application
        */
        Ptr<Server> SetupServer (Ptr<Node> server,int16_t server_port, Ipv4InterfaceContainer &interfaces, ns3::Time &timeOffset, 
            double start_time, double stop_time);
        
        /**
        * \brief Setup the clients for simulation with given attributes 
        * \param c          Container of nodes used in network
        * \param clients    Mapping from ids to client sessions 
        * \param interfaces The Ipv4 interfaces container
        * \param m_addrMap  Mapping from Ip address to id
        * \param timeOffset parameter for async setup
        * \param start_time The start time in simulation for the clients
        * \param stop_time  The stop time in simulation for the clients
        * \return           An application container of client apps
        */
        ApplicationContainer SetupClients (NodeContainer &c, std::map<int, std::shared_ptr<ClientSession>> &clients, int16_t server_port,
                          Ipv4InterfaceContainer &interfaces, std::map<Ipv4Address, int> &m_addrMap,
                          ns3::Time &timeOffset, double start_time, double stop_time);
        
        /**
        * \brief Setup the Netanim Module for optical representation
        * \param c          Container of nodes used in network
        * \param start_time The start time of packet monitoring
        * \return           The instance of the Animation Interface
        */
        AnimationInterface ConfigureAnimation(NodeContainer &c,double start_time);
        
        /**
        * \brief Extract the downlink phase results of the simulation
        * \param clients    Mapping from ids to client sessions 
        * \param interfaces The Ipv4 interfaces container
        * \param stats      An empty mapping from client address to simulation results 
        */
        void ExtractDownlinkResults (std::map<int, std::shared_ptr<ClientSession>> &clients,
                                    int16_t server_port, Ipv4InterfaceContainer &interfaces,
                                    std::map<Ipv4Address, FlwrProvider::Message> &stats);
        
        /**
        * \brief Extract the uplink phase results of the simulation
        * \param server    The Server Application 
        * \param stats     Mapping from client address to simulation results 
        */
        void ExtractUplinkResults (Ptr<Server> server, int16_t server_port,
                                  Ipv4InterfaceContainer &interfaces,
                                  std::map<Ipv4Address, FlwrProvider::Message> &stats);
        
        /**
        * \brief Process the results acquired from downlink and uplink fot final result 
        * \param m_addrMap  Mapping from Ip address to id
        * \param stats      Mapping from client address to simulation results 
        * \return           Mapping from client id to simulation results 
        */
        std::map<int, FlwrProvider::Message> ProcessResults(std::map<Ipv4Address, int> m_addrMap,std::map<Ipv4Address, FlwrProvider::Message> &stats);


        int m_numClients;                           //!< Number of clients in experiment
        int m_wifi_net_template; 
        bool m_moving_clients;
        std::string m_networkType;                  //!< Network type
        int m_maxPacketSize;                        //!< Max packet size
        double m_txGain;                            //!< TX gain (for wifi network)
        std::string m_modelType;                    //!< modelType used for training 
        std::string m_deviceType;                   //!< Device type used for training 
        double m_modelSize;                         //!< Size of model
        std::string m_dataRate;                     //!< Datarate for server
        bool m_bAsync;                              //!< Indicator bool for whether experiement is async
        FlwrProvider *m_flwrProvider;               //!< pointer to an flwr-interface (used to communicate with flower)
        FILE *m_fp;                                 //!< pointer to logfile
        std::vector<double> m_server_coordinates;   //!< coordinates of server used in NetAnim
        std::vector<double> m_router_coordinates;   //!< coordinates of server used in NetAnim
        int m_round;                                //!< experiment round
        static const int MAX_DISTANCE_FROM_ROUTER = 20;



    };
}

#endif //IOT_EXPERIMENT_H
