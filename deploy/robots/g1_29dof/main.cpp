#include "FSM/CtrlFSM.h"
#include "FSM/State_Passive.h"
#include "FSM/State_FixStand.h"
#include "FSM/State_RLBase.h"
#include "State_Mimic.h"
#include "pad_input.h"

std::unique_ptr<LowCmd_t> FSMState::lowcmd = nullptr;
std::shared_ptr<LowState_t> FSMState::lowstate = nullptr;
std::shared_ptr<Keyboard> FSMState::keyboard = std::make_shared<Keyboard>();

void init_fsm_state()
{
    auto lowcmd_sub = std::make_shared<unitree::robot::g1::subscription::LowCmd>();
    usleep(0.2 * 1e6);
    if(!lowcmd_sub->isTimeout())
    {
        spdlog::critical("The other process is using the lowcmd channel, please close it first.");
        unitree::robot::go2::shutdown();
        // exit(0);
    }
    FSMState::lowcmd = std::make_unique<LowCmd_t>();
    FSMState::lowstate = std::make_shared<LowState_t>();
    spdlog::info("Waiting for connection to robot...");
    FSMState::lowstate->wait_for_connection();
    spdlog::info("Connected to robot.");
}

int main(int argc, char** argv)
{
    // Load parameters
    auto vm = param::helper(argc, argv);

    std::cout << " --- Unitree Robotics --- \n";
    std::cout << "     G1-29dof Controller \n";

    // Unitree DDS Config
    unitree::robot::ChannelFactory::Instance()->Init(0, vm["network"].as<std::string>());

    init_fsm_state();

    // Optional laptop pad over UDP (Wi-Fi), merged with the Unitree remote. Off unless --pad-port is given.
    if(vm["pad-port"].as<int>() > 0)
    {
        PadInput::Config pc;
        pc.port = vm["pad-port"].as<int>();
        const std::string keyhex = vm["pad-key"].as<std::string>();
        bool key_ok = keyhex.size() == 16;
        for(int i = 0; key_ok && i < 8; i++)
        {
            unsigned v = 0;
            key_ok = std::sscanf(keyhex.substr(2 * i, 2).c_str(), "%2x", &v) == 1;
            pc.key[i] = static_cast<uint8_t>(v);
        }
        if(!key_ok)
        {
            spdlog::critical("--pad-port needs --pad-key with exactly 16 hex characters.");
            exit(-1);
        }
        const std::string allow = vm["pad-allow-ip"].as<std::string>();
        if(!allow.empty())
        {
            in_addr a;
            if(inet_pton(AF_INET, allow.c_str(), &a) != 1)
            {
                spdlog::critical("--pad-allow-ip: not an IPv4 address.");
                exit(-1);
            }
            pc.allow_ip = a.s_addr;
        }
        pc.timeout_s = vm["pad-timeout-ms"].as<int>() / 1000.0;
        pc.deadman = vm.count("pad-no-deadman") == 0;
        try {
            g_pad = std::make_shared<PadInput>(pc);
        } catch(const std::exception& e) {
            spdlog::critical("{}", e.what());
            exit(-1);
        }
        spdlog::info("Pad input on UDP {} (allow-ip {}, timeout {} ms, deadman {}). The Unitree remote still works.",
                     pc.port, allow.empty() ? "any" : allow, vm["pad-timeout-ms"].as<int>(), pc.deadman ? "L1" : "OFF");
    }

    FSMState::lowcmd->msg_.mode_machine() = 5; // 29dof
    if(!FSMState::lowcmd->check_mode_machine(FSMState::lowstate)) {
        spdlog::critical("Unmatched robot type.");
        exit(-1);
    }
    
    // Initialize FSM
    auto fsm = std::make_unique<CtrlFSM>(param::config["FSM"]);
    fsm->start();

    std::cout << "Press [L2 + Up] to enter FixStand mode.\n";
    std::cout << "And then press [R1 + X] to start controlling the robot.\n";

    while (true)
    {
        sleep(1);
    }
    
    return 0;
}

