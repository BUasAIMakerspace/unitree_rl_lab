#include "FSM/State_RLBase.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include <unordered_map>

namespace isaaclab
{
// keyboard velocity commands example
// change "velocity_commands" observation name in policy deploy.yaml to "keyboard_velocity_commands"
REGISTER_OBSERVATION(keyboard_velocity_commands)
{
    std::string key = FSMState::keyboard->key();
    static auto cfg = env->cfg["commands"]["base_velocity"]["ranges"];

    static std::unordered_map<std::string, std::vector<float>> key_commands = {
        {"w", {1.0f, 0.0f, 0.0f}},
        {"s", {-1.0f, 0.0f, 0.0f}},
        {"a", {0.0f, 1.0f, 0.0f}},
        {"d", {0.0f, -1.0f, 0.0f}},
        {"q", {0.0f, 0.0f, 1.0f}},
        {"e", {0.0f, 0.0f, -1.0f}}
    };
    std::vector<float> cmd = {0.0f, 0.0f, 0.0f};
    if (key_commands.find(key) != key_commands.end())
    {
        // TODO: smooth and limit the velocity commands
        cmd = key_commands[key];
    }
    return cmd;
}

}

State_RLBase::State_RLBase(int state_mode, std::string state_string)
: FSMState(state_mode, state_string) 
{
    auto cfg = param::config["FSM"][state_string];
    auto policy_dir = param::parser_policy_dir(cfg["policy_dir"].as<std::string>());

    env = std::make_unique<isaaclab::ManagerBasedRLEnv>(
        YAML::LoadFile(policy_dir / "params" / "deploy.yaml"),
        std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(FSMState::lowstate)
    );
    env->alg = std::make_unique<isaaclab::OrtRunner>(policy_dir / "exported" / "policy.onnx");

    this->registered_checks.emplace_back(
        std::make_pair(
            [&]()->bool{ return isaaclab::mdp::bad_orientation(env.get(), 1.0); },
            FSMStringMap.right.at("Passive")
        )
    );
}

void State_RLBase::run()
{
    auto& cmd = lowcmd->msg_.motor_cmd();
    if(!first_step_done)
    {
        for(
            size_t i = 0;
            i < from_q.size();
            ++i
        ) {
            cmd[i].q() = from_q[i];
            cmd[i].kp() = from_kp[i];
            cmd[i].kd() = from_kd[i];
        }
        return;
    }

    if(!switch_started)
    {
        switch_start = std::chrono::steady_clock::now();
        switch_started = true;
    }

    const double elaps = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - switch_start
    ).count();
    const float blend = std::min(1.0, elaps/ SWITCH_TIME);

    for(
        size_t i = 0;
        i < from_kp.size();
        ++i
    ) {
        cmd[i].kp() = (1 - blend) * from_kp[i] + blend * env->robot->data.joint_stiffness[i];
        cmd[i].kd() = (1 - blend) * from_kd[i] + blend * env->robot->data.joint_damping[i];
    }
    auto action = env->action_manager->processed_actions();
    for(
        int i = 0;
        i < env->robot->data.joint_ids_map.size();
        i++
    ) {
        int m = env->robot->data.joint_ids_map[i];
        cmd[m].q() = (1 - blend) * from_q[m] + blend * action[i];
    }
    // auto action = env->action_manager->processed_actions();
    // for(int i(0); i < env->robot->data.joint_ids_map.size(); i++) {
    //     lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].q() = action[i];
    // }
}