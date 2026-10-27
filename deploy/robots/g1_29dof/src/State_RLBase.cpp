#include "FSM/State_RLBase.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include <unordered_map>
#include <vector>
#include <cmath>

static std::vector<float> current_command()
{
    // Define each command with their correlated key.
    static std::unordered_map <std::string, std::vector<float>> key_commands = {
        // Forward command.
        {
            "w",
            {
                1.0f,
                0.0f,
                0.0f
            },
        },
        // Backward command.
        {
            "s",
            {
                -1.0f,
                0.0f,
                0.0f
            },
        },
        // Left side-step command.
        {
            "a",
            {
                0.0f,
                1.0f,
                0.0f
            }
        },
        // Right side-step command.
        {
            "d",
            {
                0.0f,
                -1.0f,
                0.0f
            }
        },
        // Left turn command.
        {
            "q",
            {
                0.0f,
                0.0f,
                1.0f
            }
        },
        // Right turn command.
        {
            "e",
            {
                0.0f,
                0.0f,
                -1.0f
            }
        }
    };
    // Return an empty command when no keyboard state is given.
    if(!FSMState::keyboard)
    {
        return{
            0.0f, 0.0f, 0.0f
        };
    }
    // Collect given key
    std::string key = FSMState::keyboard->key();
    // Change the key 
    auto k_cmd = key_commands.find(key);
    // If the key is not defined in key commands.
    if(k_cmd != key_commands.end())
    {
        return k_cmd->second;
    }
    return {0.0f, 0.0f, 0.0f};
}

static float command_norm(const std::vector<float>& cmd)
{
    float cmd_norm = std::sqrt(cmd[0] * cmd[0] + cmd[1] * cmd[1] + cmd[2] * cmd[2]);
    return cmd_norm;
}

namespace isaaclab
{
// keyboard velocity commands example
// change "velocity_commands" observation name in policy deploy.yaml to "keyboard_velocity_commands"
REGISTER_OBSERVATION(keyboard_velocity_commands)
{
    return current_command();
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
    // Check of current state.
    if(state_string == "Perpetual")
    {
        // Switch to Velocity state when command norm is more than 0.15.
        this->registered_checks.emplace_back(
            std::make_pair(
                [this]()->bool{
                    float cmd_norm = command_norm(current_command());
                    return cmd_norm>0.15f;
                },
                FSMStringMap.right.at("Velocity")
            )
        );
    }
    else if(state_string == "Velocity")
    {
        // Switch to Perpetual state when command norm is less than 0.15
        // for longer than defined IDLE_TIME.
        this->registered_checks.emplace_back(
            std::make_pair(
                [this]()->bool{
                    float cmd_norm = command_norm(current_command());
                    if(cmd_norm>0.1f)
                    {
                        idle_started = false;
                        return false;
                    }
                    else {
                        if(!idle_started)
                        {
                            // Initiate idle_start time.
                            idle_start = std::chrono::steady_clock::now();
                            idle_started = true;
                        }
                        const double len_quiet = std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - idle_start
                        ).count();
                        return len_quiet >= IDLE_TIME;
                    }
                },
                FSMStringMap.right.at("Perpetual")
            )
        );
    }
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
