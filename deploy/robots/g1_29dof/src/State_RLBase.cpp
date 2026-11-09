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
    is_episodic = env->cfg["episodic"].IsDefined();
    if(is_episodic)
    {
        // Switch back to Velocity policy.
        this->registered_checks.emplace_back(
            std::make_pair(
                [this]()->bool{
                    if(!first_step_done)
                    {
                        return false;
                    }
                    const auto episodic = env->cfg["episodic"];
                    const float dur = episodic["clip_durations"][env->clip_index].as<float>();
                    float end = dur;
                    // Set end as cancellation time if it is defined and more than 0. 
                    if(env->cancel_time >= 0.0f)
                    {
                        end = std::min(env->cancel_time, dur);
                    }
                    const float tail = episodic["ease_out_time"].as<float>() + episodic["hold_time"].as<float>();
                    if(env->clip_time() < end + tail)
                    {
                        return false;
                    }
                    recenter_next = true;
                    return true;
                },
                FSMStringMap.right.at("Velocity")
            )
        );
        // Cancel button.
        if(cfg["cancel"])
        {
            unitree::common::dsl::Parser parser(cfg["cancel"].as<std::string>());
            auto ast = parser.Parse();
            // Define if button is pressed.
            auto cancel_pressed = unitree::common::dsl::Compile(*ast);
            this->registered_checks.emplace_back(
                std::make_pair(
                    // Check if clip is cancelled and set the cancellation time if the case.
                    [this, cancel_pressed]()->bool{
                        if(
                            first_step_done
                            && env->cancel_time < 0.0f
                            && cancel_pressed(FSMState::lowstate->joystick)
                        )
                        {
                            env->cancel_time = env->clip_time();
                        }
                        return false;
                    },
                    FSMStringMap.right.at("Velocity")
                )
            );
        }
    }
    // Animation trigger.
    else if(cfg["animation_triggers"])
    {
        // Set conditions per clip.
        const auto conditions = cfg["animation_triggers"].as<std::vector<std::string>>();
        for(
            int i = 0;
            i < static_cast<int>(conditions.size());
            ++i
        )
        {
            unitree::common::dsl::Parser parser(conditions[i]);
            auto ast = parser.Parse();
            auto pressed = unitree::common::dsl::Compile(*ast);
            this->registered_checks.emplace_back(
                std::make_pair(
                    [this, i, pressed]()->bool{
                        const auto now = std::chrono::steady_clock::now();
                        // Save command request.
                        if(pressed(FSMState::lowstate->joystick))
                        {
                            pending_clip = i;
                            pending_since = now;
                        }
                        // Return false if clip doesn't align to current i,
                        // or the delayed steps are not yet finished.
                        if(
                            pending_clip != i
                            || !first_step_done
                            || env->still_steps < STILL_STEPS
                        )
                        {
                            return false;
                        }
                        if(std::chrono::duration<double>(now - pending_since).count() > PENDING_TIMEOUT)
                        {
                            pending_clip = -1;
                            return false;
                        }
                        requested_clip = i;
                        pending_clip = -1;
                        return true;
                    },
                    FSMStringMap.right.at("Animation")
                )
            );
        }
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
